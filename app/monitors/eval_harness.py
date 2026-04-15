"""Automated evaluation harness for Nova.

Loads a YAML task suite, runs each task through the real brain.think()
pipeline (ephemeral — no history written), computes per-category quality
metrics, detects regressions vs the previous run, and writes structured
JSON + human-readable Markdown reports.

Designed to run as a scheduled heartbeat monitor (check_type="eval").
Never mocks the core reasoning path.  Only external network calls
(web_search to live SearXNG) require the real service; calculator and
code_exec are self-contained.

Report files written to EVAL_REPORT_PATH:
  eval_<timestamp>.json     — full structured report
  eval_<timestamp>.md       — human-readable summary
  eval_history.jsonl        — one-line-per-run time-series log (appended)
  eval_baseline.json        — baseline for regression comparison (written on
                              first run; update manually to set a new baseline)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalTask:
    id: str
    category: str
    query: str
    assertions: list[dict]
    timeout: int = 60
    seed_skill: dict | None = None
    paraphrase_of: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    task_id: str
    category: str
    query: str
    passed: bool
    response_text: str
    tools_invoked: list[str]
    skill_used: str | None
    reflexion_score: float | None
    latency_seconds: float
    failed_assertions: list[str]
    error: str | None = None
    decomposed: bool = False  # True when structural multi-agent decomposition fired


@dataclass
class CategoryMetrics:
    category: str
    total: int
    passed: int
    pass_rate: float
    latency_p50: float
    latency_p95: float
    reflexion_mean: float | None = None
    reflexion_std: float | None = None
    reflexion_p10: float | None = None
    reflexion_p90: float | None = None
    # skill-match specific
    hit_rate: float | None = None
    # semantic-match specific
    recall_at_threshold: float | None = None
    # autonomous-tool specific
    multi_tool_rate: float | None = None
    # multi-agent specific
    decomposition_rate: float | None = None


@dataclass
class RegressionFlag:
    metric: str
    baseline: float
    current: float
    delta: float
    tolerance: float
    flagged: bool


@dataclass
class EvalReport:
    run_id: str
    suite_path: str
    suite_version: str
    total_tasks: int
    passed: int
    failed: int
    skipped: int
    pass_rate: float
    duration_seconds: float
    categories: dict[str, CategoryMetrics]
    task_results: list[TaskResult]
    regressions: list[RegressionFlag]
    baseline_run_id: str | None
    config_snapshot: dict
    timestamp: str


# ---------------------------------------------------------------------------
# Assertion evaluation
# ---------------------------------------------------------------------------

def check_assertion(
    assertion: dict,
    response: str,
    tools_invoked: list[str],
    skill_used: str | None,
    reflexion_score: float | None,
    decomposed: bool = False,
) -> bool:
    """Return True if the assertion passes."""
    atype = assertion.get("type", "")

    if atype == "answer_contains":
        return assertion["value"].lower() in response.lower()

    if atype == "answer_matches":
        return bool(re.search(assertion["value"], response, re.IGNORECASE))

    if atype == "answer_not_contains":
        return assertion["value"].lower() not in response.lower()

    if atype == "tool_invoked":
        return assertion["value"] in tools_invoked

    if atype == "no_tool_invoked":
        return len(tools_invoked) == 0

    if atype == "skill_matched":
        return skill_used is not None

    if atype == "skill_name_matches":
        return skill_used is not None and bool(
            re.search(assertion["value"], skill_used, re.IGNORECASE)
        )

    if atype == "reflexion_in_range":
        if reflexion_score is None:
            return False
        return float(assertion["min"]) <= reflexion_score <= float(assertion["max"])

    if atype == "reflexion_above":
        return reflexion_score is not None and reflexion_score >= float(assertion["value"])

    if atype == "reflexion_below":
        return reflexion_score is not None and reflexion_score < float(assertion["value"])

    if atype == "response_not_empty":
        return len(response.replace(" ", "").replace("\n", "")) >= 20

    if atype == "decomposition_fired":
        return decomposed

    if atype == "decomposition_not_fired":
        return not decomposed

    logger.warning("[EvalHarness] Unknown assertion type: %r", atype)
    return False


def format_assertion_failure(
    assertion: dict,
    response: str,
    tools_invoked: list[str],
    skill_used: str | None,
    reflexion_score: float | None,
) -> str:
    atype = assertion.get("type", "?")
    if atype == "answer_contains":
        snippet = response[:80].replace("\n", " ")
        return f"answer_contains({assertion['value']!r}) — got: {snippet!r}"
    if atype == "answer_matches":
        snippet = response[:80].replace("\n", " ")
        return f"answer_matches({assertion['value']!r}) — got: {snippet!r}"
    if atype == "tool_invoked":
        return f"tool_invoked({assertion['value']!r}) — tools used: {tools_invoked}"
    if atype == "skill_matched":
        return f"skill_matched — no skill matched (skill_used={skill_used!r})"
    if atype == "skill_name_matches":
        return f"skill_name_matches({assertion['value']!r}) — skill_used={skill_used!r}"
    if atype == "reflexion_in_range":
        return (
            f"reflexion_in_range([{assertion['min']}, {assertion['max']}])"
            f" — got {reflexion_score}"
        )
    if atype == "reflexion_above":
        return f"reflexion_above({assertion['value']}) — got {reflexion_score}"
    if atype == "reflexion_below":
        return f"reflexion_below({assertion['value']}) — got {reflexion_score}"
    if atype == "response_not_empty":
        return f"response_not_empty — response has {len(response.strip())} chars"
    if atype == "decomposition_fired":
        return "decomposition_fired — decomposition did not trigger"
    if atype == "decomposition_not_fired":
        return "decomposition_not_fired — decomposition unexpectedly triggered"
    return f"{atype} failed"


# ---------------------------------------------------------------------------
# Metric computation (pure functions — easy to unit-test)
# ---------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted or unsorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def compute_category_metrics(results: list[TaskResult]) -> dict[str, CategoryMetrics]:
    """Group results by category and compute per-category statistics."""
    by_cat: dict[str, list[TaskResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    metrics: dict[str, CategoryMetrics] = {}
    for cat, cat_results in by_cat.items():
        total = len(cat_results)
        passed = sum(1 for r in cat_results if r.passed)
        latencies = [r.latency_seconds for r in cat_results]
        scores = [r.reflexion_score for r in cat_results if r.reflexion_score is not None]

        cm = CategoryMetrics(
            category=cat,
            total=total,
            passed=passed,
            pass_rate=passed / total if total else 0.0,
            latency_p50=percentile(latencies, 50),
            latency_p95=percentile(latencies, 95),
        )

        if scores:
            cm.reflexion_mean = statistics.mean(scores)
            cm.reflexion_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            cm.reflexion_p10 = percentile(scores, 10)
            cm.reflexion_p90 = percentile(scores, 90)

        # Category-specific supplemental metrics
        if cat == "skill-match":
            skill_hits = sum(1 for r in cat_results if r.skill_used is not None)
            cm.hit_rate = skill_hits / total if total else 0.0

        if cat == "semantic-match":
            skill_hits = sum(1 for r in cat_results if r.skill_used is not None)
            cm.recall_at_threshold = skill_hits / total if total else 0.0

        if cat == "autonomous-tool":
            multi_tool = sum(1 for r in cat_results if len(r.tools_invoked) >= 2)
            cm.multi_tool_rate = multi_tool / total if total else 0.0

        if cat == "multi-agent":
            decomposed = sum(1 for r in cat_results if r.decomposed)
            cm.decomposition_rate = decomposed / total if total else 0.0

        metrics[cat] = cm
    return metrics


# ---------------------------------------------------------------------------
# Regression detection (pure function)
# ---------------------------------------------------------------------------

def detect_regressions(
    current: dict[str, CategoryMetrics],
    baseline: dict,
    tolerance: float = 0.10,
) -> list[RegressionFlag]:
    """Compare current metrics against baseline dict.

    Flags any metric where current < baseline - tolerance (i.e. a drop
    beyond the allowed tolerance).  Only downward regressions are flagged
    (improvements are not penalised).

    Args:
        current:   CategoryMetrics keyed by category name.
        baseline:  Raw dict from a previous report's "categories" section.
        tolerance: Absolute drop allowed before flagging (default 0.10 = 10%).

    Returns:
        List of RegressionFlag objects; only flagged items have flagged=True.
    """
    flags: list[RegressionFlag] = []

    _metric_keys = [
        ("pass_rate", "pass_rate"),
        ("hit_rate", "hit_rate"),
        ("recall_at_threshold", "recall_at_threshold"),
        ("multi_tool_rate", "multi_tool_rate"),
        ("decomposition_rate", "decomposition_rate"),
        ("reflexion_mean", "reflexion_mean"),
    ]

    for cat, cm in current.items():
        baseline_cat = baseline.get("categories", {}).get(cat, {})
        if not baseline_cat:
            continue

        for attr, key in _metric_keys:
            current_val = getattr(cm, attr, None)
            baseline_val = baseline_cat.get(key)
            if current_val is None or baseline_val is None:
                continue

            delta = current_val - float(baseline_val)
            flagged = delta < -tolerance

            flags.append(RegressionFlag(
                metric=f"{cat}.{key}",
                baseline=float(baseline_val),
                current=current_val,
                delta=delta,
                tolerance=tolerance,
                flagged=flagged,
            ))

    return flags


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _report_to_dict(report: EvalReport) -> dict:
    d = asdict(report)
    # Convert CategoryMetrics dict values
    d["categories"] = {k: asdict(v) for k, v in report.categories.items()}
    d["task_results"] = [asdict(r) for r in report.task_results]
    d["regressions"] = [asdict(f) for f in report.regressions]
    return d


def render_markdown(report: EvalReport) -> str:
    """Render a human-readable Markdown summary of the report."""
    lines: list[str] = []

    flagged = [r for r in report.regressions if r.flagged]
    status_icon = "REGRESSION" if flagged else "OK"
    lines.append(f"# Nova Eval Report [{status_icon}]")
    lines.append(f"**Run:** {report.run_id}  ")
    lines.append(f"**Suite:** {report.suite_path} v{report.suite_version}  ")
    lines.append(f"**Timestamp:** {report.timestamp}  ")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total tasks | {report.total_tasks} |")
    lines.append(f"| Passed | {report.passed} |")
    lines.append(f"| Failed | {report.failed} |")
    lines.append(f"| Pass rate | {report.pass_rate:.1%} |")
    lines.append(f"| Duration | {report.duration_seconds:.1f}s |")
    lines.append("")

    if flagged:
        lines.append("## Regressions DETECTED")
        for r in flagged:
            lines.append(
                f"- **{r.metric}**: {r.baseline:.3f} → {r.current:.3f}"
                f" (Δ={r.delta:+.3f}, tol={r.tolerance:.3f})"
            )
        lines.append("")

    lines.append("## Per-Category Results")
    lines.append("| Category | Pass | Total | Rate | Latency P50 | Latency P95 | Reflexion mean |")
    lines.append("|----------|------|-------|------|-------------|-------------|----------------|")
    for cat, cm in report.categories.items():
        ref = f"{cm.reflexion_mean:.2f}" if cm.reflexion_mean is not None else "—"
        lines.append(
            f"| {cat} | {cm.passed} | {cm.total}"
            f" | {cm.pass_rate:.1%}"
            f" | {cm.latency_p50:.1f}s"
            f" | {cm.latency_p95:.1f}s"
            f" | {ref} |"
        )
    lines.append("")

    lines.append("## Supplemental Metrics")
    for cat, cm in report.categories.items():
        extras: list[str] = []
        if cm.hit_rate is not None:
            extras.append(f"hit_rate={cm.hit_rate:.1%}")
        if cm.recall_at_threshold is not None:
            extras.append(f"semantic_recall={cm.recall_at_threshold:.1%}")
        if cm.multi_tool_rate is not None:
            extras.append(f"multi_tool_rate={cm.multi_tool_rate:.1%}")
        if cm.decomposition_rate is not None:
            extras.append(f"decomposition_rate={cm.decomposition_rate:.1%}")
        if cm.reflexion_std is not None:
            extras.append(f"reflexion_std={cm.reflexion_std:.2f}")
        if cm.reflexion_p10 is not None:
            extras.append(f"P10={cm.reflexion_p10:.2f} P90={cm.reflexion_p90:.2f}")
        if extras:
            lines.append(f"**{cat}**: {', '.join(extras)}")
    lines.append("")

    lines.append("## Failed Tasks")
    failed_results = [r for r in report.task_results if not r.passed]
    if not failed_results:
        lines.append("_All tasks passed._")
    else:
        for r in failed_results:
            lines.append(f"### {r.task_id} ({r.category})")
            lines.append(f"**Query:** {r.query}")
            if r.error:
                lines.append(f"**Error:** {r.error}")
            for fa in r.failed_assertions:
                lines.append(f"- {fa}")
            lines.append("")

    lines.append("## All Task Results")
    lines.append("| ID | Cat | Pass | Score | Latency | Tools | Skill |")
    lines.append("|----|-----|------|-------|---------|-------|-------|")
    for r in report.task_results:
        score = f"{r.reflexion_score:.2f}" if r.reflexion_score is not None else "—"
        skill = r.skill_used or "—"
        tools = ",".join(r.tools_invoked) or "—"
        icon = "✓" if r.passed else "✗"
        lines.append(
            f"| {r.task_id} | {r.category} | {icon}"
            f" | {score} | {r.latency_seconds:.1f}s | {tools} | {skill} |"
        )

    lines.append("")
    lines.append(f"*Baseline run: {report.baseline_run_id or 'none (first run)'}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EvalHarness — main class
# ---------------------------------------------------------------------------

class EvalHarness:
    """Loads a YAML eval suite and runs it against the real Nova pipeline."""

    def __init__(
        self,
        suite_path: str | Path | None = None,
        report_dir: str | Path | None = None,
        regression_tolerance: float | None = None,
    ) -> None:
        self.suite_path = Path(suite_path or config.EVAL_SUITE_PATH)
        self.report_dir = Path(report_dir or config.EVAL_REPORT_PATH)
        self.tolerance = (
            regression_tolerance
            if regression_tolerance is not None
            else config.EVAL_REGRESSION_TOLERANCE
        )

    # --- Suite loading ---

    def load_suite(self) -> list[EvalTask]:
        """Parse the YAML suite file into EvalTask objects."""
        with open(self.suite_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        tasks: list[EvalTask] = []
        for raw in doc.get("tasks", []):
            # Normalise seed_skill steps list
            seed = raw.get("seed_skill")
            if seed:
                steps = seed.get("steps", [])
                norm_steps = []
                for step in steps:
                    ns = {
                        "tool": step.get("tool", ""),
                        "args_template": step.get("args_template", {}),
                        "output_key": step.get("output_key", "result"),
                    }
                    norm_steps.append(ns)
                seed = {**seed, "steps": norm_steps}

            tasks.append(EvalTask(
                id=raw["id"],
                category=raw["category"],
                query=raw["query"],
                assertions=raw.get("assertions", []),
                timeout=raw.get("timeout", 60),
                seed_skill=seed,
                paraphrase_of=raw.get("paraphrase_of"),
                tags=raw.get("tags", []),
            ))
        return tasks

    def suite_version(self) -> str:
        """Read the version field from the YAML suite."""
        try:
            with open(self.suite_path, encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            return str(doc.get("version", "unknown"))
        except Exception:
            return "unknown"

    # --- Skill seeding ---

    def _seed_skills(self, tasks: list[EvalTask]) -> None:
        """Create any eval skills listed in task seed_skill fields.

        Idempotent: if a skill with the same name already exists (from a
        prior harness run) the creation is skipped because create_skill()
        performs name-based dedup internally.
        """
        from app.core.brain import get_services

        svc = get_services()
        if not svc.skills:
            logger.warning("[EvalHarness] SkillStore unavailable — skipping skill seeding")
            return

        seen_names: set[str] = set()
        for task in tasks:
            seed = task.seed_skill
            if not seed or seed["name"] in seen_names:
                continue
            seen_names.add(seed["name"])
            try:
                skill_id = svc.skills.create_skill(
                    name=seed["name"],
                    trigger_pattern=seed["trigger_pattern"],
                    steps=seed["steps"],
                    answer_template=seed.get("answer_template"),
                )
                if skill_id:
                    logger.debug("[EvalHarness] Seeded skill %r (id=%s)", seed["name"], skill_id)
                else:
                    logger.debug("[EvalHarness] Skill %r already exists (name dedup)", seed["name"])
            except Exception as e:
                logger.warning("[EvalHarness] Skill seed failed for %r: %s", seed["name"], e)

    # --- Single task execution ---

    async def run_task(self, task: EvalTask) -> TaskResult:
        """Run one eval task through the real brain pipeline."""
        from app.core.brain import think
        from app.core.reflexion import assess_quality
        from app.schema import EventType

        tokens: list[str] = []
        tools_invoked: list[str] = []
        skill_used: str | None = None
        decomposed: bool = False
        error: str | None = None

        start = time.monotonic()
        try:
            async with asyncio.timeout(task.timeout):
                async for event in think(query=task.query, ephemeral=True):
                    if event.type == EventType.TOKEN:
                        tokens.append(event.data.get("text", ""))
                    elif event.type == EventType.TOOL_USE:
                        tool_name = event.data.get("tool", "")
                        if tool_name:
                            tools_invoked.append(tool_name)
                    elif event.type == EventType.DONE:
                        skill_used = event.data.get("skill_used")
                        decomposed = bool(event.data.get("decomposed", False))
        except asyncio.TimeoutError:
            error = f"Timeout after {task.timeout}s"
            logger.warning("[EvalHarness] Task %s timed out", task.id)
        except Exception as e:
            error = str(e)
            logger.warning("[EvalHarness] Task %s failed: %s", task.id, e)

        latency = time.monotonic() - start
        response_text = "".join(tokens).strip()

        # Compute heuristic reflexion score (consistent, no LLM call)
        reflexion_score: float | None = None
        if response_text:
            score, _ = assess_quality(
                answer=response_text,
                tool_results=[{"tool": t, "output": ""} for t in tools_invoked],
                max_tool_rounds=config.MAX_TOOL_ROUNDS,
                query=task.query,
            )
            reflexion_score = round(score, 4)

        # Evaluate assertions
        failed_assertions: list[str] = []
        if error and not response_text:
            # Hard error with no response — all assertions fail
            failed_assertions = [f"task_error: {error}"]
        else:
            for a in task.assertions:
                if not check_assertion(
                    a, response_text, tools_invoked, skill_used, reflexion_score,
                    decomposed=decomposed,
                ):
                    failed_assertions.append(
                        format_assertion_failure(
                            a, response_text, tools_invoked, skill_used, reflexion_score
                        )
                    )

        passed = len(failed_assertions) == 0

        return TaskResult(
            task_id=task.id,
            category=task.category,
            query=task.query,
            passed=passed,
            response_text=response_text[:500],  # truncate for report storage
            tools_invoked=list(dict.fromkeys(tools_invoked)),  # dedup, preserve order
            skill_used=skill_used,
            reflexion_score=reflexion_score,
            latency_seconds=round(latency, 2),
            failed_assertions=failed_assertions,
            error=error,
            decomposed=decomposed,
        )

    # --- Full suite run ---

    async def run_all(self, tasks: list[EvalTask] | None = None) -> EvalReport:
        """Run the full suite and return a complete EvalReport."""
        if tasks is None:
            tasks = self.load_suite()

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suite_ver = self.suite_version()
        start_ts = time.monotonic()

        logger.info("[EvalHarness] Starting eval run %s — %d tasks", run_id, len(tasks))

        # Seed eval skills into live SkillStore before testing
        self._seed_skills(tasks)

        # Run tasks sequentially to avoid confounding latency metrics
        task_results: list[TaskResult] = []
        for i, task in enumerate(tasks, 1):
            logger.info(
                "[EvalHarness] Task %d/%d: %s (%s)", i, len(tasks), task.id, task.category
            )
            result = await self.run_task(task)
            task_results.append(result)
            status = "PASS" if result.passed else "FAIL"
            logger.info(
                "[EvalHarness] %s %s (score=%.2f, %.1fs)",
                status, task.id,
                result.reflexion_score or 0.0, result.latency_seconds,
            )

        duration = time.monotonic() - start_ts
        passed = sum(1 for r in task_results if r.passed)
        failed = len(task_results) - passed

        categories = compute_category_metrics(task_results)

        # Load baseline for regression comparison
        baseline_data, baseline_run_id = self._load_baseline()
        regressions = detect_regressions(categories, baseline_data, self.tolerance)

        flagged = [r for r in regressions if r.flagged]
        if flagged:
            logger.warning(
                "[EvalHarness] %d regression(s) detected: %s",
                len(flagged),
                [r.metric for r in flagged],
            )

        # Config snapshot (only eval-relevant fields)
        cfg_snap = {
            "ENABLE_SEMANTIC_SKILL_MATCHING": config.ENABLE_SEMANTIC_SKILL_MATCHING,
            "SKILL_SEMANTIC_THRESHOLD": config.SKILL_SEMANTIC_THRESHOLD,
            "ENABLE_EVAL_HARNESS": config.ENABLE_EVAL_HARNESS,
            "EVAL_REGRESSION_TOLERANCE": config.EVAL_REGRESSION_TOLERANCE,
            "LLM_MODEL": config.LLM_MODEL,
            "LLM_PROVIDER": config.LLM_PROVIDER,
        }

        report = EvalReport(
            run_id=run_id,
            suite_path=str(self.suite_path),
            suite_version=suite_ver,
            total_tasks=len(task_results),
            passed=passed,
            failed=failed,
            skipped=0,
            pass_rate=passed / len(task_results) if task_results else 0.0,
            duration_seconds=round(duration, 1),
            categories=categories,
            task_results=task_results,
            regressions=regressions,
            baseline_run_id=baseline_run_id,
            config_snapshot=cfg_snap,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return report

    # --- Baseline management ---

    def _baseline_path(self) -> Path:
        return self.report_dir / "eval_baseline.json"

    def _load_baseline(self) -> tuple[dict, str | None]:
        """Load the baseline report.  Returns (data, run_id) or ({}, None)."""
        bp = self._baseline_path()
        if not bp.exists():
            return {}, None
        try:
            with open(bp, encoding="utf-8") as f:
                data = json.load(f)
            return data, data.get("run_id")
        except Exception as e:
            logger.warning("[EvalHarness] Could not load baseline: %s", e)
            return {}, None

    def write_baseline(self, report: EvalReport) -> None:
        """Write this report as the new regression baseline."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        data = _report_to_dict(report)
        with open(self._baseline_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("[EvalHarness] Baseline written to %s", self._baseline_path())

    # --- Report persistence ---

    def write_report(self, report: EvalReport) -> tuple[Path, Path]:
        """Write JSON + Markdown reports.  Returns (json_path, md_path)."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        slug = report.run_id

        json_path = self.report_dir / f"eval_{slug}.json"
        md_path = self.report_dir / f"eval_{slug}.md"

        data = _report_to_dict(report)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown(report))

        logger.info("[EvalHarness] Report written: %s / %s", json_path.name, md_path.name)
        return json_path, md_path

    def append_history(self, report: EvalReport) -> None:
        """Append a one-line summary to the time-series history log."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        history_path = self.report_dir / "eval_history.jsonl"
        flagged = [r.metric for r in report.regressions if r.flagged]
        line = {
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "pass_rate": report.pass_rate,
            "passed": report.passed,
            "failed": report.failed,
            "total": report.total_tasks,
            "duration_s": report.duration_seconds,
            "regressions_flagged": flagged,
            "categories": {
                cat: {
                    "pass_rate": cm.pass_rate,
                    "reflexion_mean": cm.reflexion_mean,
                    "hit_rate": cm.hit_rate,
                    "recall_at_threshold": cm.recall_at_threshold,
                }
                for cat, cm in report.categories.items()
            },
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    # --- Convenience: run and persist in one call ---

    async def run_and_persist(self) -> tuple[EvalReport, Path, Path]:
        """Run the full suite, write reports, update history, write baseline if first run."""
        report = await self.run_all()
        json_path, md_path = self.write_report(report)
        self.append_history(report)

        # Write baseline if this is the first run (no existing baseline)
        if not self._baseline_path().exists():
            self.write_baseline(report)
            logger.info("[EvalHarness] First run — baseline established at %s", self._baseline_path())

        return report, json_path, md_path
