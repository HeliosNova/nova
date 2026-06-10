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
import os
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
    seed_document: dict | None = None  # {title, source, text} — seeded before retrieval tasks
    # {topic, correct_answer, lesson_text, confidence} — seeded BETWEEN the
    # before/after runs of a memory-learning task to prove the lesson causes
    # the fix. Auto-cleaned after the task (context-marker scoped).
    seed_lesson: dict | None = None
    seed_fact: dict | None = None  # {subject, predicate, object} — KG analogue (wired in WS2C)
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
    # True when the run hit its time budget before finishing. A timed-out task
    # whose partial output still satisfies every assertion counts as a pass
    # (slow-but-correct); otherwise it is outcome=timeout — excluded from
    # correctness denominators, never counted as "wrong".
    timed_out: bool = False
    # memory-learning specific (None for all other categories):
    #   before_correct — did the model answer correctly WITHOUT the seeded lesson
    #   after_correct  — did it answer correctly WITH the lesson in context
    #   caused_fix     — after_correct AND NOT before_correct (the lesson fixed it)
    memory_before_correct: bool | None = None
    memory_after_correct: bool | None = None
    memory_caused_fix: bool | None = None


@dataclass
class _Invocation:
    """Result of one brain.think() run — shared by normal + memory tasks."""
    response_text: str
    tools_invoked: list[str]
    skill_used: str | None
    decomposed: bool
    max_decomposition_depth: int
    reflexion_score: float | None
    latency_seconds: float
    error: str | None
    timed_out: bool = False


@dataclass
class CategoryMetrics:
    category: str
    total: int
    passed: int
    pass_rate: float          # passed / completed — timeouts excluded from the denominator
    latency_p50: float
    latency_p95: float
    timeouts: int = 0         # runs that hit the time budget without proving correctness
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
    # retrieval specific
    retrieval_recall: float | None = None
    # memory-learning specific
    memory_causal_fix_rate: float | None = None   # of testable pairs, fraction the lesson fixed
    memory_already_known_rate: float | None = None  # fraction the base model already knew
    memory_testable: int | None = None              # # pairs where the model did NOT already know


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
    pass_rate: float  # passed / (total - timeouts): correctness over completed runs
    duration_seconds: float
    categories: dict[str, CategoryMetrics]
    task_results: list[TaskResult]
    regressions: list[RegressionFlag]
    baseline_run_id: str | None
    config_snapshot: dict
    timestamp: str
    # Runs that hit the time budget without proving correctness. Excluded
    # from the pass_rate denominator — latency is tracked separately.
    timeouts: int = 0


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
    max_decomposition_depth: int = 0,
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

    if atype == "decomposition_depth_at_least":
        # Verifies recursive decomposition actually fired: at least one sub-agent
        # spawned its own sub-agents. Reads max_decomposition_depth from the DONE event.
        return max_decomposition_depth >= int(assertion["value"])

    if atype == "retrieval_recall":
        # Passes if the seeded fact keyword appears in the response.
        # The harness seeds the document before running the task; if retrieval
        # works correctly, the response should contain the expected term.
        # Pipe-delimited values are treated as alternatives — useful when the
        # seed doc uses synonyms (e.g. 'parameter|weight' for the gradient
        # descent task: both words appear in the seed and either confirms
        # retrieval surfaced the doc).
        target = assertion["value"]
        rl = response.lower()
        if "|" in target:
            return any(t.strip().lower() in rl for t in target.split("|") if t.strip())
        return target.lower() in rl

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
    if atype == "decomposition_depth_at_least":
        return f"decomposition_depth_at_least({assertion['value']}) — observed depth was insufficient"
    if atype == "retrieval_recall":
        snippet = response[:80].replace("\n", " ")
        return f"retrieval_recall({assertion['value']!r}) — not found in: {snippet!r}"
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
        # A timed-out run proves nothing about correctness — it is neither
        # right nor wrong. All rate denominators use completed runs only;
        # latency stats keep every run (a timeout IS latency truth).
        # (passed implies completed: slow-but-correct is a pass, not a timeout.)
        completed = [r for r in cat_results if r.passed or not r.timed_out]
        n_completed = len(completed)
        passed = sum(1 for r in cat_results if r.passed)
        latencies = [r.latency_seconds for r in cat_results]
        scores = [r.reflexion_score for r in completed if r.reflexion_score is not None]

        cm = CategoryMetrics(
            category=cat,
            total=total,
            passed=passed,
            pass_rate=passed / n_completed if n_completed else 0.0,
            latency_p50=percentile(latencies, 50),
            latency_p95=percentile(latencies, 95),
            timeouts=total - n_completed,
        )

        if scores:
            cm.reflexion_mean = statistics.mean(scores)
            cm.reflexion_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            cm.reflexion_p10 = percentile(scores, 10)
            cm.reflexion_p90 = percentile(scores, 90)

        # Category-specific supplemental metrics (over completed runs — a cut-off
        # run never received its DONE event, so skill_used/decomposed are unknown)
        if cat == "skill-match":
            skill_hits = sum(1 for r in completed if r.skill_used is not None)
            cm.hit_rate = skill_hits / n_completed if n_completed else 0.0

        if cat == "semantic-match":
            skill_hits = sum(1 for r in completed if r.skill_used is not None)
            cm.recall_at_threshold = skill_hits / n_completed if n_completed else 0.0

        if cat == "autonomous-tool":
            multi_tool = sum(1 for r in completed if len(r.tools_invoked) >= 2)
            cm.multi_tool_rate = multi_tool / n_completed if n_completed else 0.0

        if cat == "multi-agent":
            decomposed = sum(1 for r in completed if r.decomposed)
            cm.decomposition_rate = decomposed / n_completed if n_completed else 0.0

        if cat == "retrieval":
            # retrieval_recall = fraction of tasks where the seeded fact was found
            # Uses pass_rate as proxy (each retrieval task has a retrieval_recall assertion)
            cm.retrieval_recall = cm.pass_rate

        if cat in ("memory-learning", "kg-retrieval"):
            # The headline proof: of the pairs where the base model did NOT
            # already know the answer, what fraction did the seeded lesson fix?
            # (Pairs the model already knew are excluded from the denominator
            # and reported separately as already_known_rate.)
            testable = [r for r in cat_results if r.memory_before_correct is False]
            fixed = [r for r in testable if r.memory_caused_fix]
            cm.memory_testable = len(testable)
            cm.memory_causal_fix_rate = (len(fixed) / len(testable)) if testable else None
            already = sum(1 for r in cat_results if r.memory_before_correct is True)
            cm.memory_already_known_rate = already / total if total else 0.0

        metrics[cat] = cm
    return metrics


# ---------------------------------------------------------------------------
# Regression detection (pure function)
# ---------------------------------------------------------------------------

def detect_regressions(
    current: dict[str, CategoryMetrics],
    baseline: dict,
    tolerance: float | dict[str, float] = 0.10,
) -> list[RegressionFlag]:
    """Compare current metrics against baseline dict.

    Flags any metric where current < baseline - tolerance (i.e. a drop
    beyond the allowed tolerance).  Only downward regressions are flagged
    (improvements are not penalised).

    Args:
        current:   CategoryMetrics keyed by category name.
        baseline:  Raw dict from a previous report's "categories" section.
        tolerance: Either a float (uniform tolerance for all categories) or
                   a dict mapping category-name to per-category tolerance.
                   Missing categories in the dict fall back to the 0.10 default.

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
        ("retrieval_recall", "retrieval_recall"),
        ("memory_causal_fix_rate", "memory_causal_fix_rate"),
        ("reflexion_mean", "reflexion_mean"),
    ]

    def _tol_for(cat: str) -> float:
        if isinstance(tolerance, dict):
            return float(tolerance.get(cat, tolerance.get("__default__", 0.10)))
        return float(tolerance)

    for cat, cm in current.items():
        baseline_cat = baseline.get("categories", {}).get(cat, {})
        if not baseline_cat:
            continue

        cat_tol = _tol_for(cat)

        for attr, key in _metric_keys:
            current_val = getattr(cm, attr, None)
            baseline_val = baseline_cat.get(key)
            if current_val is None or baseline_val is None:
                continue

            delta = current_val - float(baseline_val)
            flagged = delta < -cat_tol

            flags.append(RegressionFlag(
                metric=f"{cat}.{key}",
                baseline=float(baseline_val),
                current=current_val,
                delta=delta,
                tolerance=cat_tol,
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
    lines.append(f"| Timeouts (excluded from pass rate) | {report.timeouts} |")
    lines.append(f"| Pass rate (over completed) | {report.pass_rate:.1%} |")
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
    lines.append("| Category | Pass | Timeout | Total | Rate | Latency P50 | Latency P95 | Reflexion mean |")
    lines.append("|----------|------|---------|-------|------|-------------|-------------|----------------|")
    for cat, cm in report.categories.items():
        ref = f"{cm.reflexion_mean:.2f}" if cm.reflexion_mean is not None else "—"
        lines.append(
            f"| {cat} | {cm.passed} | {cm.timeouts} | {cm.total}"
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
        if cm.retrieval_recall is not None:
            extras.append(f"retrieval_recall={cm.retrieval_recall:.1%}")
        if cm.memory_causal_fix_rate is not None:
            extras.append(
                f"causal_fix_rate={cm.memory_causal_fix_rate:.1%} "
                f"(testable={cm.memory_testable})"
            )
        if cm.memory_already_known_rate is not None:
            extras.append(f"already_known={cm.memory_already_known_rate:.1%}")
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

    # Prompt-module health section
    module_versions = report.config_snapshot.get("prompt_module_versions", {})
    if module_versions:
        lines.append("## Prompt Module Health")
        lines.append("| Module | Active version |")
        lines.append("|--------|---------------|")
        for mod, ver in sorted(module_versions.items()):
            marker = " (baseline)" if ver == 1 else ""
            lines.append(f"| {mod} | v{ver}{marker} |")
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
        _default_tol = (
            regression_tolerance
            if regression_tolerance is not None
            else config.EVAL_REGRESSION_TOLERANCE
        )
        # Per-category tolerance map (added 2026-05-13, task #33).
        # Read once at init from `categories_meta` in suite.yaml. Categories
        # not listed fall back to `__default__` which is the global config
        # value (or the explicit kwarg). detect_regressions consults this dict.
        cat_tols: dict[str, float] = {"__default__": float(_default_tol)}
        try:
            with open(self.suite_path, encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
            for cat, meta in (doc.get("categories_meta") or {}).items():
                if isinstance(meta, dict) and "regression_tolerance" in meta:
                    cat_tols[cat] = float(meta["regression_tolerance"])
        except Exception as e:
            logger.warning("[EvalHarness] categories_meta parse skipped: %s", e)
        self.tolerance: float | dict[str, float] = (
            cat_tols if len(cat_tols) > 1 else float(_default_tol)
        )
        # Shadow-eval module overrides (set via set_module_overrides())
        self._module_overrides: dict[str, str] = {}
        self._scoring_module_overrides: dict[str, str] = {}

    def set_module_overrides(
        self,
        overrides: dict[str, str],
        scoring_overrides: dict[str, str] | None = None,
    ) -> None:
        """Inject prompt-module overrides for all tasks in this run.

        Used by shadow-eval: the candidate module content is injected via
        ContextVar so brain.think() (and all callees) see it without any
        global state mutation.  scoring_overrides pins the critique version
        used for internal reflexion scoring to the baseline (Goodhart firewall).
        """
        self._module_overrides = overrides
        self._scoring_module_overrides = scoring_overrides or {}

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

            # Timeout multiplier — heavier base models (27B) need longer
            # per-task budgets than the suite was originally sized for (9B).
            # Set EVAL_TIMEOUT_MULTIPLIER=2.5 when running on Qwen 3.6:27b
            # or any other large base. Default 1.0 keeps 9B behavior.
            try:
                _mult = float(os.environ.get("EVAL_TIMEOUT_MULTIPLIER", "1.0"))
            except (TypeError, ValueError):
                _mult = 1.0
            _raw_timeout = raw.get("timeout", 60)
            _scaled_timeout = max(int(_raw_timeout * _mult), int(_raw_timeout))
            tasks.append(EvalTask(
                id=raw["id"],
                category=raw["category"],
                query=raw["query"],
                assertions=raw.get("assertions", []),
                timeout=_scaled_timeout,
                seed_skill=seed,
                seed_document=raw.get("seed_document"),
                seed_lesson=raw.get("seed_lesson"),
                seed_fact=raw.get("seed_fact"),
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

    # --- Document seeding ---

    async def _seed_documents(self, tasks: list[EvalTask]) -> None:
        """Ingest any eval documents listed in task seed_document fields.

        Idempotent: Retriever.ingest() uses doc_id to delete and re-insert,
        so re-running does not accumulate duplicates.
        """
        from app.core.retriever import Retriever

        retriever = Retriever()
        seen_titles: set[str] = set()
        for task in tasks:
            doc = task.seed_document
            if not doc or doc.get("title", "") in seen_titles:
                continue
            seen_titles.add(doc.get("title", ""))
            try:
                doc_id, n_chunks = await retriever.ingest(
                    text=doc["text"],
                    title=doc.get("title", "eval-seed"),
                    source=doc.get("source", "eval"),
                )
                logger.debug(
                    "[EvalHarness] Seeded document %r → %d chunk(s) (doc_id=%s)",
                    doc.get("title"), n_chunks, doc_id,
                )
            except Exception as e:
                logger.warning(
                    "[EvalHarness] Document seed failed for %r: %s", doc.get("title"), e
                )

    # --- Single task execution ---

    async def _invoke_brain(self, query: str, timeout: int) -> _Invocation:
        """Run one query through the real brain (ephemeral) and collect signals.

        Shared by run_task (normal categories) and _run_memory_task (the
        before/after runs). Honors the shadow-eval ContextVar overrides.
        """
        from app.core.brain import think
        from app.core.reflexion import assess_quality
        from app.schema import EventType
        from app.core.prompt_optimizer import _MODULE_OVERRIDES, _SCORING_OVERRIDES

        tokens: list[str] = []
        tools_invoked: list[str] = []
        skill_used: str | None = None
        decomposed: bool = False
        max_decomposition_depth: int = 0
        error: str | None = None
        timed_out: bool = False

        # Inject shadow-eval module overrides (no-op when empty)
        _token1 = _MODULE_OVERRIDES.set(self._module_overrides)
        _token2 = _SCORING_OVERRIDES.set(self._scoring_module_overrides)

        start = time.monotonic()
        try:
            async with asyncio.timeout(timeout):
                async for event in think(query=query, ephemeral=True):
                    if event.type == EventType.TOKEN:
                        tokens.append(event.data.get("text", ""))
                    elif event.type == EventType.TOOL_USE:
                        tool_name = event.data.get("tool", "")
                        if tool_name:
                            tools_invoked.append(tool_name)
                    elif event.type == EventType.DONE:
                        skill_used = event.data.get("skill_used")
                        decomposed = bool(event.data.get("decomposed", False))
                        max_decomposition_depth = int(event.data.get("max_decomposition_depth", 0) or 0)
        except asyncio.TimeoutError:
            error = f"Timeout after {timeout}s"
            timed_out = True
            logger.warning("[EvalHarness] query timed out after %ss", timeout)
        except Exception as e:
            error = str(e)
            logger.warning("[EvalHarness] brain invocation failed: %s", e)
        finally:
            # Always restore ContextVars regardless of outcome
            _MODULE_OVERRIDES.reset(_token1)
            _SCORING_OVERRIDES.reset(_token2)

        latency = time.monotonic() - start
        response_text = "".join(tokens).strip()

        # Compute heuristic reflexion score (consistent, no LLM call)
        reflexion_score: float | None = None
        if response_text:
            score, _ = assess_quality(
                answer=response_text,
                tool_results=[{"tool": t, "output": ""} for t in tools_invoked],
                max_tool_rounds=config.MAX_TOOL_ROUNDS,
                query=query,
            )
            reflexion_score = round(score, 4)

        return _Invocation(
            response_text=response_text,
            tools_invoked=tools_invoked,
            skill_used=skill_used,
            decomposed=decomposed,
            max_decomposition_depth=max_decomposition_depth,
            reflexion_score=reflexion_score,
            latency_seconds=latency,
            error=error,
            timed_out=timed_out,
        )

    def _evaluate_assertions(self, assertions: list[dict], inv: _Invocation) -> list[str]:
        """Return the list of failed-assertion descriptions for an invocation."""
        if inv.error and not inv.response_text:
            # Hard error with no response — all assertions fail
            return [f"task_error: {inv.error}"]
        failed: list[str] = []
        for a in assertions:
            if not check_assertion(
                a, inv.response_text, inv.tools_invoked, inv.skill_used,
                inv.reflexion_score,
                decomposed=inv.decomposed,
                max_decomposition_depth=inv.max_decomposition_depth,
            ):
                failed.append(
                    format_assertion_failure(
                        a, inv.response_text, inv.tools_invoked, inv.skill_used,
                        inv.reflexion_score,
                    )
                )
        return failed

    async def run_task(self, task: EvalTask) -> TaskResult:
        """Run one eval task through the real brain pipeline."""
        # memory-learning / kg-retrieval tasks use dedicated before/seed/after paths
        if task.category == "kg-retrieval" and task.seed_fact:
            return await self._run_kg_task(task)
        if task.category == "memory-learning" and task.seed_lesson:
            return await self._run_memory_task(task)

        inv = await self._invoke_brain(task.query, task.timeout)
        failed_assertions = self._evaluate_assertions(task.assertions, inv)
        passed = len(failed_assertions) == 0
        # Slow-but-correct counts as a pass; a timeout that prevented proving
        # correctness is outcome=timeout, not a wrong answer.
        timed_out = inv.timed_out and not passed

        # When an eval task fails, write a failure reflexion so the failure
        # injection path has data to retrieve on similar future queries.
        # Without this, the eval harness ran ephemeral=True (which skips
        # post-processing) and failures never surfaced into reflexions —
        # leaving the failure side of the learning loop with 0 records.
        # (memory-learning tasks never reach here — a "before" miss is expected
        # and must not pollute the reflexion store. Timeouts are budget
        # exhaustion, not quality failures — they must not pollute it either.)
        if not passed and not timed_out:
            try:
                from app.core.brain import get_services
                svc = get_services()
                if svc and svc.reflexions:
                    reason = "; ".join(failed_assertions[:3])[:400]
                    svc.reflexions.store(
                        task_summary=task.query[:500],
                        outcome="failure",
                        reflection=f"[eval-harness:{task.id}] {reason}",
                        quality_score=inv.reflexion_score if inv.reflexion_score is not None else 0.3,
                        tools_used=inv.tools_invoked,
                        revision_count=0,
                    )
            except Exception as _e:
                logger.debug("[EvalHarness] failure-reflexion write skipped: %s", _e)

        return TaskResult(
            task_id=task.id,
            category=task.category,
            query=task.query,
            passed=passed,
            response_text=inv.response_text[:2000],  # truncate for report storage
            tools_invoked=list(dict.fromkeys(inv.tools_invoked)),  # dedup, preserve order
            skill_used=inv.skill_used,
            reflexion_score=inv.reflexion_score,
            latency_seconds=round(inv.latency_seconds, 2),
            failed_assertions=failed_assertions,
            error=inv.error,
            decomposed=inv.decomposed,
            timed_out=timed_out,
        )

    # --- Memory-learning task (before / seed lesson / after) ---

    async def _run_memory_task(self, task: EvalTask) -> TaskResult:
        """Prove a seeded lesson CAUSES a corrected answer.

        1. Run the query with the lesson ABSENT      -> before_correct
        2. Seed the lesson (context-marker scoped)
        3. Run the same query with the lesson PRESENT -> after_correct
        4. Delete the seeded lesson
        caused_fix = after_correct AND NOT before_correct.

        Deterministic — no LLM judge. The task's assertions (typically
        answer_contains the seeded fact) are evaluated against BOTH runs.
        """
        from app.core.brain import get_services

        svc = get_services()
        learning = getattr(svc, "learning", None) if svc else None
        seed = task.seed_lesson or {}
        marker = f"eval-mem:{task.id}"
        start = time.monotonic()

        if not learning or not seed.get("correct_answer"):
            return TaskResult(
                task_id=task.id, category=task.category, query=task.query,
                passed=False, response_text="", tools_invoked=[], skill_used=None,
                reflexion_score=None, latency_seconds=0.0,
                failed_assertions=[
                    "memory-learning: learning engine or seed_lesson.correct_answer missing"
                ],
                error="setup_error",
            )

        # Defensive: clear any leftover eval lesson for this task id
        self._purge_eval_lessons(learning, marker)

        # 1. BEFORE — lesson absent
        before = await self._invoke_brain(task.query, task.timeout)
        before_failed = self._evaluate_assertions(task.assertions, before)
        before_correct = bool(before.response_text) and not before_failed

        # 2. SEED — context marker makes cleanup safe (never deletes real lessons)
        try:
            learning.add_knowledge_lesson(
                topic=seed.get("topic", task.id),
                correct_answer=seed["correct_answer"],
                lesson_text=seed.get("lesson_text", ""),
                context=marker,
                confidence=float(seed.get("confidence", 0.95)),
            )
        except Exception as e:
            logger.warning("[EvalHarness] memory seed failed for %s: %s", task.id, e)

        # 3. AFTER — lesson present
        after = await self._invoke_brain(task.query, task.timeout)
        after_failed = self._evaluate_assertions(task.assertions, after)
        after_correct = bool(after.response_text) and not after_failed

        # 4. CLEANUP
        self._purge_eval_lessons(learning, marker)

        latency = time.monotonic() - start

        # A leg that hit its time budget without proving correctness makes the
        # pair UNTESTABLE — we can't distinguish "lesson didn't fix it" from
        # "generation got cut off". Exclude it from causal metrics entirely
        # rather than letting budget exhaustion masquerade as a failed fix.
        pair_timed_out = (before.timed_out and not before_correct) or (
            after.timed_out and not after_correct
        )
        if pair_timed_out:
            legs = []
            if before.timed_out and not before_correct:
                legs.append("before")
            if after.timed_out and not after_correct:
                legs.append("after")
            return TaskResult(
                task_id=task.id,
                category=task.category,
                query=task.query,
                passed=False,
                response_text=after.response_text[:2000],
                tools_invoked=list(dict.fromkeys(after.tools_invoked)),
                skill_used=after.skill_used,
                reflexion_score=after.reflexion_score,
                latency_seconds=round(latency, 2),
                failed_assertions=[
                    f"memory-learning: {'/'.join(legs)} run timed out — pair untestable"
                ],
                error=after.error or before.error,
                decomposed=after.decomposed,
                timed_out=True,
                memory_before_correct=None,
                memory_after_correct=None,
                memory_caused_fix=None,
            )

        caused_fix = bool(after_correct and not before_correct)

        notes = [
            f"before_correct={before_correct} after_correct={after_correct} "
            f"caused_fix={caused_fix}"
        ]
        report_failures = ([] if after_correct else after_failed) + notes

        return TaskResult(
            task_id=task.id,
            category=task.category,
            query=task.query,
            passed=after_correct,
            response_text=after.response_text[:2000],
            tools_invoked=list(dict.fromkeys(after.tools_invoked)),
            skill_used=after.skill_used,
            reflexion_score=after.reflexion_score,
            latency_seconds=round(latency, 2),
            failed_assertions=report_failures,
            error=after.error if not after.response_text else None,
            decomposed=after.decomposed,
            memory_before_correct=before_correct,
            memory_after_correct=after_correct,
            memory_caused_fix=caused_fix,
        )

    @staticmethod
    def _purge_eval_lessons(learning, marker: str) -> None:
        """Delete only lessons tagged with our eval context marker.

        Scoped by the exact `context` marker so real user lessons are never
        touched, even if a seed deduped against an existing lesson.
        """
        try:
            rows = learning._db.fetchall(
                "SELECT id FROM lessons WHERE context = ?", (marker,)
            )
            for row in rows:
                learning.delete_lesson(int(row["id"]))
        except Exception as e:
            logger.warning("[EvalHarness] eval lesson cleanup (%s) skipped: %s", marker, e)

    # --- KG-retrieval task (before / seed fact / after) ---

    async def _run_kg_task(self, task: EvalTask) -> TaskResult:
        """Prove a seeded KNOWLEDGE-GRAPH fact is retrieved + used to fix an answer.

        Same shape as _run_memory_task but seeds a (subject, predicate, object)
        triple via kg.add_fact and retires it via kg.delete_fact (which sets
        valid_to, so the fact is auto-excluded from future retrieval — clean
        isolation). Exercises get_relevant_facts → KG prompt injection.
        """
        from app.core.brain import get_services

        svc = get_services()
        kg = getattr(svc, "kg", None) if svc else None
        seed = task.seed_fact or {}
        s, p, o = seed.get("subject"), seed.get("predicate"), seed.get("object")
        start = time.monotonic()

        if not kg or not (s and p and o):
            return TaskResult(
                task_id=task.id, category=task.category, query=task.query,
                passed=False, response_text="", tools_invoked=[], skill_used=None,
                reflexion_score=None, latency_seconds=0.0,
                failed_assertions=["kg-retrieval: kg engine or seed_fact (subject/predicate/object) missing"],
                error="setup_error",
            )

        async def _clean():
            try:
                await kg.delete_fact(s, p, o)
            except Exception as e:
                logger.warning("[EvalHarness] kg cleanup failed for %s: %s", task.id, e)

        await _clean()  # defensive pre-clean

        # 1. BEFORE — fact absent
        before = await self._invoke_brain(task.query, task.timeout)
        before_failed = self._evaluate_assertions(task.assertions, before)
        before_correct = bool(before.response_text) and not before_failed

        # 2. SEED the KG triple
        try:
            await kg.add_fact(s, p, o, confidence=float(seed.get("confidence", 0.95)),
                              source="eval", provenance="eval-kg")
        except Exception as e:
            logger.warning("[EvalHarness] kg seed failed for %s: %s", task.id, e)

        # 3. AFTER — fact present
        after = await self._invoke_brain(task.query, task.timeout)
        after_failed = self._evaluate_assertions(task.assertions, after)
        after_correct = bool(after.response_text) and not after_failed

        # 4. CLEANUP (retire the seeded fact)
        await _clean()

        latency = time.monotonic() - start

        # Same untestable rule as memory-learning: a timed-out leg means the
        # pair proves nothing about the seeded fact.
        pair_timed_out = (before.timed_out and not before_correct) or (
            after.timed_out and not after_correct
        )
        if pair_timed_out:
            legs = []
            if before.timed_out and not before_correct:
                legs.append("before")
            if after.timed_out and not after_correct:
                legs.append("after")
            return TaskResult(
                task_id=task.id,
                category=task.category,
                query=task.query,
                passed=False,
                response_text=after.response_text[:2000],
                tools_invoked=list(dict.fromkeys(after.tools_invoked)),
                skill_used=after.skill_used,
                reflexion_score=after.reflexion_score,
                latency_seconds=round(latency, 2),
                failed_assertions=[
                    f"kg-retrieval: {'/'.join(legs)} run timed out — pair untestable"
                ],
                error=after.error or before.error,
                decomposed=after.decomposed,
                timed_out=True,
                memory_before_correct=None,
                memory_after_correct=None,
                memory_caused_fix=None,
            )

        caused_fix = bool(after_correct and not before_correct)
        notes = [
            f"before_correct={before_correct} after_correct={after_correct} "
            f"caused_fix={caused_fix}"
        ]
        report_failures = ([] if after_correct else after_failed) + notes

        return TaskResult(
            task_id=task.id,
            category=task.category,
            query=task.query,
            passed=after_correct,
            response_text=after.response_text[:2000],
            tools_invoked=list(dict.fromkeys(after.tools_invoked)),
            skill_used=after.skill_used,
            reflexion_score=after.reflexion_score,
            latency_seconds=round(latency, 2),
            failed_assertions=report_failures,
            error=after.error if not after.response_text else None,
            decomposed=after.decomposed,
            memory_before_correct=before_correct,
            memory_after_correct=after_correct,
            memory_caused_fix=caused_fix,
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

        # Seed eval skills and documents into live stores before testing
        self._seed_skills(tasks)
        await self._seed_documents(tasks)

        # Run tasks sequentially to avoid confounding latency metrics
        task_results: list[TaskResult] = []
        for i, task in enumerate(tasks, 1):
            logger.info(
                "[EvalHarness] Task %d/%d: %s (%s)", i, len(tasks), task.id, task.category
            )
            result = await self.run_task(task)
            task_results.append(result)
            status = "PASS" if result.passed else ("TIMEOUT" if result.timed_out else "FAIL")
            logger.info(
                "[EvalHarness] %s %s (score=%.2f, %.1fs)",
                status, task.id,
                result.reflexion_score or 0.0, result.latency_seconds,
            )

        duration = time.monotonic() - start_ts
        passed = sum(1 for r in task_results if r.passed)
        timeouts = sum(1 for r in task_results if r.timed_out and not r.passed)
        failed = len(task_results) - passed - timeouts
        completed = len(task_results) - timeouts

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
        from app.core.prompt_optimizer import PromptModuleStore as _PMS
        try:
            _active_versions = _PMS().get_active_versions()
        except Exception:
            _active_versions = {}
        cfg_snap = {
            "ENABLE_SEMANTIC_SKILL_MATCHING": config.ENABLE_SEMANTIC_SKILL_MATCHING,
            "SKILL_SEMANTIC_THRESHOLD": config.SKILL_SEMANTIC_THRESHOLD,
            "ENABLE_EVAL_HARNESS": config.ENABLE_EVAL_HARNESS,
            "EVAL_REGRESSION_TOLERANCE": config.EVAL_REGRESSION_TOLERANCE,
            "LLM_MODEL": config.LLM_MODEL,
            "LLM_PROVIDER": config.LLM_PROVIDER,
            "prompt_module_versions": _active_versions,
        }

        report = EvalReport(
            run_id=run_id,
            suite_path=str(self.suite_path),
            suite_version=suite_ver,
            total_tasks=len(task_results),
            passed=passed,
            failed=failed,
            skipped=0,
            timeouts=timeouts,
            pass_rate=passed / completed if completed else 0.0,
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
            "timeouts": report.timeouts,
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
