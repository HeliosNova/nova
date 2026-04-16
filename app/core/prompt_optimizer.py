"""Prompt self-modification system — prompt-level optimization only.

This module implements a conservative, eval-gated prompt optimization loop.
It is NOT weight tuning (Nova runs a static GGUF on Ollama). It modifies
instruction strings fed to the model as context.

Architecture:
  1. PromptModuleStore  — SQLite-backed versioned prompt registry
  2. get_active_module()  — load active version; respects ContextVar overrides for shadow-eval
  3. with_module_overrides() — context manager for shadow-eval injection (no global state)
  4. PromptOptimizerAnalyzer — heuristic drift detection + LLM-drafted mutation proposals

SAFETY FIREWALLS (non-negotiable):
  - _SELF_MOD_ALLOWED_MODULES: frozenset gates every write path.
    IDENTITY_AND_REASONING, TOOL_EXAMPLES, safety instructions, and all harness
    prompts are structurally excluded — get_active_module() returns None for them.
  - is_baseline=1 rows are never updated by the self-mod path.
  - META_PROMPT is a hardcoded constant; its SHA-256 hash is checked in CI tests.
  - ENABLE_PROMPT_SELF_MOD=false (default) prevents proposals and promotions;
    the module loader still works so baselines are always readable.

Requires: database table `prompt_modules` (schema in database.py migration 9).
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generator

from app.config import config
from app.database import get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Firewall: explicit allow-list of tunable modules
# ---------------------------------------------------------------------------

#: The only module names that may be written to prompt_modules by the optimizer.
#: Any name NOT in this set is silently ignored by get_active_module() and
#: rejected by PromptModuleStore.write_candidate().
_SELF_MOD_ALLOWED_MODULES: frozenset[str] = frozenset({
    "critique_prompt",           # reflexion.py: _CRITIQUE_PROMPT
    "extraction_prompt",         # learning.py: _EXTRACTION_PROMPT
    "skill_extraction_prompt",   # skills.py: inline system message
    "merge_instruction_parallel",  # decomposer.py: parallel strategy template
    "merge_instruction_sequential",  # decomposer.py: sequential strategy template
    "kg_extraction_prompt",      # brain.py: KG triple extraction template
})

# Harness-internal prompts that must NEVER be allowed (explicit block, belt-and-suspenders)
_HARNESS_INTERNAL_MODULES: frozenset[str] = frozenset({
    "quiz_gen", "quiz_answer", "quiz_grade",
})

# ---------------------------------------------------------------------------
# Shadow-eval ContextVars (no global mutation — safe across async coroutines)
# ---------------------------------------------------------------------------

#: Active module overrides for the current async task (generation path).
_MODULE_OVERRIDES: ContextVar[dict[str, str]] = ContextVar(
    "_module_overrides", default={}
)

#: Scoring-path overrides: pins which prompt version is used when brain.think()
#: calls critique_response() internally, ensuring the harness grades responses
#: with the baseline critique rather than the candidate being tested.
_SCORING_OVERRIDES: ContextVar[dict[str, str]] = ContextVar(
    "_scoring_overrides", default={}
)


@contextmanager
def with_module_overrides(
    overrides: dict[str, str],
    scoring_overrides: dict[str, str] | None = None,
) -> Generator[None, None, None]:
    """Context manager: inject module overrides for one shadow-eval task.

    All code running inside this context (including brain.think() and its
    callees) will see the overridden prompt content instead of the DB version.
    ContextVars are inherited by sub-tasks but NOT shared across unrelated tasks.

    Usage::

        with with_module_overrides({"critique_prompt": candidate_text},
                                   scoring_overrides={"critique_prompt": baseline_text}):
            result = await run_task(...)
    """
    token1 = _MODULE_OVERRIDES.set(overrides)
    token2 = _SCORING_OVERRIDES.set(scoring_overrides or {})
    try:
        yield
    finally:
        _MODULE_OVERRIDES.reset(token1)
        _SCORING_OVERRIDES.reset(token2)


# ---------------------------------------------------------------------------
# Module loader — the single entry point for all callers
# ---------------------------------------------------------------------------

def get_active_module(module_name: str, *, scoring: bool = False) -> str | None:
    """Load the active content for a named prompt module.

    Returns None if:
    - module_name is not in _SELF_MOD_ALLOWED_MODULES  (firewall)
    - module_name is in _HARNESS_INTERNAL_MODULES       (firewall)
    - No active row exists in the DB and no ContextVar override is set

    Callers fall back to their hardcoded constant when None is returned,
    so the system degrades gracefully on first run or when the table is empty.

    Args:
        module_name: Name from _SELF_MOD_ALLOWED_MODULES.
        scoring: If True, checks _SCORING_OVERRIDES before _MODULE_OVERRIDES.
                 Used by critique_response() to enforce the Goodhart firewall:
                 when shadow-testing a critique_prompt candidate, the scoring
                 path uses the pinned baseline, not the candidate.
    """
    # Hard firewall: reject disallowed names silently
    if module_name not in _SELF_MOD_ALLOWED_MODULES:
        return None
    if module_name in _HARNESS_INTERNAL_MODULES:
        return None

    # ContextVar override (shadow-eval injection)
    if scoring:
        cv_overrides = _SCORING_OVERRIDES.get()
    else:
        cv_overrides = _MODULE_OVERRIDES.get()

    if cv_overrides and module_name in cv_overrides:
        return cv_overrides[module_name]

    # DB lookup — active version
    try:
        db = get_db()
        row = db.fetchone(
            "SELECT content FROM prompt_modules "
            "WHERE module_name=? AND status='active' "
            "ORDER BY version DESC LIMIT 1",
            (module_name,),
        )
        return row["content"] if row else None
    except Exception as e:
        logger.warning("prompt_optimizer: DB read failed for %r: %s", module_name, e)
        return None


# ---------------------------------------------------------------------------
# PromptModuleStore — CRUD + lifecycle state machine
# ---------------------------------------------------------------------------

@dataclass
class ModuleVersion:
    """Row from prompt_modules, deserialized for caller convenience."""
    id: int
    module_name: str
    version: int
    content: str
    is_baseline: bool
    status: str  # candidate | active | superseded | rolled_back | quarantined
    parent_version: int | None
    delta_description: str | None
    promoted_at: str | None
    promoted_eval_run_id: str | None
    rolled_back_at: str | None
    quarantined_until: str | None
    shadow_eval_metrics: dict | None
    created_at: str


class PromptModuleStore:
    """Thread-safe CRUD for the prompt_modules table.

    All writes validate module_name against _SELF_MOD_ALLOWED_MODULES before
    touching the database.  is_baseline rows are never modified by any write path.
    """

    def __init__(self, db=None) -> None:
        self._db = db or get_db()

    # --- Baseline seeding (called at startup) ---

    def seed_baseline(self) -> None:
        """Write baseline rows for all allowed modules if not already seeded.

        Idempotent.  Baseline rows have is_baseline=1 and status='active'.
        They are the immutable last-resort fallback.
        """
        from app.core.prompt_optimizer_baselines import MODULE_BASELINES
        for name, content in MODULE_BASELINES.items():
            if name not in _SELF_MOD_ALLOWED_MODULES:
                continue
            existing = self._db.fetchone(
                "SELECT id FROM prompt_modules WHERE module_name=? AND is_baseline=1",
                (name,),
            )
            if not existing:
                self._db.execute(
                    "INSERT INTO prompt_modules "
                    "(module_name, version, content, is_baseline, status) "
                    "VALUES (?, 1, ?, 1, 'active')",
                    (name, content),
                )
                logger.info("prompt_optimizer: seeded baseline for %r", name)

    # --- Reads ---

    def get_active(self, module_name: str) -> ModuleVersion | None:
        """Return active version metadata (not just content)."""
        row = self._db.fetchone(
            "SELECT * FROM prompt_modules WHERE module_name=? AND status='active' "
            "ORDER BY version DESC LIMIT 1",
            (module_name,),
        )
        return self._row_to_mv(row) if row else None

    def get_baseline(self, module_name: str) -> ModuleVersion | None:
        """Return the immutable baseline version."""
        row = self._db.fetchone(
            "SELECT * FROM prompt_modules WHERE module_name=? AND is_baseline=1",
            (module_name,),
        )
        return self._row_to_mv(row) if row else None

    def get_by_id(self, module_id: int) -> ModuleVersion | None:
        row = self._db.fetchone(
            "SELECT * FROM prompt_modules WHERE id=?", (module_id,)
        )
        return self._row_to_mv(row) if row else None

    def get_active_versions(self) -> dict[str, int]:
        """Return {module_name: version} for all active modules."""
        rows = self._db.fetchall(
            "SELECT module_name, version FROM prompt_modules "
            "WHERE status='active' ORDER BY module_name"
        )
        return {r["module_name"]: r["version"] for r in rows}

    def list_candidates(self, module_name: str) -> list[ModuleVersion]:
        rows = self._db.fetchall(
            "SELECT * FROM prompt_modules WHERE module_name=? AND status='candidate' "
            "ORDER BY version",
            (module_name,),
        )
        return [self._row_to_mv(r) for r in rows]

    # --- Safety cap checks ---

    def count_proposals_today(self, module_name: str) -> int:
        row = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM prompt_modules "
            "WHERE module_name=? AND status='candidate' "
            "AND date(created_at)=date('now')",
            (module_name,),
        )
        return row["c"] if row else 0

    def count_pending(self, module_name: str) -> int:
        row = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM prompt_modules "
            "WHERE module_name=? AND status='candidate'",
            (module_name,),
        )
        return row["c"] if row else 0

    def count_promotions_today(self) -> int:
        """System-wide promotion count for today."""
        row = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM prompt_modules "
            "WHERE status='active' AND date(promoted_at)=date('now')"
        )
        return row["c"] if row else 0

    def is_quarantined(self, module_name: str) -> bool:
        row = self._db.fetchone(
            "SELECT id FROM prompt_modules "
            "WHERE module_name=? AND status='quarantined' "
            "AND quarantined_until > datetime('now')",
            (module_name,),
        )
        return row is not None

    # --- Drift check ---

    def compute_drift(self, module_name: str, candidate_content: str) -> float:
        """Word-overlap Jaccard distance from baseline (0.0 = identical, 1.0 = disjoint).

        Uses word-level Jaccard as a proxy for cosine distance when an embedding
        model is unavailable.  Threshold 0.25 (from design doc) allows meaningful
        rephrasing without wholesale replacement.

        Note: embedding-based drift (nomic-embed-text-v2-moe) is a v2 upgrade
        that requires Ollama to be running.
        """
        baseline = self.get_baseline(module_name)
        if not baseline:
            return 0.0
        b_words = set(baseline.content.lower().split())
        c_words = set(candidate_content.lower().split())
        union = len(b_words | c_words)
        if union == 0:
            return 0.0
        return 1.0 - len(b_words & c_words) / union

    # --- Writes ---

    def write_candidate(
        self,
        module_name: str,
        content: str,
        delta_description: str,
    ) -> int | None:
        """Write a new candidate.  Returns row ID, or None if blocked by safety caps.

        Safety checks (in order):
        1. module_name must be in _SELF_MOD_ALLOWED_MODULES
        2. ENABLE_PROMPT_SELF_MOD must be true
        3. Proposals today < PROMPT_MOD_MAX_PROPOSALS_PER_DAY
        4. Pending candidates < PROMPT_MOD_MAX_PENDING
        5. Module must not be quarantined
        6. Drift from baseline <= PROMPT_MOD_MAX_DRIFT
        """
        if module_name not in _SELF_MOD_ALLOWED_MODULES:
            logger.warning(
                "prompt_optimizer: write_candidate rejected — %r not in allow-list", module_name
            )
            return None

        if not config.ENABLE_PROMPT_SELF_MOD:
            logger.debug("prompt_optimizer: write_candidate blocked — ENABLE_PROMPT_SELF_MOD=false")
            return None

        if self.count_proposals_today(module_name) >= config.PROMPT_MOD_MAX_PROPOSALS_PER_DAY:
            logger.info(
                "prompt_optimizer: daily proposal cap reached for %r", module_name
            )
            return None

        if self.count_pending(module_name) >= config.PROMPT_MOD_MAX_PENDING:
            logger.info(
                "prompt_optimizer: max pending candidates reached for %r", module_name
            )
            return None

        if self.is_quarantined(module_name):
            logger.info("prompt_optimizer: %r is quarantined — skipping proposal", module_name)
            return None

        drift = self.compute_drift(module_name, content)
        if drift > config.PROMPT_MOD_MAX_DRIFT:
            logger.warning(
                "prompt_optimizer: drift %.3f > %.3f limit for %r — rejecting",
                drift, config.PROMPT_MOD_MAX_DRIFT, module_name,
            )
            return None

        # Next version number
        row = self._db.fetchone(
            "SELECT MAX(version) AS v FROM prompt_modules WHERE module_name=?",
            (module_name,),
        )
        next_version = (row["v"] or 0) + 1

        # Current active is the parent
        active = self.get_active(module_name)
        parent_version = active.version if active else None

        cursor = self._db.execute(
            "INSERT INTO prompt_modules "
            "(module_name, version, content, parent_version, status, delta_description) "
            "VALUES (?, ?, ?, ?, 'candidate', ?)",
            (module_name, next_version, content, parent_version, delta_description),
        )
        rowid = cursor.lastrowid
        logger.info(
            "prompt_optimizer: candidate written for %r v%d (id=%d drift=%.3f)",
            module_name, next_version, rowid, drift,
        )
        return rowid

    def promote(
        self,
        module_id: int,
        eval_run_id: str,
        metrics: dict,
    ) -> bool:
        """Promote a candidate to active.

        Safety checks:
        - Row must exist with status='candidate'
        - System-wide promotions today < PROMPT_MOD_MAX_PROMOTIONS_PER_DAY
        - ENABLE_PROMPT_SELF_MOD must be true

        Returns True on success.
        """
        if not config.ENABLE_PROMPT_SELF_MOD:
            return False

        row = self._db.fetchone(
            "SELECT * FROM prompt_modules WHERE id=? AND status='candidate'",
            (module_id,),
        )
        if not row:
            return False

        if self.count_promotions_today() >= config.PROMPT_MOD_MAX_PROMOTIONS_PER_DAY:
            logger.info("prompt_optimizer: daily promotion cap reached — cannot promote id=%d", module_id)
            return False

        module_name = row["module_name"]
        metrics_json = json.dumps(metrics)

        with self._db.transaction() as tx:
            # Supersede current active(s)
            tx.execute(
                "UPDATE prompt_modules SET status='superseded' "
                "WHERE module_name=? AND status='active'",
                (module_name,),
            )
            # Promote candidate
            tx.execute(
                "UPDATE prompt_modules SET status='active', "
                "promoted_at=datetime('now'), promoted_eval_run_id=?, "
                "shadow_eval_metrics=? WHERE id=?",
                (eval_run_id, metrics_json, module_id),
            )

        logger.info(
            "prompt_optimizer: promoted %r id=%d (run=%s)", module_name, module_id, eval_run_id
        )
        return True

    def rollback(self, module_id: int) -> bool:
        """Roll back an active module to its parent; quarantine the promoted version.

        Returns True on success.
        """
        row = self._db.fetchone(
            "SELECT * FROM prompt_modules WHERE id=? AND status='active'",
            (module_id,),
        )
        if not row:
            return False

        module_name = row["module_name"]
        parent_version = row["parent_version"]

        with self._db.transaction() as tx:
            # Quarantine the currently-active promoted row
            tx.execute(
                "UPDATE prompt_modules SET status='quarantined', "
                "rolled_back_at=datetime('now'), "
                "quarantined_until=datetime('now', '+1 day') WHERE id=?",
                (module_id,),
            )
            if parent_version is not None:
                # Restore parent to active
                tx.execute(
                    "UPDATE prompt_modules SET status='active' "
                    "WHERE module_name=? AND version=?",
                    (module_name, parent_version),
                )

        logger.warning(
            "prompt_optimizer: rolled back %r id=%d (parent_version=%s)",
            module_name, module_id, parent_version,
        )
        return True

    def quarantine_candidate(self, module_id: int) -> bool:
        """Quarantine a candidate that failed shadow eval."""
        row = self._db.fetchone(
            "SELECT id FROM prompt_modules WHERE id=? AND status='candidate'",
            (module_id,),
        )
        if not row:
            return False
        self._db.execute(
            "UPDATE prompt_modules SET status='quarantined', "
            "quarantined_until=datetime('now', '+1 day') WHERE id=?",
            (module_id,),
        )
        logger.info("prompt_optimizer: quarantined candidate id=%d", module_id)
        return True

    # --- Helpers ---

    @staticmethod
    def _row_to_mv(row) -> ModuleVersion:
        metrics = None
        raw = row["shadow_eval_metrics"]
        if raw:
            try:
                metrics = json.loads(raw)
            except Exception:
                pass
        return ModuleVersion(
            id=row["id"],
            module_name=row["module_name"],
            version=row["version"],
            content=row["content"],
            is_baseline=bool(row["is_baseline"]),
            status=row["status"],
            parent_version=row["parent_version"],
            delta_description=row["delta_description"],
            promoted_at=row["promoted_at"],
            promoted_eval_run_id=row["promoted_eval_run_id"],
            rolled_back_at=row["rolled_back_at"],
            quarantined_until=row["quarantined_until"],
            shadow_eval_metrics=metrics,
            created_at=row["created_at"],
        )


# ---------------------------------------------------------------------------
# Shadow-eval runner — compares candidate vs baseline using EvalHarness
# ---------------------------------------------------------------------------

@dataclass
class ShadowEvalResult:
    """Outcome of one shadow-eval run for a candidate."""
    run_id: str
    candidate_id: int
    module_name: str
    passed: bool
    target_category: str
    baseline_metric: float
    candidate_metric: float
    delta_pp: float
    regressions: list[str] = field(default_factory=list)
    latency_ratio: float = 1.0
    calibration_ok: bool = True
    reason: str = ""


async def run_shadow_eval(
    candidate_id: int,
    target_category: str,
    target_metric: str = "pass_rate",
    store: PromptModuleStore | None = None,
) -> ShadowEvalResult:
    """Run one shadow-eval run for a candidate module.

    Injects the candidate via with_module_overrides() so no live state changes.
    For critique_prompt candidates, scoring_overrides pins the baseline version
    to avoid the Goodhart loop (candidate grades its own shadow outputs).

    Returns a ShadowEvalResult with pass/fail determination.
    """
    from app.monitors.eval_harness import EvalHarness

    if store is None:
        store = PromptModuleStore()

    mv = store.get_by_id(candidate_id)
    if not mv:
        return ShadowEvalResult(
            run_id="",
            candidate_id=candidate_id,
            module_name="",
            passed=False,
            target_category=target_category,
            baseline_metric=0.0,
            candidate_metric=0.0,
            delta_pp=0.0,
            reason=f"candidate id={candidate_id} not found",
        )

    module_name = mv.module_name
    candidate_content = mv.content

    # Goodhart firewall: if we're testing critique_prompt, score with baseline
    baseline_mv = store.get_baseline(module_name)
    scoring_overrides: dict[str, str] = {}
    if module_name == "critique_prompt" and baseline_mv:
        scoring_overrides["critique_prompt"] = baseline_mv.content

    overrides = {module_name: candidate_content}

    run_id = datetime.now(timezone.utc).strftime("shadow_%Y%m%d_%H%M%S")

    harness = EvalHarness()
    harness.set_module_overrides(overrides, scoring_overrides)

    try:
        report = await harness.run_all()
    except Exception as e:
        logger.error("prompt_optimizer: shadow eval failed: %s", e, exc_info=True)
        return ShadowEvalResult(
            run_id=run_id,
            candidate_id=candidate_id,
            module_name=module_name,
            passed=False,
            target_category=target_category,
            baseline_metric=0.0,
            candidate_metric=0.0,
            delta_pp=0.0,
            reason=f"shadow eval exception: {e}",
        )

    # Extract target metric
    cat_metrics = report.categories.get(target_category)
    candidate_metric = 0.0
    if cat_metrics:
        candidate_metric = getattr(cat_metrics, target_metric, 0.0) or 0.0

    # Load baseline metric from stored eval history
    baseline_metric = _load_baseline_metric(target_category, target_metric)
    delta_pp = (candidate_metric - baseline_metric) * 100

    # Check regressions in all non-target categories (>1pp drop disqualifies)
    regression_limit = config.PROMPT_MOD_REGRESSION_TOLERANCE_PP / 100.0
    regressions = []
    for cat, cm in report.categories.items():
        if cat == target_category:
            continue
        bl_val = _load_baseline_metric(cat, target_metric)
        cur_val = getattr(cm, target_metric, 0.0) or 0.0
        if bl_val > 0 and (cur_val - bl_val) < -regression_limit:
            regressions.append(
                f"{cat}.{target_metric}: {bl_val:.3f}→{cur_val:.3f} (Δ={cur_val-bl_val:+.3f})"
            )

    # Calibration check (only for critique_prompt; uses reflexion score distribution)
    calibration_ok = True
    if module_name == "critique_prompt":
        cal_cat = report.categories.get("reflexion-calibration")
        if cal_cat and cal_cat.reflexion_p90 is not None:
            if cal_cat.reflexion_p90 > 0.93 or (cal_cat.reflexion_mean or 0) > 0.80:
                calibration_ok = False
                regressions.append(
                    f"reflexion-calibration: P90={cal_cat.reflexion_p90:.2f} "
                    f"mean={cal_cat.reflexion_mean:.2f} — inflation detected"
                )

    # Latency check
    base_p95 = _load_baseline_latency(target_category)
    cur_p95 = cat_metrics.latency_p95 if cat_metrics else 0.0
    latency_ratio = (cur_p95 / base_p95) if base_p95 > 0 else 1.0

    if latency_ratio > config.PROMPT_MOD_LATENCY_OVERHEAD_MAX:
        regressions.append(
            f"latency P95 overhead: {latency_ratio:.2f}x > {config.PROMPT_MOD_LATENCY_OVERHEAD_MAX:.2f}x limit"
        )

    min_improvement = config.PROMPT_MOD_MIN_IMPROVEMENT_PP
    passed = (
        delta_pp >= min_improvement
        and len(regressions) == 0
        and calibration_ok
    )

    reason = (
        f"delta={delta_pp:+.2f}pp on {target_category}.{target_metric}"
        + (f" | regressions: {regressions}" if regressions else "")
        + (f" | latency_ratio={latency_ratio:.2f}x" if latency_ratio > 1.0 else "")
    )

    return ShadowEvalResult(
        run_id=run_id,
        candidate_id=candidate_id,
        module_name=module_name,
        passed=passed,
        target_category=target_category,
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        delta_pp=delta_pp,
        regressions=regressions,
        latency_ratio=latency_ratio,
        calibration_ok=calibration_ok,
        reason=reason,
    )


def _load_baseline_metric(category: str, metric_key: str) -> float:
    """Read a baseline metric from eval_baseline.json.  Returns 0.0 if unavailable."""
    import pathlib
    baseline_path = pathlib.Path(config.EVAL_REPORT_PATH) / "eval_baseline.json"
    try:
        with open(baseline_path, encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("categories", {}).get(category, {}).get(metric_key, 0.0))
    except Exception:
        return 0.0


def _load_baseline_latency(category: str) -> float:
    """Read baseline latency_p95 for a category.  Returns 0.0 if unavailable."""
    import pathlib
    baseline_path = pathlib.Path(config.EVAL_REPORT_PATH) / "eval_baseline.json"
    try:
        with open(baseline_path, encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("categories", {}).get(category, {}).get("latency_p95", 0.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# PromptOptimizerAnalyzer — heuristic drift detection + LLM mutation drafting
# ---------------------------------------------------------------------------

#: The fixed meta-prompt used to draft candidate mutations.
#: THIS STRING IS A HARDCODED CONSTANT — it is never self-modifiable.
#: Its SHA-256 hash is checked in tests/test_prompt_optimizer.py::test_meta_prompt_hash_stability.
META_PROMPT: str = (
    "You are a prompt engineer reviewing an AI assistant's instruction template.\n"
    "Given the current prompt and observed performance failures, propose a targeted improvement.\n\n"
    "RULES:\n"
    "- Make the smallest change that addresses the specific failure pattern observed.\n"
    "- Do NOT change the JSON output format contract (any {key: value} structure must stay identical).\n"
    "- Do NOT add instructions that reference internal system details (tool names, DB tables, etc.).\n"
    "- Do NOT relax safety-related language or grounding requirements.\n"
    "- Keep the revised prompt within 20% of the original length.\n\n"
    "Failures observed:\n{failures}\n\n"
    "Current prompt:\n{current_prompt}\n\n"
    "Return ONLY the revised prompt text, with no explanation or preamble."
)

#: SHA-256 of META_PROMPT encoded as UTF-8.  Verified in CI to catch accidental drift.
META_PROMPT_HASH: str = hashlib.sha256(META_PROMPT.encode()).hexdigest()

#: Heuristic rules: (category, metric_key, direction, threshold_pp, module_to_tune, target_category)
#: direction: 'down' = metric declining triggers tuning
_DRIFT_RULES: list[tuple[str, str, str, float, str, str]] = [
    ("reflexion-calibration", "reflexion_mean", "down", 3.0, "critique_prompt", "reflexion-calibration"),
    ("reflexion-calibration", "reflexion_p10", "down", 3.0, "critique_prompt", "reflexion-calibration"),
    ("reasoning", "pass_rate", "down", 3.0, "extraction_prompt", "reasoning"),
    ("tool-use", "pass_rate", "down", 3.0, "extraction_prompt", "tool-use"),
    ("skill-match", "hit_rate", "down", 3.0, "skill_extraction_prompt", "skill-match"),
    ("multi-agent", "pass_rate", "down", 3.0, "merge_instruction_parallel", "multi-agent"),
    ("semantic-match", "recall_at_threshold", "down", 3.0, "kg_extraction_prompt", "semantic-match"),
]


class PromptOptimizerAnalyzer:
    """Heuristic drift detector and LLM-mutation drafter.

    Called by the 'Prompt Optimizer' heartbeat monitor (check_type='prompt_analyzer').
    Reads eval_history.jsonl, applies drift rules, and proposes candidate mutations
    via invoke_nothink() with META_PROMPT.

    Does NOT use brain.think() — uses the raw LLM interface to avoid recursion.
    """

    def __init__(self, store: PromptModuleStore | None = None) -> None:
        self._store = store or PromptModuleStore()

    async def analyze_and_propose(
        self, history_path: str | None = None
    ) -> list[int]:
        """Read eval history, detect drift, draft candidates.

        Returns list of new candidate row IDs (may be empty if no drift detected
        or caps are hit).
        """
        if not config.ENABLE_PROMPT_SELF_MOD:
            logger.debug("prompt_optimizer: ENABLE_PROMPT_SELF_MOD=false — analyzer skipped")
            return []

        history = self._load_history(history_path)
        if len(history) < 3:
            logger.debug("prompt_optimizer: not enough eval history (%d runs)", len(history))
            return []

        # Apply drift rules to last 3+ runs
        recent = history[-4:]  # look at last 4 runs for trend
        triggered: list[tuple[str, str, str]] = []  # (module_name, failure_summary, target_cat)

        for (cat, metric, direction, threshold_pp, module_name, target_cat) in _DRIFT_RULES:
            vals = []
            for run in recent:
                v = run.get("categories", {}).get(cat, {}).get(metric)
                if v is not None:
                    vals.append(float(v))

            if len(vals) < 3:
                continue

            # Check declining trend: last value < first value - threshold (as fraction)
            first_val = vals[0]
            last_val = vals[-1]
            delta_pp = (last_val - first_val) * 100
            if direction == "down" and delta_pp <= -threshold_pp:
                failure_summary = (
                    f"{cat}.{metric} declined {delta_pp:+.1f}pp over last {len(vals)} runs "
                    f"(from {first_val:.3f} to {last_val:.3f})"
                )
                triggered.append((module_name, failure_summary, target_cat))
                logger.info(
                    "prompt_optimizer: drift detected for %r: %s", module_name, failure_summary
                )

        new_ids: list[int] = []
        for module_name, failure_summary, target_cat in triggered:
            candidate_id = await self._propose_candidate(module_name, failure_summary)
            if candidate_id:
                new_ids.append(candidate_id)

        return new_ids

    async def _propose_candidate(self, module_name: str, failure_summary: str) -> int | None:
        """Draft a candidate mutation via LLM and write it to the store."""
        from app.core import llm

        active = self._store.get_active(module_name)
        if not active:
            logger.warning("prompt_optimizer: no active module for %r — cannot propose", module_name)
            return None

        prompt_text = META_PROMPT.format(
            failures=failure_summary,
            current_prompt=active.content,
        )

        try:
            import asyncio
            raw = await asyncio.wait_for(
                llm.invoke_nothink(
                    [{"role": "user", "content": prompt_text}],
                    temperature=0.4,
                    max_tokens=1500,
                ),
                timeout=config.INTERNAL_LLM_TIMEOUT,
            )
        except Exception as e:
            logger.warning("prompt_optimizer: LLM draft failed for %r: %s", module_name, e)
            return None

        if not raw or not isinstance(raw, str) or len(raw.strip()) < 50:
            logger.warning(
                "prompt_optimizer: LLM returned empty/short candidate for %r", module_name
            )
            return None

        candidate_content = raw.strip()
        delta_description = f"Auto-proposed: {failure_summary}"

        return self._store.write_candidate(module_name, candidate_content, delta_description)

    # --- Eval history loading ---

    def _load_history(self, history_path: str | None = None) -> list[dict]:
        """Read eval_history.jsonl into a list of run dicts."""
        import pathlib
        if history_path is None:
            history_path = str(pathlib.Path(config.EVAL_REPORT_PATH) / "eval_history.jsonl")
        try:
            lines = []
            with open(history_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            lines.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return lines
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.warning("prompt_optimizer: could not load eval history: %s", e)
            return []

    # --- Post-promotion auto-rollback check ---

    async def check_and_rollback_if_needed(self) -> list[str]:
        """After a scheduled eval run, check for post-promotion regressions.

        If any module was promoted recently (within the past day) and the current
        eval shows a regression > 2pp vs pre-promotion baseline, roll it back.

        Returns list of rollback reason strings.
        """
        history = self._load_history()
        if len(history) < 2:
            return []

        rollback_reasons: list[str] = []
        # Find active modules that were promoted today
        rows = get_db().fetchall(
            "SELECT id, module_name, promoted_at FROM prompt_modules "
            "WHERE status='active' AND is_baseline=0 AND promoted_at IS NOT NULL "
            "AND promoted_at > datetime('now', '-1 day')"
        )

        current_run = history[-1]
        # Use second-to-last as pre-promotion baseline if available
        pre_run = history[-2] if len(history) >= 2 else {}

        for row in rows:
            module_id = row["id"]
            module_name = row["module_name"]

            # Find the target category for this module
            target_cat = None
            for (cat, _, _, _, mod, tcat) in _DRIFT_RULES:
                if mod == module_name:
                    target_cat = tcat
                    break

            if not target_cat:
                continue

            current_pass = current_run.get("categories", {}).get(target_cat, {}).get("pass_rate")
            pre_pass = pre_run.get("categories", {}).get(target_cat, {}).get("pass_rate")
            if current_pass is None or pre_pass is None:
                continue

            delta_pp = (current_pass - float(pre_pass)) * 100
            if delta_pp < -2.0:  # >2pp drop after promotion triggers rollback
                reason = (
                    f"{module_name} id={module_id}: {target_cat}.pass_rate "
                    f"{float(pre_pass):.3f}→{current_pass:.3f} (Δ={delta_pp:+.1f}pp)"
                )
                if self._store.rollback(module_id):
                    rollback_reasons.append(reason)
                    logger.warning("prompt_optimizer: auto-rollback: %s", reason)

        return rollback_reasons


# ---------------------------------------------------------------------------
# Startup hook: seed baselines if needed
# ---------------------------------------------------------------------------

def init_prompt_optimizer(db=None) -> None:
    """Seed baseline module rows on startup.  Safe to call multiple times."""
    try:
        store = PromptModuleStore(db=db)
        store.seed_baseline()
    except Exception as e:
        logger.warning("prompt_optimizer: startup seeding failed: %s", e)
