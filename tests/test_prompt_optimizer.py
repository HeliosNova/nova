"""Tests for the prompt self-modification system (app/core/prompt_optimizer.py).

5 empirical verification scenarios:
  (a) Favorable path  — candidate passes shadow eval → promoted to active
  (b) Rejection path  — regressing candidate → quarantined; no state change
  (c) Auto-rollback   — post-promotion regression → rolled back + quarantined
  (d) Firewall        — disallowed modules blocked at every write path
  (e) Goodhart defense — self-flattering critique_prompt caught by calibration

Plus: META_PROMPT hash stability guard and safety cap enforcement.

Nothing here hits the real LLM, eval harness, or network.
EvalHarness.run_all() is mocked in every test that exercises run_shadow_eval().
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.prompt_optimizer import (
    META_PROMPT,
    META_PROMPT_HASH,
    PromptModuleStore,
    _SELF_MOD_ALLOWED_MODULES,
    get_active_module,
    run_shadow_eval,
    with_module_overrides,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASELINE_TEXT = (
    "Evaluate the quality of this response. Consider: accuracy, "
    "clarity, helpfulness. Score from 0.0 to 1.0."
)

_CANDIDATE_TEXT = (
    "Evaluate the quality of this response carefully. Consider: accuracy, "
    "clarity, helpfulness, and completeness. Score from 0.0 to 1.0."
)


def _seed_module(db, module_name: str, content: str = _BASELINE_TEXT) -> None:
    """Insert a single baseline row for a module, bypassing the startup seeder."""
    db.execute(
        "INSERT INTO prompt_modules "
        "(module_name, version, content, is_baseline, status) VALUES (?, 1, ?, 1, 'active')",
        (module_name, content),
    )


def _make_category_metrics(
    category: str = "reasoning",
    *,
    pass_rate: float = 0.90,
    reflexion_mean: float | None = None,
    reflexion_p10: float | None = None,
    reflexion_p90: float | None = None,
    latency_p95: float = 1.0,
):
    from app.monitors.eval_harness import CategoryMetrics
    return CategoryMetrics(
        category=category,
        total=5,
        passed=round(5 * pass_rate),
        pass_rate=pass_rate,
        latency_p50=0.5,
        latency_p95=latency_p95,
        reflexion_mean=reflexion_mean,
        reflexion_std=0.05 if reflexion_mean is not None else None,
        reflexion_p10=reflexion_p10,
        reflexion_p90=reflexion_p90,
    )


def _make_eval_report(categories: dict | None = None) -> MagicMock:
    """Minimal EvalReport mock for shadow-eval tests."""
    from app.monitors.eval_harness import EvalReport
    report = MagicMock(spec=EvalReport)
    report.categories = categories or {}
    return report


# ---------------------------------------------------------------------------
# (d) Firewall
# ---------------------------------------------------------------------------

class TestFirewall:
    """Every disallowed write path must be silently blocked."""

    def test_get_active_module_rejects_identity_block(self):
        """IDENTITY_AND_REASONING is not in the allow-list → None."""
        assert get_active_module("IDENTITY_AND_REASONING") is None

    def test_get_active_module_rejects_harness_internal(self):
        """quiz_gen is in _HARNESS_INTERNAL_MODULES → None."""
        assert get_active_module("quiz_gen") is None

    def test_get_active_module_rejects_quiz_answer(self):
        assert get_active_module("quiz_answer") is None

    def test_get_active_module_rejects_quiz_grade(self):
        assert get_active_module("quiz_grade") is None

    def test_get_active_module_rejects_unknown(self):
        assert get_active_module("totally_unknown_module_xyz") is None

    def test_write_candidate_rejects_disallowed_name(self, db, monkeypatch):
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        from app.config import reset_config; reset_config()
        store = PromptModuleStore(db=db)
        result = store.write_candidate("IDENTITY_AND_REASONING", "payload", "test")
        assert result is None

    def test_write_candidate_rejects_meta_prompt_name(self, db, monkeypatch):
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        from app.config import reset_config; reset_config()
        store = PromptModuleStore(db=db)
        result = store.write_candidate("META_PROMPT", "hacked meta-prompt", "test")
        assert result is None

    def test_write_candidate_rejects_harness_internal_modules(self, db, monkeypatch):
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        from app.config import reset_config; reset_config()
        store = PromptModuleStore(db=db)
        for name in ("quiz_gen", "quiz_answer", "quiz_grade"):
            assert store.write_candidate(name, "content", "test") is None, (
                f"{name!r} should be blocked by both firewalls"
            )

    def test_allowed_modules_excludes_safety_critical_names(self):
        """The allow-list must not contain any safety-critical module names."""
        forbidden = {
            "IDENTITY_AND_REASONING", "META_PROMPT",
            "quiz_gen", "quiz_answer", "quiz_grade",
            "tool_examples", "access_tier", "safety_instructions",
        }
        overlap = _SELF_MOD_ALLOWED_MODULES & forbidden
        assert overlap == set(), f"Safety-critical name in allow-list: {overlap}"

    def test_allowed_modules_contains_expected_six(self):
        """Exactly 6 modules are in the allow-list (design doc D1)."""
        expected = {
            "critique_prompt", "extraction_prompt", "skill_extraction_prompt",
            "merge_instruction_parallel", "merge_instruction_sequential",
            "kg_extraction_prompt",
        }
        assert _SELF_MOD_ALLOWED_MODULES == expected


# ---------------------------------------------------------------------------
# META_PROMPT hash stability (CI guard)
# ---------------------------------------------------------------------------

class TestMetaPromptHashStability:
    """META_PROMPT_HASH must match SHA-256(META_PROMPT).

    If this test fails, either META_PROMPT was accidentally changed
    (revert the change) or intentionally updated (update META_PROMPT_HASH
    in app/core/prompt_optimizer.py).
    """

    def test_hash_matches_constant(self):
        expected = hashlib.sha256(META_PROMPT.encode()).hexdigest()
        assert META_PROMPT_HASH == expected, (
            "META_PROMPT has drifted from its recorded SHA-256.\n"
            f"  Computed : {expected}\n"
            f"  Constant : {META_PROMPT_HASH}\n"
            "Update META_PROMPT_HASH if the change is intentional."
        )

    def test_meta_prompt_non_empty(self):
        assert len(META_PROMPT.strip()) >= 200, "META_PROMPT looks truncated"

    def test_required_template_slots_present(self):
        assert "{failures}" in META_PROMPT
        assert "{current_prompt}" in META_PROMPT


# ---------------------------------------------------------------------------
# Safety cap enforcement
# ---------------------------------------------------------------------------

class TestSafetyCaps:
    """All safety caps must gate correctly."""

    def test_kill_switch_blocks_write(self, db, monkeypatch):
        """ENABLE_PROMPT_SELF_MOD=false prevents write_candidate."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "false")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        assert store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test") is None

    def test_kill_switch_blocks_promote(self, db, monkeypatch):
        """ENABLE_PROMPT_SELF_MOD=false prevents promote."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "false")
        from app.config import reset_config; reset_config()
        store = PromptModuleStore(db=db)
        assert store.promote(999, "run_x", {"pass_rate": 1.0}) is False

    def test_daily_proposal_cap(self, db, monkeypatch):
        """After PROMPT_MOD_MAX_PROPOSALS_PER_DAY (2) proposals, the 3rd is blocked."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_PROPOSALS_PER_DAY", "2")
        monkeypatch.setenv("PROMPT_MOD_MAX_PENDING", "99")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        id1 = store.write_candidate("critique_prompt", _CANDIDATE_TEXT + " a", "t1")
        id2 = store.write_candidate("critique_prompt", _CANDIDATE_TEXT + " b", "t2")
        id3 = store.write_candidate("critique_prompt", _CANDIDATE_TEXT + " c", "t3")
        assert id1 is not None
        assert id2 is not None
        assert id3 is None  # cap reached

    def test_max_pending_cap(self, db, monkeypatch):
        """After PROMPT_MOD_MAX_PENDING (3) pending candidates, the 4th is blocked."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_PROPOSALS_PER_DAY", "99")
        monkeypatch.setenv("PROMPT_MOD_MAX_PENDING", "3")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        for i in range(3):
            rowid = store.write_candidate("critique_prompt", f"{_CANDIDATE_TEXT} v{i}", f"t{i}")
            assert rowid is not None, f"Proposal {i} should have succeeded"
        overflow = store.write_candidate("critique_prompt", _CANDIDATE_TEXT + " overflow", "t4")
        assert overflow is None

    def test_drift_threshold_enforced(self, db, monkeypatch):
        """Candidate with Jaccard drift > MAX_DRIFT is rejected."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "0.05")
        monkeypatch.setenv("PROMPT_MOD_MAX_PENDING", "99")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        # Zero word overlap with baseline → maximum drift
        alien = "zephyr quixotic luminescence transcendent paradigm metamorphosis ephemeral"
        assert store.write_candidate("critique_prompt", alien, "drift test") is None

    def test_quarantined_module_blocks_new_proposals(self, db, monkeypatch):
        """A quarantined module cannot receive new candidate proposals."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        assert cand_id is not None
        assert store.quarantine_candidate(cand_id) is True
        assert store.is_quarantined("critique_prompt") is True
        result = store.write_candidate("critique_prompt", _CANDIDATE_TEXT + " v2", "blocked")
        assert result is None

    def test_daily_promotion_cap(self, db, monkeypatch):
        """System-wide promotion cap: max 2 promotions per day (PROMPT_MOD_MAX_PROMOTIONS_PER_DAY)."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        monkeypatch.setenv("PROMPT_MOD_MAX_PROPOSALS_PER_DAY", "99")
        monkeypatch.setenv("PROMPT_MOD_MAX_PENDING", "99")
        monkeypatch.setenv("PROMPT_MOD_MAX_PROMOTIONS_PER_DAY", "2")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt", _BASELINE_TEXT)
        _seed_module(db, "extraction_prompt", _BASELINE_TEXT)
        _seed_module(db, "kg_extraction_prompt", _BASELINE_TEXT)
        store = PromptModuleStore(db=db)

        # First two promotions should succeed (different modules)
        cid1 = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        assert store.promote(cid1, "run_001", {"pass_rate": 0.9}) is True

        cid2 = store.write_candidate("extraction_prompt", _CANDIDATE_TEXT + " ex", "test")
        assert store.promote(cid2, "run_002", {"pass_rate": 0.9}) is True

        # Third promotion is blocked by the system-wide daily cap
        cid3 = store.write_candidate("kg_extraction_prompt", _CANDIDATE_TEXT + " kg", "test")
        assert store.promote(cid3, "run_003", {"pass_rate": 0.9}) is False


# ---------------------------------------------------------------------------
# (a) Favorable path
# ---------------------------------------------------------------------------

class TestFavorablePath:
    """Candidate passes shadow eval → gets promoted to active."""

    def test_write_candidate_creates_pending_row(self, db, monkeypatch):
        """write_candidate creates a 'candidate' row with correct metadata."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "add completeness")
        assert cand_id is not None
        mv = store.get_by_id(cand_id)
        assert mv is not None
        assert mv.status == "candidate"
        assert mv.module_name == "critique_prompt"
        assert mv.version == 2
        assert mv.parent_version == 1

    def test_promote_sets_active_and_supersedes_baseline(self, db, monkeypatch):
        """Promoting a candidate marks it active and supersedes the previous active row."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        assert cand_id is not None

        ok = store.promote(cand_id, "run_favorable_001", {"pass_rate": 0.92})
        assert ok is True

        active = store.get_active("critique_prompt")
        assert active is not None
        assert active.id == cand_id
        assert active.version == 2
        assert active.status == "active"
        assert active.promoted_eval_run_id == "run_favorable_001"

        # Baseline row is superseded but still findable via get_baseline()
        baseline = store.get_baseline("critique_prompt")
        assert baseline is not None
        assert baseline.status == "superseded"
        assert baseline.content == _BASELINE_TEXT

    def test_get_active_module_returns_promoted_content(self, db, monkeypatch):
        """After promotion, get_active_module() returns the promoted content."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        store.promote(cand_id, "run_002", {"pass_rate": 0.95})

        with patch("app.core.prompt_optimizer.get_db", return_value=db):
            result = get_active_module("critique_prompt")
        assert result == _CANDIDATE_TEXT

    async def test_shadow_eval_returns_passed_on_improvement(self, db, monkeypatch):
        """run_shadow_eval returns passed=True when candidate improves the target metric."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        monkeypatch.setenv("PROMPT_MOD_MIN_IMPROVEMENT_PP", "2.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        assert cand_id is not None

        good_report = _make_eval_report({
            "reflexion-calibration": _make_category_metrics(
                "reflexion-calibration",
                pass_rate=0.90,
                reflexion_mean=0.74,
                reflexion_p10=0.60,
                reflexion_p90=0.87,  # below 0.93 ceiling → calibration ok
            ),
        })

        with (
            patch("app.monitors.eval_harness.EvalHarness") as MockHarness,
            patch("app.core.prompt_optimizer._load_baseline_metric", return_value=0.85),
            patch("app.core.prompt_optimizer._load_baseline_latency", return_value=1.0),
        ):
            mock_harness_inst = MagicMock()
            mock_harness_inst.run_all = AsyncMock(return_value=good_report)
            MockHarness.return_value = mock_harness_inst

            result = await run_shadow_eval(
                cand_id, "reflexion-calibration", store=store
            )

        assert result.passed is True
        assert result.candidate_id == cand_id
        assert result.module_name == "critique_prompt"
        assert result.delta_pp > 0
        assert result.regressions == []
        assert result.calibration_ok is True

    def test_get_active_versions_reflects_promotion(self, db, monkeypatch):
        """get_active_versions() maps each module_name to its current active version."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        store.promote(cand_id, "run_003", {})
        versions = store.get_active_versions()
        assert versions["critique_prompt"] == 2


# ---------------------------------------------------------------------------
# (b) Rejection path
# ---------------------------------------------------------------------------

class TestRejectionPath:
    """Regressing candidate → quarantined; active module unchanged."""

    async def test_shadow_eval_fails_when_sibling_category_regresses(self, db, monkeypatch):
        """passed=False when a non-target category drops more than REGRESSION_TOLERANCE_PP."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        monkeypatch.setenv("PROMPT_MOD_MIN_IMPROVEMENT_PP", "2.0")
        monkeypatch.setenv("PROMPT_MOD_REGRESSION_TOLERANCE_PP", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        assert cand_id is not None

        # Target category improves; reasoning regresses (0.85 → 0.60, Δ = -25pp)
        regression_report = _make_eval_report({
            "reflexion-calibration": _make_category_metrics(
                "reflexion-calibration",
                pass_rate=0.92,
                reflexion_mean=0.74,
                reflexion_p10=0.60,
                reflexion_p90=0.87,
            ),
            "reasoning": _make_category_metrics("reasoning", pass_rate=0.60),
        })

        with (
            patch("app.monitors.eval_harness.EvalHarness") as MockHarness,
            patch(
                "app.core.prompt_optimizer._load_baseline_metric",
                side_effect=lambda cat, _m: 0.85,
            ),
            patch("app.core.prompt_optimizer._load_baseline_latency", return_value=1.0),
        ):
            mock_harness_inst = MagicMock()
            mock_harness_inst.run_all = AsyncMock(return_value=regression_report)
            MockHarness.return_value = mock_harness_inst
            result = await run_shadow_eval(
                cand_id, "reflexion-calibration", store=store
            )

        assert result.passed is False
        assert len(result.regressions) > 0
        assert any("reasoning" in r for r in result.regressions)

    def test_quarantine_candidate_sets_status_and_expiry(self, db, monkeypatch):
        """quarantine_candidate() marks status='quarantined' with a future expiry."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        assert cand_id is not None
        assert store.quarantine_candidate(cand_id) is True
        mv = store.get_by_id(cand_id)
        assert mv.status == "quarantined"
        assert mv.quarantined_until is not None

    def test_active_module_unchanged_after_quarantine(self, db, monkeypatch):
        """After quarantining a candidate, the baseline remains the active module."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        store.quarantine_candidate(cand_id)
        active = store.get_active("critique_prompt")
        assert active is not None
        assert active.version == 1
        assert active.is_baseline is True

    async def test_shadow_eval_no_auto_promote_on_failure(self, db, monkeypatch):
        """A failed shadow eval must NOT automatically promote the candidate."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        monkeypatch.setenv("PROMPT_MOD_MIN_IMPROVEMENT_PP", "2.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")

        # No improvement: baseline=0.85, candidate=0.50 → Δ = -35pp
        bad_report = _make_eval_report({
            "reflexion-calibration": _make_category_metrics(
                "reflexion-calibration", pass_rate=0.50
            ),
        })

        with (
            patch("app.monitors.eval_harness.EvalHarness") as MockHarness,
            patch("app.core.prompt_optimizer._load_baseline_metric", return_value=0.85),
            patch("app.core.prompt_optimizer._load_baseline_latency", return_value=1.0),
        ):
            mock_harness_inst = MagicMock()
            mock_harness_inst.run_all = AsyncMock(return_value=bad_report)
            MockHarness.return_value = mock_harness_inst
            result = await run_shadow_eval(cand_id, "reflexion-calibration", store=store)

        assert result.passed is False
        # Active module is still the baseline (v1)
        active = store.get_active("critique_prompt")
        assert active.version == 1
        assert active.is_baseline is True


# ---------------------------------------------------------------------------
# (c) Auto-rollback
# ---------------------------------------------------------------------------

class TestAutoRollback:
    """Post-promotion regression triggers automatic rollback and quarantine."""

    async def test_rollback_on_regression_exceeding_threshold(self, db, monkeypatch):
        """If a recently-promoted module regresses >2pp, check_and_rollback_if_needed
        rolls it back and quarantines it."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)

        # Promote a candidate
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        ok = store.promote(cand_id, "run_x", {"pass_rate": 0.92})
        assert ok is True
        assert store.get_active("critique_prompt").id == cand_id

        # Fake eval history: pre-promotion at 0.90, post-promotion regressed to 0.85
        # (Δ = -5pp > 2pp trigger threshold)
        pre_run = {"categories": {"reflexion-calibration": {"pass_rate": 0.90}}}
        post_run = {"categories": {"reflexion-calibration": {"pass_rate": 0.85}}}

        from app.core.prompt_optimizer import PromptOptimizerAnalyzer
        analyzer = PromptOptimizerAnalyzer(store=store)

        with (
            patch.object(analyzer, "_load_history", return_value=[pre_run, post_run]),
            patch("app.core.prompt_optimizer.get_db", return_value=db),
        ):
            reasons = await analyzer.check_and_rollback_if_needed()

        assert len(reasons) == 1
        assert "critique_prompt" in reasons[0]

        # Promoted module must be quarantined
        mv = store.get_by_id(cand_id)
        assert mv.status == "quarantined"

        # Parent (baseline v1) must be restored to active
        restored = store.get_active("critique_prompt")
        assert restored is not None
        assert restored.version == 1
        assert restored.is_baseline is True

    async def test_no_rollback_within_tolerance(self, db, monkeypatch):
        """A ≤2pp drop after promotion does not trigger rollback."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        store.promote(cand_id, "run_y", {"pass_rate": 0.92})

        # Only 1pp regression → within the 2pp hard threshold
        pre_run = {"categories": {"reflexion-calibration": {"pass_rate": 0.90}}}
        post_run = {"categories": {"reflexion-calibration": {"pass_rate": 0.89}}}

        from app.core.prompt_optimizer import PromptOptimizerAnalyzer
        analyzer = PromptOptimizerAnalyzer(store=store)
        with (
            patch.object(analyzer, "_load_history", return_value=[pre_run, post_run]),
            patch("app.core.prompt_optimizer.get_db", return_value=db),
        ):
            reasons = await analyzer.check_and_rollback_if_needed()

        assert reasons == []
        # Promoted module is still active
        active = store.get_active("critique_prompt")
        assert active.id == cand_id

    async def test_insufficient_history_skips_rollback_check(self, db, monkeypatch):
        """With fewer than 2 eval runs in history, the check is skipped cleanly."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        from app.config import reset_config; reset_config()
        store = PromptModuleStore(db=db)

        from app.core.prompt_optimizer import PromptOptimizerAnalyzer
        analyzer = PromptOptimizerAnalyzer(store=store)
        with (
            patch.object(analyzer, "_load_history", return_value=[{"categories": {}}]),
            patch("app.core.prompt_optimizer.get_db", return_value=db),
        ):
            reasons = await analyzer.check_and_rollback_if_needed()

        assert reasons == []


# ---------------------------------------------------------------------------
# (e) Goodhart defense
# ---------------------------------------------------------------------------

class TestGoodhartDefense:
    """Self-flattering critique_prompt is caught by the calibration guard."""

    async def test_inflated_scores_trigger_calibration_failure(self, db, monkeypatch):
        """A critique_prompt that inflates reflexion scores fails the calibration check."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        monkeypatch.setenv("PROMPT_MOD_MIN_IMPROVEMENT_PP", "2.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "inflated")
        assert cand_id is not None

        # Pass_rate looks good, but reflexion scores are inflated beyond design caps:
        # reflexion_p90 > 0.93 OR reflexion_mean > 0.80 → calibration_ok=False
        inflated_report = _make_eval_report({
            "reflexion-calibration": _make_category_metrics(
                "reflexion-calibration",
                pass_rate=0.95,
                reflexion_mean=0.85,   # INFLATED (> 0.80 ceiling)
                reflexion_p10=0.72,
                reflexion_p90=0.96,    # INFLATED (> 0.93 ceiling)
            ),
        })

        with (
            patch("app.monitors.eval_harness.EvalHarness") as MockHarness,
            patch("app.core.prompt_optimizer._load_baseline_metric", return_value=0.80),
            patch("app.core.prompt_optimizer._load_baseline_latency", return_value=1.0),
        ):
            mock_harness_inst = MagicMock()
            mock_harness_inst.run_all = AsyncMock(return_value=inflated_report)
            MockHarness.return_value = mock_harness_inst
            result = await run_shadow_eval(cand_id, "reflexion-calibration", store=store)

        assert result.passed is False
        assert result.calibration_ok is False
        assert any("inflation" in r.lower() for r in result.regressions)

    async def test_baseline_scorer_injected_for_critique_shadow_eval(self, db, monkeypatch):
        """For critique_prompt candidates, scoring_overrides must pin the baseline content.

        This is the Goodhart firewall: the candidate is tested for generation quality,
        but graded (reflexion) using the baseline critique — not its own output.
        """
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt", _BASELINE_TEXT)
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        assert cand_id is not None

        captured_gen: dict[str, str] = {}
        captured_score: dict[str, str] = {}

        def _capture(overrides, scoring_overrides=None):
            captured_gen.update(overrides or {})
            captured_score.update(scoring_overrides or {})

        good_report = _make_eval_report({
            "reflexion-calibration": _make_category_metrics(
                "reflexion-calibration",
                pass_rate=0.92,
                reflexion_mean=0.74,
                reflexion_p10=0.60,
                reflexion_p90=0.87,
            ),
        })

        with (
            patch("app.monitors.eval_harness.EvalHarness") as MockHarness,
            patch("app.core.prompt_optimizer._load_baseline_metric", return_value=0.85),
            patch("app.core.prompt_optimizer._load_baseline_latency", return_value=1.0),
        ):
            mock_harness_inst = MagicMock()
            mock_harness_inst.run_all = AsyncMock(return_value=good_report)
            mock_harness_inst.set_module_overrides.side_effect = _capture
            MockHarness.return_value = mock_harness_inst
            await run_shadow_eval(cand_id, "reflexion-calibration", store=store)

        # Generation path: the candidate text is injected
        assert captured_gen.get("critique_prompt") == _CANDIDATE_TEXT, (
            "Candidate text must be in the generation overrides"
        )
        # Scoring path: baseline pins the scorer (not the candidate)
        assert captured_score.get("critique_prompt") == _BASELINE_TEXT, (
            "Baseline text must be pinned in scoring_overrides (Goodhart firewall)"
        )

    async def test_non_critique_module_has_empty_scoring_override(self, db, monkeypatch):
        """For non-critique_prompt modules there is no Goodhart risk → scoring_overrides={}."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "extraction_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("extraction_prompt", _CANDIDATE_TEXT, "test")
        assert cand_id is not None

        captured_score: dict[str, str] = {}

        def _capture(overrides, scoring_overrides=None):
            captured_score.update(scoring_overrides or {})

        good_report = _make_eval_report({
            "reasoning": _make_category_metrics("reasoning", pass_rate=0.95),
        })

        with (
            patch("app.monitors.eval_harness.EvalHarness") as MockHarness,
            patch("app.core.prompt_optimizer._load_baseline_metric", return_value=0.85),
            patch("app.core.prompt_optimizer._load_baseline_latency", return_value=1.0),
        ):
            mock_harness_inst = MagicMock()
            mock_harness_inst.run_all = AsyncMock(return_value=good_report)
            mock_harness_inst.set_module_overrides.side_effect = _capture
            MockHarness.return_value = mock_harness_inst
            await run_shadow_eval(cand_id, "reasoning", store=store)

        # No scoring override injected for non-critique modules
        assert captured_score == {}


# ---------------------------------------------------------------------------
# ContextVar isolation (with_module_overrides)
# ---------------------------------------------------------------------------

class TestContextVarIsolation:
    """ContextVar-based isolation: overrides must not leak between tasks."""

    def test_override_visible_inside_context(self):
        """Content set via with_module_overrides() is returned by get_active_module()."""
        override_text = "overridden critique prompt for isolation test"
        with with_module_overrides({"critique_prompt": override_text}):
            result = get_active_module("critique_prompt")
        assert result == override_text

    def test_override_not_visible_outside_context(self):
        """After the context exits, the ContextVar reverts and DB is queried."""
        with with_module_overrides({"critique_prompt": "volatile override"}):
            pass  # enter and exit immediately
        with patch("app.core.prompt_optimizer.get_db") as mock_db:
            mock_db.return_value.fetchone.return_value = None
            result = get_active_module("critique_prompt")
        assert result is None

    def test_scoring_override_separated_from_generation_override(self):
        """scoring=True reads _SCORING_OVERRIDES; scoring=False reads _MODULE_OVERRIDES."""
        gen_text = "generation path: candidate content"
        score_text = "scoring path: baseline content"
        with with_module_overrides(
            {"critique_prompt": gen_text},
            scoring_overrides={"critique_prompt": score_text},
        ):
            gen_result = get_active_module("critique_prompt", scoring=False)
            score_result = get_active_module("critique_prompt", scoring=True)
        assert gen_result == gen_text
        assert score_result == score_text
        assert gen_result != score_result

    def test_overrides_reset_on_exception(self):
        """ContextVars are reset even when the body raises an exception."""
        try:
            with with_module_overrides({"critique_prompt": "volatile"}):
                raise RuntimeError("intentional test error")
        except RuntimeError:
            pass
        with patch("app.core.prompt_optimizer.get_db") as mock_db:
            mock_db.return_value.fetchone.return_value = None
            result = get_active_module("critique_prompt")
        assert result is None

    def test_disallowed_module_not_returned_even_if_overridden(self):
        """Firewall check runs before ContextVar lookup — disallowed name stays None."""
        with with_module_overrides({"IDENTITY_AND_REASONING": "injected"}):
            result = get_active_module("IDENTITY_AND_REASONING")
        assert result is None


# ---------------------------------------------------------------------------
# PromptModuleStore lifecycle state machine
# ---------------------------------------------------------------------------

class TestModuleLifecycle:
    """Lifecycle transitions: candidate → active → superseded / quarantined / rolled_back."""

    def test_rollback_quarantines_active_and_restores_parent(self, db, monkeypatch):
        """rollback() quarantines the current active row and restores parent to active."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        store.promote(cand_id, "run_z", {})

        ok = store.rollback(cand_id)
        assert ok is True

        # Promoted row is quarantined with expiry
        mv = store.get_by_id(cand_id)
        assert mv.status == "quarantined"
        assert mv.rolled_back_at is not None
        assert mv.quarantined_until is not None

        # Parent (baseline v1) is active again
        active = store.get_active("critique_prompt")
        assert active is not None
        assert active.version == 1

    def test_rollback_requires_active_status(self, db, monkeypatch):
        """rollback() on a non-active row returns False (no-op)."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        store = PromptModuleStore(db=db)
        cand_id = store.write_candidate("critique_prompt", _CANDIDATE_TEXT, "test")
        # candidate, not active — rollback must be a no-op
        assert store.rollback(cand_id) is False

    def test_count_proposals_today_and_active_versions(self, db, monkeypatch):
        """count_proposals_today and get_active_versions work correctly."""
        monkeypatch.setenv("ENABLE_PROMPT_SELF_MOD", "true")
        monkeypatch.setenv("PROMPT_MOD_MAX_DRIFT", "1.0")
        monkeypatch.setenv("PROMPT_MOD_MAX_PENDING", "99")
        monkeypatch.setenv("PROMPT_MOD_MAX_PROPOSALS_PER_DAY", "99")
        from app.config import reset_config; reset_config()
        _seed_module(db, "critique_prompt")
        _seed_module(db, "extraction_prompt")
        store = PromptModuleStore(db=db)
        # Initially no candidates
        assert store.count_proposals_today("critique_prompt") == 0
        # Write two candidates
        store.write_candidate("critique_prompt", _CANDIDATE_TEXT + " 1", "t1")
        store.write_candidate("critique_prompt", _CANDIDATE_TEXT + " 2", "t2")
        assert store.count_proposals_today("critique_prompt") == 2
        # Unrelated module stays at 0
        assert store.count_proposals_today("extraction_prompt") == 0
        # Both modules are at version 1 (baselines)
        versions = store.get_active_versions()
        assert versions["critique_prompt"] == 1
        assert versions["extraction_prompt"] == 1
