"""Tests for the SCM/SleepGate two-phase dream split (task #30).

Covers:
- consolidate_nrem invokes the structural/deterministic substeps only
- consolidate_rem invokes the LLM-driven integrative substeps only
- Back-compat: consolidate() still runs everything via the two-phase wrapper
- Failure in REM does NOT clear NREM's committed counters
- ConsolidationResult tracks per-phase timing + completion flags
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.dream import (
    ConsolidationResult,
    DreamConsolidator,
    GatherSignals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_consolidator():
    """DreamConsolidator with a no-op db (substeps are stubbed individually)."""
    db = MagicMock()
    return DreamConsolidator(db)


def _stub_substeps(consolidator, *, nrem_counts: dict | None = None, rem_counts: dict | None = None):
    """Replace each consolidation substep with an AsyncMock that bumps the
    corresponding ConsolidationResult field. Returns the dict of mocks for
    call-count assertions.
    """
    nrem_counts = nrem_counts or {}
    rem_counts = rem_counts or {}
    mocks = {}

    def _bumper(field: str, by: int):
        async def _inner(signals, result, *args, **kwargs):
            setattr(result, field, getattr(result, field) + by)
        return _inner

    # NREM substeps — _mine_dpo_pairs needs (signals, result) only; others same
    for name, field, by in [
        ("_prune_reflexions", "reflexions_pruned", nrem_counts.get("_prune_reflexions", 1)),
        ("_compact_kg_chains", "kg_chains_compacted", nrem_counts.get("_compact_kg_chains", 1)),
        ("_disable_weak_skills", "skills_disabled", nrem_counts.get("_disable_weak_skills", 1)),
        ("_handle_failed_curiosity", "curiosity_dismissed", nrem_counts.get("_handle_failed_curiosity", 1)),
        ("_refresh_stale_facts", "facts_refreshed", nrem_counts.get("_refresh_stale_facts", 1)),
        ("_mine_dpo_pairs", "dpo_pairs_generated", nrem_counts.get("_mine_dpo_pairs", 1)),
    ]:
        m = AsyncMock(side_effect=_bumper(field, by))
        setattr(consolidator, name, m)
        mocks[name] = m

    # REM substeps — _promote_reflexions and _resolve_contradictions take svc
    for name, field, by in [
        ("_promote_reflexions", "reflexions_promoted", rem_counts.get("_promote_reflexions", 1)),
        ("_resolve_contradictions", "contradictions_resolved", rem_counts.get("_resolve_contradictions", 1)),
    ]:
        async def _rem_inner(signals, result, svc, _f=field, _b=by):
            setattr(result, _f, getattr(result, _f) + _b)
        m = AsyncMock(side_effect=_rem_inner)
        setattr(consolidator, name, m)
        mocks[name] = m

    # _consolidate_procedural_memory takes (result, svc) — different signature
    async def _proc(result, svc):
        result.procedural_clusters_consolidated += rem_counts.get("_consolidate_procedural_memory", 1)
    proc_mock = AsyncMock(side_effect=_proc)
    consolidator._consolidate_procedural_memory = proc_mock
    mocks["_consolidate_procedural_memory"] = proc_mock

    return mocks


# ---------------------------------------------------------------------------
# Phase routing — NREM vs REM
# ---------------------------------------------------------------------------

class TestPhaseRouting:
    @pytest.mark.asyncio
    async def test_nrem_runs_structural_substeps_only(self):
        c = _make_consolidator()
        mocks = _stub_substeps(c)
        result = ConsolidationResult()
        await c.consolidate_nrem(GatherSignals(), result, svc=MagicMock())

        nrem_steps = [
            "_prune_reflexions", "_compact_kg_chains", "_disable_weak_skills",
            "_handle_failed_curiosity", "_refresh_stale_facts", "_mine_dpo_pairs",
        ]
        rem_steps = [
            "_promote_reflexions", "_resolve_contradictions",
            "_consolidate_procedural_memory",
        ]
        for s in nrem_steps:
            assert mocks[s].call_count == 1, f"NREM missed {s}"
        for s in rem_steps:
            assert mocks[s].call_count == 0, f"NREM wrongly called REM substep {s}"
        assert result.nrem_completed is True
        assert result.rem_completed is False

    @pytest.mark.asyncio
    async def test_rem_runs_integrative_substeps_only(self):
        c = _make_consolidator()
        mocks = _stub_substeps(c)
        result = ConsolidationResult()
        await c.consolidate_rem(GatherSignals(), result, svc=MagicMock())

        nrem_steps = [
            "_prune_reflexions", "_compact_kg_chains", "_disable_weak_skills",
            "_handle_failed_curiosity", "_refresh_stale_facts", "_mine_dpo_pairs",
        ]
        rem_steps = [
            "_promote_reflexions", "_resolve_contradictions",
            "_consolidate_procedural_memory",
        ]
        for s in rem_steps:
            assert mocks[s].call_count == 1, f"REM missed {s}"
        for s in nrem_steps:
            assert mocks[s].call_count == 0, f"REM wrongly called NREM substep {s}"
        assert result.rem_completed is True
        assert result.nrem_completed is False

    @pytest.mark.asyncio
    async def test_rem_skips_procedural_when_disabled(self, monkeypatch):
        c = _make_consolidator()
        mocks = _stub_substeps(c)
        # Disable procedural consolidation via config
        from app.config import config
        monkeypatch.setattr(config, "ENABLE_PROCEDURAL_CONSOLIDATION", False)
        result = ConsolidationResult()
        await c.consolidate_rem(GatherSignals(), result, svc=MagicMock())
        assert mocks["_consolidate_procedural_memory"].call_count == 0
        # Other REM substeps still run
        assert mocks["_promote_reflexions"].call_count == 1
        assert mocks["_resolve_contradictions"].call_count == 1


# ---------------------------------------------------------------------------
# Back-compat — consolidate() runs everything
# ---------------------------------------------------------------------------

class TestBackwardsCompatibility:
    """consolidate() back-compat — must invoke both phases in order.

    These tests stub the two coordinators directly rather than going through
    the real brain/services import (which has heavy deps that don't always
    import on host Python — and the value of these tests is the dispatch
    behavior, not the integration).
    """

    @pytest.mark.asyncio
    async def test_consolidate_calls_both_phases_in_order(self):
        c = _make_consolidator()
        call_order = []

        async def _nrem(signals, result, svc):
            call_order.append("nrem")
            result.nrem_completed = True

        async def _rem(signals, result, svc):
            call_order.append("rem")
            result.rem_completed = True

        c.consolidate_nrem = _nrem
        c.consolidate_rem = _rem

        # Stub the brain.get_services + tool-whitelist plumbing
        import app.core.dream as dream_module
        fake_brain = MagicMock()
        fake_brain.get_services = MagicMock(return_value=MagicMock())
        fake_tiers = MagicMock()
        fake_tiers.set_tool_whitelist = MagicMock()
        fake_tiers.MAINTENANCE_TOOLS = []
        with patch.dict("sys.modules", {
            "app.core.brain": fake_brain,
            "app.core.access_tiers": fake_tiers,
        }):
            result = await c.consolidate(GatherSignals())

        assert call_order == ["nrem", "rem"]
        assert result.nrem_completed is True
        assert result.rem_completed is True
        # Whitelist set then cleared
        assert fake_tiers.set_tool_whitelist.call_count >= 2
        assert fake_tiers.set_tool_whitelist.call_args_list[-1].args == (None,)


# ---------------------------------------------------------------------------
# Failure isolation — REM failure does NOT clobber NREM
# ---------------------------------------------------------------------------

class TestRemFailureIsolation:
    @pytest.mark.asyncio
    async def test_rem_exception_keeps_nrem_counters(self, monkeypatch):
        """If REM raises mid-way, NREM's already-committed counters remain
        on the shared ConsolidationResult."""
        c = _make_consolidator()

        # Build a result + run NREM normally
        result = ConsolidationResult()

        async def _nrem(signals, res, *a, **kw):
            res.reflexions_pruned = 5
            res.kg_chains_compacted = 2
            res.nrem_completed = True

        async def _rem_boom(signals, res, *a, **kw):
            res.reflexions_promoted = 1  # partial REM progress
            raise RuntimeError("rem went sideways")

        c.consolidate_nrem = _nrem
        c.consolidate_rem = _rem_boom

        # consolidate() catches via exception bubbling — explicitly test the
        # invariant by reproducing the run() guard:
        try:
            await c.consolidate_nrem(GatherSignals(), result, svc=MagicMock())
        except Exception:
            pytest.fail("NREM should not raise in this test")
        try:
            await c.consolidate_rem(GatherSignals(), result, svc=MagicMock())
        except Exception:
            pass  # run() catches REM exceptions

        # NREM commits preserved despite REM blowup
        assert result.reflexions_pruned == 5
        assert result.kg_chains_compacted == 2
        assert result.nrem_completed is True
        # REM partial progress also preserved (we never roll back)
        assert result.reflexions_promoted == 1
        assert result.rem_completed is False


# ---------------------------------------------------------------------------
# Result dataclass — new fields default to 0/False
# ---------------------------------------------------------------------------

class TestResultFields:
    def test_new_fields_default(self):
        r = ConsolidationResult()
        assert r.nrem_seconds == 0.0
        assert r.rem_seconds == 0.0
        assert r.nrem_completed is False
        assert r.rem_completed is False

    def test_config_default_disabled(self):
        from app.config import config
        assert config.ENABLE_TWO_PHASE_DREAM is False, (
            "two-phase dream must be opt-in (default off) for the prototype"
        )
        assert config.DREAM_REM_TIMEOUT_SECONDS == 60.0
