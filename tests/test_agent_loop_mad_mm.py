"""Tests for MAD-MM (ICLR 2026) subjective memory masking in agent_loop.

Covers:
- Scratchpad.render_for_step honors keep_step_ids / keep_finding_keys filters
- _mask_prior_observations: parsing happy path, ambiguous verdicts default to keep,
  all-drop collapse falls back to keep-all, LLM failure falls back to keep-all,
  empty inputs return early without an LLM call
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.core.agent_loop import (
    Plan,
    Scratchpad,
    Step,
    STEP_DONE,
    STEP_PENDING,
    _mask_prior_observations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan_with_done_steps(n: int = 3) -> Plan:
    """Build a Plan with n DONE steps + 1 pending current step."""
    steps = []
    for i in range(1, n + 1):
        s = Step(id=i, description=f"prior step {i}")
        s.status = STEP_DONE
        s.observation = f"observation from step {i}"
        steps.append(s)
    cur = Step(id=n + 1, description="current step")
    cur.status = STEP_PENDING
    steps.append(cur)
    return Plan(goal="test goal", steps=steps)


# ---------------------------------------------------------------------------
# Scratchpad.render_for_step honors mask filters
# ---------------------------------------------------------------------------

class TestRenderRespectsMask:
    def test_no_mask_keeps_everything(self):
        """When keep_* are None (default), render unchanged."""
        plan = _make_plan_with_done_steps(3)
        cur = plan.steps[-1]
        sp = Scratchpad(query="q")
        sp.findings = {"f1": "val1", "f2": "val2"}
        out = sp.render_for_step(plan, cur)
        assert "step 1" in out
        assert "step 2" in out
        assert "step 3" in out
        assert "f1" in out
        assert "f2" in out

    def test_keep_step_subset_drops_others(self):
        plan = _make_plan_with_done_steps(3)
        cur = plan.steps[-1]
        sp = Scratchpad(query="q")
        out = sp.render_for_step(plan, cur, keep_step_ids={1, 3})
        assert "[step 1]" in out
        assert "[step 2]" not in out
        assert "[step 3]" in out

    def test_keep_finding_subset_drops_others(self):
        plan = _make_plan_with_done_steps(0)
        # Need at least one done step or render adds no PRIOR STEPS — use findings only
        sp = Scratchpad(query="q")
        sp.findings = {"keep_me": "v1", "drop_me": "v2", "also_keep": "v3"}
        out = sp.render_for_step(plan, plan.steps[0], keep_finding_keys={"keep_me", "also_keep"})
        assert "keep_me" in out
        assert "drop_me" not in out
        assert "also_keep" in out

    def test_empty_keep_set_renders_no_prior_section(self):
        """If mask drops all steps, the PRIOR STEPS header itself shouldn't appear."""
        plan = _make_plan_with_done_steps(2)
        cur = plan.steps[-1]
        sp = Scratchpad(query="q")
        out = sp.render_for_step(plan, cur, keep_step_ids=set())
        assert "PRIOR STEPS:" not in out
        assert "GOAL:" in out  # goal still rendered

    def test_previous_attempt_block_never_masked(self):
        """Even with empty keep sets, current step's revision context survives."""
        plan = _make_plan_with_done_steps(2)
        cur = plan.steps[-1]
        cur.attempts = 1
        cur.observation = "what i did last time"
        cur.action = {"answer": "wrong"}
        cur.critique = "wrong because reasons"
        sp = Scratchpad(query="q")
        out = sp.render_for_step(plan, cur, keep_step_ids=set(), keep_finding_keys=set())
        assert "PREVIOUS ATTEMPT FOR THIS STEP" in out
        assert "what i did last time" in out


# ---------------------------------------------------------------------------
# _mask_prior_observations: parse + fallback paths
# ---------------------------------------------------------------------------

class TestMaskParser:
    """All tests stub llm.invoke_nothink — no real LLM calls."""

    @pytest.mark.asyncio
    async def test_happy_path_parses_yes_no(self, monkeypatch):
        async def _stub(*a, **kw):
            return json.dumps({
                "step_1": "yes",
                "step_2": "no",
                "finding_age": "yes",
                "finding_color": "no",
            })
        monkeypatch.setattr("app.core.agent_loop.llm.invoke_nothink", _stub)
        kept_steps, kept_findings, decisions = await _mask_prior_observations(
            goal="g", step_description="d",
            prior_step_items=[(1, "obs1"), (2, "obs2")],
            finding_items=[("age", "30"), ("color", "blue")],
        )
        assert kept_steps == {1}
        assert kept_findings == {"age"}
        assert decisions["step_2"] == "no"
        assert decisions["finding_color"] == "no"

    @pytest.mark.asyncio
    async def test_ambiguous_verdict_defaults_yes(self, monkeypatch):
        """Anything that isn't yes/no — including missing keys — must keep."""
        async def _stub(*a, **kw):
            return json.dumps({
                "step_1": "maybe",     # bogus
                "step_2": "YES",       # case-insensitive
                # step_3 omitted entirely
            })
        monkeypatch.setattr("app.core.agent_loop.llm.invoke_nothink", _stub)
        kept_steps, _, _ = await _mask_prior_observations(
            goal="g", step_description="d",
            prior_step_items=[(1, "a"), (2, "b"), (3, "c")],
            finding_items=[],
        )
        assert kept_steps == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_llm_failure_keeps_all(self, monkeypatch):
        async def _stub(*a, **kw):
            raise RuntimeError("ollama unreachable")
        monkeypatch.setattr("app.core.agent_loop.llm.invoke_nothink", _stub)
        kept_steps, kept_findings, _ = await _mask_prior_observations(
            goal="g", step_description="d",
            prior_step_items=[(1, "a"), (2, "b")],
            finding_items=[("k", "v")],
        )
        assert kept_steps == {1, 2}
        assert kept_findings == {"k"}

    @pytest.mark.asyncio
    async def test_empty_llm_response_keeps_all(self, monkeypatch):
        async def _stub(*a, **kw):
            return ""
        monkeypatch.setattr("app.core.agent_loop.llm.invoke_nothink", _stub)
        kept_steps, _, _ = await _mask_prior_observations(
            goal="g", step_description="d",
            prior_step_items=[(1, "a")],
            finding_items=[],
        )
        assert kept_steps == {1}

    @pytest.mark.asyncio
    async def test_malformed_json_keeps_all(self, monkeypatch):
        async def _stub(*a, **kw):
            return "not even close to JSON"
        monkeypatch.setattr("app.core.agent_loop.llm.invoke_nothink", _stub)
        kept_steps, _, _ = await _mask_prior_observations(
            goal="g", step_description="d",
            prior_step_items=[(1, "a"), (2, "b")],
            finding_items=[],
        )
        assert kept_steps == {1, 2}

    @pytest.mark.asyncio
    async def test_all_drop_collapse_falls_back_to_keep_all(self, monkeypatch):
        """Pathological 'mask everything' response is overridden — never render empty memory."""
        async def _stub(*a, **kw):
            return json.dumps({"step_1": "no", "step_2": "no", "finding_x": "no"})
        monkeypatch.setattr("app.core.agent_loop.llm.invoke_nothink", _stub)
        kept_steps, kept_findings, _ = await _mask_prior_observations(
            goal="g", step_description="d",
            prior_step_items=[(1, "a"), (2, "b")],
            finding_items=[("x", "v")],
        )
        # Collapse guard: when LLM said drop everything, fall back to keep-all
        assert kept_steps == {1, 2}
        assert kept_findings == {"x"}

    @pytest.mark.asyncio
    async def test_empty_inputs_skip_llm_call(self, monkeypatch):
        """No items to mask = no LLM call at all (returns empty sets)."""
        called = {"count": 0}

        async def _stub(*a, **kw):
            called["count"] += 1
            return "{}"

        monkeypatch.setattr("app.core.agent_loop.llm.invoke_nothink", _stub)
        kept_steps, kept_findings, decisions = await _mask_prior_observations(
            goal="g", step_description="d",
            prior_step_items=[],
            finding_items=[],
        )
        assert called["count"] == 0
        assert kept_steps == set()
        assert kept_findings == set()
        assert decisions == {}

    @pytest.mark.asyncio
    async def test_prompt_includes_step_description_and_goal(self, monkeypatch):
        captured = {}

        async def _stub(messages, **kw):
            captured["prompt"] = messages[0]["content"]
            return json.dumps({"step_1": "yes"})

        monkeypatch.setattr("app.core.agent_loop.llm.invoke_nothink", _stub)
        await _mask_prior_observations(
            goal="OVERARCHING_GOAL_TEXT",
            step_description="UNIQUE_STEP_DESC",
            prior_step_items=[(1, "obs")],
            finding_items=[],
        )
        assert "UNIQUE_STEP_DESC" in captured["prompt"]
        assert "OVERARCHING_GOAL_TEXT" in captured["prompt"]


# ---------------------------------------------------------------------------
# Config gate (smoke)
# ---------------------------------------------------------------------------

class TestConfigDefaultsOff:
    def test_default_disabled(self):
        from app.config import config
        assert config.ENABLE_MAD_MM_MASKING is False, (
            "MAD-MM masking must be opt-in (default off) to avoid surprise latency"
        )

    def test_min_prior_steps_default_3(self):
        from app.config import config
        assert config.MAD_MM_MIN_PRIOR_STEPS == 3
