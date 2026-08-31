"""Nominate-vs-decide skill dedup for the ambiguous similarity band.

WHY A JUDGE EXISTS (all measured, live store):
  * 2026-08-18: distinct sibling skills score 0.836-0.914; true reworded dup
    0.968 -> the 0.94 scalar bar was set.
  * 2026-08-30: a TRUE induced duplicate (calculator_math vs calculator_usage,
    induced a day apart) scored 0.826 — INSIDE the 08-18 distinct range —
    while the night's genuinely distinct pairs topped out at 0.713. The bands
    OVERLAP across datasets: no scalar can separate them. The 0.94 gate let
    the pair through, and hours after a manual merge the induction pass minted
    the SAME skill again (#110 calculator_for_math_problems, 18:30).

Design under test (MemRefine, arXiv:2606.13177): similarity only NOMINATES;
a schema-pinned LLM judge DECIDES, and only inside [0.70, 0.94). Every failure
path returns "distinct" (fail-open to creation) so the 2026-08-18 over-collapse
(6 seeded skills -> 1 survivor) structurally cannot recur through this code.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.auto_skills import _judge_duplicate_skill


def store_with_neighbor(sim: float | None):
    skills = MagicMock()
    if sim is None:
        skills.nearest_skill.return_value = None
    else:
        skills.nearest_skill.return_value = {
            "id": 83, "name": "calculator_usage", "similarity": sim,
            "trigger_pattern": "When to use a calculator during problem-solving.",
            "procedure_text": "Use the calculator tool for arithmetic.",
        }
    return skills


@pytest.mark.asyncio
class TestBandRouting:
    async def test_below_floor_is_distinct_without_llm(self):
        """0.713 was the measured max for genuinely distinct pairs — below the
        0.70 floor no judge call is spent (or risked)."""
        with patch("app.core.auto_skills.llm.invoke_nothink",
                   new=AsyncMock()) as m:
            v = await _judge_duplicate_skill(
                store_with_neighbor(0.65), "weather_probe", "check weather", "")
            assert v == "distinct"
            m.assert_not_awaited()

    async def test_above_scalar_bar_is_duplicate_without_llm(self):
        """>= 0.94 is proven-dup territory (0.968 measured) — no LLM needed."""
        with patch("app.core.auto_skills.llm.invoke_nothink",
                   new=AsyncMock()) as m:
            v = await _judge_duplicate_skill(
                store_with_neighbor(0.95), "calc_math", "use calculator", "")
            assert v == "duplicate"
            m.assert_not_awaited()

    async def test_no_neighbor_is_distinct(self):
        v = await _judge_duplicate_skill(
            store_with_neighbor(None), "anything", "desc", "")
        assert v == "distinct"


@pytest.mark.asyncio
class TestInBandJudge:
    async def test_judge_duplicate_verdict_blocks(self):
        """The live case: 0.826 similarity, judge recognises the same skill."""
        with patch("app.core.auto_skills.llm.invoke_nothink",
                   new=AsyncMock(return_value='{"verdict": "duplicate"}')):
            v = await _judge_duplicate_skill(
                store_with_neighbor(0.826), "calculator_for_math_problems",
                "When to use a calculator for math problems.", "Use calculator.")
            assert v == "duplicate"

    async def test_judge_distinct_verdict_allows(self):
        with patch("app.core.auto_skills.llm.invoke_nothink",
                   new=AsyncMock(return_value='{"verdict": "distinct"}')):
            v = await _judge_duplicate_skill(
                store_with_neighbor(0.80), "unit_conversion",
                "Convert between units of measurement.", "Use calculator.")
            assert v == "distinct"

    async def test_judge_uses_json_schema(self):
        """Standing lesson: small-model JSON needs json_schema everywhere —
        the contradiction judge fail-opened 22% and KG curation had 0
        successes in 48h without it."""
        mock = AsyncMock(return_value='{"verdict": "distinct"}')
        with patch("app.core.auto_skills.llm.invoke_nothink", new=mock):
            await _judge_duplicate_skill(
                store_with_neighbor(0.80), "n", "d", "p")
        schema = mock.await_args.kwargs.get("json_schema")
        assert schema is not None, "judge must pin its output with json_schema"
        assert schema["properties"]["verdict"]["enum"] == ["duplicate", "distinct"]


@pytest.mark.asyncio
class TestFailOpen:
    """The dangerous direction is over-collapse, not under-dedup. Every
    failure must allow creation — exactly today's behavior."""

    async def test_llm_exception_is_distinct(self):
        with patch("app.core.auto_skills.llm.invoke_nothink",
                   new=AsyncMock(side_effect=RuntimeError("ollama down"))):
            v = await _judge_duplicate_skill(
                store_with_neighbor(0.85), "n", "d", "p")
            assert v == "distinct"

    async def test_garbage_verdict_is_distinct(self):
        with patch("app.core.auto_skills.llm.invoke_nothink",
                   new=AsyncMock(return_value='{"verdict": "maybe?"}')):
            v = await _judge_duplicate_skill(
                store_with_neighbor(0.85), "n", "d", "p")
            assert v == "distinct"

    async def test_nearest_skill_exception_is_distinct(self):
        skills = MagicMock()
        skills.nearest_skill.side_effect = RuntimeError("chroma sad")
        v = await _judge_duplicate_skill(skills, "n", "d", "p")
        assert v == "distinct"
