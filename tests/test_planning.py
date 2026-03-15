"""Tests for the Query Planning module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.planning import should_plan, create_plan, format_plan_for_prompt


# ===========================================================================
# Detection heuristic: should_plan()
# ===========================================================================

class TestShouldPlan:
    def test_greeting_skipped(self):
        assert not should_plan("hello there!", "greeting")

    def test_correction_skipped(self):
        assert not should_plan("actually, the capital is Paris", "correction")

    def test_short_query_skipped(self):
        assert not should_plan("what time?", "general")

    def test_simple_query_skipped(self):
        assert not should_plan("What is the capital of France?", "general")

    def test_multi_part_detected(self):
        q = "What is the current price of Bitcoin and also calculate compound interest on $10000 at 5% for 10 years"
        assert should_plan(q, "general")

    def test_reasoning_words_detected(self):
        q = "Compare the advantages and disadvantages of React versus Vue for a large-scale enterprise application"
        assert should_plan(q, "general")

    def test_numbered_list_detected(self):
        q = "I need you to do the following:\n1. Search for the latest Python release\n2. Calculate the download size\n3. Summarize the changes"
        assert should_plan(q, "general")

    def test_multiple_tool_signals(self):
        q = "Search for the current Bitcoin price and also calculate my total portfolio value for 2.5 BTC"
        assert should_plan(q, "general")

    def test_long_multi_question(self):
        q = "What is the best programming language for web development and what tools should I use, and how does it compare to other options available?"
        assert should_plan(q, "general")

    def test_single_tool_signal_not_enough(self):
        q = "Search for the latest news about Tesla"
        assert not should_plan(q, "general")


# ===========================================================================
# Plan creation: create_plan()
# ===========================================================================

class TestCreatePlan:
    @pytest.mark.asyncio
    async def test_creates_valid_plan(self):
        mock_response = json.dumps({
            "steps": [
                {"description": "Search for Bitcoin price", "tool": "web_search"},
                {"description": "Calculate portfolio value", "tool": "calculator"},
                {"description": "Synthesize final answer", "tool": "none"},
            ],
            "complexity": "multi_step",
        })

        with patch("app.core.planning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            plan = await create_plan(
                "What is Bitcoin price and calculate my portfolio",
                ["web_search", "calculator", "code_exec"],
            )

        assert plan is not None
        assert len(plan["steps"]) == 3
        assert plan["steps"][0]["tool"] == "web_search"
        assert plan["complexity"] == "multi_step"

    @pytest.mark.asyncio
    async def test_caps_at_5_steps(self):
        mock_response = json.dumps({
            "steps": [{"description": f"step {i}", "tool": "none"} for i in range(8)],
            "complexity": "multi_step",
        })

        with patch("app.core.planning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            plan = await create_plan("complex query", ["web_search"])

        assert plan is not None
        assert len(plan["steps"]) <= 5

    @pytest.mark.asyncio
    async def test_invalid_tool_replaced_with_none(self):
        mock_response = json.dumps({
            "steps": [{"description": "do something", "tool": "nonexistent_tool"}],
            "complexity": "simple",
        })

        with patch("app.core.planning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            plan = await create_plan("query", ["web_search"])

        assert plan is not None
        assert plan["steps"][0]["tool"] == "none"

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        with patch("app.core.planning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="")
            plan = await create_plan("query", ["web_search"])

        assert plan is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        with patch("app.core.planning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="not json at all")
            plan = await create_plan("query", ["web_search"])

        assert plan is None

    @pytest.mark.asyncio
    async def test_reflexions_injected(self):
        mock_response = json.dumps({
            "steps": [{"description": "use web_search", "tool": "web_search"}],
            "complexity": "simple",
        })
        captured_messages = []

        async def capture(msgs, **kwargs):
            captured_messages.extend(msgs)
            return mock_response

        with patch("app.core.planning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(side_effect=capture)
            await create_plan(
                "bitcoin price",
                ["web_search"],
                reflexions_text="Previously failed with knowledge_search",
            )

        system_msg = captured_messages[0]["content"]
        assert "Previously failed" in system_msg


# ===========================================================================
# Format plan for prompt
# ===========================================================================

class TestFormatPlan:
    def test_formats_correctly(self):
        plan = {
            "steps": [
                {"description": "Search for data", "tool": "web_search"},
                {"description": "Calculate result", "tool": "calculator"},
                {"description": "Synthesize", "tool": "none"},
            ],
            "complexity": "multi_step",
        }
        text = format_plan_for_prompt(plan)
        assert "[PLAN]" in text
        assert "using web_search" in text
        assert "using calculator" in text
        assert "3. Synthesize" in text
        assert "using none" not in text  # "none" tool shouldn't show "using none"

    def test_empty_plan(self):
        assert format_plan_for_prompt({}) == ""
        assert format_plan_for_prompt(None) == ""

    def test_no_steps(self):
        assert format_plan_for_prompt({"steps": []}) == ""
