"""Tests for the Self-Critique module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.critique import should_critique, critique_answer, format_critique_for_regeneration


# ===========================================================================
# Detection heuristic: should_critique()
# ===========================================================================

class TestShouldCritique:
    def test_greeting_skipped(self):
        assert not should_critique("hi", "Hello! How can I help?", "greeting", [])

    def test_correction_skipped(self):
        assert not should_critique("wrong", "I'll fix that", "correction", [])

    def test_short_answer_skipped(self):
        assert not should_critique("what is 2+2?", "4", "general", [])

    def test_successful_tool_results_skip(self):
        """When all tools succeeded (no failure markers), skip critique."""
        assert not should_critique(
            "bitcoin price",
            "Bitcoin is currently at $50,000 based on the search results.",
            "general",
            [{"tool": "web_search", "output": "..."}],
        )

    def test_failed_tool_results_trigger(self):
        """When a tool has failure markers, critique should trigger."""
        assert should_critique(
            "bitcoin price",
            "Bitcoin is currently at $50,000 based on the search results.",
            "general",
            [{"tool": "web_search", "output": "[tool web_search failed: timeout]"}],
        )

    def test_long_answer_triggers(self):
        answer = "Here is a comprehensive analysis. " * 20
        assert should_critique("analyze this", answer, "general", [])

    def test_planned_query_triggers(self):
        assert should_critique(
            "compare these two things",
            "Here's my detailed comparison of the two approaches with several important considerations to discuss.",
            "general",
            [],
            was_planned=True,
        )

    def test_medium_answer_no_tools_no_plan(self):
        assert not should_critique(
            "what is Python?",
            "Python is a programming language created by Guido van Rossum.",
            "general",
            [],
        )


# ===========================================================================
# Critique: critique_answer()
# ===========================================================================

class TestCritiqueAnswer:
    @pytest.mark.asyncio
    async def test_passing_critique(self):
        mock_response = json.dumps({"pass": True, "issues": []})
        with patch("app.core.critique.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            result = await critique_answer("What is Python?", "Python is a programming language.")

        assert result is not None
        assert result["pass"] is True
        assert result["issues"] == []

    @pytest.mark.asyncio
    async def test_failing_critique(self):
        mock_response = json.dumps({
            "pass": False,
            "issues": ["Missed part 2 of the question", "Unsupported claim about performance"],
        })
        with patch("app.core.critique.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            result = await critique_answer(
                "Compare Python and Rust performance and ecosystem",
                "Python is slower than Rust.",
            )

        assert result is not None
        assert result["pass"] is False
        assert len(result["issues"]) == 2

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        with patch("app.core.critique.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="")
            result = await critique_answer("query", "answer")

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        with patch("app.core.critique.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="not valid json")
            result = await critique_answer("query", "answer")

        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        with patch("app.core.critique.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(side_effect=Exception("LLM down"))
            result = await critique_answer("query", "answer")

        assert result is None


# ===========================================================================
# Format critique for regeneration
# ===========================================================================

class TestFormatCritique:
    def test_passing_critique_empty(self):
        assert format_critique_for_regeneration({"pass": True, "issues": []}) == ""

    def test_failing_critique_formatted(self):
        critique = {
            "pass": False,
            "issues": ["Missed part 2", "Unsupported claim"],
        }
        text = format_critique_for_regeneration(critique)
        assert "[SELF-CHECK FAILED]" in text
        assert "Missed part 2" in text
        assert "Unsupported claim" in text

    def test_no_issues_empty(self):
        assert format_critique_for_regeneration({"pass": False, "issues": []}) == ""

    def test_none_critique_empty(self):
        assert format_critique_for_regeneration(None) == ""

    def test_empty_dict_empty(self):
        assert format_critique_for_regeneration({}) == ""


# ===========================================================================
# Integration: Brain + Critique
# ===========================================================================

class TestCritiqueBrainIntegration:
    @pytest.mark.asyncio
    async def test_critique_triggers_on_complex_query(self, db):
        """When critique is enabled and query was planned, critique should run."""
        from app.core.brain import Services, set_services, think
        from app.core.memory import ConversationStore, UserFactStore

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
        )
        set_services(svc)

        with patch("app.core.brain.llm") as mock_llm:
            # Main generation
            mock_result = AsyncMock()
            mock_result.content = "Here is a comprehensive comparison of the two approaches. " * 5
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            # Critique call — return passing
            mock_llm.invoke_nothink = AsyncMock(return_value=json.dumps({"pass": True, "issues": []}))

            events = []
            async for event in think("Compare Python and Rust for web development and also analyze their ecosystems"):
                events.append(event)

        # Should complete without errors
        done_events = [e for e in events if e.type.value == "done"]
        assert len(done_events) == 1
