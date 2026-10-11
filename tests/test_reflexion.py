"""Tests for the Reflexion Store module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.llm import GenerationResult, StreamChunk, ToolCall, _extract_tool_calls, _strip_think_tags
from app.core.reflexion import assess_quality


# ===========================================================================
# Quality Assessment
# ===========================================================================

class TestQualityAssessment:
    def test_good_answer(self):
        score, reason = assess_quality(
            "Bitcoin is a decentralized cryptocurrency created in 2009.",
            tool_results=[],
            max_tool_rounds=5,
        )
        assert score >= 0.7
        # Reason may include positive bonuses (e.g. "+0.15 specificity bonus");
        # only reject if the reason flags a failure.
        assert "fail" not in reason.lower() and "penalty" not in reason.lower()

    def test_empty_answer(self):
        score, reason = assess_quality("", [], 5)
        assert score == 0.0
        assert "Empty" in reason

    def test_short_answer_complex_query(self):
        # Short answers to complex queries should be penalized
        score, reason = assess_quality(
            "Yes.", [], 5,
            query="Can you explain the differences between quantum computing and classical computing?",
        )
        assert score < 1.0
        assert "short" in reason.lower()

    def test_short_answer_simple_query(self):
        # Short answers to simple/short queries are legitimate (no length penalty);
        # may still attract specificity/alignment bonuses or penalties from heuristic.
        score, reason = assess_quality("Paris.", [], 5, query="Capital of France?")
        assert score >= 0.5

    def test_failure_phrases(self):
        # Hard failure phrases trigger strong penalty
        score, reason = assess_quality(
            "I cannot help with that. I wasn't able to find information. Failed to retrieve results.",
            tool_results=[],
            max_tool_rounds=5,
        )
        assert score <= 0.6
        assert "failure phrase" in reason.lower()

    def test_soft_uncertainty_phrases(self):
        # Soft uncertainty phrases trigger lighter penalty
        score, reason = assess_quality(
            "I don't know the answer and I'm not sure about that topic.",
            tool_results=[],
            max_tool_rounds=5,
        )
        assert score <= 0.95  # soft penalty only
        assert "uncertainty" in reason.lower()

    def test_tool_exhaustion(self):
        tools = [{"tool": f"t{i}", "output": "ok"} for i in range(5)]
        score, reason = assess_quality(
            "Here's what I found from searching multiple sources.",
            tool_results=tools,
            max_tool_rounds=5,
        )
        assert score < 1.0
        assert "rounds" in reason.lower()

    def test_tool_failures(self):
        tools = [
            {"tool": "web_search", "output": "Error: connection timed out"},
            {"tool": "http_fetch", "output": "Failed to fetch URL"},
        ]
        score, reason = assess_quality(
            "I tried searching but the tools had issues.",
            tool_results=tools,
            max_tool_rounds=5,
        )
        assert score < 0.8
        assert "failed" in reason.lower()

    def test_cumulative_penalties(self):
        """Multiple issues should stack penalties."""
        tools = [{"tool": "web_search", "output": "Error: failed to connect"}] * 5
        score, reason = assess_quality(
            "I don't know. I couldn't find anything.",
            tool_results=tools,
            max_tool_rounds=5,
        )
        assert score < 0.3


# ===========================================================================
# Grounded critique — independent judge path
# ===========================================================================

class TestEvidenceSection:
    def test_no_tools(self):
        from app.core.reflexion import _build_evidence_section
        assert "(none" in _build_evidence_section([])

    def test_excerpt_and_truncation_marker(self):
        from app.core.reflexion import _build_evidence_section, _EVIDENCE_PER_TOOL
        section = _build_evidence_section(
            [{"tool": "web_search", "output": "x" * (_EVIDENCE_PER_TOOL + 500)}]
        )
        assert "[1] web_search:" in section
        assert "…[truncated]" in section
        assert len(section) < _EVIDENCE_PER_TOOL + 100

    def test_error_labeled_when_no_output(self):
        from app.core.reflexion import _build_evidence_section
        section = _build_evidence_section(
            [{"tool": "http_fetch", "output": "", "error": "connection refused"}]
        )
        assert "TOOL ERROR: connection refused" in section

    def test_total_budget_omission_is_explicit(self):
        from app.core.reflexion import _build_evidence_section, _EVIDENCE_TOTAL
        many = [{"tool": f"t{i}", "output": "y" * 600} for i in range(10)]
        section = _build_evidence_section(many)
        assert "omitted for length" in section
        assert len(section) < _EVIDENCE_TOTAL + 300


class TestGroundedJudge:
    """With config.JUDGE_MODEL set, critique_response routes to the
    independent judge with the tool evidence in the prompt; on any judge
    failure it falls back to the legacy self-critique; with JUDGE_MODEL
    unset the legacy path is untouched."""

    @pytest.fixture
    def judge_on(self, monkeypatch):
        from app.config import config
        monkeypatch.setattr(config, "JUDGE_MODEL", "gemma3:4b")
        return config

    @pytest.mark.asyncio
    async def test_judge_receives_evidence_and_returns_score(self, judge_on):
        from app.core import llm as llm_mod
        from app.core.reflexion import critique_response

        judge_json = json.dumps({
            "score": 0.85, "grounded": 0.9,
            "unsupported_claims": [], "critique": "well supported",
        })
        tool_results = [{
            "tool": "web_search",
            "output": "Acme employs 12,000 people worldwide (Reuters, May 2026)",
        }]
        answer = "Acme employs about 12,000 people as of May 2026." + " More context." * 30
        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = judge_json
            score, reason = await critique_response(
                "How many employees does Acme have?", answer, tool_results,
            )
        assert score == 0.85
        assert "[judge:gemma3:4b]" in reason
        assert "grounded=0.90" in reason
        assert mock_invoke.call_count == 1  # judge succeeded — no legacy call
        kwargs = mock_invoke.call_args.kwargs
        assert kwargs["model"] == "gemma3:4b"
        assert kwargs["num_ctx"] == 8192
        assert kwargs["json_schema"] is not None
        prompt = mock_invoke.call_args.args[0][0]["content"]
        assert "12,000 people" in prompt  # evidence excerpt shown to the judge
        assert "GROUNDED" in prompt       # grounded rubric, not the legacy template

    @pytest.mark.asyncio
    async def test_low_grounded_caps_high_style_score(self, judge_on):
        from app.core import llm as llm_mod
        from app.core.reflexion import critique_response

        # Judge inconsistency guard: stylish score with unsupported claims
        # must not survive when tools ran.
        judge_json = json.dumps({
            "score": 0.9, "grounded": 0.2,
            "unsupported_claims": ["Acme has exactly 47,500 employees"],
            "critique": "specific figure not in evidence",
        })
        tool_results = [{"tool": "web_search", "output": "Acme employs about 12,000 people."}]
        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = judge_json
            score, reason = await critique_response(
                "How many employees does Acme have?",
                "Acme has exactly 47,500 employees." + " Filler sentence." * 30,
                tool_results,
            )
        assert score == 0.4  # min(0.9, 0.4) fusion cap
        assert "unsupported: Acme has exactly 47,500 employees" in reason

    @pytest.mark.asyncio
    async def test_judge_exception_falls_back_to_legacy(self, judge_on):
        from app.core import llm as llm_mod
        from app.core.reflexion import critique_response

        legacy_json = json.dumps({"score": 0.7, "critique": "fine"})
        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = [Exception("judge down"), legacy_json]
            score, reason = await critique_response(
                "q", "An answer long enough to be substantial." * 10, [],
            )
        assert score == 0.7
        assert "[judge" not in reason
        assert mock_invoke.call_count == 2
        legacy_kwargs = mock_invoke.call_args_list[1].kwargs
        assert "model" not in legacy_kwargs  # legacy call untouched

    @pytest.mark.asyncio
    async def test_judge_garbage_json_falls_back_to_legacy(self, judge_on):
        from app.core import llm as llm_mod
        from app.core.reflexion import critique_response

        legacy_json = json.dumps({"score": 0.65, "critique": "legacy graded"})
        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = ["complete garbage, no braces", legacy_json]
            score, reason = await critique_response(
                "q", "An answer long enough to be substantial." * 10, [],
            )
        assert score == 0.65
        assert mock_invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_no_judge_configured_legacy_untouched(self, monkeypatch):
        from app.config import config
        from app.core import llm as llm_mod
        from app.core.reflexion import critique_response

        monkeypatch.setattr(config, "JUDGE_MODEL", "")
        legacy_json = json.dumps({"score": 0.8, "critique": "ok"})
        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = legacy_json
            score, reason = await critique_response(
                "q", "An answer long enough to be substantial." * 10, [],
            )
        assert score == 0.8
        assert mock_invoke.call_count == 1
        kwargs = mock_invoke.call_args.kwargs
        assert "model" not in kwargs
        prompt = mock_invoke.call_args.args[0][0]["content"]
        assert "GROUNDED" not in prompt  # legacy template, not the judge rubric
