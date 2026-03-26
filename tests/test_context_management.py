"""Item 64: Test _manage_context() from brain.py.

Create a scenario where history exceeds token budget, verify summarization
is triggered, verify output fits within budget.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.core.brain import _manage_context


class TestManageContext:
    """Test context window management."""

    @pytest.mark.asyncio
    async def test_short_history_passes_through(self):
        """Short history that fits within budget should pass through unchanged."""
        system_prompt = "You are a helpful assistant."
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        query = "How are you?"

        result_history, summary = await _manage_context(system_prompt, history, query)
        assert result_history == history
        assert summary == ""

    @pytest.mark.asyncio
    async def test_long_history_triggers_summarization(self, monkeypatch):
        """History exceeding MAX_CONTEXT_TOKENS should trigger summarization."""
        # Lower the context limit to force summarization
        monkeypatch.setenv("MAX_CONTEXT_TOKENS", "200")
        monkeypatch.setenv("RECENT_MESSAGES_KEEP", "2")
        from app.config import reset_config
        reset_config()

        system_prompt = "You are a helpful assistant."
        # Create a long history that will exceed 200 tokens
        history = [
            {"role": "user", "content": f"This is message {i} with some extra padding text to make it longer and consume more tokens." * 5}
            for i in range(20)
        ]
        query = "What did we discuss?"

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="Discussed various topics in 20 messages.")

            result_history, summary = await _manage_context(system_prompt, history, query)

            # Should have been trimmed
            from app.config import config
            assert len(result_history) <= config.RECENT_MESSAGES_KEEP
            # Summary should be non-empty
            assert summary != ""

    @pytest.mark.asyncio
    async def test_summary_generated_on_overflow(self, monkeypatch):
        """LLM should be called to generate a summary when context overflows."""
        monkeypatch.setenv("MAX_CONTEXT_TOKENS", "100")
        monkeypatch.setenv("RECENT_MESSAGES_KEEP", "2")
        from app.config import reset_config
        reset_config()

        system_prompt = "You are a helpful assistant."
        history = [
            {"role": "user", "content": "Tell me about topic " + "x" * 500}
            for _ in range(20)
        ]
        query = "Summarize"

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="Summary of previous discussion.")

            result_history, summary = await _manage_context(system_prompt, history, query)

            # LLM should have been called for summarization
            mock_llm.invoke_nothink.assert_called_once()
            assert "Summary" in summary or "discussion" in summary

    @pytest.mark.asyncio
    async def test_keeps_recent_messages(self, monkeypatch):
        """After trimming, the most recent messages should be preserved."""
        monkeypatch.setenv("MAX_CONTEXT_TOKENS", "100")
        monkeypatch.setenv("RECENT_MESSAGES_KEEP", "4")
        from app.config import reset_config
        reset_config()

        system_prompt = "You are a helpful assistant."
        history = [
            {"role": "user", "content": f"Message {i} " + "x" * 200}
            for i in range(10)
        ]
        query = "What's next?"

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="Earlier discussion summary.")

            result_history, summary = await _manage_context(system_prompt, history, query)

            # The last 4 messages should be preserved
            assert len(result_history) == 4
            assert result_history[-1]["content"] == history[-1]["content"]

    @pytest.mark.asyncio
    async def test_summarization_failure_truncates(self, monkeypatch):
        """If LLM summarization fails, should still truncate with a note."""
        monkeypatch.setenv("MAX_CONTEXT_TOKENS", "100")
        monkeypatch.setenv("RECENT_MESSAGES_KEEP", "2")
        from app.config import reset_config
        reset_config()

        system_prompt = "You are a helpful assistant."
        history = [
            {"role": "user", "content": f"Message {i} " + "x" * 200}
            for i in range(10)
        ]
        query = "Continue"

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(side_effect=Exception("LLM unreachable"))

            result_history, summary = await _manage_context(system_prompt, history, query)

            # Should still truncate even if summarization fails
            assert len(result_history) == 2
            # Fallback note should be present
            assert "truncated" in summary.lower()

    @pytest.mark.asyncio
    async def test_few_messages_not_trimmed(self):
        """Fewer messages than RECENT_MESSAGES_KEEP should not be trimmed."""
        system_prompt = "You are a helpful assistant."
        history = [
            {"role": "user", "content": "One question"},
            {"role": "assistant", "content": "One answer"},
        ]
        query = "Follow up"

        result_history, summary = await _manage_context(system_prompt, history, query)
        assert result_history == history
        assert summary == ""
