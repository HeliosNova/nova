"""Streaming provider tests — verify stream_with_thinking() for the Ollama provider.

Cloud provider streaming (OpenAI/Anthropic/Google) was removed when Nova
committed to sovereign-only operation; only Ollama streaming is tested here.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.llm import StreamChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockStreamResponse:
    """Mock httpx streaming response that yields lines."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ===========================================================================
# Ollama Provider
# ===========================================================================

class TestOllamaStreaming:
    """Test OllamaProvider.stream_with_thinking() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_yields_thinking_and_content_chunks(self):
        from app.core.providers.ollama import OllamaProvider

        lines = [
            json.dumps({"message": {"thinking": "Let me think..."}, "done": False}),
            json.dumps({"message": {"thinking": "about this..."}, "done": False}),
            json.dumps({"message": {"content": "The answer "}, "done": False}),
            json.dumps({"message": {"content": "is 42."}, "done": False}),
            json.dumps({"message": {}, "done": True}),
        ]

        mock_response = MockStreamResponse(lines)

        provider = OllamaProvider.__new__(OllamaProvider)
        provider._llm_model = "test-model"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        provider._get_client = MagicMock(return_value=mock_client)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "What is 42?"}],
            [],
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)

        # Should have thinking, content, and done chunks
        thinking_chunks = [c for c in chunks if c.thinking]
        content_chunks = [c for c in chunks if c.content]
        done_chunks = [c for c in chunks if c.done]

        assert len(thinking_chunks) >= 2
        assert len(content_chunks) >= 2
        assert len(done_chunks) == 1
        assert "".join(c.thinking for c in thinking_chunks) == "Let me think...about this..."
        assert "".join(c.content for c in content_chunks) == "The answer is 42."

    @pytest.mark.asyncio
    async def test_skips_empty_lines(self):
        from app.core.providers.ollama import OllamaProvider

        lines = [
            "",
            "   ",
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            "",
            json.dumps({"message": {}, "done": True}),
        ]

        mock_response = MockStreamResponse(lines)

        provider = OllamaProvider.__new__(OllamaProvider)
        provider._llm_model = "test-model"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        provider._get_client = MagicMock(return_value=mock_client)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "Hello"}],
            [],
        ):
            chunks.append(chunk)

        assert len(chunks) == 2  # content + done
        assert chunks[0].content == "Hello"
        assert chunks[1].done is True

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        from app.core.providers.ollama import OllamaProvider

        lines = [
            "not valid json",
            json.dumps({"message": {"content": "Valid"}, "done": False}),
            "{broken json...",
            json.dumps({"message": {}, "done": True}),
        ]

        mock_response = MockStreamResponse(lines)

        provider = OllamaProvider.__new__(OllamaProvider)
        provider._llm_model = "test-model"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        provider._get_client = MagicMock(return_value=mock_client)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "Hello"}],
            [],
        ):
            chunks.append(chunk)

        # Should skip invalid JSON and still produce valid chunks
        assert any(c.content == "Valid" for c in chunks)
        assert any(c.done for c in chunks)


