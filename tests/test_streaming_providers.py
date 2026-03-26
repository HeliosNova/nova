"""Streaming provider tests — verify stream_with_thinking() for all 4 providers.

Each provider is tested with mocked HTTP responses to verify that
StreamChunk objects are yielded correctly from the streaming API.
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


# ===========================================================================
# OpenAI Provider
# ===========================================================================

class TestOpenAIStreaming:
    """Test OpenAIProvider.stream_with_thinking() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_yields_content_chunks(self):
        from app.core.providers.openai import OpenAIProvider

        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"content": "Hello "}}]}),
            'data: ' + json.dumps({"choices": [{"delta": {"content": "world!"}}]}),
            'data: [DONE]',
        ]

        mock_response = MockStreamResponse(lines)

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4o"
        provider._api_key = "test-key"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        provider._get_client = MagicMock(return_value=mock_client)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "Hello"}],
            [],
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)

        content_chunks = [c for c in chunks if c.content]
        done_chunks = [c for c in chunks if c.done]

        assert len(content_chunks) >= 2
        assert len(done_chunks) == 1
        assert "".join(c.content for c in content_chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_skips_non_data_lines(self):
        from app.core.providers.openai import OpenAIProvider

        lines = [
            ": comment line",
            'data: ' + json.dumps({"choices": [{"delta": {"content": "Result"}}]}),
            'event: some_event',
            'data: [DONE]',
        ]

        mock_response = MockStreamResponse(lines)

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4o"
        provider._api_key = "test-key"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        provider._get_client = MagicMock(return_value=mock_client)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "Test"}],
            [],
        ):
            chunks.append(chunk)

        content = "".join(c.content for c in chunks if c.content)
        assert content == "Result"


# ===========================================================================
# Anthropic Provider
# ===========================================================================

class TestAnthropicStreaming:
    """Test AnthropicProvider.stream_with_thinking() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_yields_thinking_and_content(self):
        from app.core.providers.anthropic import AnthropicProvider

        lines = [
            'data: ' + json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}),
            'data: ' + json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Analyzing..."}}),
            'data: ' + json.dumps({"type": "content_block_stop", "index": 0}),
            'data: ' + json.dumps({"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}),
            'data: ' + json.dumps({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The answer is 42."}}),
            'data: ' + json.dumps({"type": "content_block_stop", "index": 1}),
            'data: ' + json.dumps({"type": "message_stop"}),
        ]

        mock_response = MockStreamResponse(lines)

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._model = "claude-sonnet"
        provider._api_key = "test-key"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        provider._get_client = MagicMock(return_value=mock_client)

        # Need to mock _convert_messages
        provider._convert_messages = MagicMock(return_value=(
            "You are helpful.",
            [{"role": "user", "content": "What is 42?"}],
        ))
        provider._convert_tools = MagicMock(return_value=[])

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "What is 42?"}],
            [],
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)

        thinking_text = "".join(c.thinking for c in chunks if c.thinking)
        content_text = "".join(c.content for c in chunks if c.content)

        assert "Analyzing" in thinking_text
        assert "42" in content_text


# ===========================================================================
# Google Provider
# ===========================================================================

class TestGoogleStreaming:
    """Test GoogleProvider.stream_with_thinking() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_yields_content_chunks(self):
        from app.core.providers.google import GoogleProvider

        lines = [
            'data: ' + json.dumps({
                "candidates": [{"content": {"parts": [{"text": "Hello "}]}}]
            }),
            'data: ' + json.dumps({
                "candidates": [{"content": {"parts": [{"text": "from Gemini."}]}}]
            }),
            'data: ' + json.dumps({
                "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]
            }),
        ]

        mock_response = MockStreamResponse(lines)

        provider = GoogleProvider.__new__(GoogleProvider)
        provider._model = "gemini-2.0-flash"
        provider._api_key = "test-key"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        provider._get_client = MagicMock(return_value=mock_client)

        # Mock helper methods
        provider._convert_messages = MagicMock(return_value=(
            "System prompt",
            [{"role": "user", "parts": [{"text": "Hello"}]}],
        ))
        provider._convert_tools = MagicMock(return_value=[])
        provider._url = MagicMock(return_value="/v1/models/gemini-2.0-flash:streamGenerateContent")
        provider._extract_text = lambda parts: "".join(p.get("text", "") for p in parts)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "Hello"}],
            [],
        ):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)

        content = "".join(c.content for c in chunks if c.content)
        assert "Hello " in content
        assert "Gemini" in content

    @pytest.mark.asyncio
    async def test_handles_empty_candidates(self):
        from app.core.providers.google import GoogleProvider

        lines = [
            'data: ' + json.dumps({"candidates": []}),
            'data: ' + json.dumps({
                "candidates": [{"content": {"parts": [{"text": "Result"}]}}]
            }),
        ]

        mock_response = MockStreamResponse(lines)

        provider = GoogleProvider.__new__(GoogleProvider)
        provider._model = "gemini-2.0-flash"
        provider._api_key = "test-key"

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)
        provider._get_client = MagicMock(return_value=mock_client)

        provider._convert_messages = MagicMock(return_value=(
            "System", [{"role": "user", "parts": [{"text": "Test"}]}],
        ))
        provider._convert_tools = MagicMock(return_value=[])
        provider._url = MagicMock(return_value="/v1/models/gemini-2.0-flash:streamGenerateContent")
        provider._extract_text = lambda parts: "".join(p.get("text", "") for p in parts)

        chunks = []
        async for chunk in provider.stream_with_thinking(
            [{"role": "user", "content": "Test"}],
            [],
        ):
            chunks.append(chunk)

        content = "".join(c.content for c in chunks if c.content)
        assert "Result" in content
