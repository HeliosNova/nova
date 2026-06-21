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


# ===========================================================================
# invoke_nothink — JSON prefix handling and per-call options
# ===========================================================================

class TestInvokeNothinkJsonHandling:
    """Regression: the json_prefix re-prepend must be conditional.

    Prefill-continuation models (qwen35) return content WITHOUT the prefix
    and need it prepended; models whose chat template closes the assistant
    turn (gemma3) ignore the prefill and return the COMPLETE object —
    unconditional prepending corrupted that to "{{..." (found 2026-06-11
    wiring the grounded judge)."""

    def _provider(self):
        from app.core.providers.ollama import OllamaProvider
        provider = OllamaProvider.__new__(OllamaProvider)
        provider._llm_model = "test-model"
        provider._get_client = MagicMock(return_value=MagicMock())
        return provider

    @pytest.mark.asyncio
    async def test_complete_object_response_not_double_prefixed(self):
        provider = self._provider()
        complete = '{"score": 0.9, "grounded": 1.0, "unsupported_claims": [], "critique": "ok"}'
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": complete}}
        with patch(
            "app.core.providers.ollama.retry_on_transient",
            new_callable=AsyncMock, return_value=mock_resp,
        ):
            out = await provider.invoke_nothink(
                [{"role": "user", "content": "grade"}],
                json_mode=True, json_prefix="{",
                json_schema={"type": "object"},
                model="gemma3:4b",
            )
        assert json.loads(out)["score"] == 0.9

    @pytest.mark.asyncio
    async def test_continuation_response_gets_prefix_prepended(self):
        provider = self._provider()
        continuation = '"score": 0.7, "critique": "fine"}'
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": continuation}}
        with patch(
            "app.core.providers.ollama.retry_on_transient",
            new_callable=AsyncMock, return_value=mock_resp,
        ):
            out = await provider.invoke_nothink(
                [{"role": "user", "content": "grade"}],
                json_mode=True, json_prefix="{",
                model="qwen-test",
            )
        assert json.loads(out)["score"] == 0.7

    @pytest.mark.asyncio
    async def test_num_ctx_flows_into_request_options(self):
        provider = self._provider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": '{"ok": true}'}}
        with patch(
            "app.core.providers.ollama.retry_on_transient",
            new_callable=AsyncMock, return_value=mock_resp,
        ) as mock_rot:
            await provider.invoke_nothink(
                [{"role": "user", "content": "x"}],
                json_mode=True, json_prefix="{",
                model="gemma3:4b", num_ctx=8192,
            )
            payload = mock_rot.call_args.kwargs["json"]
            assert payload["options"]["num_ctx"] == 8192

            mock_rot.reset_mock()
            await provider.invoke_nothink(
                [{"role": "user", "content": "x"}],
                model="gemma3:4b",
            )
            payload = mock_rot.call_args.kwargs["json"]
            assert "num_ctx" not in payload["options"]


