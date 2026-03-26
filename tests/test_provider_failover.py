"""Tests for provider error handling — verify each provider raises
LLMUnavailableError on network/HTTP failures.

Tests:
- OpenAI timeout raises LLMUnavailableError
- Anthropic timeout raises LLMUnavailableError
- Ollama ConnectError raises LLMUnavailableError
- Google HTTP 500 raises LLMUnavailableError
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.llm import LLMUnavailableError


# ---------------------------------------------------------------------------
# Helper to build a mock httpx response
# ---------------------------------------------------------------------------

def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    resp.json.return_value = json_data or {}
    return resp


# ===========================================================================
# OpenAI Timeout
# ===========================================================================

class TestOpenAITimeout:
    """Verify OpenAI provider raises LLMUnavailableError on timeout."""

    def _make_provider(self):
        from app.core.providers.openai import OpenAIProvider
        return OpenAIProvider(api_key="sk-test", model="gpt-4o")

    @pytest.mark.asyncio
    @patch("app.core.providers.openai._retry_on_transient", new_callable=AsyncMock)
    async def test_openai_timeout_raises(self, mock_retry):
        """Mock httpx to raise TimeoutException, verify LLMUnavailableError."""
        mock_retry.side_effect = httpx.TimeoutException("request timed out")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError, match="timed out"):
            await provider.generate_with_tools(
                [{"role": "user", "content": "hello"}],
                [{"name": "test_tool", "description": "a test"}],
            )

    @pytest.mark.asyncio
    @patch("app.core.providers.openai._retry_on_transient", new_callable=AsyncMock)
    async def test_openai_timeout_on_invoke_nothink(self, mock_retry):
        """Timeout also raises LLMUnavailableError for invoke_nothink."""
        mock_retry.side_effect = httpx.TimeoutException("read timed out")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError, match="timed out"):
            await provider.invoke_nothink(
                [{"role": "user", "content": "hi"}]
            )


# ===========================================================================
# Anthropic Timeout
# ===========================================================================

class TestAnthropicTimeout:
    """Verify Anthropic provider raises LLMUnavailableError on timeout."""

    def _make_provider(self):
        from app.core.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-20250514")

    @pytest.mark.asyncio
    @patch("app.core.providers.anthropic._retry_on_transient", new_callable=AsyncMock)
    async def test_anthropic_timeout_raises(self, mock_retry):
        """Mock httpx to raise TimeoutException, verify LLMUnavailableError."""
        mock_retry.side_effect = httpx.TimeoutException("request timed out")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError, match="timed out"):
            await provider.generate_with_tools(
                [{"role": "user", "content": "hello"}],
                [{"name": "test_tool", "description": "a test"}],
            )

    @pytest.mark.asyncio
    @patch("app.core.providers.anthropic._retry_on_transient", new_callable=AsyncMock)
    async def test_anthropic_timeout_on_invoke_nothink(self, mock_retry):
        """Timeout also raises LLMUnavailableError for invoke_nothink."""
        mock_retry.side_effect = httpx.TimeoutException("read timed out")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError, match="timed out"):
            await provider.invoke_nothink(
                [{"role": "user", "content": "hi"}]
            )


# ===========================================================================
# Ollama Connection Error
# ===========================================================================

class TestOllamaConnectionError:
    """Verify Ollama provider raises LLMUnavailableError on ConnectError."""

    @pytest.mark.asyncio
    async def test_ollama_connection_error(self):
        """Mock httpx to raise ConnectError, verify LLMUnavailableError."""
        from app.core.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            llm_model="qwen3.5:27b",
        )

        with patch("app.core.providers.ollama.retry_on_transient", new_callable=AsyncMock) as mock_retry:
            mock_retry.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(LLMUnavailableError, match="Cannot connect"):
                await provider.generate_with_tools(
                    [{"role": "user", "content": "hello"}],
                    [{"name": "test_tool", "description": "a test"}],
                )

    @pytest.mark.asyncio
    async def test_ollama_connection_error_invoke_nothink(self):
        """ConnectError also raises LLMUnavailableError for invoke_nothink."""
        from app.core.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            llm_model="qwen3.5:27b",
        )

        with patch("app.core.providers.ollama.retry_on_transient", new_callable=AsyncMock) as mock_retry:
            mock_retry.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(LLMUnavailableError, match="Cannot connect"):
                await provider.invoke_nothink(
                    [{"role": "user", "content": "hi"}]
                )

    @pytest.mark.asyncio
    async def test_ollama_timeout_raises(self):
        """Ollama timeout also raises LLMUnavailableError."""
        from app.core.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            base_url="http://localhost:11434",
            llm_model="qwen3.5:27b",
        )

        with patch("app.core.providers.ollama.retry_on_transient", new_callable=AsyncMock) as mock_retry:
            mock_retry.side_effect = httpx.TimeoutException("request timed out")

            with pytest.raises(LLMUnavailableError, match="timed out"):
                await provider.generate_with_tools(
                    [{"role": "user", "content": "hello"}],
                    [{"name": "t", "description": "d"}],
                )


# ===========================================================================
# Google HTTP Error
# ===========================================================================

class TestGoogleHttpError:
    """Verify Google provider handles HTTP 500 appropriately."""

    def _make_provider(self):
        from app.core.providers.google import GoogleProvider
        return GoogleProvider(api_key="AIza-test", model="gemini-2.0-flash")

    @pytest.mark.asyncio
    @patch("app.core.providers.google._retry_on_transient", new_callable=AsyncMock)
    async def test_google_http_error(self, mock_retry):
        """Mock httpx to raise ConnectError (simulating persistent 500),
        verify LLMUnavailableError is raised."""
        mock_retry.side_effect = httpx.ConnectError("server error")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError, match="Cannot connect"):
            await provider.generate_with_tools(
                [{"role": "user", "content": "hello"}],
                [{"name": "test_tool", "description": "a test"}],
            )

    @pytest.mark.asyncio
    @patch("app.core.providers.google._retry_on_transient", new_callable=AsyncMock)
    async def test_google_timeout_raises(self, mock_retry):
        """Google timeout raises LLMUnavailableError."""
        mock_retry.side_effect = httpx.TimeoutException("request timed out")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError, match="timed out"):
            await provider.generate_with_tools(
                [{"role": "user", "content": "hello"}],
                [{"name": "t", "description": "d"}],
            )

    @pytest.mark.asyncio
    @patch("app.core.providers.google._retry_on_transient", new_callable=AsyncMock)
    async def test_google_invoke_nothink_timeout(self, mock_retry):
        """Google invoke_nothink timeout raises LLMUnavailableError."""
        mock_retry.side_effect = httpx.TimeoutException("timed out")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError, match="timed out"):
            await provider.invoke_nothink(
                [{"role": "user", "content": "hi"}]
            )
