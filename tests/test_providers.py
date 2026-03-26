"""Tests for LLM provider factory and cloud providers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.llm import (
    GenerationResult,
    LLMUnavailableError,
    ToolCall,
    create_provider,
)


# ---------------------------------------------------------------------------
# Factory: create_provider()
# ---------------------------------------------------------------------------


class TestCreateProvider:
    """Test the provider factory returns the right provider type."""

    def test_ollama_default(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        import importlib, app.config
        importlib.reload(app.config)
        from app.core.providers.ollama import OllamaProvider
        p = create_provider(app.config.config)
        assert isinstance(p, OllamaProvider)

    def test_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        import importlib, app.config
        importlib.reload(app.config)
        from app.core.providers.openai import OpenAIProvider
        p = create_provider(app.config.config)
        assert isinstance(p, OpenAIProvider)

    def test_anthropic(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        import importlib, app.config
        importlib.reload(app.config)
        from app.core.providers.anthropic import AnthropicProvider
        p = create_provider(app.config.config)
        assert isinstance(p, AnthropicProvider)

    def test_google(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "google")
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")
        monkeypatch.setenv("GOOGLE_MODEL", "gemini-2.0-flash")
        import importlib, app.config
        importlib.reload(app.config)
        from app.core.providers.google import GoogleProvider
        p = create_provider(app.config.config)
        assert isinstance(p, GoogleProvider)

    def test_unknown_falls_back_to_ollama(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        import importlib, app.config
        importlib.reload(app.config)
        from app.core.providers.ollama import OllamaProvider
        p = create_provider(app.config.config)
        assert isinstance(p, OllamaProvider)


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


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    """Test OpenAI provider with mocked HTTP responses."""

    def _make_provider(self):
        from app.core.providers.openai import OpenAIProvider
        return OpenAIProvider(api_key="sk-test", model="gpt-4o")

    @pytest.mark.asyncio
    @patch("app.core.providers.openai._retry_on_transient", new_callable=AsyncMock)
    async def test_invoke_nothink_returns_content(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {
            "choices": [{"message": {"content": "Hello world"}}]
        })
        provider = self._make_provider()

        result = await provider.invoke_nothink([{"role": "user", "content": "Hi"}])
        assert result == "Hello world"

    @pytest.mark.asyncio
    @patch("app.core.providers.openai._retry_on_transient", new_callable=AsyncMock)
    async def test_generate_with_tools_extracts_tool_call(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "test"}'
                        }
                    }]
                }
            }]
        })
        provider = self._make_provider()

        tools = [{"name": "web_search", "description": "Search", "parameters": {}}]
        result = await provider.generate_with_tools(
            [{"role": "user", "content": "search for test"}], tools
        )
        assert isinstance(result, GenerationResult)
        assert result.tool_calls
        assert result.tool_calls[0].tool == "web_search"
        assert result.tool_calls[0].args == {"query": "test"}

    @pytest.mark.asyncio
    @patch("app.core.providers.openai._retry_on_transient", new_callable=AsyncMock)
    async def test_connect_error_raises_llm_unavailable(self, mock_retry):
        mock_retry.side_effect = httpx.ConnectError("refused")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError):
            await provider.generate_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"name": "t", "description": "d"}],
            )

    @pytest.mark.asyncio
    @patch("app.core.providers.openai._retry_on_transient", new_callable=AsyncMock)
    async def test_timeout_raises_llm_unavailable(self, mock_retry):
        mock_retry.side_effect = httpx.TimeoutException("timeout")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError):
            await provider.generate_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"name": "t", "description": "d"}],
            )

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        provider = self._make_provider()
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_response(200, {"data": []}))
        provider._client = mock_client

        assert await provider.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self):
        provider = self._make_provider()
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        provider._client = mock_client

        assert await provider.check_health() is False


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Test Anthropic provider with mocked HTTP responses."""

    def _make_provider(self):
        from app.core.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-20250514")

    @pytest.mark.asyncio
    @patch("app.core.providers.anthropic._retry_on_transient", new_callable=AsyncMock)
    async def test_invoke_nothink_returns_content(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {
            "content": [{"type": "text", "text": "Hello from Claude"}]
        })
        provider = self._make_provider()

        result = await provider.invoke_nothink([{"role": "user", "content": "Hi"}])
        assert result == "Hello from Claude"

    @pytest.mark.asyncio
    @patch("app.core.providers.anthropic._retry_on_transient", new_callable=AsyncMock)
    async def test_generate_with_tools_extracts_tool_use(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {
            "content": [
                {"type": "text", "text": "Let me search for that."},
                {"type": "tool_use", "name": "web_search", "input": {"query": "test"}},
            ]
        })
        provider = self._make_provider()

        tools = [{"name": "web_search", "description": "Search", "parameters": {}}]
        result = await provider.generate_with_tools(
            [{"role": "user", "content": "search"}], tools
        )
        assert result.tool_calls
        assert result.tool_calls[0].tool == "web_search"
        assert result.tool_calls[0].args == {"query": "test"}

    @pytest.mark.asyncio
    @patch("app.core.providers.anthropic._retry_on_transient", new_callable=AsyncMock)
    async def test_thinking_extraction(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Answer"},
            ]
        })
        provider = self._make_provider()

        result = await provider.generate_with_tools(
            [{"role": "user", "content": "think"}],
            [{"name": "t", "description": "d"}],
        )
        assert result.thinking == "Let me think..."
        assert result.content == "Answer"

    @pytest.mark.asyncio
    @patch("app.core.providers.anthropic._retry_on_transient", new_callable=AsyncMock)
    async def test_connect_error_raises_llm_unavailable(self, mock_retry):
        mock_retry.side_effect = httpx.ConnectError("refused")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError):
            await provider.generate_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"name": "t", "description": "d"}],
            )

    @pytest.mark.asyncio
    async def test_check_health_429_is_healthy(self):
        provider = self._make_provider()
        mock_client = MagicMock()
        resp = _mock_response(429, {})
        resp.raise_for_status = MagicMock()  # Don't raise for health check
        mock_client.get = AsyncMock(return_value=resp)
        provider._client = mock_client

        assert await provider.check_health() is True


# ---------------------------------------------------------------------------
# Google Provider
# ---------------------------------------------------------------------------


class TestGoogleProvider:
    """Test Google Gemini provider with mocked HTTP responses."""

    def _make_provider(self):
        from app.core.providers.google import GoogleProvider
        return GoogleProvider(api_key="AIza-test", model="gemini-2.0-flash")

    @pytest.mark.asyncio
    @patch("app.core.providers.google._retry_on_transient", new_callable=AsyncMock)
    async def test_invoke_nothink_returns_content(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {
            "candidates": [{"content": {"parts": [{"text": "Hello from Gemini"}]}}]
        })
        provider = self._make_provider()

        result = await provider.invoke_nothink([{"role": "user", "content": "Hi"}])
        assert result == "Hello from Gemini"

    @pytest.mark.asyncio
    @patch("app.core.providers.google._retry_on_transient", new_callable=AsyncMock)
    async def test_generate_with_tools_extracts_function_call(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {
            "candidates": [{
                "content": {
                    "parts": [
                        {"functionCall": {"name": "web_search", "args": {"query": "test"}}}
                    ]
                }
            }]
        })
        provider = self._make_provider()

        tools = [{"name": "web_search", "description": "Search", "parameters": {}}]
        result = await provider.generate_with_tools(
            [{"role": "user", "content": "search"}], tools
        )
        assert result.tool_calls
        assert result.tool_calls[0].tool == "web_search"

    @pytest.mark.asyncio
    @patch("app.core.providers.google._retry_on_transient", new_callable=AsyncMock)
    async def test_empty_candidates_returns_empty(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {
            "candidates": []
        })
        provider = self._make_provider()

        result = await provider.generate_with_tools(
            [{"role": "user", "content": "hi"}],
            [{"name": "t", "description": "d"}],
        )
        assert result.content == ""
        assert not result.tool_calls

    @pytest.mark.asyncio
    @patch("app.core.providers.google._retry_on_transient", new_callable=AsyncMock)
    async def test_connect_error_raises_llm_unavailable(self, mock_retry):
        mock_retry.side_effect = httpx.ConnectError("refused")
        provider = self._make_provider()

        with pytest.raises(LLMUnavailableError):
            await provider.generate_with_tools(
                [{"role": "user", "content": "hi"}],
                [{"name": "t", "description": "d"}],
            )

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        provider = self._make_provider()
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_response(200, {"models": []}))
        provider._client = mock_client

        assert await provider.check_health() is True


# ---------------------------------------------------------------------------
# Retry Backoff (from test_audit_consolidated)
# ---------------------------------------------------------------------------


class TestRetryBackoff:
    """retry_on_transient uses exponential backoff + jitter."""

    @pytest.mark.asyncio
    async def test_429_jitter(self):
        from app.core.providers._retry import retry_on_transient

        call_count = 0
        sleep_delays = []

        async def mock_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count <= 2:
                resp.status_code = 429
                resp.headers = {"Retry-After": "2"}
            else:
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
            return resp

        client = MagicMock()
        client.request = mock_request

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await retry_on_transient(client, "POST", "/test")
            assert result.status_code == 200
            for call in mock_sleep.call_args_list:
                delay = call[0][0]
                sleep_delays.append(delay)
                assert delay >= 2.0

    @pytest.mark.asyncio
    async def test_500_exponential_backoff(self):
        from app.core.providers._retry import retry_on_transient

        call_count = 0
        sleep_delays = []

        async def mock_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count <= 2:
                resp.status_code = 500
                resp.text = "Internal Server Error"
            else:
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
            return resp

        client = MagicMock()
        client.request = mock_request

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await retry_on_transient(client, "POST", "/test")
            assert result.status_code == 200
            assert mock_sleep.call_count >= 1
            for call in mock_sleep.call_args_list:
                delay = call[0][0]
                sleep_delays.append(delay)
            if len(sleep_delays) >= 2:
                assert sleep_delays[1] > sleep_delays[0]


# ---------------------------------------------------------------------------
# Streaming Idle Timeout (from test_audit_consolidated)
# ---------------------------------------------------------------------------


class TestStreamingIdleTimeout:

    def _assert_has_streaming(self, provider_cls, provider_instance):
        """Verify the provider has stream_with_thinking method."""
        assert hasattr(provider_instance, "stream_with_thinking"), (
            f"{provider_cls.__name__} must have stream_with_thinking method"
        )
        assert callable(getattr(provider_instance, "stream_with_thinking"))

    def test_ollama_has_streaming(self):
        from app.core.providers.ollama import OllamaProvider
        provider = OllamaProvider()
        self._assert_has_streaming(OllamaProvider, provider)

    def test_ollama_client_timeout_has_connect(self):
        """The actual Ollama provider client should have a connect timeout of 10s."""
        from app.core.providers.ollama import OllamaProvider
        provider = OllamaProvider()
        client = provider._get_client()
        assert client.timeout.connect == 10.0

    def test_anthropic_has_streaming(self):
        from app.core.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        self._assert_has_streaming(AnthropicProvider, provider)

    def test_anthropic_client_timeout_has_connect(self):
        """The actual Anthropic provider client should have a connect timeout of 10s."""
        from app.core.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        client = provider._get_client()
        assert client.timeout.connect == 10.0

    def test_openai_has_streaming(self):
        from app.core.providers.openai import OpenAIProvider
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
        self._assert_has_streaming(OpenAIProvider, provider)

    def test_openai_client_timeout_has_connect(self):
        """The actual OpenAI provider client should have a connect timeout of 10s."""
        from app.core.providers.openai import OpenAIProvider
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")
        client = provider._get_client()
        assert client.timeout.connect == 10.0

    def test_google_has_streaming(self):
        from app.core.providers.google import GoogleProvider
        provider = GoogleProvider(api_key="AIza-test", model="gemini-2.0-flash")
        self._assert_has_streaming(GoogleProvider, provider)

    def test_google_client_timeout_has_connect(self):
        """The actual Google provider client should have a connect timeout of 10s."""
        from app.core.providers.google import GoogleProvider
        provider = GoogleProvider(api_key="AIza-test", model="gemini-2.0-flash")
        client = provider._get_client()
        assert client.timeout.connect == 10.0
