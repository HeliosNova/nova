"""The Ollama Latency Monitor says WHERE the loaded models run (2026-10-08).

For 16 hours the 9B ran 100% on the CPU (GPU discovery had timed out at
container start) and the latency check reported "ollama healthy (12ms)" every
two hours: it only timed ``/api/tags``. The daily readout had no line that
could say "CPU". Now the check also reads ``/api/ps`` and a loaded model with
no VRAM on a box whose provider can see a GPU is an error, named per model.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.providers.ollama import OllamaProvider
from app.monitors.heartbeat import HeartbeatLoop, MonitorStore


@pytest.fixture
def loop(db):
    return HeartbeatLoop(MonitorStore(db))


def _provider(models):
    p = MagicMock()
    p.check_health = AsyncMock(return_value=True)
    p.loaded_models = AsyncMock(return_value=models)
    return p


@pytest.mark.asyncio
async def test_a_model_with_no_vram_is_reported_as_cpu(loop):
    provider = _provider([{"name": "nova-ft:latest", "size": 10_000_000_000, "size_vram": 0}])
    with patch("app.core.llm.get_provider", return_value=provider):
        result = await loop._execute_ollama_latency_check()
    assert "CPU" in result, result
    assert "nova-ft" in result, result
    assert "error" in result.lower() or "❌" in result, result


@pytest.mark.asyncio
async def test_a_fully_resident_model_is_reported_as_gpu(loop):
    provider = _provider([{"name": "nova-ft:latest", "size": 10_000_000_000, "size_vram": 10_000_000_000}])
    with patch("app.core.llm.get_provider", return_value=provider):
        result = await loop._execute_ollama_latency_check()
    assert "100% GPU" in result, result
    assert "CPU" not in result, result


@pytest.mark.asyncio
async def test_a_partial_offload_shows_the_share(loop):
    provider = _provider([{"name": "qwen3.8:27b", "size": 17_000_000_000, "size_vram": 8_500_000_000}])
    with patch("app.core.llm.get_provider", return_value=provider):
        result = await loop._execute_ollama_latency_check()
    assert "50% GPU" in result, result


@pytest.mark.asyncio
async def test_nothing_loaded_is_not_an_error(loop):
    provider = _provider([])
    with patch("app.core.llm.get_provider", return_value=provider):
        result = await loop._execute_ollama_latency_check()
    assert "healthy" in result.lower(), result
    assert "CPU" not in result


@pytest.mark.asyncio
async def test_a_provider_without_the_method_still_reports_latency(loop):
    provider = MagicMock()
    provider.check_health = AsyncMock(return_value=True)
    del provider.loaded_models
    with patch("app.core.llm.get_provider", return_value=provider):
        result = await loop._execute_ollama_latency_check()
    assert "healthy" in result.lower(), result


@pytest.mark.asyncio
async def test_provider_loaded_models_reads_api_ps():
    payload = {"models": [{"name": "nova-ft:latest", "size": 10, "size_vram": 10, "details": {}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(200, json=payload)

    provider = OllamaProvider(base_url="http://ollama.test")
    provider._client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    models = await provider.loaded_models()
    assert models == [{"name": "nova-ft:latest", "size": 10, "size_vram": 10}]


@pytest.mark.asyncio
async def test_provider_loaded_models_is_empty_when_ollama_is_down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = OllamaProvider(base_url="http://ollama.test")
    provider._client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    assert await provider.loaded_models() == []
