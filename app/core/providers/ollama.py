"""Ollama LLM provider — raw HTTP, no LangChain.

Qwen3.5-specific tricks (thinking suppression, JSON prefixing) handled here.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

import httpx

from app.config import config
from app.core.llm import (
    GenerationResult,
    LLMUnavailableError,
    StreamChunk,
    ToolCall,
    _extract_tool_call,
    _find_balanced_json,
    _strip_think_tags,
)
from app.core.providers._retry import retry_on_transient

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Ollama LLM provider — raw HTTP, no LangChain.

    Qwen3.5-specific tricks (thinking suppression, JSON prefixing) handled here.
    """

    def __init__(self, base_url: str | None = None, llm_model: str | None = None, embed_model: str | None = None):
        self._base_url = base_url or config.OLLAMA_URL
        self._llm_model = llm_model or config.LLM_MODEL
        self._embed_model = embed_model or config.EMBEDDING_MODEL
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            try:
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0),
                )
            except Exception:
                self._client = None
                raise
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def check_health(self) -> bool:
        """Check if Ollama is reachable by listing models."""
        try:
            client = self._get_client()
            resp = await client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def invoke_nothink(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        json_prefix: str = "[{",
        max_tokens: int = 1000,
        temperature: float = 0.1,
        model: str | None = None,
    ) -> str:
        model = model or self._llm_model
        client = self._get_client()
        is_qwen = "qwen" in model.lower()

        ollama_messages = list(messages)

        # Assistant prefix trick: force model to skip thinking
        if is_qwen:
            prefix_content = "<think>\n\n</think>\n"
            if json_mode and json_prefix:
                prefix_content += json_prefix
            ollama_messages.append({"role": "assistant", "content": prefix_content})
        elif json_mode and json_prefix:
            ollama_messages.append({"role": "assistant", "content": json_prefix})

        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": ollama_messages,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "repeat_penalty": 1.1 if json_mode else 1.5,
            },
        }

        if json_mode:
            payload["format"] = "json"

        try:
            resp = await retry_on_transient(client, "POST", "/api/chat", json=payload)
            data = resp.json()
            content = data.get("message", {}).get("content", "")

            if json_mode and json_prefix:
                content = json_prefix + content

            content = _strip_think_tags(content)

            if json_mode:
                content = _find_balanced_json(content, json_prefix)

            return content.strip()

        except httpx.ConnectError as e:
            logger.warning("[invoke_nothink] Cannot connect to Ollama: %s", e)
            raise LLMUnavailableError(f"Cannot connect to Ollama: {e}")
        except httpx.TimeoutException as e:
            logger.warning("[invoke_nothink] Ollama request timed out: %s", e)
            raise LLMUnavailableError(f"Ollama request timed out: {e}")
        except LLMUnavailableError:
            raise
        except Exception as e:
            logger.warning("[invoke_nothink] Unexpected error: %s", e)
            return ""

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2000,
        images: list[str] | None = None,
    ) -> GenerationResult:
        model = model or self._llm_model
        client = self._get_client()

        is_qwen = "qwen" in model.lower()

        # If images are provided, ensure the last user message includes them
        send_messages = list(messages)
        if images and send_messages:
            for i in range(len(send_messages) - 1, -1, -1):
                if send_messages[i].get("role") == "user":
                    send_messages[i] = {**send_messages[i], "images": images}
                    break

        # Assistant prefix trick: suppress thinking for Qwen models (only when extended thinking is disabled)
        if is_qwen and not config.ENABLE_EXTENDED_THINKING:
            send_messages.append({"role": "assistant", "content": "<think>\n\n</think>\n"})

        try:
            resp = await retry_on_transient(client, "POST", "/api/chat", json={
                "model": model,
                "stream": False,
                "think": False,
                "messages": send_messages,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            })
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Ollama. Is it running?")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Ollama request timed out.")
        except LLMUnavailableError:
            raise

        data = resp.json()
        content = data.get("message", {}).get("content", "")
        content = _strip_think_tags(content).strip()

        tool_call = _extract_tool_call(content, tools)

        return GenerationResult(content=content, tool_call=tool_call, raw=data)

    async def stream_with_thinking(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response with thinking enabled. Yields incremental chunks."""
        model = model or self._llm_model
        client = self._get_client()

        max_stream_retries = 2
        for _stream_attempt in range(max_stream_retries + 1):
            try:
                async with client.stream(
                    "POST",
                    "/api/chat",
                    json={
                        "model": model,
                        "stream": True,
                        "think": True,
                        "messages": messages,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                    timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0, read=60.0),
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk_data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        msg = chunk_data.get("message", {})
                        thinking_delta = msg.get("thinking", "")
                        content_delta = msg.get("content", "")
                        done = chunk_data.get("done", False)

                        if thinking_delta or content_delta or done:
                            yield StreamChunk(
                                thinking=thinking_delta,
                                content=content_delta,
                                done=done,
                            )
                return  # Success — exit retry loop
            except httpx.ConnectError:
                if _stream_attempt < max_stream_retries:
                    import asyncio
                    logger.warning("Ollama stream connect error, retrying (%d/%d)", _stream_attempt + 1, max_stream_retries)
                    await asyncio.sleep(2.0)
                    continue
                raise LLMUnavailableError("Cannot connect to Ollama. Is it running?")
            except httpx.TimeoutException:
                raise LLMUnavailableError("Ollama request timed out.")
