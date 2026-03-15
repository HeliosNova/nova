"""Anthropic LLM provider — raw httpx, no SDK dependency.

Supports Claude models via the /v1/messages endpoint.
Tool calls use Anthropic's native tool_use content blocks (structured).
Supports extended thinking when available.
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
)
from app.core.providers._retry import retry_on_transient as _retry_on_transient

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.anthropic.com"
_API_VERSION = "2023-06-01"


class AnthropicProvider:
    """Anthropic Messages API provider via httpx."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self._api_key = api_key
        self._model = model
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=_BASE_URL,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": _API_VERSION,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def check_health(self) -> bool:
        """Check if Anthropic API is reachable by listing models (no token cost)."""
        try:
            client = self._get_client()
            resp = await client.get("/v1/models")
            return resp.status_code in (200, 429)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Split system message from conversation messages for Anthropic format.

        Concatenates all system messages (brain.py injects tool results as system role).
        Non-initial system messages are injected as user messages to preserve ordering.
        """
        system_parts = []
        conversation = []
        seen_first_non_system = False
        for msg in messages:
            if msg["role"] == "system":
                content = msg.get("content", "")
                if not seen_first_non_system:
                    # Accumulate leading system messages into one system prompt
                    system_parts.append(content)
                else:
                    # Mid-conversation system messages (tool results, critiques) → user role
                    conversation.append({"role": "user", "content": f"[System] {content}"})
            else:
                seen_first_non_system = True
                conversation.append({"role": msg["role"], "content": msg.get("content", "")})
        system = "\n\n".join(system_parts) if system_parts else ""
        return system, conversation

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Nova_ tool list to Anthropic tool format."""
        result = []
        for t in tools:
            result.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    def _extract_tool_call(self, content_blocks: list[dict]) -> ToolCall | None:
        """Extract a structured tool call from Anthropic tool_use content blocks."""
        for block in content_blocks:
            if block.get("type") == "tool_use":
                return ToolCall(
                    tool=block.get("name", ""),
                    args=block.get("input", {}),
                )
        return None

    def _extract_text(self, content_blocks: list[dict]) -> str:
        """Extract text content from Anthropic content blocks."""
        parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)

    def _extract_thinking(self, content_blocks: list[dict]) -> str:
        """Extract thinking content from Anthropic content blocks."""
        parts = []
        for block in content_blocks:
            if block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

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
        model = model or self._model
        client = self._get_client()

        system, conversation = self._convert_messages(messages)
        if not conversation:
            conversation = [{"role": "user", "content": "Hello"}]

        if json_mode:
            conversation.append({"role": "assistant", "content": json_prefix})

        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation,
        }
        if system:
            payload["system"] = system

        try:
            resp = await _retry_on_transient(client, "POST", "/v1/messages", json=payload)
            data = resp.json()
            content = self._extract_text(data.get("content", []))
            if json_mode:
                content = json_prefix + content
            return content.strip()
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Anthropic API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Anthropic request timed out.")
        except LLMUnavailableError:
            raise
        except Exception as e:
            logger.warning("[invoke_nothink] Anthropic call failed: %s", e)
            return ""

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 2000,
        images: list[str] | None = None,
    ) -> GenerationResult:
        model = model or self._model
        client = self._get_client()

        system, conversation = self._convert_messages(messages)
        if not conversation:
            conversation = [{"role": "user", "content": "Hello"}]

        # Convert images to Anthropic vision format
        if images and conversation:
            for i in range(len(conversation) - 1, -1, -1):
                if conversation[i].get("role") == "user":
                    text = conversation[i].get("content", "")
                    content_parts = []
                    for img in images:
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img,
                            },
                        })
                    content_parts.append({"type": "text", "text": text})
                    conversation[i] = {"role": "user", "content": content_parts}
                    break

        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._convert_tools(tools)

        try:
            resp = await _retry_on_transient(client, "POST", "/v1/messages", json=payload)
        except LLMUnavailableError:
            raise
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Anthropic API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Anthropic request timed out.")

        data = resp.json()
        content_blocks = data.get("content", [])
        content = self._extract_text(content_blocks)
        tool_call = self._extract_tool_call(content_blocks)
        thinking = self._extract_thinking(content_blocks)
        usage = data.get("usage") or None

        return GenerationResult(
            content=content.strip(),
            tool_call=tool_call,
            raw=data,
            thinking=thinking,
            usage=usage,
        )

    async def stream_with_thinking(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response from Anthropic with thinking support."""
        model = model or self._model
        client = self._get_client()

        system, conversation = self._convert_messages(messages)
        if not conversation:
            conversation = [{"role": "user", "content": "Hello"}]

        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation,
            "stream": True,
        }
        if system:
            payload["system"] = system

        # Enable extended thinking
        if temperature != 1.0:
            logger.debug("Anthropic thinking mode requires temperature=1.0 (overriding %.1f)", temperature)
        payload["temperature"] = 1.0  # Required by Anthropic when thinking is enabled
        thinking_budget = max(1024, max_tokens // 2)
        payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        # Use beta header for thinking support
        headers = {"anthropic-beta": "interleaved-thinking-2025-05-14"}

        try:
            async with client.stream(
                "POST", "/v1/messages", json=payload, headers=headers,
                timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0, read=60.0),
            ) as resp:
                resp.raise_for_status()
                in_thinking = False
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "content_block_start":
                        block = event.get("content_block", {})
                        in_thinking = block.get("type") == "thinking"

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        delta_type = delta.get("type", "")
                        if delta_type == "thinking_delta":
                            yield StreamChunk(thinking=delta.get("thinking", ""))
                        elif delta_type == "text_delta":
                            yield StreamChunk(content=delta.get("text", ""))

                    elif event_type == "content_block_stop":
                        in_thinking = False

                    elif event_type == "message_stop":
                        yield StreamChunk(done=True)
                        return

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503):
                raise LLMUnavailableError(f"Anthropic streaming error {e.response.status_code}")
            raise LLMUnavailableError(f"Anthropic error: {e}")
        except LLMUnavailableError:
            raise
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Anthropic API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Anthropic request timed out.")
