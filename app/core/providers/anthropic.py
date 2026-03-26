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

# Anthropic beta header for extended thinking — loaded from config


class AnthropicProvider:
    """Anthropic Messages API provider via httpx."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self._api_key = api_key
        self._model = model
        self._client: httpx.AsyncClient | None = None

    @property
    def capabilities(self) -> "ProviderCapabilities":
        from app.core.llm import ProviderCapabilities
        return ProviderCapabilities(
            needs_emphatic_prompts=False,
            supports_native_tools=True,
            supports_thinking=True,
            json_prefix_behavior="prepend",
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=config.ANTHROPIC_BASE_URL,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": config.ANTHROPIC_API_VERSION,
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
        # Merge consecutive same-role messages (Anthropic rejects them with 400)
        merged: list[dict] = []
        for msg in conversation:
            if merged and merged[-1]["role"] == msg["role"] and isinstance(merged[-1]["content"], str) and isinstance(msg["content"], str):
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(dict(msg))
        return system, merged

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Nova tool list to Anthropic tool format."""
        result = []
        for t in tools:
            result.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    def _extract_tool_calls(self, content_blocks: list[dict]) -> list[ToolCall]:
        """Extract all structured tool calls from Anthropic tool_use content blocks."""
        result = []
        for block in content_blocks:
            if block.get("type") == "tool_use":
                result.append(ToolCall(
                    tool=block.get("name", ""),
                    args=block.get("input", {}),
                ))
        return result

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
            conversation = [{"role": "user", "content": "Follow the system instructions."}]

        if json_mode:
            conversation.append({"role": "assistant", "content": json_prefix})

        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation,
        }
        if system:
            payload["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]

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
            logger.error("[invoke_nothink] Anthropic call failed: %s", e)
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
        tool_choice: str | None = None,
    ) -> GenerationResult:
        model = model or self._model
        client = self._get_client()

        system, conversation = self._convert_messages(messages)
        if not conversation:
            conversation = [{"role": "user", "content": "Follow the system instructions."}]

        # Convert images to Anthropic vision format
        if images and conversation:
            for i in range(len(conversation) - 1, -1, -1):
                if conversation[i].get("role") == "user":
                    text = conversation[i].get("content", "")
                    content_parts = []
                    for img in images:
                        from app.core.text_utils import detect_image_mime
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": detect_image_mime(img),
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
            payload["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]
        if tools:
            payload["tools"] = self._convert_tools(tools)
            if tool_choice is not None:
                payload["tool_choice"] = {"type": tool_choice}

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
        tool_calls = self._extract_tool_calls(content_blocks)
        thinking = self._extract_thinking(content_blocks)
        usage = data.get("usage") or None

        # Detect truncation due to token limit
        stop_reason = data.get("stop_reason", "")
        if stop_reason == "max_tokens":
            logger.warning("Anthropic response truncated (stop_reason=max_tokens)")
            content += "\n\n[Warning: Response was truncated due to token limit]"

        return GenerationResult(
            content=content.strip(),
            tool_calls=tool_calls,
            raw=data,
            thinking=thinking,
            usage=usage,
            stop_reason=data.get("stop_reason", ""),
        )

    async def stream_with_thinking(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4000,
        tool_choice: str | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response from Anthropic with thinking support."""
        model = model or self._model
        client = self._get_client()

        system, conversation = self._convert_messages(messages)
        if not conversation:
            conversation = [{"role": "user", "content": "Follow the system instructions."}]

        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation,
            "stream": True,
        }
        if system:
            payload["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]
        if tools:
            payload["tools"] = self._convert_tools(tools)
            if tool_choice is not None:
                payload["tool_choice"] = {"type": tool_choice}

        # Enable extended thinking only when configured
        headers = {}
        if config.ENABLE_EXTENDED_THINKING:
            if temperature != 1.0:
                logger.debug("Anthropic thinking mode requires temperature=1.0 (overriding %.1f)", temperature)
            payload["temperature"] = 1.0  # Required by Anthropic when thinking is enabled
            thinking_budget = max(1024, max_tokens // 2)
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            headers = {"anthropic-beta": config.ANTHROPIC_BETA_HEADER}

        _yielded_done = False
        try:
            async with client.stream(
                "POST", "/v1/messages", json=payload, headers=headers,
                timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0, read=300.0),
            ) as resp:
                resp.raise_for_status()
                in_thinking = False
                _current_tool: dict = {"name": "", "input_json": ""}
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
                        block_type = block.get("type", "")
                        in_thinking = block_type == "thinking"
                        if block_type == "tool_use":
                            _current_tool = {"name": block.get("name", ""), "input_json": ""}

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        delta_type = delta.get("type", "")
                        if delta_type == "thinking_delta":
                            yield StreamChunk(thinking=delta.get("thinking", ""))
                        elif delta_type == "text_delta":
                            yield StreamChunk(content=delta.get("text", ""))
                        elif delta_type == "input_json_delta":
                            _current_tool["input_json"] += delta.get("partial_json", "")

                    elif event_type == "content_block_stop":
                        if _current_tool.get("name"):
                            try:
                                args = json.loads(_current_tool["input_json"]) if _current_tool["input_json"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            yield StreamChunk(
                                tool_call=ToolCall(tool=_current_tool["name"], args=args)
                            )
                            _current_tool = {"name": "", "input_json": ""}
                        in_thinking = False

                    elif event_type == "message_stop":
                        _yielded_done = True
                        yield StreamChunk(done=True)
                        return

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503):
                raise LLMUnavailableError(f"Anthropic streaming error {e.response.status_code}")
            raise LLMUnavailableError(f"Anthropic error: {e}")
        except LLMUnavailableError:
            raise
        except httpx.ReadError:
            raise LLMUnavailableError("Connection lost during Anthropic streaming")
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Anthropic API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Anthropic request timed out.")
        finally:
            if not _yielded_done:
                yield StreamChunk(done=True)
