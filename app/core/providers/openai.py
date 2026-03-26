"""OpenAI LLM provider — raw httpx, no SDK dependency.

Supports GPT-4o and compatible models via the /v1/chat/completions endpoint.
Tool calls use OpenAI's native function calling (structured, no text parsing).
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


class OpenAIProvider:
    """OpenAI ChatCompletion provider via httpx."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._api_key = api_key
        self._model = model
        self._client: httpx.AsyncClient | None = None

    @property
    def capabilities(self) -> "ProviderCapabilities":
        from app.core.llm import ProviderCapabilities
        return ProviderCapabilities(
            needs_emphatic_prompts=False,
            supports_native_tools=True,
            supports_thinking=False,
            json_prefix_behavior="ignore",
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=config.OPENAI_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
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
        """Check if OpenAI API is reachable by listing models."""
        try:
            client = self._get_client()
            resp = await client.get("/v1/models")
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Nova tool list to OpenAI function-calling format."""
        functions = []
        for t in tools:
            functions.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return functions

    def _extract_tool_calls(self, message: dict) -> list[ToolCall]:
        """Extract all structured tool calls from an OpenAI response message."""
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return []
        result = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            result.append(ToolCall(tool=name, args=args))
        return result

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

        token_key = "max_completion_tokens" if config.OPENAI_BASE_URL == "https://api.openai.com" else "max_tokens"
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            token_key: max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = await _retry_on_transient(client, "POST", "/v1/chat/completions", json=payload)
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "")
            return content.strip()
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to OpenAI API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("OpenAI request timed out.")
        except LLMUnavailableError:
            raise
        except Exception as e:
            logger.error("[invoke_nothink] OpenAI call failed: %s", e)
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

        # Convert images to OpenAI vision format
        send_messages = list(messages)
        if images and send_messages:
            for i in range(len(send_messages) - 1, -1, -1):
                if send_messages[i].get("role") == "user":
                    text = send_messages[i].get("content", "")
                    content_parts = [{"type": "text", "text": text}]
                    for img in images:
                        from app.core.text_utils import detect_image_mime
                        _mime = detect_image_mime(img)
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{_mime};base64,{img}"},
                        })
                    send_messages[i] = {"role": "user", "content": content_parts}
                    break

        token_key = "max_completion_tokens" if config.OPENAI_BASE_URL == "https://api.openai.com" else "max_tokens"
        payload: dict = {
            "model": model,
            "messages": send_messages,
            "temperature": temperature,
            token_key: max_tokens,
        }

        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        try:
            resp = await _retry_on_transient(client, "POST", "/v1/chat/completions", json=payload)
        except LLMUnavailableError:
            raise
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to OpenAI API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("OpenAI request timed out.")

        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content", "") or ""
        tool_calls = self._extract_tool_calls(message)
        usage = data.get("usage") or None
        # Optionally remap token counting to use completion_tokens only
        if usage and config.OPENAI_USE_COMPLETION_TOKENS:
            usage = dict(usage)
            usage["total_tokens"] = usage.get("completion_tokens", 0)

        # Detect truncation due to token limit
        finish_reason = data["choices"][0].get("finish_reason", "")
        if finish_reason == "length":
            logger.warning("OpenAI response truncated (finish_reason=length)")
            content += "\n\n[Warning: Response was truncated due to token limit]"

        return GenerationResult(
            content=content.strip(),
            tool_calls=tool_calls,
            raw=data,
            usage=usage,
            stop_reason=data["choices"][0].get("finish_reason", ""),
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
        """Stream response from OpenAI. OpenAI doesn't have native thinking,
        so we stream content only (no thinking chunks)."""
        model = model or self._model
        client = self._get_client()

        token_key = "max_completion_tokens" if config.OPENAI_BASE_URL == "https://api.openai.com" else "max_tokens"
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            token_key: max_tokens,
            "stream": True,
        }

        # Pass tools for streaming tool calls
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        _yielded_done = False
        try:
            async with client.stream(
                "POST", "/v1/chat/completions", json=payload,
                timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0, read=300.0),
            ) as resp:
                resp.raise_for_status()
                # Track streaming tool call accumulation
                _tool_call_parts: dict[int, dict] = {}  # index -> {name, arguments}
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        # If we accumulated tool calls, emit them as structured tool_call
                        if _tool_call_parts:
                            for _idx in sorted(_tool_call_parts):
                                _tc = _tool_call_parts[_idx]
                                try:
                                    _parsed_args = json.loads(_tc.get("arguments", "{}"))
                                except json.JSONDecodeError:
                                    raw = _tc.get("arguments", "")
                                    logger.warning("OpenAI streaming: failed to parse tool call args: %s", raw)
                                    _parsed_args = {"_parse_error": True, "_raw_args": raw[:500]}
                                yield StreamChunk(
                                    tool_call=ToolCall(tool=_tc.get("name", ""), args=_parsed_args)
                                )
                        _yielded_done = True
                        yield StreamChunk(done=True)
                        return
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield StreamChunk(content=content)
                    # Handle streaming tool_calls delta
                    tc_deltas = delta.get("tool_calls", [])
                    for tc_delta in tc_deltas:
                        idx = tc_delta.get("index", 0)
                        if idx not in _tool_call_parts:
                            _tool_call_parts[idx] = {"name": "", "arguments": ""}
                        func = tc_delta.get("function", {})
                        if "name" in func:
                            _tool_call_parts[idx]["name"] = func["name"]
                        if "arguments" in func:
                            _tool_call_parts[idx]["arguments"] += func["arguments"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503):
                raise LLMUnavailableError(f"OpenAI streaming error {e.response.status_code}")
            raise LLMUnavailableError(f"OpenAI error: {e}")
        except LLMUnavailableError:
            raise
        except httpx.ReadError:
            raise LLMUnavailableError("Connection lost during OpenAI streaming")
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to OpenAI API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("OpenAI request timed out.")
        finally:
            if not _yielded_done:
                yield StreamChunk(done=True)
