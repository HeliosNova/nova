"""Google Gemini LLM provider — raw httpx, no SDK dependency.

Supports Gemini models via the REST API.
Tool calls use Gemini's native functionDeclarations (structured).
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


class GoogleProvider:
    """Google Gemini API provider via httpx."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
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
                base_url=config.GOOGLE_BASE_URL,
                headers={"x-goog-api-key": self._api_key},
                timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def check_health(self) -> bool:
        """Check if Google Gemini API is reachable by listing models."""
        try:
            client = self._get_client()
            resp = await client.get("/v1beta/models")
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert Nova messages to Gemini format (contents array).

        Concatenates leading system messages into one system instruction.
        Mid-conversation system messages (tool results, critiques) become user messages.
        """
        system_parts = []
        contents = []
        seen_first_non_system = False
        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                if not seen_first_non_system:
                    system_parts.append(text)
                else:
                    contents.append({"role": "user", "parts": [{"text": f"[System] {text}"}]})
            elif role == "assistant":
                seen_first_non_system = True
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                seen_first_non_system = True
                contents.append({"role": "user", "parts": [{"text": text}]})
        system = "\n\n".join(system_parts) if system_parts else ""
        # Merge consecutive same-role messages (Gemini rejects them with 400)
        merged: list[dict] = []
        for msg in contents:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["parts"].extend(msg["parts"])
            else:
                merged.append(dict(msg))
        return system, merged

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Nova tool list to Gemini functionDeclarations format."""
        declarations = []
        for t in tools:
            declarations.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            })
        return [{"functionDeclarations": declarations}]

    def _extract_tool_calls(self, parts: list[dict]) -> list[ToolCall]:
        """Extract all structured tool calls from Gemini functionCall parts."""
        result = []
        for part in parts:
            fc = part.get("functionCall")
            if fc:
                result.append(ToolCall(
                    tool=fc.get("name", ""),
                    args=fc.get("args", {}),
                ))
        return result

    def _extract_text(self, parts: list[dict]) -> str:
        """Extract text content from Gemini parts."""
        texts = []
        for part in parts:
            if "text" in part:
                texts.append(part["text"])
        return "\n".join(texts)

    def _url(self, model: str, method: str) -> str:
        """Build Gemini API URL."""
        return f"/v1beta/models/{model}:{method}"

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

        system, contents = self._convert_messages(messages)
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Continue with the instructions provided."}]}]

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        try:
            resp = await _retry_on_transient(client, "POST", self._url(model, "generateContent"), json=payload)
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return self._extract_text(parts).strip()
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Google Gemini API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Google Gemini request timed out.")
        except LLMUnavailableError:
            raise
        except Exception as e:
            logger.error("[invoke_nothink] Google call failed: %s", e)
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

        system, contents = self._convert_messages(messages)
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Continue with the instructions provided."}]}]

        # Convert images to Gemini inline format
        if images and contents:
            for i in range(len(contents) - 1, -1, -1):
                if contents[i].get("role") == "user":
                    parts = list(contents[i].get("parts", []))
                    from app.core.text_utils import detect_image_mime
                    for img in images:
                        parts.insert(0, {
                            "inlineData": {
                                "mimeType": detect_image_mime(img),
                                "data": img,
                            }
                        })
                    contents[i] = {"role": "user", "parts": parts}
                    break

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = self._convert_tools(tools)
            if tool_choice is not None:
                payload["toolConfig"] = {"functionCallingConfig": {"mode": tool_choice.upper()}}

        try:
            resp = await _retry_on_transient(client, "POST", self._url(model, "generateContent"), json=payload)
        except LLMUnavailableError:
            raise
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Google Gemini API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Google Gemini request timed out.")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return GenerationResult(content="", tool_calls=[], raw=data)

        parts = candidates[0].get("content", {}).get("parts", [])
        content = self._extract_text(parts)
        tool_calls = self._extract_tool_calls(parts)
        usage = data.get("usageMetadata") or None

        # Detect truncation due to token limit
        finish_reason = candidates[0].get("finishReason", "")
        if finish_reason == "MAX_TOKENS":
            logger.warning("Google Gemini response truncated (finishReason=MAX_TOKENS)")
            content += "\n\n[Warning: Response was truncated due to token limit]"

        return GenerationResult(
            content=content.strip(),
            tool_calls=tool_calls,
            raw=data,
            usage=usage,
            stop_reason=candidates[0].get("finishReason", "") if candidates else "",
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
        """Stream response from Gemini via streamGenerateContent."""
        model = model or self._model
        client = self._get_client()

        system, contents = self._convert_messages(messages)
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Continue with the instructions provided."}]}]

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = self._convert_tools(tools)
            if tool_choice is not None:
                payload["toolConfig"] = {"functionCallingConfig": {"mode": tool_choice.upper()}}

        _yielded_done = False
        try:
            # Gemini streaming uses alt=sse parameter
            url = self._url(model, "streamGenerateContent") + "?alt=sse"
            _got_sse_data = False
            async with client.stream(
                "POST", url, json=payload,
                timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0, read=300.0),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    _got_sse_data = True
                    data_str = line[6:]
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    candidates = chunk.get("candidates", [])
                    if not candidates:
                        continue

                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = self._extract_text(parts)
                    if text:
                        yield StreamChunk(content=text)

                    # Handle functionCall parts
                    for part in parts:
                        fc = part.get("functionCall")
                        if fc:
                            yield StreamChunk(
                                tool_call=ToolCall(tool=fc.get("name", ""), args=fc.get("args", {}))
                            )

                    # Check if generation is complete
                    finish = candidates[0].get("finishReason")
                    if finish and finish != "FINISH_REASON_UNSPECIFIED":
                        _yielded_done = True
                        yield StreamChunk(done=True)
                        return

            # SSE fallback: if response contained no "data: " lines, fall back to non-streaming
            if not _got_sse_data:
                logger.warning("Google streaming returned no SSE data, falling back to non-streaming")
                result = await self.generate_with_tools(
                    messages, tools, model=model, temperature=temperature,
                    max_tokens=max_tokens, tool_choice=tool_choice,
                )
                if result.content:
                    yield StreamChunk(content=result.content)
                for tc in result.tool_calls:
                    yield StreamChunk(tool_call=tc)
                _yielded_done = True
                yield StreamChunk(done=True)
                return

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503):
                raise LLMUnavailableError(f"Google streaming error {e.response.status_code}")
            raise LLMUnavailableError(f"Google error: {e}")
        except LLMUnavailableError:
            raise
        except httpx.ReadError:
            raise LLMUnavailableError("Connection lost during Google streaming")
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Google Gemini API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Google Gemini request timed out.")
        finally:
            if not _yielded_done:
                yield StreamChunk(done=True)
