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

_BASE_URL = "https://generativelanguage.googleapis.com"


class GoogleProvider:
    """Google Gemini API provider via httpx."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._api_key = api_key
        self._model = model
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=_BASE_URL,
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
        """Convert Nova_ messages to Gemini format (contents array).

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
        return system, contents

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Nova_ tool list to Gemini functionDeclarations format."""
        declarations = []
        for t in tools:
            declarations.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            })
        return [{"functionDeclarations": declarations}]

    def _extract_tool_call(self, parts: list[dict]) -> ToolCall | None:
        """Extract a structured tool call from Gemini functionCall parts."""
        for part in parts:
            fc = part.get("functionCall")
            if fc:
                return ToolCall(
                    tool=fc.get("name", ""),
                    args=fc.get("args", {}),
                )
        return None

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
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

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
            logger.warning("[invoke_nothink] Google call failed: %s", e)
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

        system, contents = self._convert_messages(messages)
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        # Convert images to Gemini inline format
        if images and contents:
            for i in range(len(contents) - 1, -1, -1):
                if contents[i].get("role") == "user":
                    parts = list(contents[i].get("parts", []))
                    for img in images:
                        parts.insert(0, {
                            "inlineData": {
                                "mimeType": "image/png",
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
            return GenerationResult(content="", tool_call=None, raw=data)

        parts = candidates[0].get("content", {}).get("parts", [])
        content = self._extract_text(parts)
        tool_call = self._extract_tool_call(parts)
        usage = data.get("usageMetadata") or None

        return GenerationResult(content=content.strip(), tool_call=tool_call, raw=data, usage=usage)

    async def stream_with_thinking(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response from Gemini via streamGenerateContent."""
        model = model or self._model
        client = self._get_client()

        system, contents = self._convert_messages(messages)
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            # Gemini streaming uses alt=sse parameter
            url = self._url(model, "streamGenerateContent") + "?alt=sse"
            async with client.stream(
                "POST", url, json=payload,
                timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0, read=60.0),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
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

                    # Check if generation is complete
                    finish = candidates[0].get("finishReason")
                    if finish and finish != "FINISH_REASON_UNSPECIFIED":
                        yield StreamChunk(done=True)
                        return

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503):
                raise LLMUnavailableError(f"Google streaming error {e.response.status_code}")
            raise LLMUnavailableError(f"Google error: {e}")
        except LLMUnavailableError:
            raise
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Google Gemini API.")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Google Gemini request timed out.")
