"""LLM abstraction — provider-agnostic interface.

Module-level functions (invoke_nothink, generate_with_tools, stream_with_thinking)
are the public API. They delegate to the current provider (default: OllamaProvider).

To swap providers: call set_provider() at startup, or set LLM_PROVIDER env var.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import AsyncGenerator, Protocol, runtime_checkable

import httpx

from app.config import config

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when the LLM backend cannot be reached or times out."""
    pass



# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A parsed tool call extracted from the model's text output."""
    tool: str
    args: dict


@dataclass
class StreamChunk:
    """An incremental chunk from a streaming generation."""
    thinking: str = ""    # Incremental thinking text
    content: str = ""     # Incremental content text
    done: bool = False


@dataclass
class GenerationResult:
    """Result of a single LLM generation."""
    content: str
    tool_call: ToolCall | None
    raw: dict
    thinking: str = ""
    usage: dict | None = None


# ---------------------------------------------------------------------------
# Provider Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """Interface that all LLM providers must implement."""

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
        """Generate text with thinking disabled. Used for background tasks."""
        ...

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 2000,
    ) -> GenerationResult:
        """Generate a response that may contain a tool call."""
        ...

    async def stream_with_thinking(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response with thinking enabled. Yields incremental chunks."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def create_provider(cfg=None) -> LLMProvider:
    """Create the LLM provider based on config. Called at startup."""
    cfg = cfg or config

    match cfg.LLM_PROVIDER:
        case "ollama":
            from app.core.providers.ollama import OllamaProvider
            return OllamaProvider()
        case "openai":
            from app.core.providers.openai import OpenAIProvider
            return OpenAIProvider(cfg.OPENAI_API_KEY, cfg.OPENAI_MODEL)
        case "anthropic":
            from app.core.providers.anthropic import AnthropicProvider
            return AnthropicProvider(cfg.ANTHROPIC_API_KEY, cfg.ANTHROPIC_MODEL)
        case "google":
            from app.core.providers.google import GoogleProvider
            return GoogleProvider(cfg.GOOGLE_API_KEY, cfg.GOOGLE_MODEL)
        case _:
            logger.warning("Unknown LLM_PROVIDER '%s', falling back to Ollama", cfg.LLM_PROVIDER)
            from app.core.providers.ollama import OllamaProvider
            return OllamaProvider()


# ---------------------------------------------------------------------------
# Provider management
# ---------------------------------------------------------------------------

_provider: LLMProvider | None = None


def set_provider(provider: LLMProvider) -> None:
    """Set the active LLM provider. Call during startup."""
    global _provider
    _provider = provider


def get_provider() -> LLMProvider:
    """Get the current LLM provider. Auto-creates OllamaProvider if none set."""
    global _provider
    if _provider is None:
        from app.core.providers.ollama import OllamaProvider
        _provider = OllamaProvider()
    return _provider


# ---------------------------------------------------------------------------
# Module-level API (delegates to current provider)
#
# These are the public functions used throughout the codebase.
# Callers import llm and call llm.invoke_nothink() — no changes needed.
# ---------------------------------------------------------------------------

async def invoke_nothink(
    messages: list[dict],
    *,
    json_mode: bool = False,
    json_prefix: str = "[{",
    max_tokens: int = 1000,
    temperature: float = 0.1,
    model: str | None = None,
) -> str:
    """Call LLM with thinking disabled. Used for background tasks."""
    return await get_provider().invoke_nothink(
        messages,
        json_mode=json_mode,
        json_prefix=json_prefix,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
    )


async def generate_with_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 2000,
    images: list[str] | None = None,
) -> GenerationResult:
    """Generate a response that may contain a tool call."""
    return await get_provider().generate_with_tools(
        messages,
        tools,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        images=images,
    )


async def stream_with_thinking(
    messages: list[dict],
    tools: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 4000,
) -> AsyncGenerator[StreamChunk, None]:
    """Stream a response with thinking enabled. Yields incremental chunks."""
    async for chunk in get_provider().stream_with_thinking(
        messages,
        tools,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        yield chunk


async def check_health() -> bool:
    """Check if the current LLM provider is reachable."""
    provider = get_provider()
    if hasattr(provider, "check_health"):
        return await provider.check_health()
    return False


async def close_client() -> None:
    """Close the current provider's resources."""
    global _provider
    if _provider is not None:
        await _provider.close()
        _provider = None


# ---------------------------------------------------------------------------
# JSON extraction utilities (provider-agnostic)
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and unmatched thinking from model output."""
    # Strip matched <think>...</think> pairs
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    # Strip content before an unmatched </think> tag (model continuing reasoning after prefix)
    last_close = text.rfind("</think>")
    if last_close != -1:
        text = text[last_close + len("</think>"):]
    return text


def _find_balanced_json(text: str, prefix: str) -> str:
    """Find the first balanced JSON structure in text."""
    if not text:
        return text

    open_char = "[" if prefix.startswith("[") else "{"
    close_char = "]" if open_char == "[" else "}"

    start = text.find(open_char)
    if start == -1:
        return text

    depth = 0
    in_string = False
    num_backslashes = 0

    for i in range(start, len(text)):
        c = text[i]
        if c == "\\" and in_string:
            num_backslashes += 1
            continue
        if c == '"':
            # Only toggle string if preceded by even number of backslashes
            if num_backslashes % 2 == 0:
                in_string = not in_string
            num_backslashes = 0
            continue
        num_backslashes = 0
        if in_string:
            continue
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text


def extract_json_object(text: str) -> dict:
    """Extract a JSON object from text that may contain surrounding content."""
    if not text:
        logger.debug("extract_json_object: empty input")
        return {}
    balanced = _find_balanced_json(text, "{")
    if not balanced or not balanced.startswith("{"):
        logger.debug("extract_json_object: no balanced JSON found in text (%d chars)", len(text))
        return {}
    try:
        return json.loads(balanced)
    except json.JSONDecodeError:
        logger.debug("extract_json_object: balanced parse failed, trying regex fallback")
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                logger.debug("extract_json_object: regex fallback also failed")
                pass
        return {}


def _extract_tool_call(content: str, tools: list[dict]) -> ToolCall | None:
    """Try to parse a single tool call from the model's text output."""
    calls = _extract_tool_calls(content, tools)
    return calls[0] if calls else None


def _validate_tool_obj(obj: dict, valid_names: set[str], lower_map: dict[str, str]) -> ToolCall | None:
    """Validate a parsed JSON object as a tool call."""
    tool_name = obj.get("tool") or obj.get("name") or obj.get("function")
    args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}

    if not tool_name:
        return None

    if tool_name not in valid_names:
        tool_name = lower_map.get(tool_name.lower())
        if not tool_name:
            return None

    if not isinstance(args, dict):
        args = {}

    return ToolCall(tool=tool_name, args=args)


def _extract_tool_calls(content: str, tools: list[dict]) -> list[ToolCall]:
    """Try to parse one or more tool calls from the model's text output.

    Handles:
    - Single JSON object: {"tool": "...", "args": {...}}
    - JSON array: [{"tool": "...", "args": {...}}, {"tool": "...", "args": {...}}]
    - Multiple JSON objects separated by text
    """
    if not content or not tools:
        return []

    valid_names = {t["name"] for t in tools}
    lower_map = {n.lower(): n for n in valid_names}

    # First, try parsing as a JSON array of tool calls
    try:
        array_text = _find_balanced_json(content, "[{")
        if array_text and array_text.startswith("["):
            parsed = json.loads(array_text)
            if isinstance(parsed, list) and len(parsed) > 0:
                calls = []
                for item in parsed:
                    if isinstance(item, dict):
                        tc = _validate_tool_obj(item, valid_names, lower_map)
                        if tc:
                            calls.append(tc)
                if calls:
                    return calls
    except (json.JSONDecodeError, Exception):
        pass

    # Fall back to finding all JSON objects in the content
    calls = []
    seen_tools = set()
    # Find all potential JSON objects by scanning for '{'
    i = 0
    while i < len(content):
        if content[i] == '{':
            balanced = _find_balanced_json(content[i:], "{")
            if balanced and balanced.startswith("{"):
                try:
                    obj = json.loads(balanced)
                    if isinstance(obj, dict):
                        tc = _validate_tool_obj(obj, valid_names, lower_map)
                        if tc and (tc.tool, json.dumps(tc.args, sort_keys=True)) not in seen_tools:
                            calls.append(tc)
                            seen_tools.add((tc.tool, json.dumps(tc.args, sort_keys=True)))
                except json.JSONDecodeError:
                    pass
                i += len(balanced)
                continue
        i += 1

    return calls
