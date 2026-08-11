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

# Context window for CHAT generations. Without an explicit num_ctx Ollama loads
# the model at its Modelfile default (4096) and SILENTLY TRUNCATES the prompt —
# found 2026-07-07: the ~11k-token system prompt (identity + lessons + KG facts
# + tools) was cut to 4096 tokens on every chat call, so injected knowledge
# never reached the model (kg-retrieval causal-fix 0.83→0.0, "I don't know X"
# with X's facts in-prompt). The compose OLLAMA_NUM_CTX env demonstrably did
# NOT apply. 24576 (was 16384): the system prompt alone measures ~15.3k real
# tokens, so 16384 left <1.1k for history + generation — one long multi-turn
# conversation re-entered silent truncation (quantified 2026-07-07). 24k fits
# the 9B Q8 with q8 KV + flash attention; Ollama clamps to VRAM if needed.
_CHAT_NUM_CTX = 24576


class OllamaProvider:
    """Ollama LLM provider — raw HTTP, no LangChain.

    Qwen3.5-specific tricks (thinking suppression, JSON prefixing) handled here.
    """

    def __init__(self, base_url: str | None = None, llm_model: str | None = None, embed_model: str | None = None):
        self._base_url = base_url or config.OLLAMA_URL
        self._llm_model = llm_model or config.LLM_MODEL
        self._embed_model = embed_model or config.EMBEDDING_MODEL
        self._client: httpx.AsyncClient | None = None

    @property
    def capabilities(self) -> "ProviderCapabilities":
        from app.core.llm import ProviderCapabilities
        return ProviderCapabilities(
            needs_emphatic_prompts=True,
            supports_native_tools=True,
            supports_thinking=True,
            json_prefix_behavior="prepend",
        )

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
        json_schema: dict | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        model: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        model = model or self._llm_model
        client = self._get_client()

        ollama_messages = list(messages)

        # Thinking suppression: rely on the native `think: false` payload param
        # below. The old "<think>\n\n</think>" assistant-prefill trick is now
        # HARMFUL on the Ollama 0.30 engine (2026-07-08): the qwen3.x renderer
        # echoes the injected tags back into the output AND, on complex rewrite
        # prompts, the prefilled assistant turn made qwen3.6:27b RE-OPEN thinking
        # and blow its token budget mid-<think> — the unterminated block then
        # leaked a whole raw reasoning monologue into a posted digest (the Science
        # digest incident). Probe: `think:false` alone returns clean output;
        # `think:false` + the prefix echoes/leaks. So only prefill the json_prefix
        # for JSON-continuation models, nothing else.
        if json_mode and json_prefix:
            ollama_messages.append({"role": "assistant", "content": json_prefix})

        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": ollama_messages,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                # 1.1 for everything. The old `1.5 if not json_mode` was an
                # unexamined v1.0.0 default that crippled the freeform background
                # gens this path actually serves (synthesis, agent merge,
                # refinement, polish) — 1.5 over-penalizes the ordinary word
                # repetition natural prose needs. The streaming path generates
                # the longest freeform prose in the system at 1.1, so 1.1 is the
                # proven-good value; JSON already required it.
                "repeat_penalty": 1.1,
            },
        }
        if num_ctx:
            # Models without a Modelfile num_ctx default low (e.g. 4096); a
            # long grading prompt would silently truncate from the head,
            # cutting the rubric. Callers with big prompts set this explicitly.
            payload["options"]["num_ctx"] = num_ctx

        if json_mode:
            # Schema enforcement (Ollama 0.17+) or generic JSON mode
            payload["format"] = json_schema if json_schema else "json"

        try:
            resp = await retry_on_transient(client, "POST", "/api/chat", json=payload)
            data = resp.json()
            content = data.get("message", {}).get("content", "")

            if json_mode and json_prefix:
                # Prefill-continuation models (qwen35) return content WITHOUT
                # the prefix — re-prepend it. Models whose chat template closes
                # the assistant turn instead (gemma3) ignore the prefill and
                # return the complete object — prepending again would corrupt
                # it to "{{...". Verified against both families 2026-06-11.
                if not content.lstrip().startswith(json_prefix.lstrip()):
                    content = json_prefix + content

            # Silent-truncation tripwire: invoke_nothink (the deep_research
            # grounding/synthesis path) never checked done_reason, so a digest
            # hitting max_tokens was cut MID-SENTENCE with no signal (2026-07-08).
            if data.get("done_reason") == "length":
                logger.warning("[truncation] invoke_nothink hit max_tokens (%d) — output cut mid-generation "
                               "(model=%s, %d chars)", max_tokens, model, len(content))

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
            logger.error("[invoke_nothink] Unexpected error: %s", e)
            raise LLMUnavailableError(f"Ollama unexpected error: {e}")

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 2000,
        images: list[str] | None = None,
        tool_choice: str | None = None,  # ignored — Ollama has no native tool calling
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

        # Thinking suppression is via the native `think: False` payload param
        # below. The old "<think>\n\n</think>" assistant-prefill trick was
        # REMOVED 2026-07-08: on the Ollama 0.30 engine the qwen3.x renderer
        # echoes the injected tags into the output and can make the model
        # re-open thinking (see invoke_nothink for the full write-up + the
        # Science-digest leak it caused on the background passes).
        _ = is_qwen  # retained for readability; no longer gates a prefill

        # Build payload with native tool calling
        payload: dict = {
            "model": model,
            "stream": False,
            "think": False,
            "messages": send_messages,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                # repeat_penalty 1.1 — see streaming path comment.
                "repeat_penalty": 1.1,
                # num_ctx: WITHOUT this, chat runs at the model's 4096 default and
                # Ollama silently drops ~2/3 of the ~11k-token system prompt —
                # observed 2026-07-07 (prompt_tokens=4096): injected KG facts and
                # lessons never reached the model, so it denied knowing entities
                # whose facts sat in its own prompt (kg-retrieval 0.83→0.0).
                "num_ctx": _CHAT_NUM_CTX,
            },
        }
        # Pass tools for native tool calling (Ollama 0.17+)
        # Disabled by default: native tools + thinking causes empty responses
        if tools and self.capabilities.supports_native_tools:
            payload["tools"] = [
                {"type": "function", "function": t} if "function" not in t else t
                for t in tools
            ]

        try:
            resp = await retry_on_transient(client, "POST", "/api/chat", json=payload)
        except httpx.ConnectError:
            raise LLMUnavailableError("Cannot connect to Ollama. Is it running?")
        except httpx.TimeoutException:
            raise LLMUnavailableError("Ollama request timed out.")
        except LLMUnavailableError:
            raise

        data = resp.json()
        msg = data.get("message", {})
        content = msg.get("content", "")
        content = _strip_think_tags(content).strip()

        # Populate usage from Ollama response
        usage = None
        if "eval_count" in data or "prompt_eval_count" in data:
            usage = {"completion_tokens": data.get("eval_count", 0), "prompt_tokens": data.get("prompt_eval_count", 0)}
            # Truncation tripwire: Ollama reports the prompt tokens it actually
            # processed, so prompt_tokens reaching num_ctx is PROOF the prompt
            # was silently cut — the failure class that ran undetected for
            # months at the 4096 default. Loud, deterministic, no estimates.
            if usage["prompt_tokens"] >= _CHAT_NUM_CTX:
                logger.error(
                    "[num_ctx] prompt_tokens=%d hit num_ctx=%d — Ollama silently "
                    "TRUNCATED this prompt; injected context was lost. Shrink the "
                    "system prompt or raise _CHAT_NUM_CTX.",
                    usage["prompt_tokens"], _CHAT_NUM_CTX,
                )

        # Detect truncation due to token limit
        done_reason = data.get("done_reason", "")
        if done_reason == "length":
            logger.warning("Ollama response truncated (done_reason=length)")
            content += "\n\n[Warning: Response was truncated due to token limit]"

        # Native tool calls from Ollama (structured, like OpenAI)
        native_tool_calls = msg.get("tool_calls", [])
        tool_calls_out: list[ToolCall] = []
        if native_tool_calls:
            for tc in native_tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if name:
                    tool_calls_out.append(ToolCall(tool=name, args=args))

        # Fallback to text parsing if no native tool calls found
        if not tool_calls_out:
            text_tool_call = _extract_tool_call(content, tools)
            if text_tool_call:
                tool_calls_out = [text_tool_call]

        return GenerationResult(
            content=content,
            tool_calls=tool_calls_out,
            raw=data,
            usage=usage,
            stop_reason=data.get("done_reason", ""),
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
        """Stream a response with thinking enabled. Yields incremental chunks.

        Tools are passed to Ollama for native tool calling (Ollama 0.17+).
        Brain.py still has text-parsing fallback for older Ollama versions.
        """
        model = model or self._llm_model
        client = self._get_client()
        use_think = True  # Fallback to False if model doesn't support thinking

        # Convert tools to Ollama format
        ollama_tools = None
        if tools:
            ollama_tools = [
                {"type": "function", "function": t} if "function" not in t else t
                for t in tools
            ]

        _yielded_done = False
        max_stream_retries = 2
        try:
            for _stream_attempt in range(max_stream_retries + 1):
                try:
                    payload = {
                            "model": model,
                            "stream": True,
                            **({"think": True} if use_think else {}),
                            "messages": messages,
                            "options": {
                                "num_predict": max_tokens,
                                "temperature": temperature,
                                # repeat_penalty 1.1 stops Qwen3.5 from falling
                                # into degenerate token-loops on complex math /
                                # LaTeX (verified case 2026-05-04: "0.1+0.2 in
                                # IEEE 754" produced 4000 chars where the same
                                # block repeated 6 times). 1.1 is mild — doesn't
                                # damage normal text. Higher values mangle JSON.
                                "repeat_penalty": 1.1,
                                # num_ctx: see generate_with_tools — the 4096
                                # default silently truncated the system prompt.
                                "num_ctx": _CHAT_NUM_CTX,
                            },
                        }
                    if ollama_tools and self.capabilities.supports_native_tools:
                        payload["tools"] = ollama_tools
                    async with client.stream(
                        "POST",
                        "/api/chat",
                        json=payload,
                        timeout=httpx.Timeout(float(config.GENERATION_TIMEOUT), connect=10.0, read=300.0),
                    ) as resp:
                        # Catch models that don't support thinking (missing RENDERER/PARSER)
                        if resp.status_code == 400 and use_think:
                            body = await resp.aread()
                            if b"does not support thinking" in body:
                                logger.warning(
                                    "Model '%s' does not support thinking API — "
                                    "falling back to think=false. Fix: add RENDERER/PARSER to Modelfile.",
                                    model,
                                )
                                use_think = False
                                continue  # Retry without think=true
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

                            # Native tool calls in stream (Ollama 0.17+)
                            stream_tool_calls = msg.get("tool_calls")
                            stream_tool: ToolCall | None = None
                            if stream_tool_calls:
                                for tc in stream_tool_calls:
                                    func = tc.get("function", {})
                                    name = func.get("name", "")
                                    args = func.get("arguments", {})
                                    if isinstance(args, str):
                                        try:
                                            args = json.loads(args)
                                        except (json.JSONDecodeError, TypeError):
                                            args = {}
                                    if name:
                                        stream_tool = ToolCall(tool=name, args=args)

                            if thinking_delta or content_delta or done or stream_tool:
                                if done:
                                    _yielded_done = True
                                yield StreamChunk(
                                    thinking=thinking_delta,
                                    content=content_delta,
                                    done=done,
                                    tool_call=stream_tool,
                                )
                    return  # Success — exit retry loop
                except httpx.ReadError:
                    raise LLMUnavailableError("Connection lost during Ollama streaming")
                except httpx.ConnectError:
                    if _stream_attempt < max_stream_retries:
                        import asyncio
                        logger.warning("Ollama stream connect error, retrying (%d/%d)", _stream_attempt + 1, max_stream_retries)
                        await asyncio.sleep(2.0)
                        continue
                    raise LLMUnavailableError("Cannot connect to Ollama. Is it running?")
                except httpx.TimeoutException:
                    raise LLMUnavailableError("Ollama request timed out.")
        finally:
            if not _yielded_done:
                yield StreamChunk(done=True)
