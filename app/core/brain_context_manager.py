"""Conversation context window management — summarize older messages when
the total token count would exceed MAX_CONTEXT_TOKENS.

Extracted from brain.py for size hygiene. Re-exported so existing
`from app.core.brain import _manage_context` keeps working.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import config
from app.core import llm
from app.core.text_utils import estimate_tokens as _estimate_tokens

logger = logging.getLogger(__name__)


async def _manage_context(
    system_prompt: str,
    history: list[dict],
    query: str,
) -> tuple[list[dict], str]:
    """Manage context window to stay within budget.

    If the total token count exceeds MAX_CONTEXT_TOKENS, summarize older
    messages and keep only the most recent RECENT_MESSAGES_KEEP messages
    verbatim.

    Returns: (trimmed_history, conversation_summary)
    """
    system_tokens = _estimate_tokens(system_prompt)
    query_tokens = _estimate_tokens(query)
    history_tokens = sum(_estimate_tokens(m.get("content", "")) for m in history)
    response_budget = config.RESPONSE_TOKEN_BUDGET

    # 20% safety buffer on estimates (heuristic ~4 chars/token is 30-50% off)
    total = int((system_tokens + history_tokens + query_tokens) * 1.2) + response_budget

    if total <= config.MAX_CONTEXT_TOKENS:
        return history, ""

    keep = config.RECENT_MESSAGES_KEEP
    if len(history) <= keep:
        return history, ""

    old_messages = history[:-keep]
    recent_messages = history[-keep:]

    summary_input = "\n".join(
        f"[{m.get('role', '?')}]: {m.get('content', '')[:300]}"
        for m in old_messages
    )

    try:
        summary = await asyncio.wait_for(
            llm.invoke_nothink(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize this conversation in 2-3 SHORT sentences. "
                            "Only key facts and decisions. No detail, no examples. "
                            "IMPORTANT: Preserve ALL dates, numbers, proper nouns, "
                            "monetary amounts, and technical values exactly as stated. "
                            "Example: 'Discussed deployment on March 15. User prefers Y. Budget is $50,000.'"
                        ),
                    },
                    {"role": "user", "content": summary_input},
                ],
                max_tokens=150,
                temperature=0.1,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )
        summary = summary.strip()
        logger.info(
            "Context managed: %d→%d messages, summary=%d chars (budget: %d/%d tokens)",
            len(history), keep, len(summary),
            system_tokens + sum(_estimate_tokens(m.get("content", "")) for m in recent_messages) + query_tokens,
            config.MAX_CONTEXT_TOKENS,
        )

        # Re-check: if still over budget, truncate the summary further.
        post_summary_tokens = (
            system_tokens
            + sum(_estimate_tokens(m.get("content", "")) for m in recent_messages)
            + query_tokens
            + _estimate_tokens(summary)
            + response_budget
        )
        if post_summary_tokens > config.MAX_CONTEXT_TOKENS and summary:
            half = len(summary) // 2
            boundary = -1
            for sep in (". ", "\n"):
                pos = summary.find(sep, half)
                if pos != -1 and (boundary == -1 or pos < boundary):
                    boundary = pos + len(sep)
            if boundary != -1 and boundary < len(summary):
                summary = summary[boundary:].lstrip()
            else:
                summary = summary[half:].lstrip()
            logger.info(
                "Post-summarization budget still exceeded — truncated summary to %d chars",
                len(summary),
            )

        return recent_messages, summary
    except (Exception, asyncio.TimeoutError) as e:
        logger.warning("Summarization failed: %s — truncating instead", e)
        truncation_note = f"[{len(old_messages)} older messages truncated due to context limits]"
        return recent_messages, truncation_note
