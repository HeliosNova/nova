"""Intent classification, model routing, and title generation — extracted
from brain.py for size hygiene.

Re-exported by brain.py so existing
`from app.core.brain import _classify_intent, _generate_title, _select_model`
keeps working for tests and callers.
"""

from __future__ import annotations

import asyncio
import logging
import re

from app.config import config
from app.core import llm
from app.core.learning import is_likely_correction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent classification (fast, no LLM call unless ambiguous)
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|sup|yo)\b",
    re.IGNORECASE,
)

# Pure greeting = greeting words + optional filler (punctuation, "there", "nova"), nothing else
_PURE_GREETING = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|howdy|sup|yo)"
    r"(?:\s+(?:there|nova|buddy|mate|friend|everyone|all))?"
    r"[!?.,:;\s]*$",
    re.IGNORECASE,
)


async def _classify_intent(query: str) -> str:
    """Intent classification — regex first, LLM tiebreaker for ambiguous greetings.

    Returns: 'greeting', 'correction', or 'general'
    """
    stripped = query.strip()

    # Single source of truth for correction detection
    if is_likely_correction(stripped):
        return "correction"

    if _GREETING_PATTERNS.match(stripped):
        if _PURE_GREETING.match(stripped):
            return "greeting"
        # Ambiguous: starts with greeting but has substantive content
        word_count = len(stripped.split())
        if word_count > 3:
            try:
                result = await asyncio.wait_for(
                    llm.invoke_nothink(
                        [{"role": "user", "content": (
                            f'Is this a simple greeting or a real question/request? '
                            f'Reply with ONE word: "greeting" or "general".\n\n"{stripped}"'
                        )}],
                        max_tokens=10,
                        temperature=0,
                    ),
                    timeout=config.INTERNAL_LLM_TIMEOUT,
                )
                if result is None:
                    return "general"
                classification = result.strip().lower().strip('"\'.')
                if classification in ("greeting", "general"):
                    return classification
            except (Exception, asyncio.TimeoutError):
                pass
            return "general"
        return "greeting"

    return "general"


# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------

async def _generate_title(query: str) -> str:
    """Generate a short conversation title from the first query."""
    try:
        result = await asyncio.wait_for(
            llm.invoke_nothink(
                [
                    {"role": "system", "content": (
                        "Generate a 3-5 word title summarizing this conversation topic. "
                        "Rules: NO emojis, NO quotes, NO punctuation, plain English only. "
                        "Return ONLY the title words, nothing else."
                    )},
                    {"role": "user", "content": query},
                ],
                max_tokens=15,
                temperature=0.2,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Title generation timed out")
        return query[:40].strip()
    if result is None:
        return query[:40].strip()
    title = result.strip().strip('"\'').strip()
    title = "".join(c for c in title if c.isascii()).strip()
    if not title or len(title) > 60:
        words = query.split()[:5]
        return " ".join(words)
    return title


# ---------------------------------------------------------------------------
# Model routing — fast model for simple queries
# ---------------------------------------------------------------------------

_QUESTION_WORDS = re.compile(
    r"(?i)^(?:who|what|when|where|why|how|which|is|are|was|were|do|does|did|can|could|will|would|should)\b"
)

_COMPLEX_PATTERNS = re.compile(
    r"(?i)\b(?:step[- ]by[- ]step|prove|derive|implement|algorithm|refactor|debug|architect|"
    r"design pattern|trade-?offs?|compare and contrast|write (?:a |an )?(?:function|class|script|program)|"
    r"solve|equation|integral|derivative|optimize|complexity|recursion|dynamic programming|"
    r"explain (?:how|why)|multi-?step|chain of thought)\b"
)

_CREATIVE_PATTERNS = re.compile(
    r"(?i)\b(?:opinion|brainstorm|creative|imagine|suggest|ideas?|write (?:a |an )?(?:poem|story|essay)|what do you think)\b"
)


def _select_model(query: str, intent: str, needs_plan: bool) -> str | None:
    """Return model override or None for default."""
    if not config.ENABLE_MODEL_ROUTING:
        return None
    stripped = query.strip()

    if config.FAST_MODEL:
        if intent == "greeting":
            return config.FAST_MODEL
        if len(stripped) < 40 and not needs_plan and "?" not in stripped and not _QUESTION_WORDS.match(stripped):
            return config.FAST_MODEL

    if config.HEAVY_MODEL:
        if needs_plan:
            return config.HEAVY_MODEL
        if _COMPLEX_PATTERNS.search(stripped):
            return config.HEAVY_MODEL

    return None
