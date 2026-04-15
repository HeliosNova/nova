"""Query Planning — decompose complex queries before generation.

Simple queries skip planning entirely. Complex queries get a single
invoke_nothink call that produces a step-by-step plan with tool assignments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.config import config
from app.core import llm
from app.core.text_utils import STOP_WORDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection heuristic (no LLM call)
# ---------------------------------------------------------------------------

_MULTI_PART_MARKERS = re.compile(
    r"\b(and also|as well as|additionally|in addition|furthermore|plus|also)\b",
    re.IGNORECASE,
)
_REASONING_WORDS = re.compile(
    r"\b(compare|contrast|analyze|explain why|what if|pros and cons|evaluate|"
    r"trade-?offs?|differences? between|advantages|disadvantages|implications)\b",
    re.IGNORECASE,
)
_QUESTION_WORDS = re.compile(r"\b(what|where|when|who|why|how|which)\b", re.IGNORECASE)
_NUMBERED_LIST = re.compile(r"(?:^|\n)\s*\d+[.)]\s", re.MULTILINE)

# Tool signal patterns — if query implies 2+ different tools, it's complex
_TOOL_SIGNALS = {
    "search": re.compile(r"\b(search|find|look up|latest|current|news|price)\b", re.IGNORECASE),
    "calculate": re.compile(r"\b(calculate|compute|how much|total|sum|percentage|convert)\b", re.IGNORECASE),
    "code": re.compile(r"\b(code|script|program|function|algorithm|write.*python)\b", re.IGNORECASE),
    "document": re.compile(r"\b(document|uploaded|my file|the pdf|the report)\b", re.IGNORECASE),
}


def should_plan(query: str, intent: str) -> bool:
    """Decide if a query needs planning. Pure heuristic, no LLM call."""
    if intent in ("greeting", "correction"):
        return False

    q = query.strip()
    if len(q) < 15:
        return False

    signals = 0

    # Multi-part markers
    if _MULTI_PART_MARKERS.search(q):
        signals += 2

    # Reasoning words
    if _REASONING_WORDS.search(q):
        signals += 2

    # Numbered list items
    if _NUMBERED_LIST.search(q):
        signals += 2

    # Long query with multiple question words
    question_matches = _QUESTION_WORDS.findall(q)
    if len(q) > 100 and len(question_matches) >= 2:
        signals += 1

    # Multiple tool signals
    tool_hits = sum(1 for pat in _TOOL_SIGNALS.values() if pat.search(q))
    if tool_hits >= 2:
        signals += 1

    # Lower threshold for personal use — quality over speed.
    # Any signal of complexity triggers planning.
    return signals >= 1


# ---------------------------------------------------------------------------
# Plan creation (1 invoke_nothink call)
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """You are a query planner. Break the user's query into sequential steps.
For each step, name the tool to use (or "none" for reasoning/synthesis).

Available tools: {tools}

Output JSON only:
{{"steps": [{{"description": "what to do", "tool": "tool_name_or_none"}}], "complexity": "simple|multi_step"}}

Rules:
- Max 5 steps
- Use exact tool names from the list above
- First step should gather information, last step should synthesize
- Keep descriptions short (under 20 words)"""


async def create_plan(
    query: str,
    tool_names: list[str],
    reflexions_text: str = "",
) -> dict | None:
    """Create a step-by-step plan for a complex query.

    Returns: {"steps": [...], "complexity": "..."} or None on failure.
    """
    system = _PLAN_SYSTEM.format(tools=", ".join(tool_names))
    if reflexions_text:
        system += f"\n\nWarnings from past failures:\n{reflexions_text}"

    try:
        raw = await asyncio.wait_for(
            llm.invoke_nothink(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                json_mode=True,
                json_prefix='{"',
                max_tokens=300,
                temperature=0.1,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )
        if not raw:
            return None

        plan = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(plan, dict) or "steps" not in plan:
            return None

        steps = plan["steps"]
        if not isinstance(steps, list) or len(steps) == 0:
            return None

        # Validate and cap at 5 steps
        valid_tools = set(tool_names) | {"none"}
        validated = []
        for step in steps[:5]:
            if not isinstance(step, dict):
                continue
            desc = str(step.get("description", "")).strip()
            tool = str(step.get("tool", "none")).strip()
            if not desc:
                continue
            if tool not in valid_tools:
                tool = "none"
            validated.append({"description": desc, "tool": tool})

        if not validated:
            return None

        return {
            "steps": validated,
            "complexity": plan.get("complexity", "multi_step"),
        }
    except Exception as e:
        logger.warning("Planning failed: %s", e)
        return None


def format_plan_for_prompt(plan: dict) -> str:
    """Format a plan as text for injection into the message list."""
    if not plan or not plan.get("steps"):
        return ""
    lines = []
    for i, step in enumerate(plan["steps"], 1):
        tool = step.get("tool", "none")
        tool_note = f" using {tool}" if tool != "none" else ""
        lines.append(f"{i}. {step['description']}{tool_note}")
    return "[PLAN]\n" + "\n".join(lines) + "\n[Follow this plan step by step.]"


def verify_plan_coverage(plan: dict, answer: str) -> list[str]:
    """Check which plan steps were not addressed in the answer.

    Uses simple keyword overlap. Returns list of missed step descriptions.
    """
    if not plan or not plan.get("steps") or not answer:
        return []

    answer_lower = answer.lower()
    missed = []

    for step in plan["steps"]:
        desc = step.get("description", "")
        if not desc:
            continue
        # Extract keywords (3+ char words) from the step description
        keywords = [w for w in re.findall(r"\b\w{3,}\b", desc.lower())
                    if w not in STOP_WORDS]
        if not keywords:
            continue
        # Step is "covered" if at least 40% of keywords appear in the answer
        hits = sum(1 for kw in keywords if kw in answer_lower)
        if hits / len(keywords) < 0.4:
            missed.append(desc)

    return missed
