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

_PLAN_SYSTEM = """You are a precise query decomposition specialist. Break complex queries into the most effective execution plan.

Think through this query before generating output:
1. Count distinct questions/tasks in the query
2. Identify step dependencies (which steps need prior results)
3. Check if any sub-questions are INDEPENDENT (each answerable without the other's results)
4. Assign the best tool to each step

Available tools: {tools}

Output JSON only:
{{"steps": [{{"description": "concrete action under 20 words", "tool": "tool_name_or_none"}}],
  "sub_questions": [{{"question": "exact sub-question text", "requires_tools": false}}],
  "complexity": "simple|multi_step|decomposable",
  "confidence": 0.1,
  "key_risk": "main thing that could go wrong"}}

complexity definitions:
- simple: one question, direct answer, at most one tool
- multi_step: steps depend on each other's outputs (sequential pipeline)
- decomposable: 2+ INDEPENDENT sub-questions answerable separately then combined (ONLY when ALL require no tools)

confidence: 0.1-1.0 — how certain you are this plan produces a correct, complete answer
sub_questions: populate ONLY when complexity=decomposable

Rules:
- Max 5 steps; use exact tool names from the list
- First step gathers info, last step synthesizes
- Only mark decomposable when sub-questions are truly independent and tool-free"""


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

        # Extract and validate sub_questions (only meaningful for decomposable)
        sub_questions: list[dict] = []
        raw_sq = plan.get("sub_questions", [])
        if isinstance(raw_sq, list):
            for sq in raw_sq[:4]:  # cap at 4
                if isinstance(sq, dict):
                    q = str(sq.get("question", "")).strip()
                    if q:
                        sub_questions.append({
                            "question": q,
                            "requires_tools": bool(sq.get("requires_tools", False)),
                        })

        # Confidence: clamp to [0.1, 1.0]
        raw_conf = plan.get("confidence", 0.8)
        try:
            confidence = max(0.1, min(1.0, float(raw_conf)))
        except (TypeError, ValueError):
            confidence = 0.8

        complexity = plan.get("complexity", "multi_step")
        if complexity not in ("simple", "multi_step", "decomposable"):
            complexity = "multi_step"

        return {
            "steps": validated,
            "sub_questions": sub_questions,
            "complexity": complexity,
            "confidence": round(confidence, 2),
            "key_risk": str(plan.get("key_risk", "")).strip()[:200],
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


_DECOMPOSABLE_RE = re.compile(
    r"\b(compare|contrast|analyze|evaluate|examine|assess)\b"
    r".{0,150}"
    r"\b(dimension|aspect|criterion|factor|angle|metric|perspective|front)\w*"
    r"\s*[:：,]",
    re.IGNORECASE | re.DOTALL,
)


def is_decomposable(query: str) -> bool:
    """Return True if the query enumerates 2+ explicit dimensions/aspects to analyze.

    Detects patterns like "compare X vs Y across N dimensions: a, b, c" or
    "analyze these aspects: latency, throughput, cost".
    """
    if not _DECOMPOSABLE_RE.search(query):
        return False
    colon_match = re.search(r"[:：]\s*(.+)", query, re.DOTALL)
    if not colon_match:
        return False
    raw = colon_match.group(1)
    items = [x.strip() for x in re.split(r",\s*|\s+and\s+", raw) if x.strip() and len(x.strip()) > 1]
    return len(items) >= 2


def _extract_sub_questions(query: str) -> list[str]:
    """Parse an enumerated-dimension query into one sub-question per dimension."""
    colon_match = re.search(r"[:：]\s*(.+)", query, re.DOTALL)
    if not colon_match:
        return []
    raw = colon_match.group(1).strip()
    items = [x.strip() for x in re.split(r",\s*|\s+and\s+", raw) if x.strip() and len(x.strip()) > 1]
    if len(items) < 2:
        return []
    base = query[: colon_match.start()].strip().rstrip(":,").strip()
    return [f"{base} — focus specifically on {item}" for item in items]


async def solve_sub_questions(
    sub_questions: list[dict],
    user_facts: str = "",
    kg_facts: str = "",
    context: str = "",
) -> str:
    """Solve independent, tool-free sub-questions in parallel.

    Called only for decomposable plans. Runs each sub-question through
    invoke_nothink() concurrently and returns a context block with the
    pre-computed answers for the main generation to synthesize.
    """
    if not sub_questions:
        return ""

    # Only solve questions that don't require tools
    solvable = [q for q in sub_questions if not q.get("requires_tools", False)]
    if len(solvable) < 2:
        return ""
    solvable = solvable[:4]  # cap to avoid token bloat

    context_prefix_parts: list[str] = []
    if user_facts:
        context_prefix_parts.append(f"User facts: {user_facts[:300]}")
    if kg_facts:
        context_prefix_parts.append(f"Known facts: {kg_facts[:300]}")
    if context:
        context_prefix_parts.append(f"Context: {context[:500]}")
    context_prefix = "\n".join(context_prefix_parts)

    system = (
        "Answer the specific question concisely and directly. "
        "Use provided facts when available. Be precise, under 150 words."
    )
    if context_prefix:
        system = context_prefix + "\n\n" + system

    async def _solve_one(q: dict) -> tuple[str, str]:
        question = q.get("question", "").strip()
        if not question:
            return "", ""
        try:
            answer = await asyncio.wait_for(
                llm.invoke_nothink(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                    ],
                    max_tokens=200,
                    temperature=0.2,
                ),
                timeout=config.INTERNAL_LLM_TIMEOUT,
            )
            return question, (answer or "").strip()
        except Exception as e:
            logger.debug("Sub-question solve failed ('%s'): %s", question[:50], e)
            return question, ""

    try:
        results = await asyncio.gather(*[_solve_one(q) for q in solvable], return_exceptions=True)
    except Exception as e:
        logger.warning("Sub-question parallel solve failed: %s", e)
        return ""

    lines: list[str] = []
    for item in results:
        if isinstance(item, Exception):
            continue
        question, answer = item  # type: ignore[misc]
        if question and answer:
            lines.append(f"Q: {question}\nA: {answer}")

    if not lines:
        return ""

    return (
        "[PRE-ANALYZED SUB-QUESTIONS]\n"
        + "\n\n".join(lines)
        + "\n[Use these pre-computed answers when constructing your final response.]"
    )


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
