"""Self-Critique — verify answer quality before streaming.

After generating, run one quick check on complex queries. If the critique
flags issues, regenerate once with the critique injected.
"""

from __future__ import annotations

import json
import logging

from app.config import config
from app.core import llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Should-critique heuristic (no LLM call)
# ---------------------------------------------------------------------------

from app.core.quality import all_tools_clean as _all_tools_succeeded


def should_critique(
    query: str,
    answer: str,
    intent: str,
    tool_results: list[dict],
    was_planned: bool = False,
    kg_facts: str = "",
    user_facts: str = "",
) -> bool:
    """Decide if an answer needs critique. Pure heuristic."""
    if intent == "correction":
        return False

    # Greetings/meta skip only if short AND query is short AND facts are grounding
    if intent in ("greeting", "meta"):
        if len(query) < 20 and (kg_facts or user_facts):
            return False

    # Short answers don't need critique
    if len(answer) < 50:
        return False

    # Tool results are ground truth — skip critique when all tools succeeded
    if tool_results and _all_tools_succeeded(tool_results):
        return False

    # KG/user facts are pre-verified — skip critique when answer is grounded in them
    if (kg_facts or user_facts) and intent == "general":
        return False

    # Triggers: failed tools, long answer, or was planned
    if was_planned:
        return True
    if tool_results:
        return True
    if len(answer) > 200:
        return True

    return False


# ---------------------------------------------------------------------------
# Critique (1 invoke_nothink call)
# ---------------------------------------------------------------------------

_CRITIQUE_SYSTEM = """You are a strict answer verifier. For each check, explain your reasoning briefly.

IMPORTANT: Owner facts and knowledge graph facts are PRE-VERIFIED. Claims matching these are NEVER hallucinations. Only flag claims with NO source support.

## Checks

1. **Query coverage**: Count how many distinct parts/questions are in the query. Verify each is addressed. List any missing parts.
2. **Source grounding**: For each factual claim, check if it's supported by the retrieved sources, owner facts, OR knowledge graph facts. A claim grounded in ANY of these is valid. Owner facts and knowledge graph entries are authoritative — referencing them is NOT hallucination.
3. **Calculation accuracy**: Verify any numbers, math, or logic in the answer.
4. **Hallucination check**: Flag specific facts (dates, names, numbers) that appear fabricated AND are not supported by any provided source category. Do NOT flag claims that match owner facts or knowledge graph entries.

## Output

Return JSON only:
{"pass": true, "issues": []}

If there are issues:
{"pass": false, "issues": ["missed part 2 of 3 in the query", "claims Python created in 1989 but source says 1991", "statistic about 40% market share has no source"]}

Be strict but fair. Short correct answers pass. Only flag real, specific issues."""


async def critique_answer(
    query: str, answer: str, sources: str = "",
    user_facts: str = "", kg_facts: str = "",
) -> dict | None:
    """Run a critique check on the answer.

    Args:
        query: The original question.
        answer: The generated answer.
        sources: Retrieved context/sources the answer should draw from.
        user_facts: Verified personal info about the user (name, employer, etc.).
        kg_facts: Verified facts from the knowledge graph.

    Returns: {"pass": bool, "issues": [...]} or None on failure.
    """
    user_content = f"Question: {query}\n\nAnswer: {answer[:config.CRITIQUE_ANSWER_LIMIT]}"
    if sources:
        user_content += f"\n\nRetrieved sources:\n{sources[:config.CRITIQUE_SOURCES_LIMIT]}"
    if user_facts:
        user_content += f"\n\nOwner facts (verified personal info about the user):\n{user_facts[:config.CRITIQUE_FACTS_LIMIT]}"
    if kg_facts:
        user_content += f"\n\nKnowledge graph facts (verified stored facts):\n{kg_facts[:config.CRITIQUE_FACTS_LIMIT]}"

    try:
        raw = await llm.invoke_nothink(
            [
                {"role": "system", "content": _CRITIQUE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            json_mode=True,
            json_prefix='{"',
            max_tokens=400,
            temperature=0.1,
        )
        if not raw:
            return None

        result = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(result, dict):
            return None

        # Normalize
        passed = result.get("pass", True)
        issues = result.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []

        return {"pass": bool(passed), "issues": issues}
    except Exception as e:
        logger.warning("Critique failed: %s", e)
        return None


def format_critique_for_regeneration(critique: dict) -> str:
    """Format critique issues as a system message for regeneration."""
    if not critique or critique.get("pass", True):
        return ""
    issues = critique.get("issues", [])
    if not issues:
        return ""
    issue_text = "\n".join(f"- {issue}" for issue in issues)
    return (
        "[SELF-CHECK FAILED]\n"
        f"Your previous answer had these issues:\n{issue_text}\n"
        "Fix these issues in your revised answer. Address every point."
    )
