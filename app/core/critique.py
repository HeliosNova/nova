"""Self-Critique — verify answer quality before streaming.

After generating, run one quick check on complex queries. If the critique
flags issues, regenerate once with the critique injected.
"""

from __future__ import annotations

import json
import logging

from app.core import llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Should-critique heuristic (no LLM call)
# ---------------------------------------------------------------------------

_TOOL_FAILURE_MARKERS = ("failed", "timed out", "error", "not available")


def _all_tools_succeeded(tool_results: list[dict]) -> bool:
    """Return True if every tool result is clean (no failure markers)."""
    for tr in tool_results:
        output = str(tr.get("output", "")).lower()
        if any(marker in output for marker in _TOOL_FAILURE_MARKERS):
            return False
        if output.startswith("[tool"):
            return False
    return True


def should_critique(
    query: str,
    answer: str,
    intent: str,
    tool_results: list[dict],
    was_planned: bool = False,
) -> bool:
    """Decide if an answer needs critique. Pure heuristic."""
    if intent in ("greeting", "correction"):
        return False

    # Short answers don't need critique
    if len(answer) < 50:
        return False

    # Tool results are ground truth — skip critique when all tools succeeded
    if tool_results and _all_tools_succeeded(tool_results):
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

## Checks

1. **Query coverage**: Count how many distinct parts/questions are in the query. Verify each is addressed. List any missing parts.
2. **Source grounding**: For each factual claim in the answer, check if it's supported by the retrieved sources. If the answer states "Python was created in 1989" but the source says "1991", flag it. If the answer makes claims not in the sources, flag them as unsupported.
3. **Calculation accuracy**: Verify any numbers, math, or logic in the answer.
4. **Hallucination check**: Flag any specific facts (dates, names, numbers, statistics) that appear fabricated or contradict the sources.

## Output

Return JSON only:
{"pass": true, "issues": []}

If there are issues:
{"pass": false, "issues": ["missed part 2 of 3 in the query", "claims Python created in 1989 but source says 1991", "statistic about 40% market share has no source"]}

Be strict but fair. Short correct answers pass. Only flag real, specific issues."""


async def critique_answer(query: str, answer: str, sources: str = "") -> dict | None:
    """Run a critique check on the answer.

    Args:
        query: The original question.
        answer: The generated answer.
        sources: Retrieved context/sources the answer should draw from.

    Returns: {"pass": bool, "issues": [...]} or None on failure.
    """
    user_content = f"Question: {query}\n\nAnswer: {answer[:1500]}"
    if sources:
        user_content += f"\n\nRetrieved sources:\n{sources[:1500]}"

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
