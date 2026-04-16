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

_CRITIQUE_SYSTEM = """You are a strict answer verifier. Work through each check systematically.

IMPORTANT: Owner facts and knowledge graph facts are PRE-VERIFIED. Claims matching these are NEVER hallucinations.

## Step 1 — Count query parts
How many distinct questions/requests are in the query? List them.

## Step 2 — Check coverage
For each part identified above: is it addressed in the answer? Mark covered/missing.

## Step 3 — Check source grounding
For each factual claim in the answer: is it supported by retrieved sources, owner facts, OR knowledge graph facts?
- Grounded in ANY of these = valid
- Owner/KG entries are authoritative — never flag these as hallucinations
- Only flag specific claims (dates, names, numbers) with ZERO source support

## Step 4 — Check calculations
Verify any arithmetic, unit conversions, or logical deductions in the answer.

## Output
Return JSON only:
{"pass": true, "issues": []}

If issues found:
{"pass": false, "issues": ["missed part 2 of 3: did not address X", "claims Python created in 1989 but source says 1991"]}

Rules: Be strict but fair. Short correct answers pass. Only flag real, specific issues with clear evidence."""


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


# ---------------------------------------------------------------------------
# Adversarial critique — dedicated logic/factual error hunter
# ---------------------------------------------------------------------------

_ADVERSARIAL_SYSTEM = """You are a devil's advocate error detector. Your SOLE JOB: find logical errors, factual mistakes, and contradictions. Ignore style, completeness, and preference.

Work through each category explicitly:

**Logic errors**: For each "because / therefore / since / so" in the answer — does the conclusion actually follow from the premise? Flag faulty inferences.

**Factual errors**: Numbers, dates, names, relationships — are they consistent with the provided context? Flag specific claims that contradict provided facts.

**Internal contradictions**: Does any part of the answer contradict another part?

**Unsupported assertions**: Specific factual claims stated with certainty that have no basis in context, tools, or common knowledge.

NEVER flag:
- Claims matching owner facts or knowledge graph entries (these are verified)
- Estimates or ranges explicitly labeled as approximate
- Opinions and recommendations
- Missing information (completeness is not your concern — only correctness)

Output JSON only:
{"flaws": [{"type": "logic|factual|contradiction|unsupported", "description": "specific issue under 30 words", "blocking": true}],
 "verdict": "pass|fail"}

verdict="fail" ONLY when 1+ flaws have blocking=true. When uncertain whether a flaw is real, set blocking=false. Default to pass."""


async def adversarial_critique(
    query: str,
    answer: str,
    sources: str = "",
    user_facts: str = "",
    kg_facts: str = "",
) -> dict | None:
    """Adversarial critique — hunts for logical/factual errors, not coverage gaps.

    Runs a separate, narrowly-focused LLM call that acts as devil's advocate.
    Distinct from standard critique (which checks coverage and grounding).
    This specifically finds WRONG claims, not missing ones.

    Returns {"flaws": [...], "blocking_flaws": [...], "verdict": "pass|fail"} or None.
    """
    user_content = f"Question: {query}\n\nAnswer to evaluate:\n{answer[:config.CRITIQUE_ANSWER_LIMIT]}"
    if sources:
        user_content += f"\n\nRetrieved sources (verified):\n{sources[:config.CRITIQUE_SOURCES_LIMIT]}"
    if user_facts:
        user_content += f"\n\nOwner facts (verified):\n{user_facts[:config.CRITIQUE_FACTS_LIMIT]}"
    if kg_facts:
        user_content += f"\n\nKnowledge graph facts (verified):\n{kg_facts[:config.CRITIQUE_FACTS_LIMIT]}"

    try:
        raw = await llm.invoke_nothink(
            [
                {"role": "system", "content": _ADVERSARIAL_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            json_mode=True,
            json_prefix='{"',
            max_tokens=500,
            temperature=0.1,
        )
        if not raw:
            return None

        result = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(result, dict):
            return None

        flaws = result.get("flaws", [])
        if not isinstance(flaws, list):
            flaws = []

        verdict = str(result.get("verdict", "pass")).lower().strip()
        if verdict not in ("pass", "fail"):
            verdict = "pass"

        blocking = [f for f in flaws if isinstance(f, dict) and f.get("blocking", False)]

        return {
            "flaws": flaws,
            "blocking_flaws": blocking,
            "verdict": verdict,
        }
    except Exception as e:
        logger.warning("Adversarial critique failed: %s", e)
        return None


def format_adversarial_for_replan(critique: dict) -> str:
    """Format adversarial critique blocking flaws as a re-generation instruction."""
    if not critique or critique.get("verdict") != "fail":
        return ""
    blocking = critique.get("blocking_flaws", [])
    if not blocking:
        return ""
    lines: list[str] = []
    for flaw in blocking:
        if not isinstance(flaw, dict):
            continue
        ftype = str(flaw.get("type", "error")).upper()
        desc = str(flaw.get("description", "")).strip()
        if desc:
            lines.append(f"- [{ftype}] {desc}")
    if not lines:
        return ""
    return (
        "[CRITICAL ERRORS FOUND — CORRECTION REQUIRED]\n"
        "The previous answer contained these verified errors:\n"
        + "\n".join(lines)
        + "\n\nGenerate a corrected answer that fixes these specific errors. "
        "Preserve everything else from the original answer that was correct."
    )


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
