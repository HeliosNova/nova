"""Auto skill creation — background extraction of reusable skills from interactions.

When a response uses 1+ tools successfully, we fire a background task that asks
the LLM to extract a reusable skill (trigger pattern + steps). Reuses ALL
existing skill guards (broadness, regex validation, capture groups).

Threshold notes:
- Single-tool interactions: extracted only when the answer looks successful
  (no failure markers). This prevents caching "I couldn't find anything" patterns.
- Multi-tool interactions (2+): extracted unconditionally (same as before).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.config import config
from app.core import llm
from app.core.skills import SkillStore, _get_tool_names, _is_too_broad, _has_capture_group_mismatch

logger = logging.getLogger(__name__)

# Phrases that indicate a failed/uncertain response — don't cache these patterns.
_FAILURE_MARKERS = (
    "couldn't find",
    "could not find",
    "failed to",
    "no results",
    "i don't know",
    "i'm not sure",
    "unable to",
    "not found",
    "error occurred",
    "an error",
    "i apologize",
    "unfortunately",
    "no information",
)


async def maybe_extract_skill(
    query: str,
    tool_results: list[dict],
    final_answer: str,
    skills: SkillStore,
    quality_score: float | None = None,
) -> None:
    """Background task: attempt to extract a reusable skill from a tool interaction.

    Runs when:
    - ENABLE_AUTO_SKILL_CREATION is true
    - 1+ tool results in the interaction
    - Single-tool interactions pass a quality gate (no failure markers, OR quality >= 0.7)
    - Multi-tool interactions extracted unconditionally
    - Not a delegate-based interaction

    quality_score: reflexion quality from the response (0.0–1.0). When >= 0.7 for
    single-tool interactions, the failure-marker check is bypassed — the reflexion
    system already confirmed the response was good.

    Failures are logged, never raised.
    """
    if not config.ENABLE_AUTO_SKILL_CREATION:
        return

    if not tool_results:
        return

    # Skip if any tool was delegate (sub-agent interactions are too complex)
    if any(tr.get("tool") == "delegate" for tr in tool_results):
        return

    # Quality gate for single-tool interactions: only extract from successful responses.
    # Multi-tool interactions are assumed successful enough to attempt extraction.
    # Exception: if the reflexion quality score is >= 0.7, the response was confirmed
    # good — bypass the failure marker check.
    if len(tool_results) == 1:
        high_quality = quality_score is not None and quality_score >= 0.7
        if not high_quality:
            answer_lower = final_answer.lower()
            if any(marker in answer_lower for marker in _FAILURE_MARKERS):
                logger.debug(
                    "Auto-skill: single-tool response contains failure marker, skipping"
                )
                return

    tool_summary = json.dumps([
        {
            "tool": tr["tool"],
            "args": tr["args"],
            "output": (tr.get("output", "") or "")[:200],
        }
        for tr in tool_results
    ], indent=2)

    try:
        result = await asyncio.wait_for(
            llm.invoke_nothink(
                [
                    {
                        "role": "system",
                        "content": (
                            "You extract reusable skills from tool interactions.\n"
                            "A skill is a trigger pattern (regex) and a sequence of tool calls "
                            "that should be repeated for similar future queries.\n\n"
                            "Given a query, the tool calls used, and the final answer, decide if "
                            "this is a reusable pattern worth caching.\n\n"
                            "Respond with JSON:\n"
                            '{"name": "short_name", "trigger_pattern": "regex_for_similar_queries", '
                            '"steps": [{"tool": "tool_name", "args_template": {"key": "{query}"}, '
                            '"output_key": "result"}], '
                            '"answer_template": "Template using {result}"}\n\n'
                            "IMPORTANT:\n"
                            "- trigger_pattern must be a valid regex (not too broad)\n"
                            f"- steps must use actual tools: {', '.join(_get_tool_names())}\n"
                            "- Use {query} as placeholder for the user's input\n"
                            "- Only extract if this is genuinely reusable for future similar queries\n\n"
                            'If NOT reusable, respond: {"skip": true}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Query: {query}\n\n"
                            f"Tool calls:\n{tool_summary}\n\n"
                            f"Answer: {final_answer[:500]}"
                        ),
                    },
                ],
                json_mode=True,
                json_prefix="{",
                max_tokens=500,
                temperature=0.2,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )

        obj = llm.extract_json_object(result)
        if not obj or obj.get("skip") or not obj.get("name"):
            logger.debug("Auto-skill: LLM decided not to extract skill")
            return

        pattern = obj.get("trigger_pattern", "")
        if not pattern:
            return

        try:
            re.compile(pattern)
        except re.error:
            logger.debug("Auto-skill: invalid regex '%s'", pattern)
            return

        if _is_too_broad(pattern):
            logger.debug("Auto-skill: pattern too broad '%s'", pattern)
            return

        steps = obj.get("steps", [])
        valid_tool_names = _get_tool_names()
        for step in steps:
            if not isinstance(step, dict):
                logger.debug("Auto-skill: step is not a dict")
                return
            if step.get("tool") not in valid_tool_names:
                logger.debug("Auto-skill: unknown tool '%s'", step.get("tool"))
                return
            if "args_template" not in step:
                logger.debug("Auto-skill: step missing args_template")
                return
            output_key = step.get("output_key", "")
            if output_key and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", output_key):
                logger.debug("Auto-skill: invalid output_key '%s'", output_key)
                return

        answer_template = obj.get("answer_template")

        if _has_capture_group_mismatch(pattern, steps, answer_template):
            logger.debug("Auto-skill: capture group mismatch")
            return

        skill_id = skills.create_skill(
            name=obj["name"],
            trigger_pattern=pattern,
            steps=steps,
            answer_template=answer_template,
            source="auto",
        )

        if skill_id:
            logger.info(
                "Auto-skill created: '%s' (id=%d, trigger=%s, tools=%d)",
                obj["name"], skill_id, pattern, len(tool_results),
            )
        else:
            logger.debug("Auto-skill rejected by guards: '%s'", obj["name"])

    except Exception as e:
        logger.debug("Auto-skill extraction failed: %s", e)
