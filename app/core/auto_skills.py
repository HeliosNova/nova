"""Auto skill creation — background extraction of reusable skills from interactions.

When a response uses 2+ tools, we fire a background task that asks the LLM
to extract a reusable skill (trigger pattern + steps). Reuses ALL existing
skill guards (broadness, regex validation, capture groups).
"""

from __future__ import annotations

import json
import logging

from app.config import config
from app.core import llm
from app.core.skills import SkillStore, _get_tool_names, _is_too_broad, _has_capture_group_mismatch

logger = logging.getLogger(__name__)


async def maybe_extract_skill(
    query: str,
    tool_results: list[dict],
    final_answer: str,
    skills: SkillStore,
) -> None:
    """Background task: attempt to extract a reusable skill from a multi-tool interaction.

    Only runs when:
    - ENABLE_AUTO_SKILL_CREATION is true
    - 2+ tool results in the interaction
    - Not a delegate-based interaction

    Failures are logged, never raised.
    """
    if not config.ENABLE_AUTO_SKILL_CREATION:
        return

    if len(tool_results) < 2:
        return

    # Skip if any tool was delegate (sub-agent interactions are too complex)
    if any(tr.get("tool") == "delegate" for tr in tool_results):
        return

    tool_summary = json.dumps([
        {"tool": tr["tool"], "args": tr["args"]}
        for tr in tool_results
    ], indent=2)

    try:
        result = await llm.invoke_nothink(
            [
                {
                    "role": "system",
                    "content": (
                        "You extract reusable skills from multi-tool interactions.\n"
                        "A skill is a trigger pattern (regex) and a sequence of tool calls "
                        "that should be repeated for similar future queries.\n\n"
                        "Given a query, the tool calls used, and the final answer, decide if "
                        "this is a reusable pattern.\n\n"
                        "Respond with JSON:\n"
                        '{"name": "short_name", "trigger_pattern": "regex_for_similar_queries", '
                        '"steps": [{"tool": "tool_name", "args_template": {"key": "{query}"}, '
                        '"output_key": "result"}], '
                        '"answer_template": "Template using {result}"}\n\n'
                        "IMPORTANT:\n"
                        "- trigger_pattern must be a valid regex (not too broad)\n"
                        "- steps must use actual tools: web_search, calculator, http_fetch, "
                        "knowledge_search, code_exec, memory_search, file_ops, shell_exec, "
                        "browser, integration\n"
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
        )

        obj = llm.extract_json_object(result)
        if obj.get("skip") or not obj.get("name"):
            logger.debug("Auto-skill: LLM decided not to extract skill")
            return

        pattern = obj.get("trigger_pattern", "")
        if not pattern:
            return

        # Validate regex
        import re
        try:
            re.compile(pattern)
        except re.error:
            logger.debug("Auto-skill: invalid regex '%s'", pattern)
            return

        # Guard: broadness check
        if _is_too_broad(pattern):
            logger.debug("Auto-skill: pattern too broad '%s'", pattern)
            return

        # Validate steps reference real tools
        steps = obj.get("steps", [])
        valid_tool_names = _get_tool_names()
        for step in steps:
            if step.get("tool") not in valid_tool_names:
                logger.debug("Auto-skill: unknown tool '%s'", step.get("tool"))
                return

        answer_template = obj.get("answer_template")

        # Guard: capture group mismatch
        if _has_capture_group_mismatch(pattern, steps, answer_template):
            logger.debug("Auto-skill: capture group mismatch")
            return

        # Create the skill
        skill_id = skills.create_skill(
            name=obj["name"],
            trigger_pattern=pattern,
            steps=steps,
            answer_template=answer_template,
        )

        if skill_id:
            logger.info("Auto-skill created: '%s' (id=%d, trigger=%s)", obj["name"], skill_id, pattern)
        else:
            logger.debug("Auto-skill rejected by guards: '%s'", obj["name"])

    except Exception as e:
        logger.debug("Auto-skill extraction failed: %s", e)
