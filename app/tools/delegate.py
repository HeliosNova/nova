"""Delegate tool — spawn sub-agent think() calls for parallel task decomposition."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import re

from app.config import config
from app.core.access_tiers import _tier as get_current_tier, set_access_tier_override, get_access_tier_override
from app.schema import EventType
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

# Request-scoped depth counter (safe for concurrent requests)
_delegation_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "delegation_depth", default=0
)


def get_delegation_depth() -> int:
    return _delegation_depth.get()


def set_delegation_depth(depth: int) -> None:
    _delegation_depth.set(depth)


class DelegateTool(BaseTool):
    name = "delegate"
    description = (
        "Delegate a sub-task to another agent that runs its own reasoning loop with full tool access (except delegate). "
        "Use for parallel independent research tasks (e.g., comparing weather in multiple cities). "
        "Each delegate produces a text result. Returns the sub-agent's response (up to 3000 chars). "
        "Do NOT use for sequential tasks or tasks that depend on each other."
    )
    parameters = "task: str, role: str = 'research assistant', findings: dict = None, source_data: dict = None"
    input_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task description for the sub-agent to execute.",
            },
            "role": {
                "type": "string",
                "description": "Role for the sub-agent (e.g., 'research assistant', 'data analyst'). Defaults to 'research assistant'.",
            },
            "findings": {
                "type": "object",
                "description": "Optional structured findings/context to inject into the delegate prompt.",
            },
            "source_data": {
                "type": "object",
                "description": "Optional structured source data to inject into the delegate prompt.",
            },
        },
        "required": ["task"],
    }

    async def execute(
        self,
        *,
        task: str = "",
        role: str = "research assistant",
        findings: dict | None = None,
        source_data: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        if not task:
            return ToolResult(output="", success=False, error="No task provided", error_category=ErrorCategory.VALIDATION)

        if not config.ENABLE_DELEGATION:
            return ToolResult(
                output="", success=False,
                error="Delegation is disabled. Set ENABLE_DELEGATION=true.",
                error_category=ErrorCategory.PERMISSION,
            )

        current_depth = get_delegation_depth()
        if current_depth >= config.MAX_DELEGATION_DEPTH:
            return ToolResult(
                output="", success=False,
                error=f"Max delegation depth ({config.MAX_DELEGATION_DEPTH}) reached.",
                error_category=ErrorCategory.PERMISSION,
            )

        # Capture the parent's effective access tier so the sub-agent
        # can never escalate beyond it.
        parent_tier = get_current_tier()

        # Import here to avoid circular imports
        from app.core.brain import think

        logger.info(
            "Delegating task (depth=%d, tier=%s): '%s' role='%s'",
            current_depth + 1, parent_tier, task[:80], role,
        )

        # Sanitize role: strip non-alphanumeric chars (keep letters, digits,
        # spaces, underscores) and truncate to 50 chars to prevent injection.
        role = re.sub(r"[^a-zA-Z0-9 _]", "", role)[:50]

        # Build the delegate prompt, injecting structured context if provided
        import json as _json
        delegate_query = f"[Role: {role}] {task}"
        if findings:
            delegate_query += f"\n\n[Findings Context]\n{_json.dumps(findings, default=str)}"
        if source_data:
            delegate_query += f"\n\n[Source Data]\n{_json.dumps(source_data, default=str)}"

        # Run a sub-agent think() in ephemeral mode
        timeout = getattr(config, "GENERATION_TIMEOUT", 480)
        set_delegation_depth(current_depth + 1)
        # Constrain sub-agent to parent's access tier (prevent escalation)
        previous_tier_override = get_access_tier_override()
        set_access_tier_override(parent_tier)
        try:

            async def _run_delegate():
                collected = []
                async for event in think(
                    query=delegate_query,
                    conversation_id=None,
                    ephemeral=True,
                ):
                    if event.type == EventType.TOKEN:
                        text = event.data.get("text", "")
                        collected.append(text)
                    elif event.type == EventType.ERROR:
                        return ToolResult(
                            output="",
                            success=False,
                            error=f"Sub-agent error: {event.data.get('message', 'unknown')}",
                            error_category=ErrorCategory.INTERNAL,
                        )
                return collected

            result_or_error = await asyncio.wait_for(
                _run_delegate(), timeout=timeout,
            )

            # _run_delegate returns a ToolResult on error, or a list on success
            if isinstance(result_or_error, ToolResult):
                return result_or_error

            result = "".join(result_or_error)
            if not result.strip():
                result = "[Sub-agent produced no output]"

            # Truncate to 3000 chars
            if len(result) > 3000:
                result = result[:3000] + "\n[... truncated]"

            return ToolResult(output=result, success=True)

        except asyncio.TimeoutError:
            logger.warning("Delegation timed out after %ds", timeout)
            return ToolResult(output="", success=False, error=f"Delegation timed out after {timeout}s", error_category=ErrorCategory.TRANSIENT)
        except Exception as e:
            logger.warning("Delegation failed: %s", e)
            return ToolResult(output="", success=False, error=f"Delegation failed: {e}", error_category=ErrorCategory.INTERNAL)
        finally:
            set_delegation_depth(current_depth)
            set_access_tier_override(previous_tier_override)
