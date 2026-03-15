"""Delegate tool — spawn sub-agent think() calls for parallel task decomposition."""

from __future__ import annotations

import contextvars
import logging

from app.config import config
from app.schema import EventType
from app.tools.base import BaseTool, ToolResult

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
        "Delegate a sub-task to another agent. Use when you need to run "
        "multiple independent research tasks in parallel (e.g., comparing "
        "weather in two cities). Each delegate runs its own think() loop "
        "with full tool access (except delegate)."
    )
    parameters = "task: str, role: str = 'research assistant'"

    async def execute(
        self,
        *,
        task: str = "",
        role: str = "research assistant",
        **kwargs,
    ) -> ToolResult:
        if not task:
            return ToolResult(output="", success=False, error="No task provided")

        if not config.ENABLE_DELEGATION:
            return ToolResult(
                output="", success=False,
                error="Delegation is disabled. Set ENABLE_DELEGATION=true.",
            )

        current_depth = get_delegation_depth()
        if current_depth >= config.MAX_DELEGATION_DEPTH:
            return ToolResult(
                output="", success=False,
                error=f"Max delegation depth ({config.MAX_DELEGATION_DEPTH}) reached.",
            )

        # Import here to avoid circular imports
        from app.core.brain import think

        logger.info(
            "Delegating task (depth=%d): '%s' role='%s'",
            current_depth + 1, task[:80], role,
        )

        # Run a sub-agent think() in ephemeral mode
        set_delegation_depth(current_depth + 1)
        try:
            collected = []
            async for event in think(
                query=f"[Role: {role}] {task}",
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
                    )

            result = "".join(collected)
            if not result.strip():
                result = "[Sub-agent produced no output]"

            # Truncate to 3000 chars
            if len(result) > 3000:
                result = result[:3000] + "\n[... truncated]"

            return ToolResult(output=result, success=True)

        except Exception as e:
            logger.warning("Delegation failed: %s", e)
            return ToolResult(output="", success=False, error=f"Delegation failed: {e}")
        finally:
            set_delegation_depth(current_depth)
