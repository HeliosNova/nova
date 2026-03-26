"""Background task tool — submit and manage async tasks without blocking the conversation."""

from __future__ import annotations

import logging

from app.schema import EventType
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)


class BackgroundTaskTool(BaseTool):
    name = "background_task"
    description = (
        "Run tasks in the background without blocking the conversation. "
        "Spawns an ephemeral reasoning loop with full tool access for long-running research or analysis. "
        "Actions: submit (start a new background task), status (check task progress by ID), "
        "list (show all tasks with status icons), cancel (stop a running task). "
        "Do NOT use for quick questions — only for tasks that would take too long to wait for."
    )
    parameters = "action: str (submit|status|list|cancel), task: str = '', task_id: str = ''"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["submit", "status", "list", "cancel"],
                "description": "Background task action to perform.",
            },
            "task": {
                "type": "string",
                "description": "Task description to submit (required for submit action).",
            },
            "task_id": {
                "type": "string",
                "description": "Task ID (required for status and cancel actions).",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        *,
        action: str = "",
        task: str = "",
        task_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not action:
            return ToolResult(output="", success=False, error="No action provided. Use: submit, status, list, cancel", error_category=ErrorCategory.VALIDATION)

        # Get TaskManager from Services
        from app.core.brain import get_services
        try:
            svc = get_services()
        except RuntimeError:
            return ToolResult(output="", success=False, error="Services not initialized", error_category=ErrorCategory.INTERNAL)

        tm = svc.task_manager
        if tm is None:
            return ToolResult(output="", success=False, error="TaskManager not available", error_category=ErrorCategory.INTERNAL)

        action = action.lower().strip()

        if action == "submit":
            return await self._submit(tm, task)
        elif action == "status":
            return self._status(tm, task_id)
        elif action == "list":
            return self._list(tm)
        elif action == "cancel":
            return self._cancel(tm, task_id)
        else:
            return ToolResult(output="", success=False, error=f"Unknown action '{action}'. Use: submit, status, list, cancel", error_category=ErrorCategory.VALIDATION)

    async def _submit(self, tm, task: str) -> ToolResult:
        if not task:
            return ToolResult(output="", success=False, error="No task provided for submit", error_category=ErrorCategory.VALIDATION)

        # Import here to avoid circular imports
        from app.core.brain import think

        # Shared list for partial result capture on failure
        partial_collector: list[str] = []

        async def _run_think() -> str:
            """Run an ephemeral think() call and collect the output."""
            async for event in think(
                query=task,
                conversation_id=None,
                ephemeral=True,
            ):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    partial_collector.append(text)
                elif event.type == EventType.ERROR:
                    raise RuntimeError(event.data.get("message", "unknown error"))

            result = "".join(partial_collector)
            if not result.strip():
                return "[Background task produced no output]"

            # Truncate to 3000 chars
            if len(result) > 3000:
                result = result[:3000] + "\n[... truncated]"
            return result

        try:
            task_id = tm.submit(_run_think(), description=task[:200], partial_collector=partial_collector)
        except RuntimeError as e:
            return ToolResult(output="", success=False, error=str(e), error_category=ErrorCategory.INTERNAL)

        return ToolResult(
            output=f"Background task submitted. ID: {task_id}\nUse background_task(action='status', task_id='{task_id}') to check progress.",
            success=True,
        )

    def _status(self, tm, task_id: str) -> ToolResult:
        if not task_id:
            return ToolResult(output="", success=False, error="No task_id provided for status", error_category=ErrorCategory.VALIDATION)

        bg = tm.get_status(task_id)
        if not bg:
            return ToolResult(output="", success=False, error=f"Task '{task_id}' not found", error_category=ErrorCategory.NOT_FOUND)

        lines = [
            f"Task: {bg.id}",
            f"Status: {bg.status}",
            f"Description: {bg.description}",
            f"Created: {bg.created_at}",
        ]
        if bg.completed_at:
            lines.append(f"Completed: {bg.completed_at}")
        if bg.result:
            lines.append(f"Result: {bg.result}")
        if bg.error:
            lines.append(f"Error: {bg.error}")
        if bg.partial_result:
            lines.append(f"Partial result: {bg.partial_result}")

        return ToolResult(output="\n".join(lines), success=True)

    def _list(self, tm) -> ToolResult:
        tasks = tm.list_tasks()
        if not tasks:
            return ToolResult(output="No background tasks.", success=True)

        lines = []
        for bg in tasks:
            status_icon = {"pending": "...", "running": ">>>", "complete": "OK", "failed": "ERR", "cancelled": "X"}.get(bg.status, "?")
            lines.append(f"[{status_icon}] {bg.id}: {bg.description[:60]} ({bg.status})")

        return ToolResult(output="\n".join(lines), success=True)

    def _cancel(self, tm, task_id: str) -> ToolResult:
        if not task_id:
            return ToolResult(output="", success=False, error="No task_id provided for cancel", error_category=ErrorCategory.VALIDATION)

        if tm.cancel(task_id):
            return ToolResult(output=f"Task '{task_id}' cancelled.", success=True)
        else:
            return ToolResult(output="", success=False, error=f"Task '{task_id}' not found or already finished", error_category=ErrorCategory.NOT_FOUND)
