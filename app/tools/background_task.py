"""Background task tool — submit and manage async tasks without blocking the conversation."""

from __future__ import annotations

import logging

from app.schema import EventType
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class BackgroundTaskTool(BaseTool):
    name = "background_task"
    description = (
        "Run a task in the background without blocking the conversation. "
        "Use for long-running research, analysis, or any task that doesn't "
        "need an immediate answer. Actions: submit (start a task), status "
        "(check a task), list (show all tasks), cancel (stop a task)."
    )
    parameters = "action: str (submit|status|list|cancel), task: str = '', task_id: str = ''"

    async def execute(
        self,
        *,
        action: str = "",
        task: str = "",
        task_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not action:
            return ToolResult(output="", success=False, error="No action provided. Use: submit, status, list, cancel")

        # Get TaskManager from Services
        from app.core.brain import get_services
        try:
            svc = get_services()
        except RuntimeError:
            return ToolResult(output="", success=False, error="Services not initialized")

        tm = svc.task_manager
        if tm is None:
            return ToolResult(output="", success=False, error="TaskManager not available")

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
            return ToolResult(output="", success=False, error=f"Unknown action '{action}'. Use: submit, status, list, cancel")

    async def _submit(self, tm, task: str) -> ToolResult:
        if not task:
            return ToolResult(output="", success=False, error="No task provided for submit")

        # Import here to avoid circular imports
        from app.core.brain import think

        async def _run_think() -> str:
            """Run an ephemeral think() call and collect the output."""
            collected = []
            async for event in think(
                query=task,
                conversation_id=None,
                ephemeral=True,
            ):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    collected.append(text)
                elif event.type == EventType.ERROR:
                    raise RuntimeError(event.data.get("message", "unknown error"))

            result = "".join(collected)
            if not result.strip():
                return "[Background task produced no output]"

            # Truncate to 3000 chars
            if len(result) > 3000:
                result = result[:3000] + "\n[... truncated]"
            return result

        try:
            task_id = tm.submit(_run_think(), description=task[:200])
        except RuntimeError as e:
            return ToolResult(output="", success=False, error=str(e))

        return ToolResult(
            output=f"Background task submitted. ID: {task_id}\nUse background_task(action='status', task_id='{task_id}') to check progress.",
            success=True,
        )

    def _status(self, tm, task_id: str) -> ToolResult:
        if not task_id:
            return ToolResult(output="", success=False, error="No task_id provided for status")

        bg = tm.get_status(task_id)
        if not bg:
            return ToolResult(output="", success=False, error=f"Task '{task_id}' not found")

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
            return ToolResult(output="", success=False, error="No task_id provided for cancel")

        if tm.cancel(task_id):
            return ToolResult(output=f"Task '{task_id}' cancelled.", success=True)
        else:
            return ToolResult(output="", success=False, error=f"Task '{task_id}' not found or already finished")
