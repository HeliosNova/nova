"""Monitor tool — lets Nova create and manage monitors via conversation.

"Monitor Bitcoin price every 30 minutes" → creates a search monitor
"Check if example.com is down every 5 minutes" → creates a url monitor
"Stop monitoring Bitcoin" → deletes monitor
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class MonitorTool(BaseTool):
    name = "monitor"
    description = (
        "Create, list, or delete background monitors that automatically check things "
        "and alert the user when changes happen. Actions: create, list, delete."
    )
    parameters = "action: str, name: str, check_type: str, config: dict, schedule_minutes: int"

    def __init__(self, monitor_store: Any = None):
        self._store = monitor_store

    async def execute(
        self,
        *,
        action: str = "",
        name: str = "",
        check_type: str = "search",
        config: dict | str = None,
        schedule_minutes: int = 30,
        **kwargs,
    ) -> ToolResult:
        if not action:
            return ToolResult(output="", success=False, error="No action specified. Use: create, list, delete")

        if not self._store:
            return ToolResult(output="", success=False, error="Monitor system not initialized")

        action = action.lower().strip()

        if action == "create":
            return await self._create(name, check_type, config, schedule_minutes)
        elif action == "list":
            return await self._list()
        elif action == "delete":
            return await self._delete(name)
        else:
            return ToolResult(output="", success=False, error=f"Unknown action '{action}'. Use: create, list, delete")

    async def _create(
        self,
        name: str,
        check_type: str,
        check_config: dict | str | None,
        schedule_minutes: int,
    ) -> ToolResult:
        if not name:
            return ToolResult(output="", success=False, error="Monitor name is required")

        # Parse config if it's a string
        if isinstance(check_config, str):
            try:
                check_config = json.loads(check_config)
            except (json.JSONDecodeError, TypeError):
                check_config = {"query": check_config}

        if not check_config:
            check_config = {}

        # Validate check_type
        valid_types = {"url", "search", "command", "query"}
        if check_type not in valid_types:
            return ToolResult(
                output="", success=False,
                error=f"Invalid check_type '{check_type}'. Use: {', '.join(valid_types)}",
            )

        # Auto-populate config based on type
        if check_type == "url" and "url" not in check_config:
            # If name looks like a URL, use it
            if name.startswith("http"):
                check_config["url"] = name
                name = name.split("//")[-1].split("/")[0]  # Use hostname as name

        if check_type == "search" and "query" not in check_config:
            check_config["query"] = name

        schedule_seconds = max(60, schedule_minutes * 60)  # Minimum 1 minute

        monitor_id = self._store.create(
            name=name,
            check_type=check_type,
            check_config=check_config,
            schedule_seconds=schedule_seconds,
            cooldown_minutes=max(1, schedule_minutes),
        )

        if monitor_id < 0:
            return ToolResult(
                output="", success=False,
                error=f"Failed to create monitor '{name}' (name may already exist)",
            )

        return ToolResult(
            output=(
                f"Monitor '{name}' created (id={monitor_id})\n"
                f"Type: {check_type}\n"
                f"Schedule: every {schedule_minutes} minutes\n"
                f"Config: {json.dumps(check_config)}\n"
                f"I'll check this automatically and alert you when something changes."
            ),
            success=True,
        )

    async def _list(self) -> ToolResult:
        monitors = self._store.list_all()
        if not monitors:
            return ToolResult(output="No monitors configured yet.", success=True)

        lines = []
        for m in monitors:
            status = "active" if m.enabled else "paused"
            schedule = f"every {m.schedule_seconds // 60}m"
            last = m.last_check_at[:16] if m.last_check_at else "never"
            lines.append(f"- [{m.id}] {m.name} ({m.check_type}, {schedule}, {status}, last: {last})")

        return ToolResult(
            output=f"Monitors ({len(monitors)}):\n" + "\n".join(lines),
            success=True,
        )

    async def _delete(self, name: str) -> ToolResult:
        if not name:
            return ToolResult(output="", success=False, error="Monitor name is required for deletion")

        # Try by name first, then by ID
        monitor = self._store.get_by_name(name)
        if not monitor:
            try:
                monitor = self._store.get(int(name))
            except (ValueError, TypeError):
                pass

        if not monitor:
            return ToolResult(output="", success=False, error=f"Monitor '{name}' not found")

        self._store.delete(monitor.id)
        return ToolResult(output=f"Monitor '{monitor.name}' (id={monitor.id}) deleted.", success=True)
