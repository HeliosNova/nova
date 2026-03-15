"""Reminder tool — one-shot reminders that reuse the monitor system.

A reminder is a monitor with notify_condition='always' and a computed
schedule_seconds = (target_time - now). After firing, the heartbeat
loop marks it done (enabled=false). Uses a 'reminder:' name prefix
to distinguish from regular monitors.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from app.database import get_db
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_REMINDER_PREFIX = "reminder:"

# Natural time patterns
_RELATIVE_RE = re.compile(
    r"in\s+(\d+)\s+(second|minute|hour|day|week)s?",
    re.IGNORECASE,
)
_TOMORROW_RE = re.compile(
    r"tomorrow\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.IGNORECASE,
)
_TODAY_RE = re.compile(
    r"today\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
    re.IGNORECASE,
)

_UNIT_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
}


def _parse_time(time_str: str) -> datetime | None:
    """Parse a time string into a datetime. Supports ISO8601 and natural language."""
    time_str = time_str.strip()

    # ISO8601
    try:
        return datetime.fromisoformat(time_str)
    except ValueError:
        pass

    now = datetime.now()

    # "in N units"
    m = _RELATIVE_RE.search(time_str)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        secs = amount * _UNIT_SECONDS.get(unit, 60)
        return now + timedelta(seconds=secs)

    # "tomorrow at H:MM am/pm"
    m = _TOMORROW_RE.search(time_str)
    if m:
        return _parse_time_of_day(now + timedelta(days=1), m)

    # "today at H:MM am/pm"
    m = _TODAY_RE.search(time_str)
    if m:
        return _parse_time_of_day(now, m)

    return None


def _parse_time_of_day(base_date: datetime, match) -> datetime:
    """Parse H:MM am/pm from a regex match onto a base date."""
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = (match.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _log_action(action_type: str, params: dict, result: str, success: bool) -> None:
    """Log an action to the action_log table."""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO action_log (action_type, params, result, success) VALUES (?, ?, ?, ?)",
            (action_type, json.dumps(params, default=str), result[:2000], 1 if success else 0),
        )
    except Exception as e:
        logger.warning("Failed to log action: %s", e)


class ReminderTool(BaseTool):
    name = "reminder"
    description = (
        "Set, list, or cancel reminders. Reminders fire at a specific time and "
        "send an alert. Supports natural time like 'in 2 hours' or 'tomorrow at 9am'. "
        "Actions: set, list, cancel."
    )
    parameters = "action: str, name: str, time: str, message: str"

    def __init__(self, monitor_store: Any = None):
        self._store = monitor_store

    async def execute(
        self,
        *,
        action: str = "",
        name: str = "",
        time: str = "",
        message: str = "",
        **kwargs,
    ) -> ToolResult:
        if not self._store:
            return ToolResult(output="", success=False, error="Monitor system not initialized (reminders use monitors)")

        if not action:
            return ToolResult(output="", success=False, error="No action specified. Use: set, list, cancel")

        action = action.lower().strip()

        if action == "set":
            return self._set(name, time, message)
        elif action == "list":
            return self._list()
        elif action == "cancel":
            return self._cancel(name)
        else:
            return ToolResult(output="", success=False, error=f"Unknown action '{action}'. Use: set, list, cancel")

    def _set(self, name: str, time_str: str, message: str) -> ToolResult:
        if not name:
            return ToolResult(output="", success=False, error="Reminder name is required")
        if not time_str:
            return ToolResult(output="", success=False, error="Time is required (e.g., 'in 2 hours', '2026-03-06T15:00:00')")

        target = _parse_time(time_str)
        if not target:
            return ToolResult(
                output="", success=False,
                error=f"Could not parse time '{time_str}'. Use ISO8601 or natural language (e.g., 'in 2 hours', 'tomorrow at 9am').",
            )

        now = datetime.now()
        delta = (target - now).total_seconds()
        if delta < 5:
            return ToolResult(output="", success=False, error="Reminder time must be in the future")

        # Cap at 2 years
        if delta > 730 * 86400:
            return ToolResult(output="", success=False, error="Reminder time must be within 2 years")

        prefixed_name = f"{_REMINDER_PREFIX}{name}"
        reminder_message = message or f"Reminder: {name}"

        monitor_id = self._store.create(
            name=prefixed_name,
            check_type="query",
            check_config={"query": reminder_message},
            schedule_seconds=int(delta),
            cooldown_minutes=0,
            notify_condition="always",
        )

        if monitor_id < 0:
            return ToolResult(
                output="", success=False,
                error=f"Failed to create reminder '{name}' (may already exist)",
            )

        params = {"name": name, "time": time_str, "message": reminder_message}
        result_msg = (
            f"Reminder set: {name}\n"
            f"Fires at: {target.strftime('%Y-%m-%d %H:%M')}\n"
            f"Message: {reminder_message}"
        )
        _log_action("reminder", params, result_msg, True)
        return ToolResult(output=result_msg, success=True)

    def _list(self) -> ToolResult:
        monitors = self._store.list_all()
        reminders = [m for m in monitors if m.name.startswith(_REMINDER_PREFIX) and m.enabled]

        if not reminders:
            return ToolResult(output="No pending reminders.", success=True)

        lines = []
        for r in reminders:
            display_name = r.name[len(_REMINDER_PREFIX):]
            msg = r.check_config.get("query", "")
            lines.append(f"- {display_name}: {msg}")

        return ToolResult(
            output=f"Pending reminders ({len(reminders)}):\n" + "\n".join(lines),
            success=True,
        )

    def _cancel(self, name: str) -> ToolResult:
        if not name:
            return ToolResult(output="", success=False, error="Reminder name is required for cancellation")

        prefixed_name = f"{_REMINDER_PREFIX}{name}"
        monitor = self._store.get_by_name(prefixed_name)
        if not monitor:
            return ToolResult(output="", success=False, error=f"Reminder '{name}' not found")

        self._store.delete(monitor.id)
        result_msg = f"Reminder '{name}' cancelled."
        _log_action("reminder", {"name": name}, result_msg, True)
        return ToolResult(output=result_msg, success=True)
