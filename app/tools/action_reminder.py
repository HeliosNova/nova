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

from app.tools.action_logging import log_action as _log_action
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

_REMINDER_PREFIX = "reminder:"

# Natural time patterns
_RELATIVE_RE = re.compile(
    r"in\s+(\d+(?:\.\d+)?)\s+(second|minute|hour|day|week)s?",
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
_NEXT_WEEKDAY_RE = re.compile(
    r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    re.IGNORECASE,
)
_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_UNIT_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
}

# Recurring patterns — return (period_seconds, first_fire_offset_seconds, label).
# First-fire offset is how long to wait until the FIRST occurrence; period is
# the recurrence interval thereafter.
_EVERY_WEEKDAY_RE = re.compile(
    r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
    re.IGNORECASE,
)
_EVERY_DAY_RE = re.compile(
    r"every\s+day(?:\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?|daily(?:\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
    re.IGNORECASE,
)
_EVERY_HOUR_RE = re.compile(r"every\s+hour|hourly", re.IGNORECASE)
_EVERY_N_RE = re.compile(
    r"every\s+(\d+)\s+(second|minute|hour|day|week)s?",
    re.IGNORECASE,
)
_WEEKDAYS_AT_RE = re.compile(
    r"weekdays?(?:\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
    re.IGNORECASE,
)


def _parse_recurrence(time_str: str) -> tuple[int, int, str] | None:
    """Detect recurring patterns. Returns (period_seconds, first_fire_offset_seconds, label) or None.

    Examples:
        "every Monday at 9am"  -> (604800, secs_until_next_monday_9am, "weekly Mon 9am")
        "daily at 7am"          -> (86400, secs_until_next_7am, "daily 7am")
        "every hour"            -> (3600, 3600, "hourly")
        "every 3 hours"         -> (10800, 10800, "every 3h")
    """
    s = time_str.strip()
    now = datetime.now()

    m = _EVERY_WEEKDAY_RE.search(s)
    if m:
        target_day = _WEEKDAY_MAP[m.group(1).lower()]
        days_ahead = (target_day - now.weekday()) % 7 or 7
        hour = int(m.group(2)) if m.group(2) else 9
        minute = int(m.group(3)) if m.group(3) else 0
        ampm = (m.group(4) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        first = (now + timedelta(days=days_ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        return (604800, max(5, int((first - now).total_seconds())), f"weekly {m.group(1)} {hour:02d}:{minute:02d}")

    m = _EVERY_DAY_RE.search(s)
    if m:
        # Group sets differ between the two alternations; check both.
        hour_str = m.group(1) or m.group(4) or "9"
        minute_str = m.group(2) or m.group(5) or "0"
        ampm_str = (m.group(3) or m.group(6) or "").lower()
        hour = int(hour_str)
        minute = int(minute_str)
        if ampm_str == "pm" and hour < 12:
            hour += 12
        first = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if first <= now:
            first += timedelta(days=1)
        return (86400, max(5, int((first - now).total_seconds())), f"daily {hour:02d}:{minute:02d}")

    if _EVERY_HOUR_RE.search(s):
        return (3600, 3600, "hourly")

    m = _EVERY_N_RE.search(s)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        period = n * _UNIT_SECONDS.get(unit, 60)
        return (period, period, f"every {n} {unit}{'s' if n != 1 else ''}")

    m = _WEEKDAYS_AT_RE.search(s)
    if m:
        hour = int(m.group(1)) if m.group(1) else 9
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        # Find next weekday-at-time occurrence
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        # Skip weekends
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return (86400, max(5, int((candidate - now).total_seconds())), f"weekdays {hour:02d}:{minute:02d}")

    return None


def _parse_time(time_str: str) -> datetime | None:
    """Parse a time string into a datetime. Supports ISO8601 and natural language."""
    time_str = time_str.strip()

    # ISO8601
    try:
        return datetime.fromisoformat(time_str)
    except ValueError:
        pass

    now = datetime.now()

    # "in N units" (supports fractional, e.g. "in 1.5 hours")
    m = _RELATIVE_RE.search(time_str)
    if m:
        amount = float(m.group(1))
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

    # "next monday", "next friday", etc.
    m = _NEXT_WEEKDAY_RE.search(time_str)
    if m:
        target_day = _WEEKDAY_MAP[m.group(1).lower()]
        current_day = now.weekday()
        days_ahead = target_day - current_day
        if days_ahead <= 0:
            days_ahead += 7  # Always go to *next* occurrence
        return (now + timedelta(days=days_ahead)).replace(hour=9, minute=0, second=0, microsecond=0)

    return None


def _parse_time_of_day(base_date: datetime, match) -> datetime:
    """Parse H:MM am/pm from a regex match onto a base date.

    If the computed time is in the past and no am/pm was specified,
    assume the user meant PM and add 12 hours.
    """
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = (match.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    result = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    # If no am/pm was specified and the resulting time is in the past, assume PM
    if not ampm and result < datetime.now() and hour < 12:
        result += timedelta(hours=12)
    return result


class ReminderTool(BaseTool):
    name = "reminder"
    description = (
        "Set, list, or cancel reminders that fire at a specific time and send an alert "
        "via configured channels. Supports natural time parsing for one-shot ('in 2 hours', "
        "'tomorrow at 9am', 'next Monday') AND recurring schedules ('every Monday at 9am', "
        "'daily at 7am', 'every 3 hours', 'weekdays at 8am', 'hourly') and ISO8601. "
        "Recurring reminders fire repeatedly until cancelled. "
        "Actions: set (create reminder), list (show pending), cancel (remove by name)."
    )
    parameters = "action: str, name: str, time: str, message: str"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set", "list", "cancel"],
                "description": "Reminder action to perform.",
            },
            "name": {
                "type": "string",
                "description": "Reminder name (required for set and cancel).",
            },
            "time": {
                "type": "string",
                "description": "When to fire. ISO8601 or natural language: 'in 2 hours', 'tomorrow at 9am'. Required for set.",
            },
            "message": {
                "type": "string",
                "description": "Optional custom message for the reminder alert.",
            },
        },
        "required": ["action"],
    }

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
            return ToolResult(output="", success=False, error="Monitor system not initialized (reminders use monitors)", error_category=ErrorCategory.INTERNAL)

        if not action:
            return ToolResult(output="", success=False, error="No action specified. Use: set, list, cancel", error_category=ErrorCategory.VALIDATION)

        action = action.lower().strip()

        if action == "set":
            return self._set(name, time, message)
        elif action == "list":
            return self._list()
        elif action == "cancel":
            return self._cancel(name)
        else:
            return ToolResult(output="", success=False, error=f"Unknown action '{action}'. Use: set, list, cancel", error_category=ErrorCategory.VALIDATION)

    def _set(self, name: str, time_str: str, message: str) -> ToolResult:
        if not name:
            return ToolResult(output="", success=False, error="Reminder name is required", error_category=ErrorCategory.VALIDATION)
        if not time_str:
            return ToolResult(output="", success=False, error="Time is required (e.g., 'in 2 hours', '2026-03-06T15:00:00', 'every Monday at 9am')", error_category=ErrorCategory.VALIDATION)

        # Recurring path takes precedence — "every Monday" should not be parsed
        # as a one-shot.
        recurrence = _parse_recurrence(time_str)
        prefixed_name = f"{_REMINDER_PREFIX}{name}"
        reminder_message = message or f"Reminder: {name}"

        if recurrence is not None:
            period_seconds, first_offset, label = recurrence
            # Cap period at 1 year for sanity
            if period_seconds > 365 * 86400:
                return ToolResult(output="", success=False, error="Recurrence period must be ≤ 1 year", error_category=ErrorCategory.VALIDATION)
            monitor_id = self._store.create(
                name=prefixed_name,
                check_type="query",
                check_config={"query": reminder_message, "recurring": True, "recurrence_label": label},
                schedule_seconds=int(period_seconds),
                cooldown_minutes=0,
                notify_condition="always",
            )
            if monitor_id < 0:
                return ToolResult(
                    output="", success=False,
                    error=f"Failed to create recurring reminder '{name}' (may already exist)",
                    error_category=ErrorCategory.INTERNAL,
                )
            first_fire = datetime.now() + timedelta(seconds=first_offset)
            params = {"name": name, "time": time_str, "message": reminder_message, "recurring": label}
            result_msg = (
                f"Recurring reminder set: {name}\n"
                f"Schedule: {label}\n"
                f"First fire: {first_fire.strftime('%Y-%m-%d %H:%M')}\n"
                f"Message: {reminder_message}"
            )
            _log_action("reminder", params, result_msg, True)
            return ToolResult(output=result_msg, success=True)

        # One-shot path
        target = _parse_time(time_str)
        if not target:
            return ToolResult(
                output="", success=False,
                error=f"Could not parse time '{time_str}'. Use ISO8601 or natural language (e.g., 'in 2 hours', 'tomorrow at 9am', 'every Monday at 9am').",
                error_category=ErrorCategory.VALIDATION,
            )

        now = datetime.now()
        delta = (target - now).total_seconds()
        if delta < 5:
            return ToolResult(output="", success=False, error="Reminder time must be in the future", error_category=ErrorCategory.VALIDATION)

        # Cap at 2 years
        if delta > 730 * 86400:
            return ToolResult(output="", success=False, error="Reminder time must be within 2 years", error_category=ErrorCategory.VALIDATION)

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
                error_category=ErrorCategory.INTERNAL,
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
            cfg = r.check_config or {}
            msg = cfg.get("query", "")
            if cfg.get("recurring"):
                label = cfg.get("recurrence_label", "recurring")
                lines.append(f"- {display_name} [{label}]: {msg}")
            else:
                lines.append(f"- {display_name}: {msg}")

        return ToolResult(
            output=f"Pending reminders ({len(reminders)}):\n" + "\n".join(lines),
            success=True,
        )

    def _cancel(self, name: str) -> ToolResult:
        if not name:
            return ToolResult(output="", success=False, error="Reminder name is required for cancellation", error_category=ErrorCategory.VALIDATION)

        prefixed_name = f"{_REMINDER_PREFIX}{name}"
        monitor = self._store.get_by_name(prefixed_name)
        if not monitor:
            return ToolResult(output="", success=False, error=f"Reminder '{name}' not found", error_category=ErrorCategory.NOT_FOUND)

        self._store.delete(monitor.id)
        result_msg = f"Reminder '{name}' cancelled."
        _log_action("reminder", {"name": name}, result_msg, True)
        return ToolResult(output=result_msg, success=True)
