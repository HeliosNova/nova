"""Calendar tool — .ics event management using the ics library.

Creates, lists, searches, and deletes calendar events stored
in a local .ics file. Pure-Python, RFC 5545 compliant.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import config
from app.database import get_db
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Limits
_MAX_DURATION_MINUTES = 24 * 60  # 24 hours
_MAX_DATE_RANGE_DAYS = 730       # 2 years


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


def _safe_calendar_path() -> Path:
    """Return the calendar path, sandboxed to /data/."""
    path = Path(config.CALENDAR_PATH)
    # Sandbox: must be under /data/
    resolved = path.resolve()
    data_dir = Path("/data").resolve()
    if not str(resolved).startswith(str(data_dir)):
        # Fall back to safe default
        path = data_dir / "calendar.ics"
    return path


def _load_calendar(path: Path):
    """Load or create a calendar from an .ics file."""
    from ics import Calendar

    if path.exists():
        with open(path, "r") as f:
            return Calendar(f.read())
    return Calendar()


def _save_calendar(cal, path: Path) -> None:
    """Save calendar to .ics file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(cal.serialize_iter())


class CalendarTool(BaseTool):
    name = "calendar"
    description = (
        "Manage calendar events in a local .ics file. "
        "Actions: create, list, search, delete."
    )
    parameters = (
        "action: str, title: str, start: str (ISO8601), "
        "duration_minutes: int, description: str, location: str, "
        "days: int (for list), query: str (for search), uid: str (for delete)"
    )

    async def execute(
        self,
        *,
        action: str = "",
        title: str = "",
        start: str = "",
        duration_minutes: int = 60,
        description: str = "",
        location: str = "",
        days: int = 7,
        query: str = "",
        uid: str = "",
        **kwargs,
    ) -> ToolResult:
        if not config.ENABLE_CALENDAR:
            return ToolResult(
                output="", success=False,
                error="Calendar is disabled. Set ENABLE_CALENDAR=true.",
            )

        if not action:
            return ToolResult(
                output="", success=False,
                error="No action specified. Use: create, list, search, delete",
            )

        action = action.lower().strip()

        if action == "create":
            return self._create(title, start, duration_minutes, description, location)
        elif action == "list":
            return self._list(days)
        elif action == "search":
            return self._search(query)
        elif action == "delete":
            return self._delete(uid)
        else:
            return ToolResult(
                output="", success=False,
                error=f"Unknown action '{action}'. Use: create, list, search, delete",
            )

    def _create(
        self,
        title: str,
        start: str,
        duration_minutes: int,
        description: str,
        location: str,
    ) -> ToolResult:
        if not title:
            return ToolResult(output="", success=False, error="Title is required")
        if not start:
            return ToolResult(output="", success=False, error="Start time is required (ISO8601)")

        # Validate duration
        if duration_minutes < 1:
            duration_minutes = 60
        if duration_minutes > _MAX_DURATION_MINUTES:
            return ToolResult(
                output="", success=False,
                error=f"Duration cannot exceed {_MAX_DURATION_MINUTES} minutes (24 hours)",
            )

        # Parse start time
        try:
            start_dt = datetime.fromisoformat(start)
        except ValueError:
            return ToolResult(
                output="", success=False,
                error=f"Invalid start time '{start}'. Use ISO8601 format (e.g., 2026-03-06T15:00:00)",
            )

        # Date range check
        now = datetime.now()
        if abs((start_dt - now).days) > _MAX_DATE_RANGE_DAYS:
            return ToolResult(
                output="", success=False,
                error=f"Date must be within {_MAX_DATE_RANGE_DAYS} days of today",
            )

        from ics import Event

        path = _safe_calendar_path()
        cal = _load_calendar(path)

        event = Event()
        event.name = title
        event.begin = start_dt
        event.duration = timedelta(minutes=duration_minutes)
        event.uid = str(uuid.uuid4())
        if description:
            event.description = description
        if location:
            event.location = location

        cal.events.add(event)
        _save_calendar(cal, path)

        params = {"title": title, "start": start, "duration_minutes": duration_minutes}
        result_msg = (
            f"Event created: {title}\n"
            f"Start: {start_dt.strftime('%Y-%m-%d %H:%M')}\n"
            f"Duration: {duration_minutes} minutes\n"
            f"UID: {event.uid}"
        )
        if location:
            result_msg += f"\nLocation: {location}"

        _log_action("calendar", params, result_msg, True)
        return ToolResult(output=result_msg, success=True)

    def _list(self, days: int) -> ToolResult:
        if days < 1:
            days = 7
        if days > _MAX_DATE_RANGE_DAYS:
            days = _MAX_DATE_RANGE_DAYS

        path = _safe_calendar_path()
        cal = _load_calendar(path)

        now = datetime.now()
        cutoff = now + timedelta(days=days)

        upcoming = []
        for event in cal.events:
            event_start = event.begin.datetime if hasattr(event.begin, 'datetime') else event.begin
            # Make naive for comparison
            if hasattr(event_start, 'tzinfo') and event_start.tzinfo is not None:
                event_start = event_start.replace(tzinfo=None)
            if now <= event_start <= cutoff:
                upcoming.append(event)

        upcoming.sort(key=lambda e: e.begin)

        if not upcoming:
            return ToolResult(output=f"No events in the next {days} days.", success=True)

        lines = []
        for e in upcoming:
            event_start = e.begin.datetime if hasattr(e.begin, 'datetime') else e.begin
            if hasattr(event_start, 'tzinfo') and event_start.tzinfo is not None:
                event_start = event_start.replace(tzinfo=None)
            start_str = event_start.strftime("%Y-%m-%d %H:%M")
            dur = int(e.duration.total_seconds() / 60) if e.duration else "?"
            loc = f" @ {e.location}" if e.location else ""
            lines.append(f"- {start_str} | {e.name} ({dur}min){loc} [uid: {e.uid}]")

        return ToolResult(
            output=f"Upcoming events (next {days} days, {len(upcoming)} found):\n" + "\n".join(lines),
            success=True,
        )

    def _search(self, query: str) -> ToolResult:
        if not query:
            return ToolResult(output="", success=False, error="Search query is required")

        path = _safe_calendar_path()
        cal = _load_calendar(path)

        query_lower = query.lower()
        matches = []
        for event in cal.events:
            searchable = " ".join(filter(None, [
                event.name or "",
                event.description or "",
                event.location or "",
            ])).lower()
            if query_lower in searchable:
                matches.append(event)

        matches.sort(key=lambda e: e.begin)

        if not matches:
            return ToolResult(output=f"No events matching '{query}'.", success=True)

        lines = []
        for e in matches:
            event_start = e.begin.datetime if hasattr(e.begin, 'datetime') else e.begin
            if hasattr(event_start, 'tzinfo') and event_start.tzinfo is not None:
                event_start = event_start.replace(tzinfo=None)
            start_str = event_start.strftime("%Y-%m-%d %H:%M")
            lines.append(f"- {start_str} | {e.name} [uid: {e.uid}]")

        return ToolResult(
            output=f"Events matching '{query}' ({len(matches)} found):\n" + "\n".join(lines),
            success=True,
        )

    def _delete(self, uid: str) -> ToolResult:
        if not uid:
            return ToolResult(output="", success=False, error="Event UID is required for deletion")

        path = _safe_calendar_path()
        cal = _load_calendar(path)

        target = None
        for event in cal.events:
            if event.uid == uid:
                target = event
                break

        if not target:
            return ToolResult(output="", success=False, error=f"Event with UID '{uid}' not found")

        cal.events.remove(target)
        _save_calendar(cal, path)

        result_msg = f"Event '{target.name}' (uid: {uid}) deleted."
        _log_action("calendar", {"uid": uid}, result_msg, True)
        return ToolResult(output=result_msg, success=True)
