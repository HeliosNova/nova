"""Tests for action_reminder — time parsing, monitor delegation, cancel, edge cases."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.tools.action_reminder import ReminderTool, _parse_time


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fresh_db():
    from app.database import SafeDB
    db = SafeDB(":memory:")
    db.init_schema()
    return db


def _make_store():
    from app.monitors.heartbeat import MonitorStore
    return MonitorStore(_fresh_db())


# ===========================================================================
# Relative time parsing
# ===========================================================================


class TestRelativeTimeParsing:
    """Test 'in N unit' patterns including fractional values."""

    def test_in_30_minutes(self):
        result = _parse_time("in 30 minutes")
        assert result is not None
        delta = (result - datetime.now()).total_seconds()
        assert 1750 < delta < 1850  # ~1800s

    def test_in_2_hours(self):
        result = _parse_time("in 2 hours")
        assert result is not None
        delta = (result - datetime.now()).total_seconds()
        assert 7100 < delta < 7300  # ~7200s

    def test_in_1_5_hours_fractional(self):
        """Fractional hours like 1.5 should produce 5400 seconds."""
        result = _parse_time("in 1.5 hours")
        assert result is not None
        delta = (result - datetime.now()).total_seconds()
        assert 5300 < delta < 5500  # ~5400s

    def test_in_1_day(self):
        result = _parse_time("in 1 day")
        assert result is not None
        delta = (result - datetime.now()).total_seconds()
        assert 86300 < delta < 86500


# ===========================================================================
# Absolute time parsing
# ===========================================================================


class TestAbsoluteTimeParsing:
    """Test ISO8601 and natural absolute time strings."""

    def test_iso8601(self):
        result = _parse_time("2028-06-15T14:30:00")
        assert result is not None
        assert result.year == 2028
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30

    def test_tomorrow_at_9am(self):
        result = _parse_time("tomorrow at 9am")
        assert result is not None
        expected = (datetime.now() + timedelta(days=1)).replace(
            hour=9, minute=0, second=0, microsecond=0
        )
        assert abs((result - expected).total_seconds()) < 2

    def test_tomorrow_at_time_with_minutes(self):
        result = _parse_time("tomorrow at 10:30 pm")
        assert result is not None
        expected = (datetime.now() + timedelta(days=1)).replace(
            hour=22, minute=30, second=0, microsecond=0
        )
        assert abs((result - expected).total_seconds()) < 2


# ===========================================================================
# "next [weekday]" parsing
# ===========================================================================


class TestNextWeekdayParsing:
    """Test 'next monday', 'next friday' etc. — Phase 3 addition."""

    def test_next_monday(self):
        result = _parse_time("next monday")
        assert result is not None
        assert result.weekday() == 0  # Monday
        # Should always be in the future
        assert result > datetime.now()
        # Default time is 9:00 AM
        assert result.hour == 9
        assert result.minute == 0

    def test_next_friday(self):
        result = _parse_time("next friday")
        assert result is not None
        assert result.weekday() == 4  # Friday
        assert result > datetime.now()
        assert result.hour == 9

    def test_next_weekday_always_future(self):
        """Even if today is the target weekday, 'next X' should go to next week."""
        now = datetime.now()
        weekday_names = ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"]
        today_name = weekday_names[now.weekday()]

        result = _parse_time(f"next {today_name}")
        assert result is not None
        # Must be at least 1 day in the future (goes to next week)
        delta = (result - now).total_seconds()
        assert delta > 86000  # At least ~1 day ahead

    def test_next_weekday_case_insensitive(self):
        result = _parse_time("next WEDNESDAY")
        assert result is not None
        assert result.weekday() == 2


# ===========================================================================
# Monitor delegation
# ===========================================================================


class TestMonitorDelegation:
    """Test that setting a reminder creates a monitor with correct config."""

    @pytest.mark.asyncio
    async def test_set_creates_monitor(self):
        store = _make_store()
        tool = ReminderTool(monitor_store=store)
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            result = await tool.execute(
                action="set", name="Doctor Appt",
                time="in 2 hours", message="Time for your appointment",
            )
        assert result.success
        assert "Doctor Appt" in result.output

        # Verify monitor was created with reminder: prefix
        monitors = store.list_all()
        reminder_monitors = [m for m in monitors if m.name.startswith("reminder:")]
        assert len(reminder_monitors) == 1
        m = reminder_monitors[0]
        assert m.name == "reminder:Doctor Appt"
        assert m.check_type == "query"
        assert m.notify_condition == "always"
        assert m.check_config["query"] == "Time for your appointment"

    @pytest.mark.asyncio
    async def test_set_default_message(self):
        """When no custom message is provided, default is 'Reminder: <name>'."""
        store = _make_store()
        tool = ReminderTool(monitor_store=store)
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            result = await tool.execute(
                action="set", name="Standup",
                time="in 30 minutes",
            )
        assert result.success
        assert "Reminder: Standup" in result.output


# ===========================================================================
# Cancel action
# ===========================================================================


class TestCancelAction:
    @pytest.mark.asyncio
    async def test_cancel_existing_reminder(self):
        store = _make_store()
        tool = ReminderTool(monitor_store=store)
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            await tool.execute(action="set", name="CancelMe", time="in 1 hour")
            result = await tool.execute(action="cancel", name="CancelMe")
        assert result.success
        assert "cancelled" in result.output.lower()

        # Verify it's gone from the list
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            result = await tool.execute(action="list")
        assert "No pending" in result.output

    @pytest.mark.asyncio
    async def test_cancel_missing_name(self):
        store = _make_store()
        tool = ReminderTool(monitor_store=store)
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            result = await tool.execute(action="cancel", name="")
        assert not result.success
        assert "required" in result.error.lower()


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_past_time_returns_parsed_but_set_rejects(self):
        """_parse_time returns a datetime even if past; _set rejects it."""
        result = _parse_time("2020-01-01T00:00:00")
        assert result is not None  # Parser returns it
        assert result < datetime.now()  # But it's in the past

    @pytest.mark.asyncio
    async def test_set_rejects_past_time(self):
        store = _make_store()
        tool = ReminderTool(monitor_store=store)
        result = await tool.execute(
            action="set", name="Past", time="2020-01-01T00:00:00",
        )
        assert not result.success
        assert "future" in result.error.lower()

    def test_invalid_format_returns_none(self):
        assert _parse_time("whenever you feel like it") is None
        assert _parse_time("") is None
        assert _parse_time("    ") is None

    @pytest.mark.asyncio
    async def test_far_future_rejected(self):
        """Reminders more than 2 years out are rejected."""
        store = _make_store()
        tool = ReminderTool(monitor_store=store)
        far_future = (datetime.now() + timedelta(days=800)).strftime("%Y-%m-%dT%H:%M:%S")
        result = await tool.execute(
            action="set", name="TooFar", time=far_future,
        )
        assert not result.success
        assert "2 years" in result.error

    def test_fractional_seconds(self):
        """Fractional amounts like 0.5 minutes = 30 seconds."""
        result = _parse_time("in 0.5 minutes")
        assert result is not None
        delta = (result - datetime.now()).total_seconds()
        assert 25 < delta < 35  # ~30s
