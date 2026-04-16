"""Tests for Phase 3 — Action Execution tools (email, calendar, reminder, webhook)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from app.database import SafeDB
from app.tools.base import ToolResult


# ===========================================================================
# Helpers
# ===========================================================================


def _fresh_db() -> SafeDB:
    """Create a fresh in-memory database with schema."""
    db = SafeDB(":memory:")
    db.init_schema()
    return db


# ===========================================================================
# EmailSendTool
# ===========================================================================


class TestEmailSendTool:
    """Tests for app.tools.action_email.EmailSendTool."""

    @pytest.fixture(autouse=True)
    def _reset_rate_limit(self):
        """Clear email rate limit between tests."""
        from app.tools.action_email import _email_timestamps
        _email_timestamps.clear()
        yield
        _email_timestamps.clear()

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        from app.tools.action_email import EmailSendTool
        tool = EmailSendTool()
        result = await tool.execute(to="test@example.com", subject="Hi", body="Hello")
        assert not result.success
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("to_,subject_,body_", [
        ("", "Hi", "Hello"),
        ("test@test.com", "", "Hello"),
        ("test@test.com", "Hi", ""),
    ], ids=["missing_to", "missing_subject", "missing_body"])
    async def test_missing_required_fields(self, to_, subject_, body_):
        from app.tools.action_email import EmailSendTool
        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = ""
            tool = EmailSendTool()
            result = await tool.execute(to=to_, subject=subject_, body=body_)
            assert not result.success
            assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_smtp_host(self):
        from app.tools.action_email import EmailSendTool
        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = ""
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = ""
            tool = EmailSendTool()
            result = await tool.execute(to="test@test.com", subject="Hi", body="Hello")
            assert not result.success
            assert "SMTP" in result.error

    @pytest.mark.asyncio
    async def test_recipient_allowlist_blocks(self):
        from app.tools.action_email import EmailSendTool
        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "allowed@test.com, admin@test.com"
            tool = EmailSendTool()
            result = await tool.execute(to="blocked@evil.com", subject="Hi", body="Hello")
            assert not result.success
            assert "not in the allowed" in result.error

    @pytest.mark.asyncio
    async def test_recipient_allowlist_allows(self):
        from app.tools.action_email import EmailSendTool
        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_SMTP_PORT = 587
            mock_cfg.EMAIL_SMTP_USER = "user"
            mock_cfg.EMAIL_SMTP_PASS = "pass"
            mock_cfg.EMAIL_SMTP_TLS = False
            mock_cfg.EMAIL_FROM = "nova@test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "allowed@test.com"
            mock_cfg.EMAIL_RATE_LIMIT = 20
            mock_cfg.DB_PATH = ":memory:"

            tool = EmailSendTool()
            with patch.object(tool, "_send_smtp", return_value="Email sent"):
                with patch("app.tools.action_logging.get_db") as mock_db:
                    mock_db.return_value = MagicMock()
                    result = await tool.execute(to="allowed@test.com", subject="Hi", body="Hello")
                    assert result.success
                    assert "Email sent" in result.output

    @pytest.mark.asyncio
    async def test_empty_allowlist_denies_all(self):
        """Empty allowlist = deny all recipients (security: no open relay)."""
        from app.tools.action_email import _is_recipient_allowed
        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = ""
            assert not _is_recipient_allowed("anyone@anywhere.com")

    @pytest.mark.asyncio
    async def test_rate_limit(self):
        from app.tools.action_email import EmailSendTool, _email_timestamps
        now = time.time()
        _email_timestamps.extend([now] * 20)  # default EMAIL_RATE_LIMIT

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "test@test.com"
            mock_cfg.EMAIL_RATE_LIMIT = 20
            mock_cfg.DB_PATH = ":memory:"

            tool = EmailSendTool()
            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()
                result = await tool.execute(to="test@test.com", subject="Hi", body="Hello")
                assert not result.success
                assert "rate limit" in result.error.lower()

    def test_allowlist_case_insensitive_and_whitespace(self):
        from app.tools.action_email import _is_recipient_allowed
        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = " Admin@Test.com , bob@test.com "
            assert _is_recipient_allowed("admin@test.com")
            assert _is_recipient_allowed("ADMIN@TEST.COM")
            assert _is_recipient_allowed("bob@test.com")
            assert not _is_recipient_allowed("charlie@test.com")


# ===========================================================================
# CalendarTool
# ===========================================================================


class TestCalendarTool:
    """Tests for app.tools.action_calendar.CalendarTool."""

    @pytest.fixture
    def tmp_calendar(self, tmp_path):
        """Provide a temp calendar path and patch _safe_calendar_path."""
        from pathlib import Path
        cal_path = tmp_path / "test_calendar.ics"
        with patch("app.tools.action_calendar._safe_calendar_path", return_value=cal_path):
            yield str(cal_path)

    @pytest.mark.asyncio
    async def test_disabled(self):
        from app.tools.action_calendar import CalendarTool
        with patch("app.tools.action_calendar.config") as mock_cfg:
            mock_cfg.ENABLE_CALENDAR = False
            tool = CalendarTool()
            result = await tool.execute(action="list")
            assert not result.success
            assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_action(self):
        from app.tools.action_calendar import CalendarTool
        with patch("app.tools.action_calendar.config") as mock_cfg:
            mock_cfg.ENABLE_CALENDAR = True
            tool = CalendarTool()
            result = await tool.execute()
            assert not result.success
            assert "action" in result.error.lower()

    @pytest.mark.asyncio
    async def test_create_validation_errors(self):
        from app.tools.action_calendar import CalendarTool
        with patch("app.tools.action_calendar.config") as mock_cfg:
            mock_cfg.ENABLE_CALENDAR = True
            tool = CalendarTool()
            # Missing title
            result = await tool.execute(action="create", start="2026-03-06T15:00:00")
            assert not result.success
            assert "title" in result.error.lower()
            # Missing start
            result = await tool.execute(action="create", title="Test Event")
            assert not result.success
            assert "start" in result.error.lower()
            # Invalid start
            result = await tool.execute(action="create", title="Test", start="not-a-date")
            assert not result.success
            assert "invalid" in result.error.lower()
            # Excessive duration
            result = await tool.execute(
                action="create", title="Test", start="2026-03-06T15:00:00",
                duration_minutes=2000,
            )
            assert not result.success
            assert "24 hours" in result.error

    @pytest.mark.asyncio
    async def test_create_and_list(self, tmp_calendar):
        from app.tools.action_calendar import CalendarTool
        with patch("app.tools.action_calendar.config") as mock_cfg:
            mock_cfg.ENABLE_CALENDAR = True
            mock_cfg.CALENDAR_PATH = tmp_calendar

            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()

                tool = CalendarTool()

                # Create an event in the future
                tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
                result = await tool.execute(
                    action="create",
                    title="Test Meeting",
                    start=tomorrow,
                    duration_minutes=60,
                    location="Office",
                )
                assert result.success
                assert "Test Meeting" in result.output
                assert "UID:" in result.output

                # List events
                result = await tool.execute(action="list", days=7)
                assert result.success
                assert "Test Meeting" in result.output

    @pytest.mark.asyncio
    async def test_create_and_delete(self, tmp_calendar):
        from app.tools.action_calendar import CalendarTool
        with patch("app.tools.action_calendar.config") as mock_cfg:
            mock_cfg.ENABLE_CALENDAR = True
            mock_cfg.CALENDAR_PATH = tmp_calendar

            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()

                tool = CalendarTool()

                tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
                result = await tool.execute(
                    action="create", title="Delete Me",
                    start=tomorrow, duration_minutes=30,
                )
                assert result.success
                uid = result.output.split("UID: ")[1].strip()

                # Delete
                result = await tool.execute(action="delete", uid=uid)
                assert result.success
                assert "deleted" in result.output.lower()

                # Verify gone
                result = await tool.execute(action="search", query="Delete Me")
                assert "No events" in result.output

    @pytest.mark.asyncio
    async def test_delete_not_found(self, tmp_path):
        from app.tools.action_calendar import CalendarTool
        cal = tmp_path / "delete_not_found.ics"
        with patch("app.tools.action_calendar.config") as mock_cfg:
            mock_cfg.ENABLE_CALENDAR = True
            mock_cfg.CALENDAR_PATH = str(cal)
            with patch("app.tools.action_calendar._safe_calendar_path", return_value=cal):
                tool = CalendarTool()
                result = await tool.execute(action="delete", uid="nonexistent-uid")
                assert not result.success
                assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_search_no_query(self):
        from app.tools.action_calendar import CalendarTool
        with patch("app.tools.action_calendar.config") as mock_cfg:
            mock_cfg.ENABLE_CALENDAR = True
            tool = CalendarTool()
            result = await tool.execute(action="search")
            assert not result.success
            assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_empty_calendar(self, tmp_path):
        from app.tools.action_calendar import CalendarTool
        empty_cal = tmp_path / "empty_calendar.ics"
        with patch("app.tools.action_calendar.config") as mock_cfg:
            mock_cfg.ENABLE_CALENDAR = True
            mock_cfg.CALENDAR_PATH = str(empty_cal)
            with patch("app.tools.action_calendar._safe_calendar_path", return_value=empty_cal):
                tool = CalendarTool()
                result = await tool.execute(action="list", days=7)
                assert result.success
                assert "No events" in result.output

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        from app.tools.action_calendar import CalendarTool
        with patch("app.tools.action_calendar.config") as mock_cfg:
            mock_cfg.ENABLE_CALENDAR = True
            tool = CalendarTool()
            result = await tool.execute(action="bogus")
            assert not result.success
            assert "unknown" in result.error.lower()


# ===========================================================================
# ReminderTool
# ===========================================================================


class TestReminderTool:
    """Tests for app.tools.action_reminder.ReminderTool."""

    @pytest.fixture
    def store(self):
        """Create a MonitorStore with fresh DB."""
        from app.monitors.heartbeat import MonitorStore
        db = _fresh_db()
        return MonitorStore(db)

    @pytest.mark.asyncio
    async def test_no_store(self):
        from app.tools.action_reminder import ReminderTool
        tool = ReminderTool(monitor_store=None)
        result = await tool.execute(action="set", name="Test", time="in 1 hour")
        assert not result.success
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_action(self):
        from app.tools.action_reminder import ReminderTool
        tool = ReminderTool(monitor_store=MagicMock())
        result = await tool.execute()
        assert not result.success
        assert "action" in result.error.lower()

    @pytest.mark.asyncio
    async def test_set_validation_errors(self, store):
        from app.tools.action_reminder import ReminderTool
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            tool = ReminderTool(monitor_store=store)
            # Missing name
            result = await tool.execute(action="set", time="in 1 hour")
            assert not result.success
            assert "name" in result.error.lower()
            # Missing time
            result = await tool.execute(action="set", name="Test")
            assert not result.success
            assert "time" in result.error.lower()
            # Invalid time
            result = await tool.execute(action="set", name="Test", time="blah blah")
            assert not result.success
            assert "could not parse" in result.error.lower()
            # Past time
            result = await tool.execute(action="set", name="Test", time="2020-01-01T00:00:00")
            assert not result.success
            assert "future" in result.error.lower()

    @pytest.mark.asyncio
    async def test_set_relative_time(self, store):
        from app.tools.action_reminder import ReminderTool
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            tool = ReminderTool(monitor_store=store)
            result = await tool.execute(
                action="set", name="Check oven",
                time="in 2 hours", message="Time to check!",
            )
            assert result.success
            assert "Check oven" in result.output
            assert "Time to check!" in result.output

    @pytest.mark.asyncio
    async def test_set_and_list(self, store):
        from app.tools.action_reminder import ReminderTool
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            tool = ReminderTool(monitor_store=store)

            await tool.execute(
                action="set", name="Test Reminder",
                time="in 1 hour", message="Test message",
            )

            result = await tool.execute(action="list")
            assert result.success
            assert "Test Reminder" in result.output

    @pytest.mark.asyncio
    async def test_set_and_cancel(self, store):
        from app.tools.action_reminder import ReminderTool
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            tool = ReminderTool(monitor_store=store)

            await tool.execute(
                action="set", name="Cancel Me",
                time="in 1 hour",
            )

            result = await tool.execute(action="cancel", name="Cancel Me")
            assert result.success
            assert "cancelled" in result.output.lower()

            # Verify gone
            result = await tool.execute(action="list")
            assert "No pending" in result.output

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, store):
        from app.tools.action_reminder import ReminderTool
        with patch("app.tools.action_logging.get_db") as mock_db:
            mock_db.return_value = MagicMock()
            tool = ReminderTool(monitor_store=store)
            result = await tool.execute(action="cancel", name="Nonexistent")
            assert not result.success
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self, store):
        from app.tools.action_reminder import ReminderTool
        tool = ReminderTool(monitor_store=store)
        result = await tool.execute(action="bogus")
        assert not result.success
        assert "unknown" in result.error.lower()


# ===========================================================================
# Time Parsing (consolidated)
# ===========================================================================


class TestTimeParsing:
    """Test the natural time parser in ReminderTool."""

    def test_relative_units(self):
        """Relative time with minutes, hours, days all work."""
        from app.tools.action_reminder import _parse_time
        result = _parse_time("in 30 minutes")
        assert result is not None
        delta = (result - datetime.now()).total_seconds()
        assert 1700 < delta < 1900

        result = _parse_time("in 2 hours")
        assert result is not None
        delta = (result - datetime.now()).total_seconds()
        assert 7100 < delta < 7300

        result = _parse_time("in 3 days")
        assert result is not None
        delta = (result - datetime.now()).total_seconds()
        assert 250000 < delta < 260000

    def test_tomorrow_at(self):
        from app.tools.action_reminder import _parse_time
        result = _parse_time("tomorrow at 9am")
        assert result is not None
        expected = (datetime.now() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        assert abs((result - expected).total_seconds()) < 2

    def test_iso8601(self):
        from app.tools.action_reminder import _parse_time
        result = _parse_time("2026-06-15T14:30:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 6
        assert result.hour == 14

    def test_unparseable(self):
        from app.tools.action_reminder import _parse_time
        result = _parse_time("whenever you feel like it")
        assert result is None


# ===========================================================================
# WebhookTool
# ===========================================================================


class TestWebhookTool:
    """Tests for app.tools.action_webhook.WebhookTool."""

    @pytest.fixture(autouse=True)
    def _reset_rate_limit(self):
        """Clear webhook rate limit between tests."""
        from app.tools.action_webhook import _webhook_timestamps
        _webhook_timestamps.clear()
        yield
        _webhook_timestamps.clear()

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        from app.tools.action_webhook import WebhookTool
        tool = WebhookTool()
        result = await tool.execute(action="call", url="https://example.com/hook")
        assert not result.success
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_url(self):
        from app.tools.action_webhook import WebhookTool
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            tool = WebhookTool()
            result = await tool.execute(action="call", url="")
            assert not result.success
            assert "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_method(self):
        from app.tools.action_webhook import WebhookTool
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://example.com"
            tool = WebhookTool()
            result = await tool.execute(action="call", url="https://example.com/hook", method="TRACE")
            assert not result.success
            assert "not allowed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_url_not_in_allowlist(self):
        from app.tools.action_webhook import WebhookTool
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://myserver.com"
            mock_cfg.DB_PATH = ":memory:"

            tool = WebhookTool()
            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()
                result = await tool.execute(action="call", url="https://evil.com/hook")
                assert not result.success
                assert "not in the allowed" in result.error

    @pytest.mark.asyncio
    async def test_ssrf_protection(self):
        from app.tools.action_webhook import WebhookTool
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "http://127.0.0.1:8080"
            mock_cfg.DB_PATH = ":memory:"

            tool = WebhookTool()
            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()
                result = await tool.execute(action="call", url="http://127.0.0.1:8080/secret")
                assert not result.success
                assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rate_limit(self):
        from app.tools.action_webhook import WebhookTool, _webhook_timestamps, _WEBHOOK_RATE_LIMIT
        now = time.time()
        _webhook_timestamps.extend([now] * _WEBHOOK_RATE_LIMIT)

        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://example.com"
            mock_cfg.DB_PATH = ":memory:"

            tool = WebhookTool()
            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()
                result = await tool.execute(action="call", url="https://example.com/hook")
                assert not result.success
                assert "rate limit" in result.error.lower()

    @pytest.mark.asyncio
    async def test_successful_call(self):
        from app.tools.action_webhook import WebhookTool
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.ENABLE_WEBHOOKS = True
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://myserver.com"
            mock_cfg.DB_PATH = ":memory:"

            tool = WebhookTool()
            with patch.object(tool, "_call", new_callable=AsyncMock, return_value="HTTP 200 OK\n{\"status\": \"ok\"}"):
                with patch("app.tools.action_logging.get_db") as mock_db:
                    mock_db.return_value = MagicMock()
                    result = await tool.execute(
                        action="call",
                        url="https://myserver.com/deploy",
                        method="POST",
                        body={"action": "deploy"},
                    )
                    assert result.success
                    assert "200" in result.output

    def test_url_prefix_matching(self):
        from app.tools.action_webhook import _is_url_allowed
        with patch("app.tools.action_webhook.config") as mock_cfg:
            mock_cfg.WEBHOOK_ALLOWED_URLS = "https://api.myserver.com, https://hooks.slack.com"
            assert _is_url_allowed("https://api.myserver.com/deploy")
            assert _is_url_allowed("https://hooks.slack.com/services/T00/B00/XXX")
            assert not _is_url_allowed("https://evil.com")
            assert not _is_url_allowed("https://myserver.com")  # prefix doesn't match


# ===========================================================================
# Action Log
# ===========================================================================


class TestActionLog:
    """Tests for action logging helper."""

    def test_log_action(self):
        from app.tools.action_logging import log_action
        db = _fresh_db()
        with patch("app.tools.action_logging.get_db", return_value=db):
            log_action("email", {"to": "test@test.com"}, "Email sent", True)

        row = db.fetchone("SELECT * FROM action_log WHERE action_type = 'email'")
        assert row is not None
        assert row["success"] == 1
        assert "test@test.com" in row["params"]

    def test_log_action_failure(self):
        from app.tools.action_logging import log_action
        db = _fresh_db()
        with patch("app.tools.action_logging.get_db", return_value=db):
            log_action("webhook", {"url": "https://evil.com"}, "blocked", False)

        row = db.fetchone("SELECT * FROM action_log WHERE action_type = 'webhook'")
        assert row is not None
        assert row["success"] == 0


# ===========================================================================
# Actions API
# ===========================================================================


class TestActionsAPI:
    """Tests for the actions API router."""

    @pytest.fixture
    def db(self):
        return _fresh_db()

    def test_list_actions_empty(self, db):
        with patch("app.api.actions.get_db", return_value=db):
            import asyncio
            from app.api.actions import list_actions
            result = asyncio.get_event_loop().run_until_complete(list_actions())
            assert result["count"] == 0
            assert result["actions"] == []

    def test_list_actions_filtered(self, db):
        db.execute(
            "INSERT INTO action_log (action_type, params, result, success) VALUES (?, ?, ?, ?)",
            ("email", '{}', "sent", 1),
        )
        db.execute(
            "INSERT INTO action_log (action_type, params, result, success) VALUES (?, ?, ?, ?)",
            ("webhook", '{}', "called", 1),
        )
        with patch("app.api.actions.get_db", return_value=db):
            import asyncio
            from app.api.actions import list_actions
            result = asyncio.get_event_loop().run_until_complete(list_actions(action_type="email"))
            assert result["count"] == 1
            assert result["actions"][0]["action_type"] == "email"

    def test_get_action_not_found(self, db):
        with patch("app.api.actions.get_db", return_value=db):
            import asyncio
            from app.api.actions import get_action
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(get_action(9999))
            assert exc_info.value.status_code == 404


# ===========================================================================
# Rate Limit Helper
# ===========================================================================


class TestRateLimit:
    def test_under_limit(self):
        from app.tools.action_email import _check_rate_limit
        timestamps = []
        assert _check_rate_limit(timestamps, 10, 3600) is True

    def test_at_limit(self):
        from app.tools.action_email import _check_rate_limit
        now = time.time()
        timestamps = [now] * 10
        assert _check_rate_limit(timestamps, 10, 3600) is False

    def test_old_entries_pruned(self):
        from app.tools.action_email import _check_rate_limit
        old = time.time() - 7200  # 2 hours ago
        timestamps = [old] * 10
        assert _check_rate_limit(timestamps, 10, 3600) is True
        assert len(timestamps) == 0  # Old entries pruned
