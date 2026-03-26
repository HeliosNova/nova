"""Tests for EmailSendTool — allowlist, rate limiting, SMTP, HTML, edge cases."""

from __future__ import annotations

import time
from email.message import EmailMessage
from unittest.mock import patch, MagicMock

import pytest

from app.tools.base import ToolResult, ErrorCategory


class TestEmailAllowlist:
    """Allowlist enforcement for email recipients."""

    def test_allowed_recipient_passes(self):
        from app.tools.action_email import _is_recipient_allowed

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "alice@example.com, bob@example.com"
            assert _is_recipient_allowed("alice@example.com") is True

    def test_disallowed_recipient_rejected(self):
        from app.tools.action_email import _is_recipient_allowed

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "alice@example.com, bob@example.com"
            assert _is_recipient_allowed("mallory@evil.com") is False

    def test_case_insensitive_matching(self):
        from app.tools.action_email import _is_recipient_allowed

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "Alice@Example.COM"
            assert _is_recipient_allowed("alice@example.com") is True
            assert _is_recipient_allowed("ALICE@EXAMPLE.COM") is True

    def test_whitespace_trimming(self):
        from app.tools.action_email import _is_recipient_allowed

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "  alice@example.com ,  bob@test.com  "
            assert _is_recipient_allowed("alice@example.com") is True
            assert _is_recipient_allowed("bob@test.com") is True


class TestEmailDenyAllOnEmptyAllowlist:
    """Empty allowlist must deny all recipients (no open relay)."""

    def test_empty_string_denies(self):
        from app.tools.action_email import _is_recipient_allowed

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = ""
            assert _is_recipient_allowed("anyone@anywhere.com") is False

    def test_whitespace_only_denies(self):
        from app.tools.action_email import _is_recipient_allowed

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "   "
            assert _is_recipient_allowed("anyone@anywhere.com") is False


class TestEmailRateLimiting:
    """Rate limiting prevents too many emails in a short period."""

    @pytest.fixture(autouse=True)
    def _reset_rate_limit(self):
        from app.tools.action_email import _email_timestamps
        _email_timestamps.clear()
        yield
        _email_timestamps.clear()

    def test_under_limit_allows(self):
        from app.tools.action_email import _check_rate_limit
        timestamps = [time.time()] * 5
        assert _check_rate_limit(timestamps, 20, 3600) is True

    def test_at_limit_rejects(self):
        from app.tools.action_email import _check_rate_limit
        now = time.time()
        timestamps = [now] * 20
        assert _check_rate_limit(timestamps, 20, 3600) is False

    def test_old_entries_pruned(self):
        from app.tools.action_email import _check_rate_limit
        old = time.time() - 7200  # 2 hours ago, outside window
        timestamps = [old] * 20
        assert _check_rate_limit(timestamps, 20, 3600) is True
        assert len(timestamps) == 0

    async def test_rate_limit_integration(self):
        """Full integration: fill timestamps to capacity, then verify execute rejects."""
        from app.tools.action_email import EmailSendTool, _email_timestamps
        now = time.time()
        _email_timestamps.extend([now] * 20)  # default EMAIL_RATE_LIMIT

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "user@test.com"
            mock_cfg.EMAIL_RATE_LIMIT = 20

            tool = EmailSendTool()
            with patch("app.tools.action_logging.get_db") as mock_db:
                mock_db.return_value = MagicMock()
                result = await tool.execute(to="user@test.com", subject="Hi", body="Hello")
                assert not result.success
                assert "rate limit" in result.error.lower()
                assert result.error_category == ErrorCategory.PERMISSION


class TestEmailSMTPMock:
    """Mock the SMTP connection and verify email is built and sent correctly."""

    @pytest.fixture(autouse=True)
    def _reset_rate_limit(self):
        from app.tools.action_email import _email_timestamps
        _email_timestamps.clear()
        yield
        _email_timestamps.clear()

    async def test_smtp_send_plain_text(self):
        from app.tools.action_email import EmailSendTool

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_SMTP_PORT = 587
            mock_cfg.EMAIL_SMTP_USER = "user@test.com"
            mock_cfg.EMAIL_SMTP_PASS = "secret"
            mock_cfg.EMAIL_SMTP_TLS = True
            mock_cfg.EMAIL_FROM = "nova@test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "recipient@test.com"
            mock_cfg.EMAIL_RATE_LIMIT = 20

            tool = EmailSendTool()

            # Mock _send_smtp to capture the EmailMessage
            captured_msgs = []

            def fake_send_smtp(msg: EmailMessage) -> str:
                captured_msgs.append(msg)
                return f"Email sent to {msg['To']} with subject '{msg['Subject']}'"

            with patch.object(tool, "_send_smtp", side_effect=fake_send_smtp):
                with patch("app.tools.action_logging.get_db") as mock_db:
                    mock_db.return_value = MagicMock()
                    result = await tool.execute(
                        to="recipient@test.com",
                        subject="Test Subject",
                        body="Hello World",
                    )

            assert result.success
            assert "Email sent" in result.output
            assert len(captured_msgs) == 1
            msg = captured_msgs[0]
            assert msg["To"] == "recipient@test.com"
            assert msg["From"] == "nova@test.com"
            assert msg["Subject"] == "[Nova] Test Subject"
            assert "Hello World" in msg.get_content()


class TestEmailHTMLMode:
    """HTML email generation."""

    @pytest.fixture(autouse=True)
    def _reset_rate_limit(self):
        from app.tools.action_email import _email_timestamps
        _email_timestamps.clear()
        yield
        _email_timestamps.clear()

    async def test_html_email_content_type(self):
        from app.tools.action_email import EmailSendTool

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_SMTP_PORT = 587
            mock_cfg.EMAIL_SMTP_USER = "user@test.com"
            mock_cfg.EMAIL_SMTP_PASS = "secret"
            mock_cfg.EMAIL_SMTP_TLS = True
            mock_cfg.EMAIL_FROM = "nova@test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "recipient@test.com"
            mock_cfg.EMAIL_RATE_LIMIT = 20

            tool = EmailSendTool()
            captured_msgs = []

            def fake_send_smtp(msg: EmailMessage) -> str:
                captured_msgs.append(msg)
                return f"Email sent to {msg['To']} with subject '{msg['Subject']}'"

            with patch.object(tool, "_send_smtp", side_effect=fake_send_smtp):
                with patch("app.tools.action_logging.get_db") as mock_db:
                    mock_db.return_value = MagicMock()
                    result = await tool.execute(
                        to="recipient@test.com",
                        subject="HTML Test",
                        body="<h1>Hello</h1><p>World</p>",
                        html=True,
                    )

            assert result.success
            msg = captured_msgs[0]
            assert msg.get_content_type() == "text/html"
            assert "<h1>Hello</h1>" in msg.get_content()

    async def test_plain_text_default(self):
        from app.tools.action_email import EmailSendTool

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_SMTP_PORT = 587
            mock_cfg.EMAIL_SMTP_USER = "user@test.com"
            mock_cfg.EMAIL_SMTP_PASS = "secret"
            mock_cfg.EMAIL_SMTP_TLS = True
            mock_cfg.EMAIL_FROM = "nova@test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "recipient@test.com"
            mock_cfg.EMAIL_RATE_LIMIT = 20

            tool = EmailSendTool()
            captured_msgs = []

            def fake_send_smtp(msg: EmailMessage) -> str:
                captured_msgs.append(msg)
                return f"Email sent to {msg['To']} with subject '{msg['Subject']}'"

            with patch.object(tool, "_send_smtp", side_effect=fake_send_smtp):
                with patch("app.tools.action_logging.get_db") as mock_db:
                    mock_db.return_value = MagicMock()
                    result = await tool.execute(
                        to="recipient@test.com",
                        subject="Plain Test",
                        body="Just plain text",
                    )

            assert result.success
            msg = captured_msgs[0]
            assert msg.get_content_type() == "text/plain"


class TestEmailEdgeCases:
    """Edge cases: missing fields, disabled feature, SMTP errors."""

    async def test_disabled_by_default(self):
        from app.tools.action_email import EmailSendTool

        tool = EmailSendTool()
        result = await tool.execute(to="test@example.com", subject="Hi", body="Hello")
        assert not result.success
        assert "disabled" in result.error.lower()
        assert result.error_category == ErrorCategory.PERMISSION

    async def test_missing_subject_rejected(self):
        from app.tools.action_email import EmailSendTool

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            tool = EmailSendTool()
            result = await tool.execute(to="test@test.com", subject="", body="Hello")
            assert not result.success
            assert "subject" in result.error.lower()
            assert result.error_category == ErrorCategory.VALIDATION

    async def test_missing_body_rejected(self):
        from app.tools.action_email import EmailSendTool

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            tool = EmailSendTool()
            result = await tool.execute(to="test@test.com", subject="Hi", body="")
            assert not result.success
            assert "body" in result.error.lower()
            assert result.error_category == ErrorCategory.VALIDATION

    async def test_missing_to_rejected(self):
        from app.tools.action_email import EmailSendTool

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            tool = EmailSendTool()
            result = await tool.execute(to="", subject="Hi", body="Hello")
            assert not result.success
            assert "required" in result.error.lower()
            assert result.error_category == ErrorCategory.VALIDATION

    async def test_missing_smtp_host_rejected(self):
        from app.tools.action_email import EmailSendTool

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = ""
            tool = EmailSendTool()
            result = await tool.execute(to="test@test.com", subject="Hi", body="Hello")
            assert not result.success
            assert "SMTP" in result.error

    @pytest.fixture(autouse=True)
    def _reset_rate_limit(self):
        from app.tools.action_email import _email_timestamps
        _email_timestamps.clear()
        yield
        _email_timestamps.clear()

    async def test_smtp_failure_returns_transient_error(self):
        """SMTP connection failure results in a TRANSIENT error category."""
        from app.tools.action_email import EmailSendTool

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_SMTP_PORT = 587
            mock_cfg.EMAIL_SMTP_USER = "user@test.com"
            mock_cfg.EMAIL_SMTP_PASS = "secret"
            mock_cfg.EMAIL_SMTP_TLS = True
            mock_cfg.EMAIL_FROM = "nova@test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "recipient@test.com"
            mock_cfg.EMAIL_RATE_LIMIT = 20

            tool = EmailSendTool()
            with patch.object(tool, "_send_smtp", side_effect=ConnectionRefusedError("Connection refused")):
                with patch("app.tools.action_logging.get_db") as mock_db:
                    mock_db.return_value = MagicMock()
                    result = await tool.execute(
                        to="recipient@test.com",
                        subject="Test",
                        body="Body",
                    )

            assert not result.success
            assert "SMTP send failed" in result.error
            assert result.error_category == ErrorCategory.TRANSIENT

    async def test_subject_auto_prefixed_with_nova(self):
        """Subject line is automatically prefixed with [Nova]."""
        from app.tools.action_email import EmailSendTool

        with patch("app.tools.action_email.config") as mock_cfg:
            mock_cfg.ENABLE_EMAIL_SEND = True
            mock_cfg.EMAIL_SMTP_HOST = "smtp.test.com"
            mock_cfg.EMAIL_SMTP_PORT = 587
            mock_cfg.EMAIL_SMTP_USER = "user@test.com"
            mock_cfg.EMAIL_SMTP_PASS = "secret"
            mock_cfg.EMAIL_SMTP_TLS = True
            mock_cfg.EMAIL_FROM = "nova@test.com"
            mock_cfg.EMAIL_ALLOWED_RECIPIENTS = "recipient@test.com"
            mock_cfg.EMAIL_RATE_LIMIT = 20

            tool = EmailSendTool()
            captured_msgs = []

            def fake_send_smtp(msg: EmailMessage) -> str:
                captured_msgs.append(msg)
                return f"Email sent to {msg['To']} with subject '{msg['Subject']}'"

            with patch.object(tool, "_send_smtp", side_effect=fake_send_smtp):
                with patch("app.tools.action_logging.get_db") as mock_db:
                    mock_db.return_value = MagicMock()
                    await tool.execute(
                        to="recipient@test.com",
                        subject="Important Update",
                        body="Some content",
                    )

            msg = captured_msgs[0]
            assert msg["Subject"] == "[Nova] Important Update"
