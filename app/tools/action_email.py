"""Email send tool — SMTP email sending with security controls.

Config-gated (ENABLE_EMAIL_SEND=false by default), rate-limited,
allowlist-protected. Uses built-in smtplib — zero new dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import smtplib
import time
from email.message import EmailMessage

from app.config import config
from app.database import get_db
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Rate limiting: max 20 emails per hour
_EMAIL_RATE_LIMIT = 20
_EMAIL_RATE_WINDOW = 3600  # seconds
_email_timestamps: list[float] = []


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


def _check_rate_limit(timestamps: list[float], max_count: int, window: int) -> bool:
    """Return True if under the rate limit. Prunes old entries."""
    now = time.time()
    cutoff = now - window
    timestamps[:] = [t for t in timestamps if t > cutoff]
    return len(timestamps) < max_count


def _is_recipient_allowed(to: str) -> bool:
    """Check if recipient is on the allowlist (empty = deny all)."""
    allowlist = config.EMAIL_ALLOWED_RECIPIENTS.strip()
    if not allowlist:
        return False
    allowed = {addr.strip().lower() for addr in allowlist.split(",") if addr.strip()}
    return to.strip().lower() in allowed


class EmailSendTool(BaseTool):
    name = "email_send"
    description = (
        "Send an email via SMTP. Requires ENABLE_EMAIL_SEND=true and SMTP credentials configured. "
        "Subject is auto-prefixed with [Nova]."
    )
    parameters = "to: str, subject: str, body: str, html: bool (optional, default false)"

    async def execute(
        self,
        *,
        to: str = "",
        subject: str = "",
        body: str = "",
        html: bool = False,
        **kwargs,
    ) -> ToolResult:
        if not config.ENABLE_EMAIL_SEND:
            return ToolResult(
                output="", success=False,
                error="Email sending is disabled. Set ENABLE_EMAIL_SEND=true and configure SMTP credentials.",
            )

        if not to:
            return ToolResult(output="", success=False, error="Recipient (to) is required")
        if not subject:
            return ToolResult(output="", success=False, error="Subject is required")
        if not body:
            return ToolResult(output="", success=False, error="Body is required")

        if not config.EMAIL_SMTP_HOST:
            return ToolResult(output="", success=False, error="EMAIL_SMTP_HOST not configured")

        # Allowlist check
        if not _is_recipient_allowed(to):
            _log_action("email", {"to": to, "subject": subject}, "blocked: recipient not in allowlist", False)
            return ToolResult(
                output="", success=False,
                error=f"Recipient '{to}' is not in the allowed recipients list",
            )

        # Rate limit
        if not _check_rate_limit(_email_timestamps, _EMAIL_RATE_LIMIT, _EMAIL_RATE_WINDOW):
            _log_action("email", {"to": to, "subject": subject}, "blocked: rate limit exceeded", False)
            return ToolResult(
                output="", success=False,
                error=f"Email rate limit exceeded ({_EMAIL_RATE_LIMIT}/hour). Try again later.",
            )

        # Build message
        msg = EmailMessage()
        msg["From"] = config.EMAIL_FROM or config.EMAIL_SMTP_USER
        msg["To"] = to
        msg["Subject"] = f"[Nova] {subject}"

        if html:
            msg.set_content(body, subtype="html")
        else:
            msg.set_content(body)

        # Send via SMTP in a thread to avoid blocking
        params = {"to": to, "subject": subject, "html": html}
        try:
            result = await asyncio.to_thread(self._send_smtp, msg)
            _email_timestamps.append(time.time())
            _log_action("email", params, result, True)
            return ToolResult(output=result, success=True)
        except Exception as e:
            error_msg = f"SMTP send failed: {e}"
            _log_action("email", params, error_msg, False)
            return ToolResult(output="", success=False, error=error_msg)

    @staticmethod
    def _send_smtp(msg: EmailMessage) -> str:
        """Synchronous SMTP send (runs in thread)."""
        host = config.EMAIL_SMTP_HOST
        port = config.EMAIL_SMTP_PORT
        user = config.EMAIL_SMTP_USER
        password = config.EMAIL_SMTP_PASS
        use_tls = config.EMAIL_SMTP_TLS

        if use_tls:
            server = smtplib.SMTP(host, port, timeout=15)
            server.starttls()
        else:
            server = smtplib.SMTP(host, port, timeout=15)

        try:
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        finally:
            server.quit()

        return f"Email sent to {msg['To']} with subject '{msg['Subject']}'"
