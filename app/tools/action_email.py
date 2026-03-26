"""Email send tool — SMTP email sending with security controls.

Config-gated (ENABLE_EMAIL_SEND=false by default), rate-limited,
allowlist-protected. Uses built-in smtplib — zero new dependencies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import smtplib
import ssl
import time
from email.message import EmailMessage

from app.config import config
from app.tools.action_logging import log_action as _log_action
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

# Rate limiting: configurable via config.EMAIL_RATE_LIMIT (default 20/hour)
_EMAIL_RATE_WINDOW = 3600  # seconds
_email_timestamps: list[float] = []
_email_rate_lock = asyncio.Lock()


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
        "Send emails via SMTP. Subject is auto-prefixed with [Nova]. "
        "Requires ENABLE_EMAIL_SEND=true and SMTP credentials. "
        "Recipients must be on the EMAIL_ALLOWED_RECIPIENTS allowlist. "
        "Rate limited to 20 emails/hour. Supports plain text and HTML body. "
        "Do NOT use for webhook notifications (use webhook tool)."
    )
    parameters = "to: str, subject: str, body: str, html: bool (optional, default false)"
    input_schema = {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient email address. Must be on the allowed recipients list.",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line. Will be prefixed with [Nova].",
            },
            "body": {
                "type": "string",
                "description": "Email body content (plain text or HTML).",
            },
            "html": {
                "type": "boolean",
                "description": "If true, send body as HTML. Defaults to false.",
            },
        },
        "required": ["to", "subject", "body"],
    }

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
                error_category=ErrorCategory.PERMISSION,
            )

        if not to:
            return ToolResult(output="", success=False, error="Recipient (to) is required", error_category=ErrorCategory.VALIDATION)
        if not subject:
            return ToolResult(output="", success=False, error="Subject is required", error_category=ErrorCategory.VALIDATION)
        if not body:
            return ToolResult(output="", success=False, error="Body is required", error_category=ErrorCategory.VALIDATION)

        if not config.EMAIL_SMTP_HOST:
            return ToolResult(output="", success=False, error="EMAIL_SMTP_HOST not configured", error_category=ErrorCategory.PERMISSION)

        # Allowlist check
        if not _is_recipient_allowed(to):
            _log_action("email", {"to": to, "subject": subject}, "blocked: recipient not in allowlist", False)
            return ToolResult(
                output="", success=False,
                error=f"Recipient '{to}' is not in the allowed recipients list",
                error_category=ErrorCategory.PERMISSION,
            )

        # Rate limit (thread-safe) — hold lock for entire send-and-record
        async with _email_rate_lock:
            if not _check_rate_limit(_email_timestamps, config.EMAIL_RATE_LIMIT, _EMAIL_RATE_WINDOW):
                _log_action("email", {"to": to, "subject": subject}, "blocked: rate limit exceeded", False)
                return ToolResult(
                    output="", success=False,
                    error=f"Email rate limit exceeded ({config.EMAIL_RATE_LIMIT}/hour). Try again later.",
                    error_category=ErrorCategory.PERMISSION,
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
                return ToolResult(output="", success=False, error=error_msg, retriable=True, error_category=ErrorCategory.TRANSIENT)

    @staticmethod
    def _send_smtp(msg: EmailMessage) -> str:
        """Synchronous SMTP send (runs in thread)."""
        host = config.EMAIL_SMTP_HOST
        port = config.EMAIL_SMTP_PORT
        user = config.EMAIL_SMTP_USER
        password = config.EMAIL_SMTP_PASS
        use_tls = config.EMAIL_SMTP_TLS

        ssl_ctx = ssl.create_default_context()
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=15, context=ssl_ctx)
        else:
            server = smtplib.SMTP(host, port, timeout=15)
            if use_tls:
                server.starttls(context=ssl_ctx)

        try:
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        finally:
            server.quit()

        return f"Email sent to {msg['To']} with subject '{msg['Subject']}'"
