"""Webhook tool — trigger external webhooks/APIs with security controls.

Config-gated (ENABLE_WEBHOOKS=false by default), URL-allowlisted,
rate-limited, SSRF-protected. Uses existing httpx dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx

from app.config import config
from app.database import get_db
from app.tools.base import BaseTool, ToolResult
from app.tools.http_fetch import _is_safe_url

logger = logging.getLogger(__name__)

# Rate limiting: max 30 webhook calls per hour
_WEBHOOK_RATE_LIMIT = 30
_WEBHOOK_RATE_WINDOW = 3600  # seconds
_webhook_timestamps: list[float] = []
_webhook_lock = asyncio.Lock()

_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_TIMEOUT = 15.0


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
    """Return True if under the rate limit."""
    now = time.time()
    cutoff = now - window
    timestamps[:] = [t for t in timestamps if t > cutoff]
    return len(timestamps) < max_count


def _is_url_allowed(url: str) -> bool:
    """Check if URL matches an allowed prefix. Empty allowlist = block all."""
    allowlist = config.WEBHOOK_ALLOWED_URLS.strip()
    if not allowlist:
        return False  # No allowlist configured = block all
    prefixes = [p.strip() for p in allowlist.split(",") if p.strip()]
    return any(url.startswith(prefix) for prefix in prefixes)


class WebhookTool(BaseTool):
    name = "webhook"
    description = (
        "Call an external webhook or API endpoint. Requires ENABLE_WEBHOOKS=true "
        "and URL must match WEBHOOK_ALLOWED_URLS prefixes. "
        "Actions: call."
    )
    parameters = "action: str, url: str, method: str (GET/POST), headers: dict, body: dict or str"

    async def execute(
        self,
        *,
        action: str = "",
        url: str = "",
        method: str = "POST",
        headers: dict | None = None,
        body: dict | str | None = None,
        **kwargs,
    ) -> ToolResult:
        if not config.ENABLE_WEBHOOKS:
            return ToolResult(
                output="", success=False,
                error="Webhooks are disabled. Set ENABLE_WEBHOOKS=true and configure WEBHOOK_ALLOWED_URLS.",
            )

        if not action or action.lower().strip() != "call":
            return ToolResult(output="", success=False, error="Use action='call' to trigger a webhook")

        if not url:
            return ToolResult(output="", success=False, error="URL is required")

        method = method.upper().strip()
        if method not in _ALLOWED_METHODS:
            return ToolResult(
                output="", success=False,
                error=f"Method '{method}' not allowed. Use: {', '.join(sorted(_ALLOWED_METHODS))}",
            )

        # URL allowlist check
        if not _is_url_allowed(url):
            _log_action("webhook", {"url": url, "method": method}, "blocked: URL not in allowlist", False)
            return ToolResult(
                output="", success=False,
                error=f"URL '{url}' is not in the allowed webhook URLs. Configure WEBHOOK_ALLOWED_URLS.",
            )

        # SSRF protection (reuse from http_fetch)
        if not _is_safe_url(url):
            _log_action("webhook", {"url": url, "method": method}, "blocked: SSRF protection", False)
            return ToolResult(
                output="", success=False,
                error="URL blocked for security (internal/private address)",
            )

        # Rate limit (locked to prevent concurrent bypass)
        async with _webhook_lock:
            if not _check_rate_limit(_webhook_timestamps, _WEBHOOK_RATE_LIMIT, _WEBHOOK_RATE_WINDOW):
                _log_action("webhook", {"url": url, "method": method}, "blocked: rate limit exceeded", False)
                return ToolResult(
                    output="", success=False,
                    error=f"Webhook rate limit exceeded ({_WEBHOOK_RATE_LIMIT}/hour). Try again later.",
                )
            _webhook_timestamps.append(time.time())

        params = {"url": url, "method": method}
        try:
            result = await self._call(url, method, headers or {}, body)
            _log_action("webhook", params, result, True)
            return ToolResult(output=result, success=True)
        except Exception as e:
            error_msg = f"Webhook call failed: {e}"
            _log_action("webhook", params, error_msg, False)
            return ToolResult(output="", success=False, error=error_msg)

    @staticmethod
    async def _call(
        url: str,
        method: str,
        headers: dict,
        body: dict | str | None,
    ) -> str:
        """Execute the HTTP call."""
        request_headers = {"User-Agent": "Nova/1.0"}
        request_headers.update(headers)

        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
        ) as client:
            # Build kwargs
            kw: dict = {"headers": request_headers}
            if body is not None and method in ("POST", "PUT", "PATCH"):
                if isinstance(body, dict):
                    kw["json"] = body
                else:
                    kw["content"] = str(body)
                    if "Content-Type" not in request_headers:
                        kw["headers"]["Content-Type"] = "text/plain"

            resp = await client.request(method, url, **kw)

            # Check final URL for SSRF after redirects
            final_url = str(resp.url)
            if final_url != url and not _is_safe_url(final_url):
                raise ValueError("Redirect target blocked (internal/private address)")

            status = resp.status_code
            response_text = resp.text
            if len(response_text) > 5000:
                response_text = response_text[:5000] + "\n[... truncated]"

            return f"HTTP {status} {resp.reason_phrase}\n{response_text}"
