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
from app.tools.action_logging import log_action as _log_action
from app.tools.base import BaseTool, ToolResult, ErrorCategory
from app.tools.http_fetch import _is_safe_url

logger = logging.getLogger(__name__)

# Rate limiting: max 30 webhook calls per hour
_WEBHOOK_RATE_LIMIT = 30
_WEBHOOK_RATE_WINDOW = 3600  # seconds
_webhook_timestamps: list[float] = []
_webhook_lock = asyncio.Lock()

_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_FORBIDDEN_HEADERS = frozenset({"host", "transfer-encoding", "connection", "content-length", "te", "upgrade"})
_TIMEOUT = 15.0


def _check_rate_limit(timestamps: list[float], max_count: int, window: int) -> bool:
    """Return True if under the rate limit."""
    now = time.time()
    cutoff = now - window
    timestamps[:] = [t for t in timestamps if t > cutoff]
    return len(timestamps) < max_count


def _is_url_allowed(url: str) -> bool:
    """Check if URL matches an allowed prefix. Empty allowlist = block all."""
    from urllib.parse import urlparse
    allowlist = config.WEBHOOK_ALLOWED_URLS.strip()
    if not allowlist:
        return False
    parsed = urlparse(url)
    for raw_prefix in allowlist.split(","):
        raw_prefix = raw_prefix.strip()
        if not raw_prefix:
            continue
        prefix = urlparse(raw_prefix)
        if (parsed.scheme == prefix.scheme and parsed.netloc == prefix.netloc
                and parsed.path.startswith(prefix.path)):
            # Ensure the path match ends at a boundary to prevent
            # /api matching /api-internal
            rest = parsed.path[len(prefix.path):]
            if rest and rest[0] not in ('/', '?', '#'):
                continue
            return True
    return False


class WebhookTool(BaseTool):
    name = "webhook"
    description = (
        "Trigger external webhook or API endpoints with full HTTP support. "
        "Requires ENABLE_WEBHOOKS=true and URL must match WEBHOOK_ALLOWED_URLS prefixes. "
        "SSRF-protected and rate-limited (30 calls/hour). "
        "Supports GET, POST, PUT, PATCH, DELETE with custom headers and body. "
        "Do NOT use for general HTTP fetching (use http_fetch) or email notifications (use email_send)."
    )
    parameters = "action: str, url: str, method: str (GET/POST), headers: dict, body: dict or str"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["call"],
                "description": "Webhook action. Currently only 'call' is supported.",
            },
            "url": {
                "type": "string",
                "description": "Webhook URL. Must match WEBHOOK_ALLOWED_URLS prefixes.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "description": "HTTP method. Defaults to POST.",
            },
            "headers": {
                "type": "object",
                "description": "Additional HTTP headers as key-value pairs.",
            },
            "body": {
                "description": "Request body. Dict sends as JSON, string sends as text.",
            },
        },
        "required": ["action", "url"],
    }

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
                error_category=ErrorCategory.PERMISSION,
            )

        if not action or action.lower().strip() != "call":
            return ToolResult(output="", success=False, error="Use action='call' to trigger a webhook", error_category=ErrorCategory.VALIDATION)

        if not url:
            return ToolResult(output="", success=False, error="URL is required", error_category=ErrorCategory.VALIDATION)

        method = method.upper().strip()
        if method not in _ALLOWED_METHODS:
            return ToolResult(
                output="", success=False,
                error=f"Method '{method}' not allowed. Use: {', '.join(sorted(_ALLOWED_METHODS))}",
                error_category=ErrorCategory.VALIDATION,
            )

        # URL allowlist check
        if not _is_url_allowed(url):
            _log_action("webhook", {"url": url, "method": method}, "blocked: URL not in allowlist", False)
            return ToolResult(
                output="", success=False,
                error=f"URL '{url}' is not in the allowed webhook URLs. Configure WEBHOOK_ALLOWED_URLS.",
                error_category=ErrorCategory.PERMISSION,
            )

        # SSRF protection (reuse from http_fetch)
        if not _is_safe_url(url):
            _log_action("webhook", {"url": url, "method": method}, "blocked: SSRF protection", False)
            return ToolResult(
                output="", success=False,
                error="URL blocked for security (internal/private address)",
                error_category=ErrorCategory.PERMISSION,
            )

        # Rate limit (locked to prevent concurrent bypass)
        async with _webhook_lock:
            if not _check_rate_limit(_webhook_timestamps, _WEBHOOK_RATE_LIMIT, _WEBHOOK_RATE_WINDOW):
                _log_action("webhook", {"url": url, "method": method}, "blocked: rate limit exceeded", False)
                return ToolResult(
                    output="", success=False,
                    error=f"Webhook rate limit exceeded ({_WEBHOOK_RATE_LIMIT}/hour). Try again later.",
                    error_category=ErrorCategory.PERMISSION,
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
            return ToolResult(output="", success=False, error=error_msg, retriable=True, error_category=ErrorCategory.TRANSIENT)

    @staticmethod
    async def _call(
        url: str,
        method: str,
        headers: dict,
        body: dict | str | None,
        _max_redirects: int = 10,
    ) -> str:
        """Execute the HTTP call with manual redirect following for SSRF safety."""
        request_headers = {"User-Agent": "Nova/1.0"}
        filtered = {k: v for k, v in headers.items() if k.lower() not in _FORBIDDEN_HEADERS}
        request_headers.update(filtered)

        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=False,
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

            # Manually follow redirects, checking each hop for SSRF
            redirects = 0
            current_method = method
            while resp.is_redirect and redirects < _max_redirects:
                redirects += 1
                location = resp.headers.get("location")
                if not location:
                    break
                # Resolve relative URLs against the current request URL
                next_url = str(resp.url.join(location))
                if not _is_safe_url(next_url):
                    raise ValueError(f"Redirect #{redirects} blocked (internal/private address)")
                # 307/308 preserve original method; 301/302/303 switch to GET
                status_code = resp.status_code
                if status_code in (301, 302, 303):
                    current_method = "GET"
                # else 307/308: keep current_method unchanged
                resp = await client.request(current_method, next_url, headers={"User-Agent": "Nova/1.0"})

            if resp.is_redirect:
                raise ValueError(f"Too many redirects (>{_max_redirects})")

            status = resp.status_code
            response_text = resp.text
            if len(response_text) > 5000:
                response_text = response_text[:5000] + "\n[... truncated]"

            return f"HTTP {status} {resp.reason_phrase}\n{response_text}"
