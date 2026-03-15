"""HTTP fetch tool — full HTTP client with SSRF protection."""

from __future__ import annotations

import base64
import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx

from app.config import config
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Block internal/private IPs
_BLOCKED_HOSTS = {
    "localhost", "0.0.0.0", "::1",
    "metadata.google.internal",
}

# RFC 1918 + loopback + link-local (IPv4 + IPv6)
_PRIVATE_NETWORKS = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv6Network("::1/128"),           # loopback
    ipaddress.IPv6Network("fe80::/10"),          # link-local
    ipaddress.IPv6Network("fc00::/7"),           # unique local (RFC 4193)
    ipaddress.IPv6Network("::ffff:127.0.0.0/104"),  # IPv4-mapped loopback
    ipaddress.IPv6Network("::ffff:10.0.0.0/104"),   # IPv4-mapped 10/8
    ipaddress.IPv6Network("::ffff:172.16.0.0/108"), # IPv4-mapped 172.16/12
    ipaddress.IPv6Network("::ffff:192.168.0.0/112"), # IPv4-mapped 192.168/16
]

_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}

_MAX_RESPONSE_SIZE = 50_000  # 50KB (up from 10KB)


def _is_private_ip(host: str) -> bool:
    """Check if a hostname resolves to a private/reserved IP (IPv4 + IPv6)."""
    # Check if host is a literal IP address
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in _PRIVATE_NETWORKS if type(addr) == type(net.network_address))
    except ValueError:
        pass
    # Not a literal IP — resolve hostname (both IPv4 and IPv6)
    try:
        results = socket.getaddrinfo(host, None)
        for family, _type, _proto, _canonname, sockaddr in results:
            try:
                addr = ipaddress.ip_address(sockaddr[0])
                if any(addr in net for net in _PRIVATE_NETWORKS if type(addr) == type(net.network_address)):
                    return True
            except ValueError:
                continue
    except (socket.gaierror, ValueError):
        pass
    return False


def _is_safe_url(url: str) -> bool:
    """Check URL isn't targeting internal services."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host in _BLOCKED_HOSTS:
            return False
        if _is_private_ip(host):
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        return True
    except Exception:
        return False


def _build_auth_header(auth: dict | None) -> dict:
    """Build authentication header from auth config.

    Supported types:
      {"type": "bearer", "token": "xxx"}
      {"type": "basic", "username": "u", "password": "p"}
      {"type": "api_key", "header": "X-Api-Key", "key": "xxx"}
    """
    if not auth:
        return {}
    auth_type = auth.get("type", "").lower()

    if auth_type == "bearer":
        token = auth.get("token", "")
        if token:
            return {"Authorization": f"Bearer {token}"}

    elif auth_type == "basic":
        username = auth.get("username", "")
        password = auth.get("password", "")
        if username:
            creds = base64.b64encode(f"{username}:{password}".encode()).decode()
            return {"Authorization": f"Basic {creds}"}

    elif auth_type == "api_key":
        header = auth.get("header", "X-Api-Key")
        key = auth.get("key", "")
        if key:
            return {header: key}

    return {}


def _sanitize_for_log(auth: dict | None) -> str:
    """Sanitize auth info for logging (mask tokens)."""
    if not auth:
        return "none"
    auth_type = auth.get("type", "unknown")
    return f"{auth_type}(***)"


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return a module-level singleton HTTP client for connection reuse."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
        )
    return _client


async def close_http_client() -> None:
    """Close the singleton HTTP client. Called during shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


class HttpFetchTool(BaseTool):
    name = "http_fetch"
    description = (
        "Make HTTP requests to URLs. Supports GET, POST, PUT, PATCH, DELETE. "
        "Use for fetching web content and calling external APIs."
    )
    parameters = "url: str, method: str = 'GET', headers: dict = None, body: dict|str = None, auth: dict = None"

    async def execute(
        self,
        *,
        url: str = "",
        method: str = "GET",
        headers: dict | None = None,
        body: dict | str | None = None,
        auth: dict | None = None,
        **kwargs,
    ) -> ToolResult:
        if not url:
            return ToolResult(output="", success=False, error="No URL provided")

        method = method.upper()
        if method not in _ALLOWED_METHODS:
            return ToolResult(
                output="",
                success=False,
                error=f"Method '{method}' not allowed. Use: {', '.join(sorted(_ALLOWED_METHODS))}",
            )

        if not _is_safe_url(url):
            return ToolResult(
                output="",
                success=False,
                error="URL blocked for security (internal/private address)",
            )

        # Log non-GET requests
        if method != "GET":
            logger.info(
                "http_fetch: %s %s auth=%s",
                method, url, _sanitize_for_log(auth),
            )

        # Build headers
        req_headers = {"User-Agent": "Nova/1.0"}
        auth_headers = _build_auth_header(auth)
        req_headers.update(auth_headers)
        if headers:
            req_headers.update(headers)

        # Build request kwargs
        request_kwargs: dict = {
            "method": method,
            "url": url,
            "headers": req_headers,
        }

        if body is not None and method not in ("GET", "HEAD"):
            if isinstance(body, dict):
                request_kwargs["json"] = body
            else:
                request_kwargs["content"] = str(body)
                if "Content-Type" not in req_headers:
                    req_headers["Content-Type"] = "text/plain"

        try:
            client = _get_client()
            resp = await client.request(**request_kwargs)

            # Check final URL after redirects for SSRF (also catches DNS rebinding on redirect)
            final_url = str(resp.url)
            if not _is_safe_url(final_url):
                return ToolResult(
                    output="",
                    success=False,
                    error="Response target blocked (internal/private address)",
                )

            # For non-2xx responses, still return the body with status info
            text = resp.text
            if len(text) > _MAX_RESPONSE_SIZE:
                total_len = len(text)
                content_type = resp.headers.get("content-type", "")
                truncated = text[:_MAX_RESPONSE_SIZE]
                if "json" in content_type:
                    # Try to truncate at a valid JSON boundary
                    for ch in ('}', ']', '"'):
                        last_pos = truncated.rfind(ch)
                        if last_pos > _MAX_RESPONSE_SIZE // 2:
                            truncated = truncated[:last_pos + 1]
                            break
                else:
                    # Truncate at last newline for readability
                    last_nl = truncated.rfind('\n')
                    if last_nl > _MAX_RESPONSE_SIZE // 2:
                        truncated = truncated[:last_nl]
                text = truncated + f"\n\n[... truncated: showing {len(truncated)} of {total_len} chars. Use http_fetch for full content.]"

            if resp.status_code >= 400:
                return ToolResult(
                    output=f"[HTTP {resp.status_code}]\n{text}",
                    success=False,
                    error=f"HTTP {resp.status_code}",
                    retriable=resp.status_code in (429, 500, 502, 503),
                )

            if config.ENABLE_INJECTION_DETECTION:
                from app.core.injection import sanitize_content
                text = sanitize_content(text, context="fetched")
            return ToolResult(output=text, success=True)

        except Exception as e:
            return ToolResult(output="", success=False, error=f"Fetch failed: {e}", retriable=True)
