"""HTTP fetch tool — full HTTP client with SSRF protection."""

from __future__ import annotations

import base64
import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import httpx

from app.config import config
from app.tools.base import BaseTool, ToolResult, ErrorCategory

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
    resolved = _resolve_and_check(host)
    return resolved is None


def _resolve_and_check(host: str) -> str | None:
    """Resolve hostname and check for private IPs.

    Returns the first safe resolved IP address as a string, or None if the host
    resolves to a private/blocked address (or resolution fails).
    """
    # Check if host is a literal IP address
    try:
        addr = ipaddress.ip_address(host)
        if any(addr in net for net in _PRIVATE_NETWORKS if type(addr) == type(net.network_address)):
            return None
        return host
    except ValueError:
        pass
    # Not a literal IP — resolve hostname (both IPv4 and IPv6)
    try:
        results = socket.getaddrinfo(host, None)
        first_safe_ip = None
        for family, _type, _proto, _canonname, sockaddr in results:
            try:
                addr = ipaddress.ip_address(sockaddr[0])
                if any(addr in net for net in _PRIVATE_NETWORKS if type(addr) == type(net.network_address)):
                    return None  # Any private resolution → block
                if first_safe_ip is None:
                    first_safe_ip = sockaddr[0]
            except ValueError:
                continue
        return first_safe_ip
    except (socket.gaierror, ValueError):
        # DNS resolution failed — return hostname as-is (fail-open for unresolvable hosts,
        # matching original behavior; the actual HTTP request will fail if host is invalid)
        return host


def _safe_url_with_pinned_ip(url: str) -> tuple[str, str, str | None] | None:
    """Validate URL safety and return (original_url, ip_pinned_url, original_host).

    Returns None if the URL is blocked. The ip_pinned_url has the hostname
    replaced with the resolved IP to prevent DNS rebinding (TOCTOU).
    If the host is already a literal IP, ip_pinned_url == original_url.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host in _BLOCKED_HOSTS:
            return None
        resolved_ip = _resolve_and_check(host)
        if resolved_ip is None:
            return None
        if parsed.scheme not in ("http", "https"):
            return None
        # If host is already the resolved IP, no pinning needed
        if host == resolved_ip:
            return (url, url, None)
        # HTTPS: skip DNS pinning — SSL cert validation already prevents SSRF
        # (pinning breaks CDN sites where resolved IP doesn't match cert)
        if parsed.scheme == "https":
            return (url, url, None)
        # HTTP only: replace hostname with resolved IP to prevent DNS rebinding
        port_suffix = f":{parsed.port}" if parsed.port else ""
        pinned_netloc = f"{resolved_ip}{port_suffix}"
        pinned_url = urlunparse(parsed._replace(netloc=pinned_netloc))
        return (url, pinned_url, host)
    except Exception:
        return None


def _is_safe_url(url: str) -> bool:
    """Check URL isn't targeting internal services."""
    return _safe_url_with_pinned_ip(url) is not None


# Sensitive query parameter names to redact from log output
_SENSITIVE_PARAMS_RE = re.compile(
    r"^(api_key|token|secret|password|key|auth|access_token|client_secret)$",
    re.IGNORECASE,
)


def _sanitize_url_for_log(url: str) -> str:
    """Strip sensitive query parameters from a URL before logging."""
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        params = parse_qs(parsed.query, keep_blank_values=True)
        sanitized = {}
        for k, v in params.items():
            if _SENSITIVE_PARAMS_RE.match(k):
                sanitized[k] = ["***"]
            else:
                sanitized[k] = v
        # Rebuild query string (flatten single-value lists)
        flat_params = []
        for k, vals in sanitized.items():
            for val in vals:
                flat_params.append((k, val))
        new_query = urlencode(flat_params)
        return urlunparse(parsed._replace(query=new_query))
    except Exception:
        return url


_FORBIDDEN_HEADERS = frozenset({"host", "transfer-encoding", "connection", "content-length", "te", "upgrade"})


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
            if header.lower() in _FORBIDDEN_HEADERS:
                logger.warning("Blocked reserved header in api_key auth: %s", header)
                return {}
            return {header: key}

    return {}


def _sanitize_for_log(auth: dict | None) -> str:
    """Sanitize auth info for logging (mask tokens)."""
    if not auth:
        return "none"
    auth_type = auth.get("type", "unknown")
    return f"{auth_type}(***)"


_client: httpx.AsyncClient | None = None


_MAX_REDIRECTS = 10

# HTTP status codes that require method change to GET on redirect
_REDIRECT_TO_GET = {301, 302, 303}
# HTTP status codes that preserve the original method on redirect
_REDIRECT_PRESERVE_METHOD = {307, 308}


def _get_client() -> httpx.AsyncClient:
    """Return a module-level singleton HTTP client for connection reuse."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(config.TOOL_TIMEOUT), connect=10.0),
            follow_redirects=False,
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
        "Make HTTP requests to external URLs. Supports GET, POST, PUT, PATCH, DELETE with optional headers, body, and "
        "authentication (bearer, basic, api_key). Returns response body (up to 50KB). "
        "Use when you have a specific URL to fetch or an API to call. "
        "Do NOT use for general web searches (use web_search instead). "
        "SSRF-protected: internal/private addresses are blocked."
    )
    parameters = "url: str, method: str = 'GET', headers: dict = None, body: dict|str = None, auth: dict = None"
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch. Must be http:// or https://.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                "description": "HTTP method. Defaults to GET.",
            },
            "headers": {
                "type": "object",
                "description": "Additional HTTP headers as key-value pairs.",
            },
            "body": {
                "description": "Request body. Dict sends as JSON, string sends as text. Ignored for GET/HEAD.",
            },
            "auth": {
                "type": "object",
                "description": "Authentication config: {type: 'bearer', token: '...'} or {type: 'basic', username: '...', password: '...'} or {type: 'api_key', header: '...', key: '...'}.",
            },
        },
        "required": ["url"],
    }

    def trim_output(self, output: str) -> str:
        """Keep status + first 2000 chars of content."""
        if len(output) <= 2000:
            return output
        # Try to keep status line (e.g., [HTTP 200]) and URL info at the top
        lines = output.split('\n', 3)
        header = '\n'.join(lines[:2]) if len(lines) > 2 else ''
        remaining = 2000 - len(header)
        if remaining > 200:
            return header + '\n' + output[len(header):len(header) + remaining] + '\n[...truncated]'
        return output[:2000] + '\n[...truncated]'

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
            return ToolResult(output="", success=False, error="No URL provided", error_category=ErrorCategory.VALIDATION)

        method = method.upper()
        if method not in _ALLOWED_METHODS:
            return ToolResult(
                output="",
                success=False,
                error=f"Method '{method}' not allowed. Use: {', '.join(sorted(_ALLOWED_METHODS))}",
                error_category=ErrorCategory.VALIDATION,
            )

        pin_result = _safe_url_with_pinned_ip(url)
        if pin_result is None:
            return ToolResult(
                output="",
                success=False,
                error="URL blocked for security (internal/private address)",
                error_category=ErrorCategory.PERMISSION,
            )
        _original_url, pinned_url, original_host = pin_result

        # Log non-GET requests (redact sensitive query params)
        if method != "GET":
            logger.info(
                "http_fetch: %s %s auth=%s",
                method, _sanitize_url_for_log(url), _sanitize_for_log(auth),
            )

        # Build headers
        req_headers = {"User-Agent": "Nova/1.0"}
        # Pin the original Host header when using IP-based URL to prevent DNS rebinding
        if original_host:
            req_headers["Host"] = original_host
        auth_headers = _build_auth_header(auth)
        req_headers.update(auth_headers)
        if headers:
            filtered = {k: v for k, v in headers.items() if k.lower() not in _FORBIDDEN_HEADERS}
            req_headers.update(filtered)

        # Build request kwargs — use pinned IP URL to prevent DNS rebinding
        request_kwargs: dict = {
            "method": method,
            "url": pinned_url,
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

            # Manual redirect loop: validate each hop via _is_safe_url() BEFORE
            # sending the request, preventing SSRF via open redirects.
            redirects = 0
            while resp.is_redirect and redirects < _MAX_REDIRECTS:
                redirects += 1
                location = resp.headers.get("location")
                if not location:
                    break
                # Resolve relative URLs against the current request URL
                next_url = str(resp.url.join(location))
                redirect_pin = _safe_url_with_pinned_ip(next_url)
                if redirect_pin is None:
                    return ToolResult(
                        output="",
                        success=False,
                        error=f"Redirect #{redirects} blocked (internal/private address)",
                        error_category=ErrorCategory.PERMISSION,
                    )
                _redir_orig, redir_pinned, redir_host = redirect_pin
                # 307/308 preserve original method; 301/302/303 switch to GET
                if resp.status_code in _REDIRECT_PRESERVE_METHOD:
                    next_method = method
                else:
                    next_method = "GET"
                # Follow the redirect (strip body for method changes to GET)
                # Use pinned IP URL + Host header to prevent DNS rebinding
                redir_headers = dict(req_headers)
                if redir_host:
                    redir_headers["Host"] = redir_host
                elif "Host" in redir_headers:
                    del redir_headers["Host"]
                redirect_kwargs: dict = {
                    "method": next_method,
                    "url": redir_pinned,
                    "headers": redir_headers,
                }
                if next_method != "GET" and body is not None:
                    if isinstance(body, dict):
                        redirect_kwargs["json"] = body
                    else:
                        redirect_kwargs["content"] = str(body)
                resp = await client.request(**redirect_kwargs)

            if resp.is_redirect:
                return ToolResult(
                    output="",
                    success=False,
                    error=f"Too many redirects (>{_MAX_REDIRECTS})",
                    error_category=ErrorCategory.TRANSIENT,
                )

            # For non-2xx responses, still return the body with status info
            text = resp.text
            if len(text) > _MAX_RESPONSE_SIZE:
                total_len = len(text)
                truncated = text[:_MAX_RESPONSE_SIZE]
                # Truncate at last newline for readability (all content types)
                last_nl = truncated.rfind('\n')
                if last_nl > _MAX_RESPONSE_SIZE // 2:
                    truncated = truncated[:last_nl]
                text = truncated + f"\n\n[...truncated: showing {len(truncated)} of {total_len} chars]"

            if resp.status_code >= 400:
                if resp.status_code == 404:
                    _cat = ErrorCategory.NOT_FOUND
                elif resp.status_code in (429, 500, 502, 503):
                    _cat = ErrorCategory.TRANSIENT
                else:
                    _cat = ErrorCategory.VALIDATION
                return ToolResult(
                    output=f"[HTTP {resp.status_code}]\n{text}",
                    success=False,
                    error=f"HTTP {resp.status_code}",
                    retriable=resp.status_code in (429, 500, 502, 503),
                    error_category=_cat,
                )

            if config.ENABLE_INJECTION_DETECTION:
                from app.core.injection import sanitize_content
                text = sanitize_content(text, context="fetched")
            return ToolResult(output=text, success=True)

        except Exception as e:
            return ToolResult(output="", success=False, error=f"Fetch failed: {e}", retriable=True, error_category=ErrorCategory.TRANSIENT)
