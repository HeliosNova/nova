"""Authentication middleware — Bearer token validation with rate-limiting."""

from __future__ import annotations

import hmac
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import config

_bearer = HTTPBearer(auto_error=False)

# Per-IP auth failure tracking: ip -> list of failure timestamps
_auth_failures: dict[str, list[float]] = defaultdict(list)
_AUTH_WINDOW = 60        # sliding window in seconds
_AUTH_MAX_FAILURES = 10  # max failures per window
_AUTH_LOCKOUT = 300      # lockout duration in seconds (5 min)
_lockouts: dict[str, float] = {}


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For behind reverse proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> None:
    """Raise 429 if IP has exceeded auth failure limit."""
    now = time.monotonic()

    # Check if currently locked out
    if ip in _lockouts:
        if now < _lockouts[ip]:
            raise HTTPException(
                status_code=429,
                detail="Too many authentication failures. Try again later.",
            )
        else:
            del _lockouts[ip]

    # Prune old failures outside the window
    cutoff = now - _AUTH_WINDOW
    _auth_failures[ip] = [t for t in _auth_failures[ip] if t > cutoff]


def _record_failure(ip: str) -> None:
    """Record an auth failure and lock out if threshold exceeded."""
    now = time.monotonic()
    _auth_failures[ip].append(now)

    if len(_auth_failures[ip]) >= _AUTH_MAX_FAILURES:
        _lockouts[ip] = now + _AUTH_LOCKOUT
        _auth_failures[ip].clear()


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Bearer token auth. If API_KEY is empty, auth is disabled (dev mode)."""
    if not config.API_KEY:
        return

    ip = _get_client_ip(request)
    _check_rate_limit(ip)

    if credentials is None or not hmac.compare_digest(credentials.credentials, config.API_KEY):
        _record_failure(ip)
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
