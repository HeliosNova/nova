"""Authentication middleware — Bearer token validation with rate-limiting."""

from __future__ import annotations

import hmac
import json
import logging
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import config

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

# Per-IP auth failure tracking: ip -> list of failure timestamps (wall clock)
_auth_failures: dict[str, list[float]] = defaultdict(list)
_AUTH_WINDOW = 60        # sliding window in seconds
_lockouts: dict[str, float] = {}
_MAX_TRACKED_IPS = None  # Use config.AUTH_MAX_TRACKED_IPS

# Lazy DB handle for lockout persistence
_lockout_db = None


def _get_db():
    """Lazily get a SafeDB instance for lockout persistence."""
    global _lockout_db
    if _lockout_db is None:
        try:
            from app.database import get_db
            _lockout_db = get_db()
        except Exception as e:
            logger.warning("Could not initialize lockout DB: %s", e)
    return _lockout_db


def load_lockouts_from_db() -> None:
    """Load persisted lockout state from DB into in-memory cache.

    Call this on application startup to restore lockout state across restarts.
    """
    db = _get_db()
    if db is None:
        return
    try:
        rows = db.fetchall("SELECT ip, failures, locked_until FROM auth_lockouts")
    except Exception as e:
        logger.warning("Failed to load lockouts from DB: %s", e)
        return

    now = time.time()
    loaded_lockouts = 0
    loaded_failures = 0

    for row in rows:
        ip = row["ip"]
        locked_until = row["locked_until"]
        failures_json = row["failures"] or "[]"

        # Restore active lockouts (skip expired)
        if locked_until and locked_until > now:
            _lockouts[ip] = locked_until
            loaded_lockouts += 1

        # Restore recent failures within the sliding window
        try:
            failure_times = json.loads(failures_json)
        except (json.JSONDecodeError, TypeError):
            failure_times = []

        cutoff = now - _AUTH_WINDOW
        recent = [t for t in failure_times if t > cutoff]
        if recent:
            _auth_failures[ip] = recent
            loaded_failures += 1

    # Clean expired entries from DB
    try:
        db.execute(
            "DELETE FROM auth_lockouts WHERE "
            "(locked_until IS NOT NULL AND locked_until <= ?) AND "
            "(failures = '[]' OR failures IS NULL)",
            (now,),
        )
    except Exception:
        pass

    if loaded_lockouts or loaded_failures:
        logger.info(
            "Loaded auth lockout state from DB: %d active lockouts, %d IPs with failures",
            loaded_lockouts, loaded_failures,
        )


def _sync_to_db(ip: str) -> None:
    """Persist current lockout/failure state for an IP to the database."""
    db = _get_db()
    if db is None:
        return
    try:
        failures = _auth_failures.get(ip, [])
        locked_until = _lockouts.get(ip)
        if not failures and locked_until is None:
            # Clean up — no state to persist
            db.execute("DELETE FROM auth_lockouts WHERE ip = ?", (ip,))
        else:
            db.execute(
                "INSERT OR REPLACE INTO auth_lockouts (ip, failures, locked_until, updated_at) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (ip, json.dumps(failures), locked_until),
            )
    except Exception as e:
        logger.warning("Failed to sync lockout state to DB for %s: %s", ip, e)


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For only behind a trusted proxy."""
    if config.TRUSTED_PROXY and request.client and request.client.host == config.TRUSTED_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _evict_oldest(d: dict, max_size: int) -> None:
    """Evict oldest entries from a dict when it exceeds max_size."""
    if len(d) <= max_size:
        return
    # Remove excess entries (oldest first by insertion order)
    excess = len(d) - max_size
    keys_to_remove = list(d.keys())[:excess]
    for k in keys_to_remove:
        del d[k]


def _cleanup_expired_entries() -> None:
    """Remove expired lockouts and stale failure entries from memory and DB."""
    now = time.time()
    # Remove expired lockouts
    expired = [ip for ip, expiry in _lockouts.items() if now >= expiry]
    for ip in expired:
        del _lockouts[ip]
        _sync_to_db(ip)
    # Remove failure entries with no recent failures (older than AUTH_LOCKOUT_SECONDS)
    stale_cutoff = now - config.AUTH_LOCKOUT_SECONDS
    stale = [ip for ip, times in _auth_failures.items() if not times or max(times) < stale_cutoff]
    for ip in stale:
        del _auth_failures[ip]
        _sync_to_db(ip)


def _check_rate_limit(ip: str) -> None:
    """Raise 429 if IP has exceeded auth failure limit."""
    now = time.time()

    # Periodic cleanup of expired entries
    _cleanup_expired_entries()

    # Check if currently locked out
    if ip in _lockouts:
        if now < _lockouts[ip]:
            raise HTTPException(
                status_code=429,
                detail="Too many authentication failures. Try again later.",
            )
        else:
            del _lockouts[ip]
            _sync_to_db(ip)

    # Prune old failures outside the window
    cutoff = now - _AUTH_WINDOW
    _auth_failures[ip] = [t for t in _auth_failures[ip] if t > cutoff]


def _record_failure(ip: str) -> None:
    """Record an auth failure and lock out if threshold exceeded."""
    now = time.time()
    _auth_failures[ip].append(now)

    # Evict oldest entries if tracking dicts grow too large
    max_ips = config.AUTH_MAX_TRACKED_IPS
    _evict_oldest(_auth_failures, max_ips)
    _evict_oldest(_lockouts, max_ips)

    if len(_auth_failures[ip]) >= config.AUTH_MAX_FAILURES:
        _lockouts[ip] = now + config.AUTH_LOCKOUT_SECONDS
        _auth_failures[ip].clear()

    # Persist to DB on every failure and lockout event
    _sync_to_db(ip)


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Bearer token auth. If API_KEY is empty, behavior depends on REQUIRE_AUTH."""
    if not config.API_KEY:
        if config.REQUIRE_AUTH:
            raise HTTPException(
                status_code=503,
                detail="API key not configured. Set NOVA_API_KEY to enable access.",
            )
        if not getattr(require_auth, "_warned_no_key", False):
            logger.critical(
                "API_KEY is empty — authentication disabled! "
                "All endpoints are publicly accessible. "
                "Set NOVA_API_KEY for production."
            )
            require_auth._warned_no_key = True
        return

    ip = _get_client_ip(request)
    _check_rate_limit(ip)

    if credentials is None or not hmac.compare_digest(credentials.credentials, config.API_KEY):
        _record_failure(ip)
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
