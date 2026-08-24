"""Authentication middleware — Bearer token validation with rate-limiting.

Failure tracking dicts are bounded: max AUTH_MAX_TRACKED_IPS entries (default 10k),
evicted on every auth check and every recorded failure.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import time

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import config

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

# Per-IP auth failure tracking: ip -> list of failure timestamps (wall clock)
# Regular dict (NOT defaultdict) — prevents unbounded growth from auto-creating
# entries on every auth check.  Entries are only created on actual failures.
_auth_failures: dict[str, list[float]] = {}
_AUTH_WINDOW = 60        # sliding window in seconds
_lockouts: dict[str, float] = {}

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
    except Exception as e:
        logger.warning("Failed to clean expired auth lockouts: %s", e)

    if loaded_lockouts or loaded_failures:
        logger.info(
            "Loaded auth lockout state from DB: %d active lockouts, %d IPs with failures",
            loaded_lockouts, loaded_failures,
        )


def _sync_to_db(ip: str) -> None:
    """Persist current lockout/failure state for an IP to the database.

    Called from require_auth (the event loop) on every auth failure and
    cleanup — the in-memory state is snapshotted here (single-threaded on
    the loop, so consistent) and the DB write is handed to the default
    executor fire-and-forget so the request path never waits on the
    SQLite lock. Off-loop callers (startup) write inline.
    """
    db = _get_db()
    if db is None:
        return
    failures = list(_auth_failures.get(ip, []))
    locked_until = _lockouts.get(ip)

    def _write() -> None:
        try:
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

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _write()
        return
    loop.run_in_executor(None, _write)


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For only behind a trusted proxy."""
    if config.TRUSTED_PROXY and request.client and request.client.host == config.TRUSTED_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


_DOCKER_BRIDGE = ipaddress.ip_network("172.16.0.0/12")


def _is_nat_collapsed_ip(ip: str) -> bool:
    """True for the loopback + Docker-bridge addresses that host->published-port
    traffic collapses to under Docker's NAT — EVERY client looks like one gateway/
    container IP there (the live auth_lockouts row was 172.18.0.8). Locking such an
    address out is pure self-DoS with no security value (it can't tell clients
    apart), so these are exempt from the failure-lockout. Genuine LAN (10/8,
    192.168/16) and public IPs are NOT exempt: when a trusted proxy supplies real
    per-client IPs via X-Forwarded-For the lockout still protects them. The 401 on
    a wrong key is unaffected — only the accumulate-and-lock path is skipped.
    (audit 2026-08-22)"""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_loopback or addr in _DOCKER_BRIDGE


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
    # NAT-collapsed addresses can't be meaningfully rate-limited (all clients
    # share them) and locking them out only DoSes the owner — skip.
    if _is_nat_collapsed_ip(ip):
        return
    now = time.time()

    # Periodic cleanup of expired entries (removes stale IPs)
    _cleanup_expired_entries()

    # Hard cap: evict oldest if either dict exceeds max tracked IPs.
    # This runs on every auth check, not just on failure recording.
    max_ips = config.AUTH_MAX_TRACKED_IPS
    _evict_oldest(_auth_failures, max_ips)
    _evict_oldest(_lockouts, max_ips)

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

    # Prune old failures outside the window — only if this IP has entries.
    # Using .get() avoids creating empty entries for IPs that never failed.
    existing = _auth_failures.get(ip)
    if existing:
        cutoff = now - _AUTH_WINDOW
        pruned = [t for t in existing if t > cutoff]
        if pruned:
            _auth_failures[ip] = pruned
        else:
            del _auth_failures[ip]


def _record_failure(ip: str) -> None:
    """Record an auth failure and lock out if threshold exceeded."""
    # Don't accumulate failures for NAT-collapsed addresses — locking the shared
    # gateway/loopback out is self-DoS, not protection (see _is_nat_collapsed_ip).
    if _is_nat_collapsed_ip(ip):
        return
    now = time.time()

    # Explicit entry creation (no defaultdict auto-creation)
    if ip not in _auth_failures:
        _auth_failures[ip] = []
    _auth_failures[ip].append(now)

    # Evict oldest entries if tracking dicts grow too large
    max_ips = config.AUTH_MAX_TRACKED_IPS
    _evict_oldest(_auth_failures, max_ips)
    _evict_oldest(_lockouts, max_ips)

    failures = _auth_failures.get(ip, [])
    if len(failures) >= config.AUTH_MAX_FAILURES:
        _lockouts[ip] = now + config.AUTH_LOCKOUT_SECONDS
        _auth_failures.pop(ip, None)  # Clear failures for this IP

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
        # Defense-in-depth for the keyless out-of-box path (full-system
        # exploration 2026-07-09): keyless mode exists so a fresh localhost
        # install works without a key (ports bind 127.0.0.1 in compose). But if
        # the owner later exposes the port (bind 0.0.0.0 / a reverse proxy) while
        # still keyless, the WHOLE API — chat, KG, exports, config-write — is
        # open. Peer IP is useless here (Docker NAT rewrites it to the bridge
        # gateway), but the Host header passes through NAT unchanged. Serve
        # keyless ONLY to a loopback Host; a non-loopback Host means the request
        # arrived over a network path → refuse and demand a key. Not a hard
        # boundary (Host is spoofable by a deliberate attacker), but it fails the
        # exposure CLOSED for drive-by/scanner/browser traffic while keeping
        # localhost frictionless. Setting NOVA_API_KEY remains the real control.
        # "testserver" is the ASGI TestClient's default Host — allow it so the
        # keyless test suite (conftest sets NOVA_API_KEY="") isn't 401'd. No real
        # HTTP client ever emits it, so allowing it costs zero real protection:
        # the gate's job is stopping drive-by/scanner/browser traffic, which
        # carries the server's actual Host, not this sentinel.
        _LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "testserver")
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip().lower().strip("[]")
        if host and host not in _LOCAL_HOSTS:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for non-local access. Set NOVA_API_KEY.",
            )
        return

    ip = _get_client_ip(request)
    _check_rate_limit(ip)

    if credentials is None or not hmac.compare_digest(credentials.credentials, config.API_KEY):
        _record_failure(ip)
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
