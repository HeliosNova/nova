"""Quiet window — pause every LLM-driven monitor for a bounded time (2026-09-02).

A/B replays, model swaps and long evals need the GPU to themselves. Until now
that meant hand-disabling 39 domain monitors and remembering to re-enable them
(the 2026-09-01 A/B ran 4.5h on a "~2h pause" because the guardian had to
re-enable them itself). A quiet window is one row in ``system_state``
(``quiet_until`` ISO timestamp + reason) that the heartbeat honors every tick:
while it is active the slow (LLM) lane is skipped and ``last_check_at`` is not
advanced, so everything catches up the moment the window expires; the fast
deterministic lane keeps running. Windows are capped at 24h so a forgotten
one cannot silence Nova for days.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

KEY_UNTIL = "quiet_until"
KEY_REASON = "quiet_reason"
MAX_HOURS = 24.0
MIN_HOURS = 0.05


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _put(db, key: str, value: str) -> None:
    db.execute(
        "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
        (key, value, _now().strftime("%Y-%m-%d %H:%M:%S")),
    )


def set_quiet(db, hours: float, reason: str = "") -> datetime:
    """Open (or extend/shorten) the quiet window; returns its end (naive UTC)."""
    hours = max(MIN_HOURS, min(MAX_HOURS, float(hours)))
    until = _now() + timedelta(hours=hours)
    _put(db, KEY_UNTIL, until.strftime("%Y-%m-%d %H:%M:%S"))
    _put(db, KEY_REASON, (reason or "")[:200])
    logger.info("[Quiet] window open until %s UTC (%s)", until.strftime("%H:%M"), reason or "no reason given")
    return until


def clear_quiet(db) -> bool:
    row = db.fetchone("SELECT value FROM system_state WHERE key = ?", (KEY_UNTIL,))
    db.execute("DELETE FROM system_state WHERE key IN (?, ?)", (KEY_UNTIL, KEY_REASON))
    if row:
        logger.info("[Quiet] window cleared")
    return bool(row)


def quiet_until(db) -> datetime | None:
    """End of the active window, or None when no window is open / it expired."""
    try:
        row = db.fetchone("SELECT value FROM system_state WHERE key = ?", (KEY_UNTIL,))
    except Exception:
        return None
    if not row or not row["value"]:
        return None
    try:
        until = datetime.strptime(str(row["value"])[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return until if until > _now() else None


def quiet_status(db) -> dict:
    until = quiet_until(db)
    reason = None
    if until is not None:
        row = db.fetchone("SELECT value FROM system_state WHERE key = ?", (KEY_REASON,))
        reason = row["value"] if row else None
    return {
        "active": until is not None,
        "until": until.strftime("%Y-%m-%d %H:%M:%S") if until else None,
        "remaining_minutes": round((until - _now()).total_seconds() / 60, 1) if until else 0.0,
        "reason": reason,
    }


def quiet_active(db) -> bool:
    return quiet_until(db) is not None
