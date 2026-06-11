"""Cross-monitor deduplication.

When the Iran-Israel ceasefire shows up in Geopolitics, Middle East, AND
Current Events monitors all in the same day, the user sees the same story
3 times. This module hashes the salient claims of a monitor result and
skips reposting if the same claims were posted in the last N hours by
ANY monitor.

Storage: small SQLite table `monitor_dedup_log` (created on first use).
Hash strategy: extract the top-3 noun-phrase-ish substrings from each
result, normalize, hash. Two results with overlapping core claims
collide.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3

logger = logging.getLogger(__name__)


_NOUN_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){1,3})\b"
)
_DEDUP_WINDOW_HOURS = 18


def _ensure_table(db) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS monitor_dedup_log ("
            " hash TEXT PRIMARY KEY, "
            " monitor_name TEXT NOT NULL, "
            " sample TEXT NOT NULL, "
            " created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_monitor_dedup_created "
            "ON monitor_dedup_log(created_at)"
        )
    except sqlite3.Error as e:
        logger.warning("[Dedup] table create failed: %s", e)


def _content_hash(value: str) -> str:
    """Hash the top noun-phrases of a value. Two values that share their
    top topics will collide."""
    if not value:
        return ""
    phrases = _NOUN_PHRASE_RE.findall(value)
    if not phrases:
        # Fall back to a hash of the first 200 chars normalized
        norm = re.sub(r"\s+", " ", value.lower()).strip()[:200]
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]
    # Normalize + dedupe + sort + take top 5
    seen: set[str] = set()
    norm: list[str] = []
    for p in phrases:
        k = p.lower().strip()
        if k in seen:
            continue
        seen.add(k)
        norm.append(k)
        if len(norm) >= 5:
            break
    blob = "|".join(sorted(norm))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def is_duplicate(db, monitor_name: str, value: str) -> bool:
    """Return True if a result with the same content hash was logged in
    the last _DEDUP_WINDOW_HOURS hours. Otherwise log this hash and return
    False so the caller can post it."""
    if not value or len(value.strip()) < 50:
        return False  # too short to dedupe meaningfully

    h = _content_hash(value)
    if not h:
        return False

    _ensure_table(db)

    # Prune old entries opportunistically
    try:
        db.execute(
            "DELETE FROM monitor_dedup_log "
            "WHERE created_at < datetime('now', '-48 hours')"
        )
    except sqlite3.Error:
        pass

    try:
        existing = db.fetchone(
            "SELECT monitor_name, created_at FROM monitor_dedup_log "
            "WHERE hash = ? "
            "AND created_at > datetime('now', ?)",
            (h, f"-{_DEDUP_WINDOW_HOURS} hours"),
        )
    except sqlite3.Error as e:
        logger.warning("[Dedup] lookup failed: %s", e)
        return False

    if existing:
        logger.info(
            "[Dedup] '%s' duplicates content posted by '%s' at %s — skipping",
            monitor_name, existing["monitor_name"], existing["created_at"],
        )
        return True

    # Log this hash
    try:
        db.execute(
            "INSERT OR REPLACE INTO monitor_dedup_log "
            "(hash, monitor_name, sample) VALUES (?, ?, ?)",
            (h, monitor_name, value[:200]),
        )
    except sqlite3.Error as e:
        logger.warning("[Dedup] log insert failed: %s", e)

    return False
