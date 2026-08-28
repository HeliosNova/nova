"""Tests for the 2026-08-26 curiosity-churn fixes.

Live symptom: a 7-day-old urgency-0.9 curiosity item was re-researched every
daemon tick (3 full GPU runs in 20 minutes), failing its closure check each
time, and the dream consolidator's unconditional attempts=0 reset guaranteed
it never terminally failed. Three fixes:

1. dream._handle_failed_curiosity grants at most ONE reset per item
   (the [dream-reset] marker in `resolution`).
2. daemon._decide enforces a 30-min cooldown between daemon-initiated
   curiosity research runs.
3. native_search.search_health() exposes a rolling empty-result rate so the
   closure path can tell "unanswerable topic" from "search outage" and skip
   the attempt burn (heartbeat_loop CURIOSITY DEFERRED path).
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from unittest.mock import MagicMock

import pytest

from app.core.dream import ConsolidationResult, DreamConsolidator, GatherSignals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeAsyncDB:
    """Minimal async facade over an in-memory sqlite3 connection."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE curiosity_queue ("
            " id INTEGER PRIMARY KEY, topic TEXT, source TEXT, urgency REAL,"
            " status TEXT, attempts INTEGER DEFAULT 0, resolution TEXT,"
            " created_at TEXT DEFAULT (datetime('now')), resolved_at TEXT)"
        )

    async def execute(self, sql, params=()):
        self._conn.execute(sql, params)
        self._conn.commit()

    async def fetchall(self, sql, params=()):
        return self._conn.execute(sql, params).fetchall()

    def row(self, item_id):
        return self._conn.execute(
            "SELECT * FROM curiosity_queue WHERE id=?", (item_id,)
        ).fetchone()

    def seed(self, *, topic, status="failed", attempts=3, resolution=None):
        cur = self._conn.execute(
            "INSERT INTO curiosity_queue (topic, source, urgency, status, attempts, resolution)"
            " VALUES (?, 'test', 0.9, ?, ?, ?)",
            (topic, status, attempts, resolution),
        )
        self._conn.commit()
        return cur.lastrowid


def _run_handle_failed(db, items):
    consolidator = DreamConsolidator(db)
    signals = GatherSignals()
    signals.failed_curiosity = items
    result = ConsolidationResult()
    asyncio.run(
        consolidator._handle_failed_curiosity(signals, result)
    )
    return result


# ---------------------------------------------------------------------------
# 1. Dream reset-once
# ---------------------------------------------------------------------------

class TestDreamResetOnce:
    def test_first_failure_gets_reset_with_marker(self):
        db = _FakeAsyncDB()
        item_id = db.seed(topic="What powers the Fed's new framework?")
        result = _run_handle_failed(
            db, [{"id": item_id, "topic": "What powers the Fed's new framework?",
                  "resolution": None}])
        row = db.row(item_id)
        assert result.curiosity_reset == 1
        assert row["status"] == "pending"
        assert row["attempts"] == 0
        assert (row["resolution"] or "").startswith("[dream-reset")

    def test_second_failure_stays_failed(self):
        db = _FakeAsyncDB()
        item_id = db.seed(topic="What powers the Fed's new framework?",
                          resolution="[dream-reset 2026-08-25 10:00:00]")
        result = _run_handle_failed(
            db, [{"id": item_id, "topic": "What powers the Fed's new framework?",
                  "resolution": "[dream-reset 2026-08-25 10:00:00]"}])
        row = db.row(item_id)
        assert result.curiosity_reset == 0
        assert row["status"] == "failed"

    def test_subjective_topic_still_dismissed(self):
        db = _FakeAsyncDB()
        item_id = db.seed(topic="best programming language for beginners")
        result = _run_handle_failed(
            db, [{"id": item_id, "topic": "best programming language for beginners",
                  "resolution": None}])
        row = db.row(item_id)
        assert result.curiosity_dismissed == 1
        assert row["status"] == "dismissed"


class TestPendingAgeOut:
    def _seed_pending(self, db, *, topic, age_days):
        item_id = db.seed(topic=topic, status="pending", attempts=0)
        db._conn.execute(
            "UPDATE curiosity_queue SET created_at = datetime('now', ?) WHERE id=?",
            (f"-{age_days} days", item_id))
        db._conn.commit()
        return item_id

    def test_stale_pending_expires_at_14d(self):
        db = _FakeAsyncDB()
        old_id = self._seed_pending(db, topic="ancient open question", age_days=15)
        result = _run_handle_failed(db, [])
        row = db.row(old_id)
        assert row["status"] == "dismissed"
        assert (row["resolution"] or "").startswith("[expired")
        assert result.curiosity_dismissed == 1

    def test_fresh_pending_survives(self):
        db = _FakeAsyncDB()
        new_id = self._seed_pending(db, topic="fresh open question", age_days=3)
        result = _run_handle_failed(db, [])
        assert db.row(new_id)["status"] == "pending"
        assert result.curiosity_dismissed == 0


# ---------------------------------------------------------------------------
# 2. Daemon curiosity cooldown
# ---------------------------------------------------------------------------

class TestDaemonCuriosityCooldown:
    def _context(self):
        return {
            "idle_minutes": 60, "hours_since_dream": 1, "pending_events": 0,
            "alerts_unsent": 0, "recent_log": [], "recent_failures": 0,
            "critical_curiosity": 2, "pending_goals": 0,
        }

    def _daemon(self):
        from app.monitors.daemon import DaemonOrchestrator
        return DaemonOrchestrator(MagicMock())

    def test_researches_when_cold(self):
        d = self._daemon()
        decision = asyncio.run(
            d._decide(self._context(), "full"))
        assert decision == {"action": "research_curiosity"}

    def test_cooldown_blocks_immediate_rerun(self):
        d = self._daemon()
        d._last_curiosity_research = time.monotonic()  # just ran
        decision = asyncio.run(
            d._decide(self._context(), "full"))
        assert (decision or {}).get("action") != "research_curiosity"

    def test_cooldown_expires_after_30min(self):
        d = self._daemon()
        d._last_curiosity_research = time.monotonic() - 1801
        decision = asyncio.run(
            d._decide(self._context(), "full"))
        assert decision == {"action": "research_curiosity"}


# ---------------------------------------------------------------------------
# 3. search_health rolling signal
# ---------------------------------------------------------------------------

class TestSearchHealth:
    def setup_method(self):
        from app.tools import native_search
        native_search._RECENT_OUTCOMES.clear()

    teardown_method = setup_method

    def test_cold_start_assumes_healthy(self):
        from app.tools.native_search import search_health
        assert search_health() == 1.0

    def test_reflects_recent_outcomes(self):
        from app.tools import native_search
        native_search._RECENT_OUTCOMES.extend([True, False, False, False])
        assert native_search.search_health() == pytest.approx(0.25)

    def test_degraded_threshold(self):
        from app.tools import native_search
        native_search._RECENT_OUTCOMES.extend([False] * 9 + [True])
        assert native_search.search_health() < 0.25
