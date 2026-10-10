"""Event-loop DB-write regression tests (2026-08-31).

The `_warn_if_event_loop` tripwire logged 22 distinct WRITE statements running
on the event-loop thread on 2026-08-31 alone (78 sites total that day; the
2026-06-11 lock-convoy incident and the 54h freeze are this class). The fix
moved each async→sync boundary onto a worker thread with asyncio.to_thread —
one wrap per call chain, not per statement.

These tests run the real code paths inside a live event loop against an
instrumented DB and assert (a) no write executes on the loop thread and
(b) the writes still happen (no over-threading that drops behavior).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from app.tools.base import EPHEMERAL_REQUEST


@pytest.fixture(autouse=True)
def reset_ephemeral():
    token = EPHEMERAL_REQUEST.set(False)
    yield
    EPHEMERAL_REQUEST.reset(token)


class LoopGuardDB:
    """Wraps a SafeDB; records which thread ran each write statement."""

    def __init__(self, real):
        self._real = real
        self.loop_thread_writes: list[str] = []
        self.loop_thread_id: int | None = None

    def _check(self, sql: str):
        if self.loop_thread_id is not None and threading.get_ident() == self.loop_thread_id:
            head = sql.lstrip()[:10].upper()
            if head.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE")):
                self.loop_thread_writes.append(sql[:80])

    def execute(self, sql, params=()):
        self._check(sql)
        return self._real.execute(sql, params)

    def executemany(self, sql, params_list):
        self._check(sql)
        return self._real.executemany(sql, params_list)

    def fetchone(self, sql, params=()):
        return self._real.fetchone(sql, params)

    def fetchall(self, sql, params=()):
        return self._real.fetchall(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def guard_db(db):
    # curiosity_queue is created by its store's DDL, not init_schema — seed
    # it before arming the loop guard.
    from app.core.curiosity import CuriosityQueue
    for stmt in CuriosityQueue._SCHEMA.strip().split(";"):
        if stmt.strip():
            db.execute(stmt)
    return LoopGuardDB(db)


class TestStorylineForecastOffLoop:
    @pytest.mark.asyncio
    async def test_update_story_records_forecast_off_loop(self, guard_db, monkeypatch):
        from unittest.mock import AsyncMock

        import app.core.storylines as sl
        guard_db.execute(
            "INSERT INTO storylines (story_key, title, summary, monitors_csv, "
            "update_count, last_updated) VALUES ('opec','OPEC supply','prior',"
            "'Energy',1,datetime('now'))")
        out = ("OPEC extended cuts through Q1.\nCHANGED: cuts extended\n"
               "FORECAST: Brent stays above $80 through November (confidence 0.6)")
        monkeypatch.setattr(sl, "_bg_invoke", AsyncMock(return_value=out))
        guard_db.loop_thread_id = threading.get_ident()
        res = await sl._update_story(
            guard_db, {"key": "opec", "title": "OPEC supply",
                       "developments": ["cuts extended at Vienna meeting"],
                       "monitors": ["Energy Watch"]})
        assert res is not None
        assert guard_db.loop_thread_writes == [], \
            f"writes ran on the event loop: {guard_db.loop_thread_writes}"
