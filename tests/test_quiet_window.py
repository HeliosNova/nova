"""Quiet window (2026-09-02): pause the LLM lane for a bounded time."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.monitors import quiet as Q
from app.monitors import heartbeat_loop as hb
from app.monitors.heartbeat_loop import HeartbeatLoop
from app.monitors.monitor_store import MonitorStore


def test_set_clear_and_expiry(db):
    assert Q.quiet_status(db) == {"active": False, "until": None, "remaining_minutes": 0.0, "reason": None}
    until = Q.set_quiet(db, 2, "priming A/B")
    st = Q.quiet_status(db)
    assert st["active"] and st["reason"] == "priming A/B" and 110 < st["remaining_minutes"] <= 120
    assert Q.quiet_until(db) == until.replace(microsecond=0)
    assert Q.clear_quiet(db) is True
    assert not Q.quiet_active(db)
    assert Q.clear_quiet(db) is False


def test_window_is_capped_and_expired_rows_read_as_inactive(db):
    until = Q.set_quiet(db, 999, "runaway")
    assert until - datetime.utcnow() <= timedelta(hours=24, seconds=5)
    db.execute("UPDATE system_state SET value = ? WHERE key = ?",
               ((datetime.utcnow() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"), Q.KEY_UNTIL))
    assert not Q.quiet_active(db)
    assert Q.quiet_status(db)["active"] is False


class _Shim:
    def __init__(self, real, stop):
        self._real, self._stop = real, stop

    def __getattr__(self, name):
        return getattr(self._real, name)

    async def sleep(self, delay, *a, **k):
        if delay == 10:
            return
        if delay >= 60:
            self._stop()
            return
        await self._real.sleep(min(delay, 0.01))


@pytest.mark.asyncio
async def test_tick_skips_the_llm_lane_while_quiet(monkeypatch):
    from app.database import get_db
    db = get_db()
    db.init_schema()
    store = MonitorStore(db)
    store.create("System Health", "system_health", {}, 7200, 60, "on_change")
    store.create("Storyline Tracker", "storyline", {}, 28800, 60, "on_change")
    store.create("Lesson Quiz", "quiz", {}, 21600, 60, "on_change")
    Q.set_quiet(db, 1, "test")

    loop = HeartbeatLoop(store)
    ran: list[str] = []

    async def _spy(monitor):
        ran.append(monitor.name)
    loop._check_monitor = _spy
    monkeypatch.setattr(hb, "asyncio", _Shim(asyncio, lambda: setattr(loop, "_running", False)))
    loop._running = True
    await asyncio.wait_for(loop._loop(), timeout=30)
    assert ran == ["System Health"], ran          # fast lane only
    # nothing advanced for the skipped monitors — they are still due
    assert {m.name for m in store.get_due()} >= {"Storyline Tracker", "Lesson Quiz"}

    Q.clear_quiet(db)
    ran.clear()
    loop._running = True
    await asyncio.wait_for(loop._loop(), timeout=30)
    assert set(ran) == {"System Health", "Storyline Tracker", "Lesson Quiz"}


def test_api_routes_registered_before_the_id_routes():
    from app.api.monitors import router
    paths = [(r.path, tuple(sorted(r.methods or []))) for r in router.routes]
    names = [p for p, _ in paths]
    assert names.index("/monitors/quiet") < names.index("/monitors/{monitor_id}")
    assert ("/monitors/quiet", ("DELETE",)) in paths and ("/monitors/quiet", ("POST",)) in paths


def test_status_schema_carries_quiet_until():
    from app.schema import StatusResponse
    assert StatusResponse().quiet_until is None
    assert StatusResponse(quiet_until="2026-09-02 09:00:00").quiet_until
