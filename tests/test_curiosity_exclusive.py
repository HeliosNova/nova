"""One curiosity researcher at a time (2026-09-22).

The hourly Curiosity Research monitor and the daemon's opportunistic run both
call HeartbeatLoop._execute_curiosity_research. Live 07:46-08:03 UTC they ran
concurrently: the daemon's think() wanted the 9B, the monitor's evidence-first
pass wanted the 27B, Ollama evicted one for the other five times in fifteen
minutes, and the monitor's research timed out at 900 s having lost the card
for four of them. The daemon's run now yields when research is in progress;
the scheduled run waits its turn instead of skipping an hour.
"""
from __future__ import annotations

import asyncio

import pytest

import app.monitors.heartbeat_loop as hb
from app.monitors import daemon as daemon_mod


def _loop(monkeypatch, gate: asyncio.Event, calls: list):
    lp = object.__new__(hb.HeartbeatLoop)

    async def _inner(_self, cfg):
        calls.append(dict(cfg))
        await gate.wait()
        return f"done {len(calls)}"

    monkeypatch.setattr(hb.HeartbeatLoop, "_execute_curiosity_research_unlocked", _inner)
    return lp


@pytest.mark.asyncio
async def test_opportunistic_run_skips_while_a_run_is_active(monkeypatch):
    gate = asyncio.Event()
    calls: list = []
    lp = _loop(monkeypatch, gate, calls)
    scheduled = asyncio.create_task(lp._execute_curiosity_research({}))
    await asyncio.sleep(0)                      # let it take the lock
    out = await lp._execute_curiosity_research({"opportunistic": True})
    assert "already running" in out
    assert len(calls) == 1                      # the daemon's run never started
    gate.set()
    assert await scheduled == "done 1"


@pytest.mark.asyncio
async def test_scheduled_run_waits_for_an_opportunistic_one(monkeypatch):
    gate = asyncio.Event()
    calls: list = []
    lp = _loop(monkeypatch, gate, calls)
    first = asyncio.create_task(lp._execute_curiosity_research({"opportunistic": True}))
    await asyncio.sleep(0)
    second = asyncio.create_task(lp._execute_curiosity_research({}))
    await asyncio.sleep(0.05)
    assert len(calls) == 1                      # waiting for the lock, not skipped
    assert not second.done()
    gate.set()
    assert await first == "done 1"
    assert await second == "done 2"


@pytest.mark.asyncio
async def test_two_scheduled_runs_serialize(monkeypatch):
    gate = asyncio.Event()
    calls: list = []
    lp = _loop(monkeypatch, gate, calls)
    a = asyncio.create_task(lp._execute_curiosity_research({}))
    b = asyncio.create_task(lp._execute_curiosity_research({}))
    await asyncio.sleep(0.05)
    assert len(calls) == 1
    gate.set()
    assert await a == "done 1" and await b == "done 2"


@pytest.mark.asyncio
async def test_daemon_marks_its_run_opportunistic(monkeypatch):
    seen: list = []

    class _HB:
        async def _execute_curiosity_research(self, cfg):
            seen.append(dict(cfg))
            return "ok"

    class _Svc:
        curiosity = object()
        heartbeat = _HB()

    import app.core.brain as brain
    monkeypatch.setattr(brain, "get_services", lambda: _Svc())
    d = daemon_mod.DaemonOrchestrator.__new__(daemon_mod.DaemonOrchestrator)
    d._last_curiosity_research = None
    d._log = lambda *a, **k: None
    await d._research_curiosity()
    assert seen == [{"opportunistic": True}]
