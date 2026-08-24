"""Daemon loop supervisor (audit 2026-08-22).

The daemon orchestrator's asyncio task had no done-callback: if _loop died on an
unhandled exception, proactive dream/curiosity/goal work stopped silently (the
KG dead-man switches live in the heartbeat loop, not the daemon). These tests
exercise _on_task_done directly: it must respawn only on an unexpected death
while running, never on an intentional cancel or after stop().
"""

import asyncio

import pytest

from app.monitors.daemon import DaemonOrchestrator


@pytest.mark.asyncio
async def test_respawns_loop_on_unhandled_death(monkeypatch):
    orch = DaemonOrchestrator(db=None)
    orch._running = True
    spawned = {"n": 0}

    async def fake_loop():
        spawned["n"] += 1
        orch._running = False  # stop after one respawn so the test can't loop forever

    monkeypatch.setattr(orch, "_loop", fake_loop)

    async def boom():
        raise RuntimeError("loop crashed")

    crashed = asyncio.ensure_future(boom())
    try:
        await crashed
    except RuntimeError:
        pass

    orch._on_task_done(crashed)
    await asyncio.sleep(0.05)  # let the respawned task run

    assert spawned["n"] == 1, "supervisor must respawn the loop after an unhandled death"
    assert orch._task is not crashed


@pytest.mark.asyncio
async def test_no_respawn_after_stop(monkeypatch):
    orch = DaemonOrchestrator(db=None)
    orch._running = False  # intentionally stopped
    spawned = {"n": 0}

    async def fake_loop():
        spawned["n"] += 1

    monkeypatch.setattr(orch, "_loop", fake_loop)

    async def boom():
        raise RuntimeError("x")

    crashed = asyncio.ensure_future(boom())
    try:
        await crashed
    except RuntimeError:
        pass

    orch._on_task_done(crashed)
    await asyncio.sleep(0.02)
    assert spawned["n"] == 0, "a stopped daemon must not respawn"


@pytest.mark.asyncio
async def test_no_respawn_on_cancel(monkeypatch):
    orch = DaemonOrchestrator(db=None)
    orch._running = True
    spawned = {"n": 0}

    async def fake_loop():
        spawned["n"] += 1

    monkeypatch.setattr(orch, "_loop", fake_loop)

    async def sleeper():
        await asyncio.sleep(10)

    cancelled = asyncio.ensure_future(sleeper())
    cancelled.cancel()
    try:
        await cancelled
    except asyncio.CancelledError:
        pass

    orch._on_task_done(cancelled)
    await asyncio.sleep(0.02)
    assert spawned["n"] == 0, "cancellation is an intentional stop — no respawn"
