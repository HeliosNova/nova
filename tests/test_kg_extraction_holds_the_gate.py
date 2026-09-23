"""A monitor's KG extraction is model work; the residency slot waits for it
(2026-09-23).

The extraction of a finished digest ran as a detached task on the 27B.
At 01:59 UTC the nightly judge (gemma) took the card the second the last
digest released the gate, while that digest's extraction was still queued
on the 27B: the two models reloaded each other six times in half an hour.
The gated check now waits, bounded, for the extraction it started.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import app.monitors.heartbeat_loop as hb
from app.monitors.heartbeat_loop import HeartbeatLoop
from app.monitors.monitor_store import MonitorStore


@pytest.mark.asyncio
async def test_the_slot_waits_for_the_extraction_it_started():
    lp = object.__new__(HeartbeatLoop)
    lp._kg_task_by_monitor = {}
    gate = asyncio.Event()

    async def _extract():
        await gate.wait()
        return "done"

    task = asyncio.create_task(_extract())
    lp._kg_task_by_monitor[7] = task
    waiter = asyncio.create_task(lp._await_kg_extraction(7))
    await asyncio.sleep(0.05)
    assert not waiter.done()                        # still holding the slot
    gate.set()
    await waiter
    assert task.done() and 7 not in lp._kg_task_by_monitor


@pytest.mark.asyncio
async def test_no_extraction_means_no_wait():
    lp = object.__new__(HeartbeatLoop)
    lp._kg_task_by_monitor = {}
    await asyncio.wait_for(lp._await_kg_extraction(1), timeout=0.5)


@pytest.mark.asyncio
async def test_a_stuck_extraction_releases_the_slot_after_the_bound_and_keeps_running():
    lp = object.__new__(HeartbeatLoop)
    lp._kg_task_by_monitor = {}
    never = asyncio.Event()
    task = asyncio.create_task(never.wait())
    lp._kg_task_by_monitor[3] = task
    await asyncio.wait_for(lp._await_kg_extraction(3, timeout=0.1), timeout=1.0)
    assert not task.done()                          # not cancelled, just no longer held
    task.cancel()


@pytest.mark.asyncio
async def test_a_query_monitor_registers_its_extraction(db, monkeypatch):
    """The registry is what the gated wrapper waits on."""
    import app.core.brain as brain
    started = asyncio.Event()

    async def _fake_extract(*a, **k):
        started.set()

    monkeypatch.setattr(brain, "get_services", lambda: SimpleNamespace(kg=object()))
    monkeypatch.setattr(brain, "_extract_kg_triples", _fake_extract)
    store = MonitorStore(db)
    loop = HeartbeatLoop(store)
    mid = store.create("Domain Study: Probe", "query", {"query": "q"}, schedule_seconds=3600,
                       cooldown_minutes=0, notify_condition="always")
    monitor = store.get(mid)
    with patch.object(loop, "_execute_check", new_callable=AsyncMock) as mock_exec, \
         patch.object(loop, "_send_alert", new_callable=AsyncMock) as mock_send:
        mock_exec.return_value = "A briefing long enough to be banked. " * 10
        mock_send.return_value = True
        await loop._check_monitor(monitor)
        assert mid in loop._kg_task_by_monitor
        await loop._await_kg_extraction(mid)
    assert started.is_set()
    assert mid not in loop._kg_task_by_monitor
