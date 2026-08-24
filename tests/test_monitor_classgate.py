"""_ClassGate — model-aware monitor concurrency (2026-08-12).

Digest-class (27B) monitors may overlap at width 2 so one digest's network
gather / CPU MiniCheck phases fill the GPU idle left by another's synthesis;
different classes (9B brain monitors) NEVER overlap a digest — 27B+9B
co-residency exceeds the 24GB card (the documented thrash ceiling). Fairness
is drain-and-switch: a waiting different-class monitor blocks later same-class
entrants from jumping the FIFO.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.monitors.heartbeat_loop import HeartbeatLoop, _ClassGate


@pytest.mark.asyncio
async def test_same_class_overlaps_to_width():
    gate = _ClassGate({"digest": 2})
    await gate.acquire("digest")
    # a second digest is admitted without waiting
    await asyncio.wait_for(gate.acquire("digest"), timeout=1)
    # a third queues (width 2)
    third = asyncio.create_task(gate.acquire("digest"))
    await asyncio.sleep(0.05)
    assert not third.done()
    await gate.release()
    await asyncio.wait_for(third, timeout=1)
    await gate.release()
    await gate.release()


@pytest.mark.asyncio
async def test_cross_class_never_overlaps():
    gate = _ClassGate({"digest": 2})
    await gate.acquire("digest")
    other = asyncio.create_task(gate.acquire("other"))
    await asyncio.sleep(0.05)
    assert not other.done()          # blocked while a digest runs
    await gate.release()
    await asyncio.wait_for(other, timeout=1)
    # and while "other" runs, a digest is blocked
    digest = asyncio.create_task(gate.acquire("digest"))
    await asyncio.sleep(0.05)
    assert not digest.done()
    await gate.release()
    await asyncio.wait_for(digest, timeout=1)
    await gate.release()


@pytest.mark.asyncio
async def test_drain_and_switch_fairness():
    # A waiting other-class monitor must not be starved by a stream of
    # same-class entrants arriving behind it.
    gate = _ClassGate({"digest": 2})
    await gate.acquire("digest")
    order: list[str] = []

    async def _entrant(cls, tag):
        await gate.acquire(cls)
        order.append(tag)

    other = asyncio.create_task(_entrant("other", "other"))
    await asyncio.sleep(0.02)                      # "other" is queued first
    late = asyncio.create_task(_entrant("digest", "late-digest"))
    await asyncio.sleep(0.05)
    # width allows a second digest, but it must NOT jump the queued "other"
    assert not late.done()
    await gate.release()                           # drain the running digest
    await asyncio.wait_for(other, timeout=1)
    assert order == ["other"]
    await gate.release()                           # "other" completes
    await asyncio.wait_for(late, timeout=1)
    assert order == ["other", "late-digest"]
    await gate.release()


def test_monitor_class_routing():
    lp = HeartbeatLoop.__new__(HeartbeatLoop)
    mk = lambda name, ct="query": SimpleNamespace(name=name, check_type=ct)
    assert lp._monitor_class(mk("Domain Study: Finance")) == "digest"
    assert lp._monitor_class(mk("Auto: Rust Language")) == "digest"
    # feed-backed query monitors route through the deep-research runner —
    # including Morning Check-in, which has curated feeds in _FEEDS
    assert lp._monitor_class(mk("World Awareness")) == "digest"
    assert lp._monitor_class(mk("Morning Check-in")) == "digest"
    # feedless brain.think query monitors and non-query monitors stay exclusive
    assert lp._monitor_class(mk("[Reminder]: call the dentist")) == "other"
    assert lp._monitor_class(mk("Lesson Quiz", ct="quiz")) == "other"
    assert lp._monitor_class(mk("Knowledge Consolidation", ct="consolidation")) == "other"
