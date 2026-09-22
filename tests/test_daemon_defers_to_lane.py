"""The daemon's opportunistic LLM work defers to the heartbeat's residency
lane (2026-09-22).

The tick gates monitors by the model they drive so the card holds one class
at a time. The daemon enters from outside the tick: twenty seconds into a
three-wide 27B digest batch it launched a curiosity think() on the 9B and the
card swapped six times in five minutes. The loop now publishes its gate and
the daemon asks lane_busy() before curiosity or dream.
"""
from __future__ import annotations

import pytest

import app.monitors.heartbeat_loop as hb
from app.monitors import daemon as daemon_mod


@pytest.mark.asyncio
async def test_lane_busy_follows_the_gate():
    lp = object.__new__(hb.HeartbeatLoop)
    assert lp.lane_busy() is False                     # no tick has run
    gate = hb._ClassGate({"digest": 3}, default=1)
    lp._lane_gate = gate
    assert lp.lane_busy() is False
    await gate.acquire("digest")
    assert lp.lane_busy() is True
    await gate.acquire("digest")
    await gate.release()
    assert lp.lane_busy() is True                      # one digest still holds it
    await gate.release()
    assert lp.lane_busy() is False


def _daemon(monkeypatch, busy: bool):
    class _HB:
        def lane_busy(self):
            return busy

    class _Svc:
        heartbeat = _HB()

    import app.core.brain as brain
    monkeypatch.setattr(brain, "get_services", lambda: _Svc())
    d = daemon_mod.DaemonOrchestrator.__new__(daemon_mod.DaemonOrchestrator)
    d._last_curiosity_research = None
    d._dream_running = False
    return d


def _context(**over):
    ctx = {"alerts_unsent": 0, "idle_minutes": 60, "hours_since_dream": 1,
           "critical_curiosity": 2, "pending_events": 0, "recent_failures": 0}
    ctx.update(over)
    return ctx


@pytest.mark.asyncio
async def test_curiosity_waits_for_an_idle_lane(monkeypatch):
    busy = _daemon(monkeypatch, busy=True)
    assert await busy._decide(_context(), daemon_mod.BUDGET_FULL) is None
    idle = _daemon(monkeypatch, busy=False)
    out = await idle._decide(_context(), daemon_mod.BUDGET_FULL)
    assert out == {"action": "research_curiosity"}


@pytest.mark.asyncio
async def test_dream_waits_for_an_idle_lane(monkeypatch):
    ctx = _context(hours_since_dream=13, critical_curiosity=0)
    busy = _daemon(monkeypatch, busy=True)
    assert await busy._decide(ctx, daemon_mod.BUDGET_FULL) is None
    idle = _daemon(monkeypatch, busy=False)
    assert (await idle._decide(ctx, daemon_mod.BUDGET_FULL))["action"] == "dream"


def test_no_heartbeat_means_not_busy(monkeypatch):
    class _Svc:
        heartbeat = None

    import app.core.brain as brain
    monkeypatch.setattr(brain, "get_services", lambda: _Svc())
    d = daemon_mod.DaemonOrchestrator.__new__(daemon_mod.DaemonOrchestrator)
    assert d._lane_busy() is False
