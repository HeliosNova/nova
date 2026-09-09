"""A tick dispatches the most overdue few, then looks again (2026-09-09).

A tick computed `due` once and awaited the WHOLE slow batch before scanning
again. After a restart that is 30-50 monitors, three digests wide at 30-60
minutes each, so the loop did not look at what had become due for hours:
measured 2026-09-08 00:14-01:16 UTC, "26 monitor(s) due -> batch of 26",
and the hourly Curiosity Research plus the 8-hourly Storyline Tracker, both
due at 01:09, had not run by 01:16 and would wait behind the rest of the
batch. That is why the hourly lane delivers ~27% of its declared cadence and
every 4-8h monitor reads "a bit late": the loop's granularity, not the model.

`get_due` already orders by overdue ratio and `_class_floor_order` promotes
starved 9B-class monitors to the front, so the fix is to dispatch only the
first `_MAX_SLOW_PER_TICK` of that order per tick and let the next scan pick
up whatever became due meanwhile. The class gate still forbids cross-class
overlap, so this costs no extra model swaps; it only shortens how long a
newly-due monitor waits for its turn.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.monitors import heartbeat_loop as hb

ROOT = Path(__file__).resolve().parents[1]


class _Shim:
    """The loop's own sleeps trimmed to one tick (same trick as the e2e test)."""

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


def _monitor(i, name, check_type, schedule_s, overdue_ratio):
    last = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=schedule_s * overdue_ratio)
    return SimpleNamespace(id=i, name=name, check_type=check_type, schedule_seconds=schedule_s,
                           last_check_at=last.strftime("%Y-%m-%d %H:%M:%S"), enabled=True,
                           check_config={}, category="content", notify_condition="always")


def _due_list():
    """20 slow monitors, most overdue first (get_due's contract): 18 digests
    plus one starved hourly 9B monitor and one merely-due one."""
    mons = [_monitor(100 + i, f"Domain Study: Area {i:02d}", "query", 8 * 3600, 3.0 - i * 0.1)
            for i in range(18)]
    mons.append(_monitor(1, "Curiosity Research", "curiosity", 3600, 2.0))     # starved 'other'
    mons.append(_monitor(2, "Lesson Quiz", "quiz", 6 * 3600, 1.05))          # just due
    return sorted(mons, key=lambda m: -(3600 * 8 if m.name.startswith("Domain") else m.schedule_seconds))


@pytest.mark.asyncio
async def test_a_tick_dispatches_only_the_most_overdue_few(monkeypatch):
    due = _due_list()
    checked: list[str] = []

    async def _check(_self, monitor):
        checked.append(monitor.name)

    async def _noop(_self, *a, **k):
        return None

    monkeypatch.setattr(hb.HeartbeatLoop, "_check_monitor", _check)
    monkeypatch.setattr(hb.HeartbeatLoop, "_flush_digest", _noop)
    monkeypatch.setattr(hb.HeartbeatLoop, "_recover_pending_deliveries", _noop)
    import app.monitors.quiet as quiet
    monkeypatch.setattr(quiet, "quiet_status", lambda db: {"active": False})
    import app.core.llm as llm
    monkeypatch.setattr(llm, "interactive_active", lambda: False)

    store = SimpleNamespace(get_due=lambda: list(due), get_due_instructions=lambda: [],
                            _db=SimpleNamespace(fetchall=lambda *a, **k: []))
    loop = object.__new__(hb.HeartbeatLoop)
    loop.store = store
    loop._digest_enabled = False
    loop._digest_buffer = []
    loop._kg_bg_tasks = set()
    loop._discord = loop._telegram = loop._whatsapp = loop._signal = None
    monkeypatch.setattr(hb, "asyncio", _Shim(asyncio, lambda: setattr(loop, "_running", False)))
    loop._running = True

    await asyncio.wait_for(loop._loop(), timeout=30)

    cap = hb._MAX_SLOW_PER_TICK
    assert len(checked) == cap, f"expected {cap} dispatched, got {len(checked)}: {checked}"
    assert "Curiosity Research" in checked, "the starved hourly monitor must make the cut"
    digests = [n for n in checked if n.startswith("Domain Study")]
    assert digests == [f"Domain Study: Area {i:02d}" for i in range(len(digests))], \
        "the digests dispatched must be the most overdue ones, in order"


def test_the_cap_leaves_room_for_two_digest_rounds():
    """Two rounds of the digest width plus a couple of 9B-class monitors: enough
    work to keep the card busy between scans, small enough that a newly-due
    hourly monitor waits for one round, not for the whole backlog."""
    assert hb._MAX_SLOW_PER_TICK == 2 * hb._MAX_CONCURRENT_DIGEST_MONITORS + 2


def test_the_cap_is_applied_after_the_floor_and_before_batching():
    src = (ROOT / "app" / "monitors" / "heartbeat_loop.py").read_text(encoding="utf-8")
    i = src.index("slow = _class_floor_order(")
    j = src.index("_MAX_SLOW_PER_TICK]")
    k = src.index("slow = _batch_by_class(slow, self._monitor_class)")
    assert i < j < k, "cap the floor-ordered list, then group what remains by class"
    assert re.search(r"most overdue this tick", src), "the tick must say when it holds work back"
