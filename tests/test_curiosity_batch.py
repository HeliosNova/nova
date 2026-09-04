"""Curiosity drains a batch per run, not one question (2026-09-04).

Measured that day: 50 questions pending, ~1.5 resolved a DAY, and a 7-to-9 day
wait between a dossier asking a question and curiosity answering it (created
08-23, resolved 09-02). The monitor declares an hourly cadence and delivers 20%
of it, so the queue was being drained about five times a day, one item each.

The scarce thing was the scheduling slot, not the research: the tick has already
paid to get here. So take several questions while we hold it, bounded by a wall
clock so one slow think() cannot hold the tick behind it.

These test the LOOP, with the per-item research stubbed. What one question does
is covered elsewhere; what this changed is how many get asked.
"""
from __future__ import annotations

import pytest

import app.monitors.heartbeat_loop as hb


class _Curiosity:
    pass


class _Svc:
    curiosity = _Curiosity()


def _loop(monkeypatch, outcomes):
    """A HeartbeatLoop whose per-item research returns canned results."""
    lp = object.__new__(hb.HeartbeatLoop)
    seen = {"n": 0}
    seq = list(outcomes)

    async def _one(_self, _svc):
        seen["n"] += 1
        return seq[min(seen["n"] - 1, len(seq) - 1)]

    monkeypatch.setattr(hb.HeartbeatLoop, "_research_one_curiosity", _one, raising=False)
    monkeypatch.setattr(hb, "get_services", lambda: _Svc(), raising=False)
    import app.core.brain as brain
    monkeypatch.setattr(brain, "get_services", lambda: _Svc())
    return lp, seen


RESOLVED = "CURIOSITY RESOLVED | topic=a | findings=x"
UNRESOLVED = "CURIOSITY UNRESOLVED | topic=b | reason=closure_check_failed"


@pytest.mark.asyncio
async def test_a_run_researches_the_whole_batch(monkeypatch):
    lp, seen = _loop(monkeypatch, [RESOLVED])
    out = await lp._execute_curiosity_research({})
    assert seen["n"] == hb._CURIOSITY_BATCH, "one question per run was the bottleneck"
    assert out.startswith("CURIOSITY BATCH |")
    assert f"{hb._CURIOSITY_BATCH}/{hb._CURIOSITY_BATCH} resolved" in out


@pytest.mark.asyncio
async def test_an_empty_queue_stops_the_run_immediately(monkeypatch):
    lp, seen = _loop(monkeypatch, ["[No pending curiosity items — skipped]"])
    out = await lp._execute_curiosity_research({})
    assert seen["n"] == 1, "an empty queue must not be asked three times"
    assert "No pending" in out


@pytest.mark.asyncio
async def test_a_dead_model_stops_the_run_instead_of_burning_the_batch(monkeypatch):
    lp, seen = _loop(monkeypatch, ["[Curiosity skipped — LLM unavailable, will retry]"])
    await lp._execute_curiosity_research({})
    assert seen["n"] == 1, "a down model must not be retried twice more in one run"


@pytest.mark.asyncio
async def test_unresolved_questions_do_not_stop_the_batch(monkeypatch):
    """A closure failure is a normal outcome; the next question still gets asked."""
    lp, seen = _loop(monkeypatch, [UNRESOLVED])
    out = await lp._execute_curiosity_research({})
    assert seen["n"] == hb._CURIOSITY_BATCH
    assert "0/" in out.split("|", 2)[1]


@pytest.mark.asyncio
async def test_a_single_item_run_keeps_its_old_shape(monkeypatch):
    """Change detection reads this string; a one-item run must not suddenly
    grow a batch header and read as a change."""
    lp, _ = _loop(monkeypatch, [RESOLVED])
    monkeypatch.setattr(hb, "_CURIOSITY_BATCH", 1)
    out = await lp._execute_curiosity_research({})
    assert out == RESOLVED


@pytest.mark.asyncio
async def test_the_wall_clock_bounds_the_run(monkeypatch):
    """One slow think() may overrun the budget; a batch must not compound it."""
    lp, seen = _loop(monkeypatch, [RESOLVED])
    clock = {"t": 0.0}
    monkeypatch.setattr(hb.time, "monotonic", lambda: clock["t"])

    orig = hb.HeartbeatLoop._research_one_curiosity

    async def _slow(self, svc):
        clock["t"] += 1000.0          # each item blows the budget
        return await orig(self, svc)

    monkeypatch.setattr(hb.HeartbeatLoop, "_research_one_curiosity", _slow, raising=False)
    await lp._execute_curiosity_research({})
    assert seen["n"] == 1, "the budget is checked between items"


@pytest.mark.asyncio
async def test_the_batch_size_is_configurable_per_monitor(monkeypatch):
    lp, seen = _loop(monkeypatch, [RESOLVED])
    await lp._execute_curiosity_research({"batch": 2})
    assert seen["n"] == 2
