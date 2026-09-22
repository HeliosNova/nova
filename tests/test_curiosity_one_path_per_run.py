"""A curiosity run takes one research path, in the lane that path uses
(2026-09-22).

Evidence-first questions (dossier sources) hold the 27B; think() questions
hold the 9B. Live 18:04 UTC one hourly run alternated them, then handed a
27B-resident card to the 9B Storyline Tracker beside it: four model loads
inside a batch that had budgeted one. Now the first pick sets the run's
path, later picks stay on it, and the tick classes the run by that path so
an evidence-first run rides the digest lane.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.monitors.heartbeat_loop as hb
from app.core.curiosity import CuriosityQueue
from app.monitors.monitor_store import Monitor

EV = hb._EVIDENCE_FIRST_SOURCES


def test_get_next_can_be_held_to_a_source_group(db):
    q = CuriosityQueue(db)
    a = q.add("What did the Fed decide at its September meeting?", source="dossier_open_question", urgency=0.6)
    b = q.add("Re-research and verify: rate limiter design", source="quiz_feedback", urgency=0.7)
    assert a > 0 and b > 0
    assert q.get_next().id == b                                   # urgency wins unconstrained
    assert q.get_next(sources=EV).id == a
    assert q.get_next(exclude_sources=EV).id == b
    assert q.get_next(sources=EV, exclude_ids={a}) is None


def _runner(monkeypatch, sources_seq):
    """A loop whose per-item research is scripted: each call reports the
    source of the item it 'picked' and records the path filter it was given."""
    lp = object.__new__(hb.HeartbeatLoop)
    calls: list = []
    seq = list(sources_seq)

    async def _one(_self, _svc, tried=None, *, sources=None, exclude_sources=None):
        calls.append({"sources": sources, "exclude_sources": exclude_sources})
        if not seq:
            return "[No pending curiosity items — skipped]"
        _self._last_curiosity_source = seq.pop(0)
        return "CURIOSITY RESOLVED | x"

    monkeypatch.setattr(hb.HeartbeatLoop, "_research_one_curiosity", _one)
    import app.core.brain as brain
    monkeypatch.setattr(brain, "get_services", lambda: SimpleNamespace(curiosity=object()))
    return lp, calls


@pytest.mark.asyncio
async def test_an_evidence_first_run_stays_evidence_first(monkeypatch):
    lp, calls = _runner(monkeypatch, ["dossier_open_question"] * 3)
    out = await lp._execute_curiosity_research({})
    assert calls[0] == {"sources": None, "exclude_sources": None}
    assert calls[1]["sources"] == EV and calls[2]["sources"] == EV
    assert "3/3 resolved" in out


@pytest.mark.asyncio
async def test_a_think_run_leaves_evidence_items_for_next_hour(monkeypatch):
    lp, calls = _runner(monkeypatch, ["quiz_feedback", "agent_failure"])
    out = await lp._execute_curiosity_research({})
    assert calls[1]["exclude_sources"] == EV and calls[2]["exclude_sources"] == EV
    assert "2/2 resolved" in out            # the exhausted path is not counted as an item


def test_the_tick_classes_a_curiosity_run_by_its_path():
    lp = object.__new__(hb.HeartbeatLoop)
    mon = Monitor(id=1, name="Curiosity Research", check_type="curiosity", check_config={},
                  schedule_seconds=3600, enabled=True, cooldown_minutes=0,
                  notify_condition="on_change", last_check_at=None, last_alert_at=None,
                  last_result=None, created_at="2026-01-01T00:00:00")
    assert lp._monitor_class(mon) == "other"                      # no peek yet: 9B, as before
    lp._curiosity_class_hint = "digest"
    assert lp._monitor_class(mon) == "digest"
    lp._curiosity_class_hint = "other"
    assert lp._monitor_class(mon) == "other"


@pytest.mark.asyncio
async def test_the_peek_reads_the_next_item_source(monkeypatch):
    lp = object.__new__(hb.HeartbeatLoop)
    mon = SimpleNamespace(check_type="curiosity")
    import app.core.brain as brain

    def _svc(src):
        q = SimpleNamespace(get_next=lambda **kw: SimpleNamespace(source=src) if src else None)
        return SimpleNamespace(curiosity=q)

    monkeypatch.setattr(brain, "get_services", lambda: _svc("dossier_open_question"))
    assert await lp._peek_curiosity_class([mon]) == "digest"
    monkeypatch.setattr(brain, "get_services", lambda: _svc("quiz_feedback"))
    assert await lp._peek_curiosity_class([mon]) == "other"
    monkeypatch.setattr(brain, "get_services", lambda: _svc(None))
    assert await lp._peek_curiosity_class([mon]) is None
    assert await lp._peek_curiosity_class([SimpleNamespace(check_type="query")]) is None
