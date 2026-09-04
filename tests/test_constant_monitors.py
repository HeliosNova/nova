"""A monitor that always says the same thing is spending a slot to say nothing.

Found 2026-09-04 by one ad-hoc query: Training Job Watch had returned the
identical string "no training history yet" 101 times in 14 days, hourly, for a
weight trainer archived in June — while the schedule was delivering 37% of what
it demanded. Nothing watched for that, because a pathway check asks whether a
writer still writes, and this one wrote faithfully every hour.

Deliberately a REPORT and not an alarm. A constant monitor can be legitimately
idle for want of input; what it always is, is a candidate for a longer cadence.
"""
from __future__ import annotations

from app.monitors.pathways import CONSTANT_MIN_RUNS, constant_monitors


class _DB:
    """Returns rows shaped like the join in constant_monitors."""

    def __init__(self, rows):
        self._rows = [{"name": n, "value": v} for n, v in rows]

    def fetchall(self, _q, _a=()):
        return self._rows


def _same(name, n, text="nothing to report"):
    return [(name, text)] * n


def test_a_monitor_repeating_itself_is_reported():
    rows = _same("Training Job Watch", CONSTANT_MIN_RUNS + 2)
    found = constant_monitors(_DB(rows))
    assert [f["name"] for f in found] == ["Training Job Watch"]
    assert found[0]["runs"] == CONSTANT_MIN_RUNS + 2
    assert found[0]["distinct"] == 1


def test_a_monitor_whose_output_changes_is_not_reported():
    rows = [("Domain Study: Finance", f"briefing number {i}") for i in range(20)]
    assert constant_monitors(_DB(rows)) == []


def test_one_differing_run_clears_a_monitor():
    """Constant means constant. A single real change is a working monitor."""
    rows = _same("Ollama Latency", CONSTANT_MIN_RUNS + 5) + [("Ollama Latency", "slow")]
    assert constant_monitors(_DB(rows)) == []


def test_too_few_runs_is_not_yet_evidence():
    """Three identical runs is a quiet week, not a dead monitor."""
    rows = _same("KG Growth Rate", CONSTANT_MIN_RUNS - 1)
    assert constant_monitors(_DB(rows)) == []


def test_the_worst_offender_comes_first():
    rows = _same("Chatty", CONSTANT_MIN_RUNS + 1) + _same("Worst", CONSTANT_MIN_RUNS + 40)
    assert [f["name"] for f in constant_monitors(_DB(rows))] == ["Worst", "Chatty"]


def test_a_null_value_does_not_crash_the_scan():
    db = _DB([])
    db._rows = [{"name": "X", "value": None}] * (CONSTANT_MIN_RUNS + 1)
    found = constant_monitors(db)
    assert found and found[0]["name"] == "X"


def test_a_broken_store_returns_nothing_rather_than_raising():
    """This runs inside the liveness monitor; it must never be the thing that
    breaks the canary."""
    class _Broken:
        def fetchall(self, *_a, **_k):
            raise RuntimeError("no such table")

    assert constant_monitors(_Broken()) == []


def test_the_liveness_report_surfaces_it(monkeypatch):
    from app.monitors import pathways as pw

    monkeypatch.setattr(pw, "snapshot", lambda db, **kw: [
        {"name": "kg_growth", "verdict": "alive", "age_hours": 1.0, "window_hours": 48.0}])
    monkeypatch.setattr(pw, "schedule_pressure", lambda db, **kw: {
        "ratio": 0.9, "delivered": 100, "demanded": 111, "starved": []})
    monkeypatch.setattr(pw, "constant_monitors", lambda db, **kw: [
        {"name": "Training Job Watch", "runs": 101, "distinct": 1}])

    status, summary, fields = pw.liveness_report(_DB([]))
    assert status == "info" and pw.HEALTHY_MARKER in summary
    assert "Training Job Watch (101x identical)" in fields["saying_nothing"]
