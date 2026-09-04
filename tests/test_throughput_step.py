"""A step down in delivered work, noticed by the system rather than by a person.

Runs per active hour fell 6.5 -> 4.1 on 2026-08-28 and stayed there for a week
with nothing noticing. Nothing was broken in the way monitoring looks for: the
app was up around 24 hours a day, digests kept their length, their sources and
their judge scores, and every monitor merely ran a little less often. Schedule
pressure could not see it either, because it reports demand against delivery and
has no before.

Per ACTIVE hour, not per day, so a real outage reads as downtime rather than as
a cost regression — the distinction that matters, because the fix for one is
restarting a container and the fix for the other is finding what got expensive.
"""
from __future__ import annotations

from app.monitors.pathways import THROUGHPUT_STEP_DROP, throughput_step


class _DB:
    def __init__(self, days):
        # days: list of (label, runs, active_hours)
        self._rows = [{"day": d, "runs": r, "hrs": h} for d, r, h in days]

    def fetchall(self, _q, _a=()):
        return self._rows


def _flat(n, rate=6.0, hrs=24):
    return [(f"2026-08-{i + 1:02d}", int(rate * hrs), hrs) for i in range(n)]


def _stepped(n, before=6.5, after=4.0, hrs=24):
    out = []
    for i in range(n):
        rate = before if i < n // 2 else after
        out.append((f"2026-08-{i + 1:02d}", int(rate * hrs), hrs))
    return out


def test_a_flat_record_is_not_a_step():
    got = throughput_step(_DB(_flat(15) + [("today", 10, 4)]))
    assert got and not got["stepped_down"], got


def test_a_real_step_down_is_caught():
    got = throughput_step(_DB(_stepped(16) + [("today", 10, 4)]))
    assert got["stepped_down"]
    assert got["before"] > got["after"]
    assert got["change"] <= -THROUGHPUT_STEP_DROP


def test_today_is_excluded_so_a_partial_day_is_not_a_regression():
    """A day that is four hours old always reads as a fall."""
    full = _flat(15)
    partial = full + [("2026-09-04", 8, 2)]     # same rate, but tiny sample
    assert throughput_step(_DB(partial))["stepped_down"] is False


def test_downtime_is_not_mistaken_for_cost():
    """Half a day offline halves the RUNS but not the runs per ACTIVE hour."""
    days = _flat(15)
    days[10] = ("2026-08-11", 6 * 12, 12)       # outage: 12 active hours
    got = throughput_step(_DB(days + [("today", 10, 4)]))
    assert not got["stepped_down"], got


def test_too_short_a_record_says_nothing():
    assert throughput_step(_DB(_flat(5))) is None


def test_a_broken_store_returns_none_rather_than_raising():
    class _Broken:
        def fetchall(self, *_a, **_k):
            raise RuntimeError("no such table")

    assert throughput_step(_Broken()) is None


def test_a_day_with_no_active_hours_is_skipped_not_divided_by():
    days = _flat(15) + [("2026-08-16", 0, 0), ("today", 10, 4)]
    assert throughput_step(_DB(days)) is not None


def test_the_liveness_report_surfaces_a_step(monkeypatch):
    from app.monitors import pathways as pw

    monkeypatch.setattr(pw, "snapshot", lambda db, **kw: [
        {"name": "kg_growth", "verdict": "alive", "age_hours": 1.0, "window_hours": 48.0}])
    monkeypatch.setattr(pw, "schedule_pressure", lambda db, **kw: {
        "ratio": 0.9, "delivered": 100, "demanded": 111, "starved": []})
    monkeypatch.setattr(pw, "constant_monitors", lambda db, **kw: [])
    monkeypatch.setattr(pw, "throughput_step", lambda db, **kw: {
        "before": 6.5, "after": 4.1, "change": -0.37, "days": 18, "stepped_down": True})

    _status, _summary, fields = pw.liveness_report(_DB([]))
    assert "6.5 -> 4.1 runs/active hour" in fields["throughput_step"]
    assert "costs more per run" in fields["throughput_step"]


def test_a_healthy_report_does_not_carry_the_step_field(monkeypatch):
    from app.monitors import pathways as pw

    monkeypatch.setattr(pw, "snapshot", lambda db, **kw: [
        {"name": "kg_growth", "verdict": "alive", "age_hours": 1.0, "window_hours": 48.0}])
    monkeypatch.setattr(pw, "schedule_pressure", lambda db, **kw: {
        "ratio": 0.9, "delivered": 100, "demanded": 111, "starved": []})
    monkeypatch.setattr(pw, "constant_monitors", lambda db, **kw: [])
    monkeypatch.setattr(pw, "throughput_step", lambda db, **kw: {
        "before": 6.5, "after": 6.4, "change": -0.02, "days": 18, "stepped_down": False})

    _status, _summary, fields = pw.liveness_report(_DB([]))
    assert "throughput_step" not in fields
