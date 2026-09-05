"""One readout that speaks whether or not anything is wrong (2026-09-05).

The 2026-08-28 throughput regression ran for a WEEK. Nothing was broken in the
way monitoring looks for — the app was up, digests kept their length and their
scores, every monitor merely ran a little less often. The signal existed, spread
across twenty system monitors and five operator scripts, each reporting its own
slice against its own threshold, and none of them owned "is Nova delivering less
than it was".

So the two properties worth testing are not the numbers, which come from
functions tested elsewhere. They are: it ALWAYS delivers, and the things that
crossed a bar are named in the order worth looking at.
"""
from __future__ import annotations

import pytest

from app.monitors import engineering_report as er


class _DB:
    def fetchone(self, _q, _a=()):
        return {"c": 0, "oldest": None, "d": None}

    def fetchall(self, _q, _a=()):
        return []


def _quiet(monkeypatch):
    """Every sub-measurement healthy, so only the report's own logic is under test."""
    monkeypatch.setattr(er, "cascade_support", lambda *a, **k: None)
    import app.monitors.pathways as pw
    monkeypatch.setattr(pw, "throughput_step", lambda db, **k: {
        "before": 6.5, "after": 6.4, "change": -0.02, "days": 18, "stepped_down": False})
    monkeypatch.setattr(pw, "schedule_pressure", lambda db, **k: {
        "ratio": 0.95, "delivered": 100, "demanded": 105, "starved": []})
    monkeypatch.setattr(pw, "constant_monitors", lambda db, **k: [])
    monkeypatch.setattr(pw, "snapshot", lambda db, **k: [
        {"name": "kg_growth", "verdict": "alive"}])
    import app.monitors.health_checks as hc
    monkeypatch.setattr(hc, "entail_gate_totals", lambda d, *a, **k: (0, 0))
    import app.core.forecasts as fc
    monkeypatch.setattr(fc, "calibration", lambda db, **k: None)


def test_it_speaks_when_nothing_is_wrong(monkeypatch):
    """The whole point. A report that only appears on a threshold is another
    thing that can stay quiet for a week."""
    _quiet(monkeypatch)
    status, summary, fields = er.build_report(_DB())
    assert status == "info"
    assert "nothing crossed a bar" in summary
    assert fields, "a quiet day still has numbers worth carrying"
    assert not [k for k in fields if k.startswith("look_at")]


def test_a_dead_pathway_leads_the_list_and_is_an_error(monkeypatch):
    _quiet(monkeypatch)
    import app.monitors.pathways as pw
    monkeypatch.setattr(pw, "snapshot", lambda db, **k: [
        {"name": "kg_growth", "verdict": "alive"},
        {"name": "storylines", "verdict": "dead"}])
    monkeypatch.setattr(pw, "throughput_step", lambda db, **k: {
        "before": 6.5, "after": 4.0, "change": -0.38, "days": 18, "stepped_down": True})
    status, summary, fields = er.build_report(_DB())
    assert status == "error"
    assert "storylines" in summary, "a dead writer outranks a slow one"
    assert fields["look_at_1"].startswith("pathway(s) DEAD")
    assert list(fields)[0] == "look_at_1", "actionable fields must come FIRST"


def test_the_regression_that_started_all_this_would_be_named(monkeypatch):
    _quiet(monkeypatch)
    import app.monitors.pathways as pw
    monkeypatch.setattr(pw, "throughput_step", lambda db, **k: {
        "before": 6.5, "after": 4.1, "change": -0.37, "days": 18, "stepped_down": True})
    status, summary, fields = er.build_report(_DB())
    assert status == "warning"
    assert "delivery is DOWN" in fields["look_at_1"]
    assert "costs more per run" in fields["look_at_1"]


def test_a_drop_rate_jump_is_called_out(monkeypatch):
    _quiet(monkeypatch)
    import app.monitors.health_checks as hc
    monkeypatch.setattr(hc, "entail_gate_totals",
                        lambda d, *a, **k: (8000, 4240) if d == 7 else (500, 400))
    _status, _summary, fields = er.build_report(_DB())
    assert any("jumped to 80%" in str(v) for k, v in fields.items()
               if k.startswith("look_at"))


def test_ordinary_daily_variation_is_not_called_out(monkeypatch):
    _quiet(monkeypatch)
    import app.monitors.health_checks as hc
    monkeypatch.setattr(hc, "entail_gate_totals",
                        lambda d, *a, **k: (8000, 4240) if d == 7 else (500, 285))
    status, _summary, fields = er.build_report(_DB())
    assert status == "info"
    assert not [k for k in fields if k.startswith("look_at")]


def test_a_broken_sub_measurement_does_not_kill_the_report(monkeypatch):
    """It runs inside a monitor; it must never be the thing that breaks."""
    _quiet(monkeypatch)

    class _Broken(_DB):
        def fetchone(self, _q, _a=()):
            raise RuntimeError("no such table")

    status, _summary, fields = er.build_report(_Broken())
    assert status in ("info", "warning", "error")
    assert "pathways" in fields


# ---------------------------------------------------------------------------
# cascade_support: the pre-registered check for the 2026-09-04 chrome rules.
# Navigation menus were winning evidence windows the article should have won,
# so removing them should RAISE narrow support. Nothing parsed this line.
# ---------------------------------------------------------------------------
LINE = ("{d} 09:00:00,000 [INFO] app.monitors.deep_research []: [entail-cascade] "
        "topic/{site}: 33 pair(s), 8 scored narrow (support {pct}%), 32 read at full width\n")


def test_narrow_support_is_read_per_call_site(tmp_path):
    p = tmp_path / "nova-app.log"
    p.write_text(LINE.format(d="2026-09-05", site="gate", pct=18)
                 + LINE.format(d="2026-09-05", site="clause", pct=4),
                 encoding="utf-8")
    got = er.cascade_support(1, str(p), today="2026-09-05")
    assert got["by_site"] == {"clause": 4, "gate": 18}
    assert got["runs"] == {"clause": 1, "gate": 1}


def test_support_is_averaged_across_runs(tmp_path):
    p = tmp_path / "nova-app.log"
    p.write_text(LINE.format(d="2026-09-05", site="gate", pct=10)
                 + LINE.format(d="2026-09-05", site="gate", pct=20),
                 encoding="utf-8")
    assert er.cascade_support(1, str(p), today="2026-09-05")["by_site"]["gate"] == 15


def test_older_lines_fall_outside_the_window(tmp_path):
    p = tmp_path / "nova-app.log"
    p.write_text(LINE.format(d="2026-08-01", site="gate", pct=70), encoding="utf-8")
    assert er.cascade_support(1, str(p), today="2026-09-05") is None


def test_no_cascade_lines_yet_is_none_not_zero(tmp_path):
    """Absent evidence must not read as 0% support."""
    (tmp_path / "nova-app.log").write_text("nothing here\n", encoding="utf-8")
    assert er.cascade_support(1, str(tmp_path / "nova-app.log"),
                              today="2026-09-05") is None


def test_a_missing_log_is_not_an_error(tmp_path):
    assert er.cascade_support(1, str(tmp_path / "nope*"), today="2026-09-05") is None


def test_the_actionable_fields_are_ordered_first(monkeypatch):
    """The renderer caps the line at 400 characters and drops fields from the
    END. On the first live run `look_at` was last and vanished entirely, so the
    one thing the report exists to say did not survive being delivered."""
    _quiet(monkeypatch)
    import app.monitors.pathways as pw
    monkeypatch.setattr(pw, "throughput_step", lambda db, **k: {
        "before": 6.5, "after": 4.1, "change": -0.37, "days": 18, "stepped_down": True})
    _status, _summary, fields = er.build_report(_DB())
    keys = list(fields)
    assert keys[0].startswith("look_at")
    assert keys.index("delivery") > 0, "numbers are context; they yield to findings"


def test_the_summary_carries_the_findings_not_a_count(monkeypatch):
    """80 characters, so it must spend them on what to look at."""
    _quiet(monkeypatch)
    import app.monitors.pathways as pw
    monkeypatch.setattr(pw, "throughput_step", lambda db, **k: {
        "before": 6.5, "after": 4.1, "change": -0.37, "days": 18, "stepped_down": True})
    _status, summary, _fields = er.build_report(_DB())
    assert "delivery is DOWN" in summary
