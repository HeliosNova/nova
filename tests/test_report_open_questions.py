"""The report carries the questions I could not answer today (2026-09-06).

Every other field in the Engineering Report describes a thing already
understood. These two exist because a specific question is open and a night of
running is the only way to close it:

  curiosity_stage1   Half of curiosity's closure failures came from a cheap
                     pre-filter that logged nothing (171 of 340 since 08-20).
                     The suspicion is that markers like "unclear from" kill
                     answers that settled the question but hedged one sub-part.
                     A marker dominating this list is the suspect; "too short"
                     dominating means something else entirely.
  planner_health     The planner's ceiling went 512 -> 900, justified by an A/B
                     that never reproduced the long queries which truncate. A
                     "900" appearing here says the raise was not enough. The
                     TimeoutError count is the larger, untouched problem: 60s
                     against a GPU the digest chain owns.

Both read /data/logs rather than the container's own log, because the container
log is lost on every restart and this session restarted nova-app nine times in
a day.
"""
from __future__ import annotations

from app.monitors import engineering_report as er

STAGE1 = ('{d} 09:00:00,000 [INFO] app.monitors.heartbeat_loop []: '
          '[Curiosity] stage-1 reject (deflection {marker!r} at 812 chars): Some topic\n')
SHORT = ('{d} 09:00:00,000 [INFO] app.monitors.heartbeat_loop []: '
         '[Curiosity] stage-1 reject (too short: 41 chars): Some topic\n')
TRUNC = ('{d} 09:00:00,000 [WARNING] app.core.providers.ollama []: '
         '[truncation] invoke_nothink hit max_tokens ({cap}) — output cut\n')
PLANFAIL = ('{d} 09:00:00,000 [WARNING] app.core.planning []: '
            'Planning failed: TimeoutError()\n')


def _log(tmp_path, text):
    p = tmp_path / "nova-app.log"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_the_marker_that_killed_an_answer_is_counted(tmp_path):
    g = _log(tmp_path, STAGE1.format(d="2026-09-06", marker="unclear from") * 3
             + STAGE1.format(d="2026-09-06", marker="i don't have"))
    got = er.curiosity_stage1(1, g, today="2026-09-06")
    assert got == {"'unclear from'": 3, "'i don\\'t have'": 1} or "unclear from" in str(got), got


def test_too_short_is_a_different_reason_from_a_marker(tmp_path):
    g = _log(tmp_path, SHORT.format(d="2026-09-06") * 2)
    got = er.curiosity_stage1(1, g, today="2026-09-06")
    assert got == {"too short": 2}


def test_stage1_outside_the_window_is_ignored(tmp_path):
    g = _log(tmp_path, STAGE1.format(d="2026-08-01", marker="unclear from"))
    assert er.curiosity_stage1(1, g, today="2026-09-06") is None


def test_no_stage1_lines_yet_reads_as_none_not_zero(tmp_path):
    """Absent evidence must not look like a clean bill of health."""
    g = _log(tmp_path, "nothing here\n")
    assert er.curiosity_stage1(1, g, today="2026-09-06") is None


def test_planner_truncations_are_counted_by_ceiling(tmp_path):
    """Which ceiling matters: a 900 means the 2026-09-05 raise was not enough."""
    g = _log(tmp_path, TRUNC.format(d="2026-09-06", cap=900) * 2
             + TRUNC.format(d="2026-09-06", cap=512))
    got = er.planner_health(1, g, today="2026-09-06")
    assert got["truncations"] == {"900": 2, "512": 1}


def test_plan_failures_are_counted_by_type(tmp_path):
    g = _log(tmp_path, PLANFAIL.format(d="2026-09-06") * 4)
    got = er.planner_health(1, g, today="2026-09-06")
    assert got["plan_failures"] == {"TimeoutError": 4}


def test_a_quiet_planner_reads_as_none(tmp_path):
    assert er.planner_health(1, _log(tmp_path, "quiet\n"), today="2026-09-06") is None


def test_a_900_truncation_becomes_something_to_look_at(monkeypatch):
    """The pre-registered check for yesterday's ceiling raise."""
    monkeypatch.setattr(er, "planner_health", lambda *a, **k: {
        "truncations": {"900": 5}, "plan_failures": {}})
    monkeypatch.setattr(er, "curiosity_stage1", lambda *a, **k: None)
    monkeypatch.setattr(er, "cascade_support", lambda *a, **k: None)
    import app.monitors.pathways as pw
    monkeypatch.setattr(pw, "throughput_step", lambda db, **k: None)
    monkeypatch.setattr(pw, "schedule_pressure", lambda db, **k: {
        "ratio": 0.95, "delivered": 100, "demanded": 105, "starved": []})
    monkeypatch.setattr(pw, "constant_monitors", lambda db, **k: [])
    monkeypatch.setattr(pw, "snapshot", lambda db, **k: [])
    import app.monitors.health_checks as hc
    monkeypatch.setattr(hc, "entail_gate_totals", lambda d, *a, **k: (0, 0))
    import app.core.forecasts as fc
    monkeypatch.setattr(fc, "calibration", lambda db, **k: None)

    class _DB:
        def fetchone(self, _q, _a=()):
            return {"c": 0, "oldest": None, "d": None}

        def fetchall(self, _q, _a=()):
            return []

    _status, _summary, fields = er.build_report(_DB())
    assert "STILL truncating at its new 900 ceiling" in fields["look_at_1"]


# ---------------------------------------------------------------------------
# The record, as distinct from the message.
# ---------------------------------------------------------------------------
def test_the_full_field_set_survives_the_400_char_cap(tmp_path):
    """The delivered line drops fields from the end; the record must not.

    Comparing this morning against last week is the entire reason the report
    exists, and rendered at 400 characters roughly half its fields never reach
    monitor_results at all.
    """
    import json
    p = tmp_path / "eng.jsonl"
    fields = {f"k{i}": "x" * 40 for i in range(14)}
    assert er.append_snapshot("warning", "something", fields, path=str(p))
    row = json.loads(p.read_text(encoding="utf-8").strip())
    assert row["status"] == "warning" and row["summary"] == "something"
    assert all(row[k] == v for k, v in fields.items())
    assert row["at"], "a record with no timestamp cannot be compared to anything"


def test_each_run_appends_rather_than_replacing(tmp_path):
    p = tmp_path / "eng.jsonl"
    er.append_snapshot("info", "day one", {"delivery": "6.5"}, path=str(p))
    er.append_snapshot("info", "day two", {"delivery": "6.2"}, path=str(p))
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2, "a trend needs both days"


def test_an_unwritable_path_does_not_break_the_monitor(tmp_path):
    """It runs inside a monitor. Recording is never worth failing the report."""
    assert er.append_snapshot("info", "s", {"a": "b"},
                              path=str(tmp_path / "no" / "such" / "dir" / "f")) is False


def test_a_field_that_is_not_json_is_still_recorded(tmp_path):
    import json
    from datetime import date
    p = tmp_path / "eng.jsonl"
    assert er.append_snapshot("info", "s", {"when": date(2026, 9, 6)}, path=str(p))
    assert json.loads(p.read_text(encoding="utf-8"))["when"] == "2026-09-06"


def test_the_executor_records_before_it_renders():
    """Order matters: the renderer is what truncates."""
    import inspect
    from app.monitors import health_checks as hc
    src = inspect.getsource(hc.HealthChecksMixin._execute_engineering_report)
    assert src.index("append_snapshot") < src.index("format_monitor_result")
