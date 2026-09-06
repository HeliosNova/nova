"""Delivering the same amount, worse (2026-09-06).

`throughput_step` answers "is Nova delivering LESS". Nothing owned the other
regression. On 2026-09-04 digests citing "(deep analysis)" climbed 10% -> 34%
-> 45% -> 55% across three days while the suite stayed green, every throughput
number stayed flat, and the OWNER found it by reading a digest.

These measures are not new — `scripts/quality_panel.py` has computed them since
that day. What is new is that the daily report reads them, so nobody has to
remember to run a script. The absolute values move with topic mix; a STEP
against the trailing week is what a prompt or gate change looks like.
"""
from __future__ import annotations

from app.monitors import engineering_report as er

CLEAN = ("Markets moved on the rate decision (reuters.com) and chipmakers "
         "followed (bloomberg.com). " * 40)
PSEUDO = CLEAN + " The 'so what' here (deep analysis) is margin compression."
LEAKY = CLEAN + " As an AI, I cannot access that page."
LINKONLY = "See the story: http://example.com/a and http://example.com/b"


class _DB:
    """recent rows, then baseline rows — in the order build_report asks."""

    def __init__(self, recent, base=None):
        self._q = [recent, base if base is not None else []]

    def fetchall(self, _sql, _args=()):
        return [{"value": v} for v in self._q.pop(0)]

    def fetchone(self, _sql, _args=()):
        return {"c": 0, "oldest": None, "d": None}


# --- digest_shape: the fingerprint -----------------------------------------
def test_a_self_citation_is_counted_and_a_real_one_is_not():
    """"(deep analysis)" is the briefing citing its own reasoning;
    "(reuters.com)" is a source. One parenthesis apart."""
    assert er.digest_shape("x (deep analysis) y")["pseudo"] == 1
    assert er.digest_shape("x (reuters.com) y")["pseudo"] == 0
    assert er.digest_shape("x (reuters.com) y")["cites"] == 1


def test_the_owners_complaint_has_a_number():
    """A short digest that is mostly URLs is the link-only failure."""
    assert er.digest_shape(LINKONLY)["linkonly"] == 1
    assert er.digest_shape(CLEAN)["linkonly"] == 0


def test_every_leak_pattern_the_panel_had_is_still_here():
    """The first version of this copy had FOUR of the six and would have
    silently stopped detecting `</tool_call>` and "I cannot access"."""
    assert len(er._LEAKS) == 6
    for text in ("As an AI, ", "step 2/5", "not specified here", "search results",
                 "</tool_call>", "I cannot access"):
        assert er.digest_shape("body " + text + " tail")["leaks"] >= 1, text


def test_the_panel_and_the_report_share_ONE_definition():
    """Two copies of a regex is how `strip_markup` came to be defined twice
    with the second silently winning."""
    import scripts.quality_panel as qp
    assert qp.digest_shape is er.digest_shape
    src = open("scripts/quality_panel.py", encoding="utf-8").read()
    for name in ("_PAREN = ", "_CITE = ", "_LEAKS = ", "_DOMAINISH = "):
        assert name not in src, f"{name} was redefined in the panel"


# --- product_quality: the comparison ---------------------------------------
def test_a_quiet_day_reports_shape_without_alarm():
    got = er.product_quality(_DB([CLEAN] * 6, [CLEAN] * 20))
    assert got["n"] == 6 and got["pseudo"] == 0 and got["linkonly"] == 0
    assert got["base_n"] == 20


def test_no_digests_reads_as_none_not_zero():
    assert er.product_quality(_DB([], [])) is None


def test_too_thin_a_baseline_is_omitted_rather_than_trusted():
    got = er.product_quality(_DB([CLEAN] * 3, [CLEAN] * 2))
    assert got is not None and "base_pseudo" not in got


def test_a_broken_store_does_not_raise():
    class _Broken:
        def fetchall(self, *_a, **_k):
            raise RuntimeError("no such table")

    assert er.product_quality(_Broken()) is None


# --- the report's attention rules ------------------------------------------
def _quiet(monkeypatch):
    monkeypatch.setattr(er, "cascade_support", lambda *a, **k: None)
    monkeypatch.setattr(er, "curiosity_stage1", lambda *a, **k: None)
    monkeypatch.setattr(er, "planner_health", lambda *a, **k: None)
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


def _look(fields):
    return " | ".join(str(v) for k, v in fields.items() if k.startswith("look_at"))


def test_the_2026_09_04_epidemic_would_have_been_NAMED(monkeypatch):
    """The pre-registered check: self-citation rising against its own week."""
    _quiet(monkeypatch)
    monkeypatch.setattr(er, "product_quality", lambda db, **k: {
        "n": 20, "chars": 7400, "cites": 5.0, "pseudo": 0.55, "leaks": 0,
        "linkonly": 0, "thin": 0, "base_pseudo": 0.10, "base_chars": 7400,
        "base_cites": 5.0, "base_n": 90})
    _status, _summary, fields = er.build_report(_DB([], []))
    assert "citing their own analysis" in _look(fields)


def test_a_rate_below_the_floor_and_flat_is_quiet(monkeypatch):
    """Written first as "a steady 0.30 is not an alarm", which was wrong.

    That premise came from the pre-fix era, where 0.64-2.36 was normal and only
    a step meant anything. Post-fix the measured rate is 0.00 across 125
    digests, so a sustained 0.30 IS a regression. The floor is what stays quiet
    on genuine noise, not on a third of digests citing themselves.
    """
    _quiet(monkeypatch)
    monkeypatch.setattr(er, "product_quality", lambda db, **k: {
        "n": 20, "chars": 7400, "cites": 5.0, "pseudo": 0.10, "leaks": 0,
        "linkonly": 0, "thin": 0, "base_pseudo": 0.08, "base_chars": 7400,
        "base_cites": 5.0, "base_n": 90})
    status, _summary, fields = er.build_report(_DB([], []))
    assert not _look(fields) and status == "info"
    assert er._PSEUDO_FLOOR > 0.10


def test_one_bare_link_digest_is_enough_to_report(monkeypatch):
    """The owner has raised this four separate times. It has no acceptable rate."""
    _quiet(monkeypatch)
    monkeypatch.setattr(er, "product_quality", lambda db, **k: {
        "n": 20, "chars": 7400, "cites": 5.0, "pseudo": 0.0, "leaks": 0,
        "linkonly": 1, "thin": 0})
    _status, _summary, fields = er.build_report(_DB([], []))
    assert "bare links" in _look(fields)


def test_digests_getting_shorter_points_at_the_ceiling_not_the_schedule(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(er, "product_quality", lambda db, **k: {
        "n": 20, "chars": 4000, "cites": 5.0, "pseudo": 0.0, "leaks": 0,
        "linkonly": 0, "thin": 12, "base_chars": 7400, "base_pseudo": 0.0,
        "base_cites": 5.0, "base_n": 90})
    _status, _summary, fields = er.build_report(_DB([], []))
    assert "got shorter" in _look(fields)


# ---------------------------------------------------------------------------
# Against a REAL database, because the stub above cannot see broken SQL.
#
# Every test in this file passed while the query read "AN+" instead of "AND" —
# a `.replace()` in the patch script that wrote it had corrupted the keyword,
# and `except Exception: return None` turned the syntax error into an empty
# result. The live run found it in one reading. A measurement whose failure
# mode is "returns nothing" needs one test that runs the query for real.
# ---------------------------------------------------------------------------
def _real_db(tmp_path, recent, base=()):
    import sqlite3
    from app.database import SafeDB
    p = str(tmp_path / "t.db")
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE monitors (id INTEGER PRIMARY KEY, category TEXT, "
                "check_type TEXT)")
    con.execute("CREATE TABLE monitor_results (id INTEGER PRIMARY KEY, "
                "monitor_id INTEGER, value TEXT, created_at TEXT)")
    con.execute("INSERT INTO monitors VALUES (1, 'content', 'query')")
    # Same category, NOT a briefing: a curiosity answer averages 470 chars.
    con.execute("INSERT INTO monitors VALUES (2, 'content', 'curiosity')")
    for v in recent:
        con.execute("INSERT INTO monitor_results (monitor_id, value, created_at) "
                    "VALUES (1, ?, datetime('now', '-2 hours'))", (v,))
    for v in base:
        con.execute("INSERT INTO monitor_results (monitor_id, value, created_at) "
                    "VALUES (1, ?, datetime('now', '-4 days'))", (v,))
    con.execute("INSERT INTO monitor_results (monitor_id, value, created_at) "
                "VALUES (2, ?, datetime('now', '-2 hours'))", ("s" * 900,))
    con.commit()
    con.close()
    return SafeDB(p)


def test_the_query_actually_runs(tmp_path):
    db = _real_db(tmp_path, [CLEAN] * 4, [CLEAN] * 20)
    try:
        got = er.product_quality(db)
    finally:
        db.close()
    assert got is not None, "the SQL did not execute — check the WHERE clause"
    assert got["base_n"] == 20
    assert got["n"] == 4, (
        "a curiosity answer is not a briefing. Measured over the whole content "
        "category, raising _CURIOSITY_BATCH to 3 read as digests getting "
        "shorter and the thin count tripling — a deliberate change arriving as "
        "a quality alarm.")


def test_the_windows_do_not_overlap(tmp_path):
    """A row must be in exactly one of recent / baseline, or a step compares
    yesterday against itself and can never fire."""
    db = _real_db(tmp_path, [PSEUDO] * 4, [CLEAN] * 20)
    try:
        got = er.product_quality(db)
    finally:
        db.close()
    assert got["pseudo"] == 1.0 and got["base_pseudo"] == 0.0


def test_short_rows_are_excluded_so_a_stub_result_is_not_a_digest(tmp_path):
    db = _real_db(tmp_path, ["too short"] * 4, [CLEAN] * 20)
    try:
        assert er.product_quality(db) is None
    finally:
        db.close()


def test_a_regression_to_the_PRE_FIX_level_still_fires(monkeypatch):
    """The gap a step-only rule leaves open.

    The 2026-09-04 fix took self-citation to 0.00 across 125 digests while the
    trailing week still averaged 1.67. Under a step-only rule, a regression all
    the way back to 1.0/digest computes as an IMPROVEMENT against that baseline
    and says nothing — for as long as the pre-fix era stays in the window.
    """
    _quiet(monkeypatch)
    monkeypatch.setattr(er, "product_quality", lambda db, **k: {
        "n": 90, "chars": 7000, "cites": 11.0, "pseudo": 1.00, "leaks": 0,
        "linkonly": 0, "thin": 0, "base_pseudo": 1.67, "base_chars": 7000,
        "base_cites": 11.0, "base_n": 300})
    _status, _summary, fields = er.build_report(_DB([], []))
    assert "citing their own analysis" in _look(fields)


def test_todays_actual_reading_stays_quiet(monkeypatch):
    """0.00 against a 1.67 baseline is the fix working, not something to say."""
    _quiet(monkeypatch)
    monkeypatch.setattr(er, "product_quality", lambda db, **k: {
        "n": 91, "chars": 6903, "cites": 11.18, "pseudo": 0.0, "leaks": 0,
        "linkonly": 0, "thin": 5, "base_pseudo": 1.67, "base_chars": 7434,
        "base_cites": 11.39, "base_n": 300})
    status, _summary, fields = er.build_report(_DB([], []))
    assert not _look(fields) and status == "info"


def test_one_stray_self_citation_in_ninety_digests_is_not_an_alarm(monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(er, "product_quality", lambda db, **k: {
        "n": 90, "chars": 7000, "cites": 11.0, "pseudo": 1 / 90, "leaks": 0,
        "linkonly": 0, "thin": 0, "base_pseudo": 0.0, "base_chars": 7000,
        "base_cites": 11.0, "base_n": 300})
    _status, _summary, fields = er.build_report(_DB([], []))
    assert not _look(fields)
