"""Pathway liveness registry (2026-09-02).

Storylines were dead for five weeks and KG extraction for three days before
anyone noticed, because a background writer's failure mode is silence. The
registry in app/monitors/pathways.py lists every optional writer with the
silence it is allowed; the Pathway Liveness monitor turns that into a
deterministic fast-lane verdict.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.monitors import pathways
from app.monitors.heartbeat_loop import HeartbeatLoop, _CANARY_NORMAL_MARKERS
from app.monitors.monitor_store import MonitorStore

import app as _app_pkg

ROOT = Path(_app_pkg.__file__).resolve().parents[1]   # the tree the code actually imports from
NOW = datetime(2026, 9, 2, 12, 0, 0)


def _cfg(**over):
    base = {p.flag: True for p in pathways.PATHWAYS if p.flag}
    base["EVAL_REPORT_PATH"] = "/nonexistent/eval_reports"
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def seeded(db):
    store = MonitorStore(db)
    store.seed_defaults()
    # A month-old install: silence is a fault, not warm-up.
    db.execute("UPDATE schema_version SET applied_at = ?",
               ((NOW - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"),))
    return db, store


def _by_name(rows):
    return {r["name"]: r for r in rows}


# ---------------------------------------------------------------- registry

def _lazy_create_statement(table: str) -> str:
    """Some writers CREATE their table on first use instead of in init_schema;
    find that statement's column block in the source tree."""
    rx = re.compile(r"CREATE TABLE IF NOT EXISTS\s+" + re.escape(table) + r"(.{0,1500})", re.S)
    for path in (ROOT / "app").rglob("*.py"):
        m = rx.search(path.read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(1)
    return ""


def test_every_registry_table_and_column_exists_in_the_fresh_schema(db):
    for p in pathways.PATHWAYS:
        if p.path:
            continue
        cols = {r["name"] for r in db.fetchall(f"PRAGMA table_info({p.table})")}
        if cols:
            assert p.time_col in cols, f"{p.name}: {p.table}.{p.time_col} missing"
            continue
        stmt = _lazy_create_statement(p.table)
        assert stmt, f"{p.name}: table {p.table} is neither in init_schema nor lazily created"
        assert re.search(r"\b" + re.escape(p.time_col) + r"\b", stmt), (
            f"{p.name}: {p.table}.{p.time_col} not in its CREATE statement")


def test_missing_lazy_table_is_never_a_probe_error(seeded):
    db, _ = seeded
    rows = _by_name(pathways.snapshot(db, cfg=_cfg(), now=NOW))
    assert rows["curiosity_intake"]["verdict"] == "dead"      # old install, never wrote
    assert "error" not in rows["curiosity_intake"]


def test_every_registry_flag_and_monitor_is_real(db):
    from app.config import config
    store = MonitorStore(db)
    store.seed_defaults()
    names = {m.name for m in store.list_all()}
    for p in pathways.PATHWAYS:
        if p.flag:
            assert hasattr(config, p.flag), f"{p.name}: unknown flag {p.flag}"
        if p.monitor:
            assert p.monitor in names, f"{p.name}: unknown monitor {p.monitor!r}"


def test_registry_covers_the_knowing_tier():
    names = {p.name for p in pathways.PATHWAYS}
    for must in ("storylines", "dossiers", "dossier_history", "forecast_minting",
                 "forecast_resolution", "curiosity_research", "kg_digest_facts",
                 "kg_extracted_facts", "output_eval", "alert_delivery", "eval_harness"):
        assert must in names


# ---------------------------------------------------------------- verdicts

def test_silent_writer_on_an_old_install_is_dead(seeded):
    db, _ = seeded
    rows = _by_name(pathways.snapshot(db, cfg=_cfg(), now=NOW))
    assert rows["storylines"]["verdict"] == "dead"
    assert rows["storylines"]["last_at"] is None
    assert rows["dossiers"]["verdict"] == "dead"


def test_recent_row_makes_the_pathway_alive(seeded):
    db, _ = seeded
    db.execute("INSERT INTO storylines (story_key, title, summary, monitors_csv, update_count, last_updated) "
               "VALUES ('k', 'T', 's', 'm', 1, ?)", (NOW.strftime("%Y-%m-%d %H:%M:%S"),))
    db.execute("INSERT INTO storyline_events (storyline_id, summary, source_monitor, is_new, created_at) "
               "VALUES (1, 'moved', 'm', 1, ?)", ((NOW - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),))
    row = _by_name(pathways.snapshot(db, cfg=_cfg(), now=NOW))["storylines"]
    assert row["verdict"] == "alive"
    assert row["recent_rows"] == 1
    assert row["age_hours"] == 5.0


def test_row_outside_the_window_is_dead_with_its_age(seeded):
    db, _ = seeded
    db.execute("INSERT INTO storylines (story_key, title, summary, monitors_csv, update_count, last_updated) "
               "VALUES ('k', 'T', 's', 'm', 1, ?)", (NOW.strftime("%Y-%m-%d %H:%M:%S"),))
    db.execute("INSERT INTO storyline_events (storyline_id, summary, source_monitor, is_new, created_at) "
               "VALUES (1, 'moved', 'm', 1, ?)", ((NOW - timedelta(hours=90)).strftime("%Y-%m-%d %H:%M:%S"),))
    row = _by_name(pathways.snapshot(db, cfg=_cfg(), now=NOW))["storylines"]
    assert row["verdict"] == "dead"
    assert row["age_hours"] == 90.0


def test_window_stretches_to_twice_the_driving_monitor_schedule(seeded):
    db, store = seeded
    m = store.get_by_name("Storyline Tracker")
    store.update(m.id, schedule_seconds=48 * 3600)
    row = _by_name(pathways.snapshot(db, cfg=_cfg(), now=NOW))["storylines"]
    assert row["window_hours"] == 97.0  # 2 x 48h + 1h > the 36h cadence


def test_usage_gated_silence_is_idle_not_dead(seeded):
    db, _ = seeded
    rows = _by_name(pathways.snapshot(db, cfg=_cfg(), now=NOW))
    assert rows["chat_messages"]["verdict"] == "idle"
    assert rows["lessons"]["verdict"] == "idle"


def test_flag_off_and_disabled_driver_are_off(seeded):
    db, store = seeded
    rows = _by_name(pathways.snapshot(db, cfg=_cfg(ENABLE_FORECASTS=False), now=NOW))
    assert rows["forecast_minting"]["verdict"] == "off"
    assert rows["forecast_resolution"]["verdict"] == "off"
    m = store.get_by_name("Storyline Tracker")
    store.update(m.id, enabled=False)
    rows = _by_name(pathways.snapshot(db, cfg=_cfg(), now=NOW))
    assert rows["storylines"]["verdict"] == "off"


def test_fresh_install_is_warming_not_dead(db):
    MonitorStore(db).seed_defaults()
    rows = _by_name(pathways.snapshot(db, cfg=_cfg(), now=datetime.utcnow()))
    assert rows["storylines"]["verdict"] == "warming"
    assert rows["dossiers"]["verdict"] == "warming"


def test_file_probe_uses_mtime(seeded, tmp_path):
    db, _ = seeded
    hist = tmp_path / "eval_history.jsonl"
    hist.write_text("{}\n")
    rows = _by_name(pathways.snapshot(db, cfg=_cfg(EVAL_REPORT_PATH=str(tmp_path)),
                                      now=datetime.utcnow()))
    assert rows["eval_harness"]["verdict"] == "alive"
    rows = _by_name(pathways.snapshot(db, cfg=_cfg(EVAL_REPORT_PATH=str(tmp_path / "nope")), now=NOW))
    assert rows["eval_harness"]["verdict"] == "dead"


def test_timestamp_spellings_all_parse():
    assert pathways._parse_ts("2026-09-01 19:30:34") == datetime(2026, 9, 1, 19, 30, 34)
    assert pathways._parse_ts("2026-09-01T19:30:34.317717") == datetime(2026, 9, 1, 19, 30, 34)
    assert pathways._parse_ts("2026-08-31T03:50:49.650926+00:00") == datetime(2026, 8, 31, 3, 50, 49)
    assert pathways._parse_ts(None) is None
    assert pathways._parse_ts("garbage") is None


# ------------------------------------------------------------------ report

def test_report_names_the_dead_pathways_and_carries_their_silence(seeded):
    db, _ = seeded
    status, summary, fields = pathways.liveness_report(db, cfg=_cfg(), now=NOW)
    assert status == "error"
    assert "DEAD" in summary and "storylines" in summary
    assert fields["storylines"].startswith("never wrote")
    assert pathways.HEALTHY_MARKER not in summary


def test_healthy_report_carries_the_marker_and_no_drifting_numbers(seeded, monkeypatch):
    db, _ = seeded
    small = tuple(p for p in pathways.PATHWAYS if p.name in ("storylines", "chat_messages"))
    monkeypatch.setattr(pathways, "PATHWAYS", small)
    db.execute("INSERT INTO storylines (story_key, title, summary, monitors_csv, update_count, last_updated) "
               "VALUES ('k', 'T', 's', 'm', 1, ?)", (NOW.strftime("%Y-%m-%d %H:%M:%S"),))
    db.execute("INSERT INTO storyline_events (storyline_id, summary, source_monitor, is_new, created_at) "
               "VALUES (1, 'moved', 'm', 1, ?)", ((NOW - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),))
    status, summary, fields = pathways.liveness_report(db, cfg=_cfg(), now=NOW)
    assert status == "info"
    assert summary.startswith(pathways.HEALTHY_MARKER)
    assert fields == {"alive": 1, "idle": 1, "off": 0}
    assert not re.search(r"\d+h", summary)


# ------------------------------------------------------------------ wiring

def test_monitor_is_wired_into_the_fast_lane_and_the_canary_gate(db):
    assert "pathway_liveness" in HeartbeatLoop._CHECK_DISPATCH
    assert _CANARY_NORMAL_MARKERS["pathway_liveness"] == pathways.HEALTHY_MARKER
    src = (ROOT / "app" / "monitors" / "heartbeat_loop.py").read_text(encoding="utf-8")
    m = re.search(r"_FAST_TYPES = \{([^}]*)\}", src)
    assert m and "pathway_liveness" in m.group(1)
    store = MonitorStore(db)
    store.seed_defaults()
    mon = store.get_by_name("Pathway Liveness")
    assert mon is not None and mon.enabled
    assert mon.check_type == "pathway_liveness"
    assert mon.category == "system"
    assert mon.schedule_seconds == 6 * 3600
    assert "Pathway Liveness" in MonitorStore._CORE_ENABLED


@pytest.mark.asyncio
async def test_executor_returns_a_formatted_verdict(monkeypatch):
    from app.database import get_db
    db = get_db()
    db.init_schema()
    store = MonitorStore(db)
    store.seed_defaults()
    out = await HeartbeatLoop(store)._execute_pathway_liveness()
    # Fresh install: everything warming, nothing dead — the healthy line.
    assert pathways.HEALTHY_MARKER in out
    db.execute("UPDATE schema_version SET applied_at = '2026-01-01 00:00:00'")
    out = await HeartbeatLoop(store)._execute_pathway_liveness()
    assert "DEAD" in out and "storylines" in out


def test_status_schema_accepts_the_snapshot(seeded):
    from app.schema import StatusResponse
    db, _ = seeded
    snap = pathways.snapshot(db, cfg=_cfg(), now=NOW)
    resp = StatusResponse(pathways=snap)
    assert {r["name"] for r in resp.pathways} == {p.name for p in pathways.PATHWAYS}
    assert StatusResponse().pathways == []
