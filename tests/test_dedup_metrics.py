"""Tests for app.core.dedup_metrics — Jaccard dedup-decision instrumentation."""

from __future__ import annotations

import pytest

from app.core import dedup_metrics
from app.database import SafeDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Fresh DB with migration 21 applied, wired into get_db()."""
    db = SafeDB(str(tmp_path / "test.db"))
    db.init_schema()
    # Patch BOTH the source (app.database.get_db) and the already-bound
    # name in dedup_metrics — `from app.database import get_db` binds a
    # local reference that ignores changes to app.database.get_db alone.
    from app import database as db_mod
    monkeypatch.setattr(db_mod, "get_db", lambda: db)
    monkeypatch.setattr(dedup_metrics, "get_db", lambda: db)
    return db


def test_migration_21_creates_table(db):
    # Migration 21 should have created the dedup_decisions table.
    cols = {row[1] for row in db.fetchall("PRAGMA table_info(dedup_decisions)")}
    assert {"entity_type", "jaccard_score", "threshold", "decision", "created_at"} <= cols


def test_record_decision_inserts_row(db):
    ok = dedup_metrics.record_decision("lesson", 0.72, 0.55, "merged")
    assert ok is True
    rows = db.fetchall("SELECT * FROM dedup_decisions")
    assert len(rows) == 1
    r = dict(rows[0])
    assert r["entity_type"] == "lesson"
    assert r["jaccard_score"] == pytest.approx(0.72)
    assert r["threshold"] == pytest.approx(0.55)
    assert r["decision"] == "merged"


def test_record_decision_rejects_unknown_entity(db):
    assert dedup_metrics.record_decision("unknown_type", 0.5, 0.5, "merged") is False
    assert len(db.fetchall("SELECT * FROM dedup_decisions")) == 0


def test_record_decision_rejects_unknown_decision(db):
    assert dedup_metrics.record_decision("lesson", 0.5, 0.5, "skipped") is False
    assert len(db.fetchall("SELECT * FROM dedup_decisions")) == 0


def test_record_decision_clamps_score(db):
    # >1.0 clamps to 1.0; <0.0 clamps to 0.0
    dedup_metrics.record_decision("lesson", 1.5, 0.5, "merged")
    dedup_metrics.record_decision("lesson", -0.2, 0.5, "inserted_new")
    rows = [dict(r) for r in db.fetchall("SELECT jaccard_score FROM dedup_decisions ORDER BY id")]
    assert rows[0]["jaccard_score"] == pytest.approx(1.0)
    assert rows[1]["jaccard_score"] == pytest.approx(0.0)


def test_record_decision_rejects_nan(db):
    assert dedup_metrics.record_decision("lesson", float("nan"), 0.5, "merged") is False
    assert dedup_metrics.record_decision("lesson", 0.5, float("nan"), "merged") is False
    assert len(db.fetchall("SELECT * FROM dedup_decisions")) == 0


def test_summarize_buckets_by_entity_and_decision(db):
    dedup_metrics.record_decision("lesson", 0.90, 0.85, "merged")
    dedup_metrics.record_decision("lesson", 0.40, 0.85, "inserted_new")
    dedup_metrics.record_decision("lesson", 0.50, 0.85, "inserted_new")
    dedup_metrics.record_decision("curiosity", 0.70, 0.60, "merged")

    out = dedup_metrics.summarize()
    assert "lesson" in out and "curiosity" in out
    assert out["lesson"]["merged"]["n"] == 1
    assert out["lesson"]["inserted_new"]["n"] == 2
    assert out["lesson"]["inserted_new"]["mean"] == pytest.approx(0.45)
    assert out["curiosity"]["merged"]["n"] == 1


def test_summarize_near_threshold_pct(db):
    # 3 rows: 2 within ±0.05 of threshold (0.85), 1 well below.
    dedup_metrics.record_decision("lesson", 0.88, 0.85, "merged")        # +0.03 — near
    dedup_metrics.record_decision("lesson", 0.81, 0.85, "inserted_new")  # -0.04 — near
    dedup_metrics.record_decision("lesson", 0.40, 0.85, "inserted_new")  # -0.45 — far

    out = dedup_metrics.summarize("lesson")
    assert out["lesson"]["merged"]["near_threshold_pct"] == pytest.approx(1.0)
    assert out["lesson"]["inserted_new"]["near_threshold_pct"] == pytest.approx(0.5)


def test_summarize_filters_by_entity(db):
    dedup_metrics.record_decision("lesson", 0.5, 0.5, "merged")
    dedup_metrics.record_decision("curiosity", 0.7, 0.6, "merged")
    out = dedup_metrics.summarize(entity_type="lesson")
    assert "lesson" in out
    assert "curiosity" not in out


def test_summarize_returns_empty_for_unknown_entity(db):
    dedup_metrics.record_decision("lesson", 0.5, 0.5, "merged")
    assert dedup_metrics.summarize(entity_type="unknown") == {}
