"""Tests for app.core.rlvr — verifiable signal collection."""

from __future__ import annotations

import json
import pytest

from app.core import rlvr
from app.database import get_db, _instances


@pytest.fixture
def signals_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "rlvr.db"))
    _instances.clear()
    db = get_db()
    db.init_schema()
    yield db
    db.close()
    _instances.clear()


# ---- record_signal ------------------------------------------------------

def test_record_signal_unknown_type_returns_false(signals_db):
    assert rlvr.record_signal("not_a_real_type", 1.0) is False


def test_record_signal_nan_inf_dropped(signals_db):
    assert rlvr.record_signal("tool_correct", float("nan")) is False
    assert rlvr.record_signal("tool_correct", float("inf")) is False
    assert rlvr.record_signal("tool_correct", float("-inf")) is False


def test_record_signal_string_value_dropped(signals_db):
    assert rlvr.record_signal("tool_correct", "high") is False


def test_record_signal_clamps_out_of_range(signals_db):
    assert rlvr.record_signal("tool_correct", 1.5) is True
    assert rlvr.record_signal("tool_correct", -0.5) is True
    rows = signals_db.fetchall("SELECT signal_value FROM verifiable_signals ORDER BY id")
    vals = [float(r["signal_value"]) for r in rows]
    assert 1.0 in vals  # clamped from 1.5
    assert 0.0 in vals  # clamped from -0.5


def test_record_signal_truncates_long_strings(signals_db):
    long_query = "x" * 5000
    long_response = "y" * 10000
    long_evidence = "z" * 5000
    assert rlvr.record_signal(
        "tool_correct", 1.0,
        query=long_query, response=long_response, evidence=long_evidence,
    ) is True
    row = signals_db.fetchone("SELECT query, response, evidence FROM verifiable_signals")
    assert len(row["query"]) <= 2000
    assert len(row["response"]) <= 4000
    assert len(row["evidence"]) <= 2000


def test_record_signal_basic_round_trip(signals_db):
    rlvr.record_signal(
        "tool_correct", 1.0,
        query="add 2 and 3", response="5",
        evidence="tool=calculator", conversation_id="conv-1",
    )
    row = signals_db.fetchone("SELECT * FROM verifiable_signals")
    assert row["signal_type"] == "tool_correct"
    assert float(row["signal_value"]) == 1.0
    assert row["query"] == "add 2 and 3"
    assert row["conversation_id"] == "conv-1"
    assert int(row["consumed_for_training"]) == 0


# ---- query_signals ------------------------------------------------------

def test_query_signals_filters_by_type(signals_db):
    rlvr.record_signal("tool_correct", 1.0)
    rlvr.record_signal("json_valid", 0.0)
    rlvr.record_signal("tool_correct", 0.5)
    out = rlvr.query_signals("tool_correct")
    assert len(out) == 2
    for s in out:
        assert s.signal_type == "tool_correct"


def test_query_signals_unknown_type_empty(signals_db):
    rlvr.record_signal("tool_correct", 1.0)
    assert rlvr.query_signals("garbage") == []


def test_query_signals_only_unconsumed(signals_db):
    rlvr.record_signal("tool_correct", 1.0)
    rlvr.record_signal("tool_correct", 0.0)
    rows = rlvr.query_signals("tool_correct")
    rlvr.mark_consumed([rows[0].id])
    after = rlvr.query_signals("tool_correct", only_unconsumed=True)
    assert len(after) == 1
    assert after[0].consumed_for_training is False


# ---- aggregate ----------------------------------------------------------

def test_aggregate_returns_per_type_stats(signals_db):
    for v in (0.0, 0.5, 0.5, 1.0):
        rlvr.record_signal("tool_correct", v)
    rlvr.record_signal("json_valid", 1.0)
    out = rlvr.aggregate()
    assert "tool_correct" in out
    tc = out["tool_correct"]
    assert tc["n"] == 4.0
    assert 0.0 < tc["mean"] < 1.0
    assert "json_valid" in out


def test_aggregate_empty(signals_db):
    assert rlvr.aggregate() == {}


# ---- mark_consumed ------------------------------------------------------

def test_mark_consumed_idempotent(signals_db):
    rlvr.record_signal("tool_correct", 1.0)
    rlvr.record_signal("tool_correct", 1.0)
    rows = rlvr.query_signals("tool_correct")
    n = rlvr.mark_consumed([r.id for r in rows])
    assert n == 2
    # Second call doesn't error or double-count
    n2 = rlvr.mark_consumed([r.id for r in rows])
    assert n2 == 2  # rowcount of UPDATE; the rows are already 1 → still match


def test_mark_consumed_empty(signals_db):
    assert rlvr.mark_consumed([]) == 0


# ---- export_grpo_jsonl --------------------------------------------------

def test_export_grpo_jsonl_writes_unconsumed(signals_db, tmp_path):
    rlvr.record_signal("tool_correct", 1.0, query="q1", response="r1")
    rlvr.record_signal("tool_correct", 0.0, query="q2", response="r2")
    out_path = tmp_path / "grpo.jsonl"
    n = rlvr.export_grpo_jsonl(str(out_path), signal_types=["tool_correct"])
    assert n == 2
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    obj = json.loads(lines[0])
    assert obj["signal_type"] == "tool_correct"
    assert "query" in obj and "response" in obj and "reward" in obj


def test_export_grpo_jsonl_min_value_filter(signals_db, tmp_path):
    rlvr.record_signal("tool_correct", 1.0)
    rlvr.record_signal("tool_correct", 0.0)
    out_path = tmp_path / "grpo.jsonl"
    n = rlvr.export_grpo_jsonl(
        str(out_path), signal_types=["tool_correct"], min_value=0.5,
    )
    assert n == 1


def test_export_grpo_jsonl_skips_consumed(signals_db, tmp_path):
    rlvr.record_signal("tool_correct", 1.0)
    rows = rlvr.query_signals("tool_correct")
    rlvr.mark_consumed([rows[0].id])
    out_path = tmp_path / "grpo.jsonl"
    n = rlvr.export_grpo_jsonl(str(out_path))
    assert n == 0
