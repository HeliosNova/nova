"""P1: every scored entailment pair is recorded with its evidence window."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.monitors import gate_record as gr


@pytest.fixture
def rec_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "RECORD_DIR", tmp_path)
    return tmp_path


def test_record_writes_one_line_per_pair_with_every_field(rec_dir):
    n = gr.record("Domain Study: X", "gate", [
        {"claim": "Acme bought Beta.", "hosts": ["reuters.com"], "doc": "Acme has acquired Beta.", "prob": 0.91, "supported": True, "narrow": True},
        {"claim": "Beta bought Acme.", "hosts": ["reuters.com"], "doc": "Acme has acquired Beta.", "prob": 0.12, "supported": False, "narrow": False},
    ])
    assert n == 2
    files = list(rec_dir.glob("gate-*.jsonl"))
    assert len(files) == 1
    assert files[0].name == f"gate-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    rows = [json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines()]
    assert [set(r) for r in rows] == [set(gr.FIELDS)] * 2
    assert rows[0]["site"] == "gate" and rows[0]["label"] == "Domain Study: X"
    assert rows[0]["supported"] is True and rows[1]["supported"] is False
    assert rows[0]["doc"] == "Acme has acquired Beta." and rows[0]["narrow"] is True


def test_record_appends_across_calls(rec_dir):
    gr.record("a", "gate", [{"claim": "c1", "doc": "d"}])
    gr.record("a", "clause", [{"claim": "c2", "doc": "d"}])
    rows = list(gr.read())
    assert [r["site"] for r in rows] == ["gate", "clause"]


def test_writer_never_raises_into_the_gate(tmp_path, monkeypatch):
    # a FILE where the directory should be: mkdir fails
    blocker = tmp_path / "entail"
    blocker.write_text("not a dir")
    monkeypatch.setattr(gr, "RECORD_DIR", blocker)
    assert gr.record("a", "gate", [{"claim": "c", "doc": "d"}]) == 0


def test_old_files_are_pruned_on_write(rec_dir):
    old = rec_dir / f"gate-{(datetime.now(timezone.utc) - timedelta(days=gr.KEEP_DAYS + 3)):%Y-%m-%d}.jsonl"
    old.write_text('{"claim": "stale"}\n')
    keep = rec_dir / f"gate-{(datetime.now(timezone.utc) - timedelta(days=2)):%Y-%m-%d}.jsonl"
    keep.write_text('{"claim": "recent"}\n')
    gr.record("a", "gate", [{"claim": "c", "doc": "d"}])
    assert not old.exists() and keep.exists()


def test_read_skips_corrupt_lines(rec_dir):
    p = rec_dir / f"gate-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    p.write_text('{"claim": "ok"}\nnot json\n\n{"claim": "ok2"}\n', encoding="utf-8")
    assert [r["claim"] for r in gr.read()] == ["ok", "ok2"]


def test_every_cascade_verdict_is_recorded():
    """The hook sits in _check_cascade so gate, clause and alt all pass through it."""
    from app.monitors import deep_research as dr
    src = inspect.getsource(dr._entailment_gate)
    assert src.count("_record_gate(") >= 2, "both cascade return paths must record"
    assert "def _record_gate(" in inspect.getsource(dr)
