"""Scheduler v2 — model-residency batching (2026-09-02, Phase 4.3).

The 24GB card holds one of the 27B / 9B / judge models. Overdue-ratio order
interleaves residency classes freely, so a tick could swap models on almost
every monitor. Batches are now grouped by class (first-appearance order,
stable within a class) after the starvation floor.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import app as _app_pkg
from app.monitors.heartbeat_loop import (
    HeartbeatLoop,
    _batch_by_class,
    _class_floor_order,
)

ROOT = Path(_app_pkg.__file__).resolve().parents[1]


def _mk(name, ct="query"):
    return SimpleNamespace(name=name, check_type=ct)


def test_batch_groups_by_class_in_first_appearance_order():
    order = [_mk("d1"), _mk("o1", "quiz"), _mk("d2"), _mk("j1", "output_eval"), _mk("o2", "quiz"), _mk("d3")]
    classify = lambda m: {"quiz": "other", "output_eval": "judge"}.get(m.check_type, "digest")
    out = [m.name for m in _batch_by_class(order, classify)]
    assert out == ["d1", "d2", "d3", "o1", "o2", "j1"]


def test_batch_is_stable_and_bounds_swaps():
    order = [_mk(f"{'d' if i % 2 else 'o'}{i}", "query" if i % 2 else "quiz") for i in range(12)]
    classify = lambda m: "digest" if m.check_type == "query" else "other"
    out = _batch_by_class(order, classify)
    classes = [classify(m) for m in out]
    swaps = sum(1 for a, b in zip(classes, classes[1:]) if a != b)
    assert swaps == 1                       # 11 alternations collapsed to one
    assert [m.name for m in out if classify(m) == "other"] == [f"o{i}" for i in range(0, 12, 2)]


def test_starvation_floor_still_leads_after_batching():
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    def _mon(name, ct, sched, ago):
        return SimpleNamespace(name=name, check_type=ct, schedule_seconds=sched,
                               last_check_at=(now - timedelta(seconds=ago)).strftime("%Y-%m-%d %H:%M:%S"))

    digest_a = _mon("digest-a", "query", 14400, 30000)
    digest_b = _mon("digest-b", "query", 14400, 26000)
    eval_m = _mon("eval", "eval", 86400, 115000)          # starved daily
    quiz = _mon("quiz", "quiz", 21600, 22000)
    classify = lambda m: "digest" if m.check_type == "query" else "other"
    floored = _class_floor_order([digest_a, digest_b, quiz, eval_m], classify, now)
    out = [m.name for m in _batch_by_class(floored, classify)]
    # the starved daily leads, and its class (other) runs contiguously with it
    assert out == ["eval", "quiz", "digest-a", "digest-b"]


def test_residency_classes():
    lp = HeartbeatLoop.__new__(HeartbeatLoop)
    assert lp._monitor_class(_mk("Domain Study: Finance")) == "digest"
    assert lp._monitor_class(_mk("Knowledge Consolidation", "consolidation")) == "digest"
    assert lp._monitor_class(_mk("Cross-Monitor Synthesis", "synthesis")) == "digest"
    assert lp._monitor_class(_mk("Output Quality Eval", "output_eval")) == "judge"
    assert lp._monitor_class(_mk("Storyline Tracker", "storyline")) == "other"
    assert lp._monitor_class(_mk("Forecast Resolution", "forecast_resolve")) == "other"
    assert lp._monitor_class(_mk("Lesson Quiz", "quiz")) == "other"


def test_loop_batches_after_the_floor():
    src = (ROOT / "app" / "monitors" / "heartbeat_loop.py").read_text(encoding="utf-8")
    i = src.index("slow = _class_floor_order(")
    j = src.index("slow = _batch_by_class(slow, self._monitor_class)")
    assert i < j
    assert re.search(r"model swap\(s\)", src)
