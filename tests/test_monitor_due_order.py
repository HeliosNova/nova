"""get_due ordering (2026-08-14): most-overdue-ratio first.

list_all() returns ORDER BY id, and get_due used to pass that through — so
late-seeded monitors (Dream Consolidation) sat LAST in every deep batch and
the knowing cycle waited hours behind every digest. The due list now sorts by
overdue RATIO (elapsed/schedule) so cadence differences don't punish slow
monitors, and never-run monitors lead.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.monitors.monitor_store import MonitorStore


def _mk(store, name, schedule_s, last_ago_s):
    mid = store.create(name=name, check_type="query", check_config={"query": "q"},
                       schedule_seconds=schedule_s)
    if last_ago_s is not None:
        last = (datetime.now(timezone.utc).replace(tzinfo=None)
                - timedelta(seconds=last_ago_s)).strftime("%Y-%m-%d %H:%M:%S")
        store.update(mid, last_check_at=last)
    return mid


def test_due_sorted_by_overdue_ratio(db):
    store = MonitorStore(db)
    for m in store.list_all():          # clear seeded defaults for a clean field
        store.update(m.id, enabled=False)
    # 4h monitor 5h late (ratio 1.25), seeded FIRST (lowest id)
    _mk(store, "digest-ish", 14400, 18000)
    # 6h monitor 11h late (ratio ~1.83), seeded LAST (highest id) — the
    # Dream Consolidation shape; must sort FIRST despite its id
    _mk(store, "consolidation-ish", 21600, 39600)
    due = [m.name for m in store.get_due()]
    assert due == ["consolidation-ish", "digest-ish"]


def test_never_run_leads(db):
    store = MonitorStore(db)
    for m in store.list_all():
        store.update(m.id, enabled=False)
    _mk(store, "late", 14400, 20000)
    _mk(store, "never-run", 14400, None)
    assert [m.name for m in store.get_due()][0] == "never-run"


def test_class_floor_promotes_starved_daily():
    # 2026-08-14: digest ratios (1.5-2+) outrank a daily's 1.3 for most of its
    # life, and restart-truncated batch tails starved eval 31h — a starved
    # other-class monitor must jump ahead of higher-ratio digests.
    from types import SimpleNamespace
    from app.monitors.heartbeat_loop import _class_floor_order

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    def _mon(name, sched, ago):
        return SimpleNamespace(
            name=name, schedule_seconds=sched,
            last_check_at=(now - timedelta(seconds=ago)).strftime("%Y-%m-%d %H:%M:%S"))

    digest_a = _mon("digest-a", 14400, 30000)   # ratio ~2.1
    digest_b = _mon("digest-b", 14400, 26000)   # ratio ~1.8
    eval_m = _mon("eval", 86400, 115000)        # ratio ~1.33 — starved daily
    fresh_other = _mon("quiz", 21600, 22000)    # ratio ~1.02 — not starved

    classify = lambda m: "other" if m.name in ("eval", "quiz") else "digest"
    out = _class_floor_order([digest_a, digest_b, eval_m, fresh_other], classify, now)
    assert out[0].name == "eval"                          # starved daily first
    assert [m.name for m in out[1:]] == ["digest-a", "digest-b", "quiz"]
