"""Regression: monitor_results storage must NOT amputate rich digests.

A 4000-char cap in monitor_store silently truncated every stored digest to a
third of what Discord posted (found 2026-07-06): the report grader scored
stumps (Science support 0.417 — trailing citations cut mid-bullet, sections
gone) and storylines/cross-monitor consumed the same amputated rows. The cap
must stay ≥ the posted-digest cap (12000, heartbeat_loop)."""
from __future__ import annotations

from app.database import SafeDB
from app.monitors.heartbeat import MonitorStore


def _store():
    db = SafeDB(":memory:")
    db.init_schema()
    return MonitorStore(db), db


def test_add_result_keeps_rich_digest_intact():
    store, db = _store()
    mid = store.create(name="t-monitor", check_type="query", check_config={})
    digest = ("## overview\n" + ("A cited, substantive sentence with detail (host.com). " * 220))
    assert 4000 < len(digest) <= 12000
    store.add_result(mid, "changed", value=digest, message=digest)
    row = db.fetchone("SELECT value, message FROM monitor_results WHERE monitor_id=?", (mid,))
    assert row["value"] == digest, "stored digest was truncated below the posted-digest cap"
    assert row["message"] == digest


def test_result_cap_at_least_posted_cap():
    assert MonitorStore._RESULT_CAP >= 12000


def test_record_check_keeps_rich_result():
    store, db = _store()
    mid = store.create(name="t-monitor2", check_type="query", check_config={})
    result = "R" * 9000
    store.record_check(mid, result)
    row = db.fetchone("SELECT last_result FROM monitors WHERE id=?", (mid,))
    assert row["last_result"] == result
