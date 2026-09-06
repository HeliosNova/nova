"""A week of waiting should count for something (2026-09-06).

Three questions minted on 2026-08-30 were answered on 09-06, having waited
7.4, 7.4 and 7.5 days — and every one resolved on its FIRST attempt. They were
never hard. They were behind a stream of slightly-more-urgent work: seven fresh
0.69 items were minted that week, and each one jumped a backlog of about forty
0.6s. A pure `ORDER BY urgency DESC` has no term for how long something has
waited, so a small steady supply of 0.69s defers the 0.6s indefinitely.

The aging is applied in the ORDER BY and NEVER written back, which is the part
that matters most here. The stored `urgency` column is read elsewhere as a
THRESHOLD: daemon.py counts pending rows at >= 0.7 to decide whether to run its
idle ~5-minute critical loop, and a stream sitting in that band once drove ~124
unresolvable brain.think() calls a day (2026-08-18). Aging a row's stored
urgency would recreate that outage exactly.
"""
from __future__ import annotations

import pytest

from app.core.curiosity import AGING_CAP, AGING_PER_DAY, CuriosityQueue


@pytest.fixture
def db(tmp_path):
    from app.database import SafeDB
    d = SafeDB(str(tmp_path / "c.db"))
    d.init_schema()
    d.execute(
        "CREATE TABLE IF NOT EXISTS curiosity_queue ("
        "id INTEGER PRIMARY KEY, topic TEXT, source TEXT, urgency REAL, "
        "status TEXT, attempts INT DEFAULT 0, resolution TEXT, "
        "created_at TEXT DEFAULT CURRENT_TIMESTAMP, resolved_at TEXT)")
    yield d
    d.close()


def _add(db, topic, urgency, days_old):
    db.execute(
        "INSERT INTO curiosity_queue (topic, source, urgency, status, attempts, created_at) "
        "VALUES (?, 'dossier_open_question', ?, 'pending', 0, "
        "datetime('now', ?))", (topic, urgency, f"-{days_old} days"))


def test_a_fresh_urgent_item_still_wins_at_first(db):
    """Urgency has to keep meaning something, or this is just FIFO."""
    _add(db, "old but ordinary", 0.6, 1)
    _add(db, "fresh and urgent", 0.69, 0)
    assert CuriosityQueue(db).get_next().topic == "fresh and urgent"


def test_a_week_old_question_is_no_longer_jumped(db):
    """The measured case: 0.6 minted 08-30, 0.69 minted 09-05, answered 09-06."""
    _add(db, "waiting since 08-30", 0.6, 7)
    _add(db, "minted yesterday", 0.69, 1)
    assert CuriosityQueue(db).get_next().topic == "waiting since 08-30"


def test_the_gap_closes_in_about_three_days(db):
    """0.09 of urgency at 0.03/day. Before that the urgent item leads."""
    _add(db, "backlog", 0.6, 2)
    _add(db, "urgent", 0.69, 0)
    assert CuriosityQueue(db).get_next().topic == "urgent"
    db.execute("UPDATE curiosity_queue SET created_at = datetime('now', '-4 days') "
               "WHERE topic = 'backlog'")
    assert CuriosityQueue(db).get_next().topic == "backlog"


def test_nothing_is_deferred_for_ever(db):
    """The cap spans the whole 0.5-0.7 range, so even the least urgent item
    reaches the front eventually."""
    _add(db, "lowest urgency, ancient", 0.5, 30)
    _add(db, "highest urgency, fresh", 0.7, 0)
    assert CuriosityQueue(db).get_next().topic == "lowest urgency, ancient"
    assert AGING_CAP > 0.20, (
        "the cap must EXCEED the 0.5-0.7 urgency range, not merely match it: at "
        "exactly 0.20 an aged 0.5 reaches 0.70 and ties a fresh 0.70, and the "
        "winner is then decided by float representation")


def test_the_aging_is_bounded(db):
    """Without a cap a stale item would outrank everything for ever, which is
    just the same starvation pointed the other way."""
    _add(db, "ancient", 0.5, 365)
    _add(db, "also ancient", 0.5, 400)
    _add(db, "recent but far more urgent", 0.5 + AGING_CAP + 0.05, 0)
    assert CuriosityQueue(db).get_next().topic == "recent but far more urgent"


def test_ties_go_to_whoever_waited_longer(db):
    _add(db, "older", 0.6, 5)
    _add(db, "newer", 0.6, 4)
    assert CuriosityQueue(db).get_next().topic == "older"


def test_the_STORED_urgency_is_never_modified(db):
    """daemon.py fires its idle 5-minute critical loop on pending urgency>=0.7.
    Aging a row INTO that band would recreate the 2026-08-18 outage where a
    stream of them drove ~124 unresolvable brain.think() calls a day."""
    _add(db, "ancient and ordinary", 0.6, 90)
    q = CuriosityQueue(db)
    for _ in range(5):
        q.get_next()
    rows = db.fetchall("SELECT urgency FROM curiosity_queue")
    assert [r["urgency"] for r in rows] == [0.6]
    critical = db.fetchone(
        "SELECT COUNT(*) c FROM curiosity_queue WHERE status='pending' AND urgency >= 0.7")
    assert critical["c"] == 0, "aging must never promote a row into the critical band"


def test_the_attempt_cap_still_applies(db):
    """Aging must not resurrect an item that has exhausted its attempts."""
    _add(db, "burned out", 0.6, 60)
    db.execute("UPDATE curiosity_queue SET attempts = 3 WHERE topic = 'burned out'")
    _add(db, "still eligible", 0.6, 0)
    assert CuriosityQueue(db).get_next().topic == "still eligible"


def test_an_empty_queue_is_still_none(db):
    assert CuriosityQueue(db).get_next() is None


def test_a_null_created_at_does_not_break_the_sort(db):
    """Defensive: a row with no timestamp must not NULL the whole expression
    and silently sort every aged item behind it."""
    _add(db, "normal", 0.6, 3)
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, attempts, "
               "created_at) VALUES ('no timestamp', 'x', 0.6, 'pending', 0, NULL)")
    assert CuriosityQueue(db).get_next().topic == "normal"


def test_the_aging_rate_is_documented_where_it_is_used():
    import inspect
    from app.core import curiosity
    src = inspect.getsource(curiosity.CuriosityQueue.get_next)
    assert "never written back" in src.lower() or "ORDER BY ONLY" in src
    assert AGING_PER_DAY > 0
