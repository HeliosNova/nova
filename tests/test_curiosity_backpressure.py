"""Curiosity queue: backpressure instead of silent destruction (2026-09-03).

Measured on the live install: the queue sat pinned at MAX_CURIOSITY_QUEUE_SIZE
while the researcher drained about five items a day, and every new topic
evicted the OLDEST pending one — 152 logged evictions of questions that had
never been researched once. The same evict-the-wrong-end shape as the KG ring
buffer. MAX_CURIOSITY_PENDING existed in config and was read nowhere.

Background work is now refused at the high-water mark (the open-questions
ledger keeps the question and offers it again later), and the hard ceiling
evicts by value rather than by age.
"""
from __future__ import annotations

import pytest

from app.core.curiosity import CuriosityQueue


def _fill(db, n, *, urgency=0.6, source="dossier_open_question", prefix="topic"):
    for i in range(n):
        db.execute(
            "INSERT INTO curiosity_queue (topic, source, urgency, status, created_at) "
            "VALUES (?, ?, ?, 'pending', datetime('now', ?))",
            (f"{prefix} number {i} about a distinct subject", source, urgency, f"-{n - i} hours"),
        )


@pytest.fixture
def q(db):
    return CuriosityQueue(db)


def _pending(db):
    return db.fetchone("SELECT COUNT(*) c FROM curiosity_queue WHERE status='pending'")["c"]


# ------------------------------------------------------------- sanitising

def test_markdown_is_stripped_before_it_reaches_a_search_query(q, db):
    qid = q.add("How much of the **$7.5 billion CAD** package is `allocated` to steel?",
                source="dossier_open_question")
    assert qid > 0
    topic = db.fetchone("SELECT topic FROM curiosity_queue WHERE id = ?", (qid,))["topic"]
    assert "**" not in topic and "`" not in topic
    assert "$7.5 billion CAD" in topic
    assert "  " not in topic


def test_leading_bullets_and_headings_are_stripped(q, db):
    qid = q.add("- ## What is the current capacity of the Rubin line?", source="dossier_open_question")
    topic = db.fetchone("SELECT topic FROM curiosity_queue WHERE id = ?", (qid,))["topic"]
    assert topic.startswith("What is the current capacity")


# ---------------------------------------------------------- backpressure

def test_background_work_is_refused_at_the_high_water_mark(q, db, monkeypatch):
    monkeypatch.setenv("MAX_CURIOSITY_PENDING", "10")
    from app.config import reset_config
    reset_config()
    _fill(db, 10)
    assert _pending(db) == 10

    refused = q.add("A brand new background question about tungsten supply",
                    source="dossier_open_question", urgency=0.6)
    assert refused == -1, "background work must be refused, not swallowed"
    assert _pending(db) == 10, "nothing may be destroyed to make room"


def test_urgent_work_still_gets_in_above_the_mark(q, db, monkeypatch):
    monkeypatch.setenv("MAX_CURIOSITY_PENDING", "10")
    monkeypatch.setenv("MAX_CURIOSITY_QUEUE_SIZE", "100")
    from app.config import reset_config
    reset_config()
    _fill(db, 10)
    urgent = q.add("An owner-facing gap that matters right now", source="gap_detection", urgency=0.8)
    assert urgent > 0
    assert _pending(db) == 11


# ------------------------------------------------------- ceiling eviction

def test_the_ceiling_evicts_the_least_valuable_not_the_oldest(q, db, monkeypatch):
    monkeypatch.setenv("MAX_CURIOSITY_PENDING", "100")
    monkeypatch.setenv("MAX_CURIOSITY_QUEUE_SIZE", "4")
    from app.config import reset_config
    reset_config()
    # oldest row is also the MOST urgent — FIFO would have destroyed it
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, status, created_at) "
               "VALUES ('oldest but most urgent question', 'gap_detection', 0.75, 'pending', "
               "datetime('now','-9 hours'))")
    db.execute("INSERT INTO curiosity_queue (topic, source, urgency, attempts, status, created_at) "
               "VALUES ('stale low value question', 'dossier_tension', 0.40, 3, 'pending', "
               "datetime('now','-2 hours'))")
    _fill(db, 2, urgency=0.6, prefix="middling")
    assert _pending(db) == 4

    q.add("A critical new question that must displace something", source="gap_detection", urgency=0.9)
    rows = {r["topic"] for r in db.fetchall("SELECT topic FROM curiosity_queue WHERE status='pending'")}
    assert "oldest but most urgent question" in rows, "age alone must not decide the victim"
    assert not any(t.startswith("stale low value") for t in rows), "the least valuable row goes"


# ------------------------------------------------------------ the ledger

def test_a_refused_topic_never_marks_the_ledger_queued(db, monkeypatch):
    """add() returns -1, which is truthy — the ledger used to record it."""
    from app.core.questions import mark_queued, sync_questions
    db.execute("INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count) "
               "VALUES ('domain', 'alpha', 'Alpha', ?, 'x', 1)",
               ("## Current understanding\nx\n## Open questions\n"
                "- What is the current capacity of the Rubin line?\n",))
    did = db.fetchone("SELECT id FROM dossiers WHERE dkey='alpha'")["id"]
    sync_questions(db, did, "alpha", db.fetchone("SELECT body FROM dossiers WHERE id=?", (did,))["body"])
    question = "What is the current capacity of the Rubin line?"

    assert mark_queued(db, "alpha", question, -1) is False
    assert mark_queued(db, "alpha", question, 0) is False
    row = db.fetchone("SELECT status, curiosity_id FROM dossier_questions WHERE dkey='alpha'")
    assert row["status"] == "open" and row["curiosity_id"] is None

    assert mark_queued(db, "alpha", question, 42) is True
    row = db.fetchone("SELECT status, curiosity_id FROM dossier_questions WHERE dkey='alpha'")
    assert row["status"] == "queued" and row["curiosity_id"] == 42


def test_pending_cap_setting_is_actually_read():
    """MAX_CURIOSITY_PENDING was defined in config and read nowhere."""
    import inspect

    from app.core import curiosity
    src = inspect.getsource(curiosity.CuriosityQueue.add)
    assert "MAX_PENDING" in src, "the pending cap must gate add(), not just exist"
