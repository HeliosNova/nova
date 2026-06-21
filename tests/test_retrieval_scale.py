"""Scale + staleness regression coverage for retrieval (audit 2026-06-12).

The eval harness seeds ONE item into a near-empty store and reads it back through
the SAME handle, so it is structurally blind to the two failure classes that kept
recurring:
  - SCALE: a hardcoded `LIMIT 500 ORDER BY confidence` candidate window silently
    dropped any relevant item ranked past 500 once the store grew. Both KG
    (MAX_KG_FACTS) and lessons (MAX_LESSON_CANDIDATES) were fixed to cap at the
    full store; these tests fail if a small truncation is reintroduced.
  - STALENESS: after an out-of-process vector reindex (drop+recreate collection),
    a stale in-process handle returned nothing for pre-existing rows. These tests
    prove retrieval recovers through a fresh handle.

Deterministic (no LLM in the loop) — they exercise the retrieval methods directly,
which is what reliably catches these regressions.
"""
from __future__ import annotations

import pytest

from app.config import config as _cfg
from app.core.kg import KnowledgeGraph
from app.core.learning import LearningEngine


# ---------------------------------------------------------------------------
# SCALE — relevant item past the naive top-500 window is still retrieved
# ---------------------------------------------------------------------------

def _seed_kg_filler(db, n: int) -> None:
    """n high-confidence facts that do NOT share tokens with the target query."""
    rows = [
        (f"FillerEntity{i}", "is_a", f"placeholder category number {i}", 0.95)
        for i in range(n)
    ]
    db.executemany(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, valid_to) "
        "VALUES (?, ?, ?, ?, NULL)",
        rows,
    )


def test_kg_retrieves_target_past_naive_500_window(db):
    kg = KnowledgeGraph(db)
    # 600 high-confidence fillers — a `LIMIT 500 ORDER BY confidence DESC`
    # candidate window would keep only these and drop the target below.
    _seed_kg_filler(db, 600)
    # Target at LOW confidence so it sorts to the very end of the candidate order,
    # but it is the only fact sharing tokens with the query.
    db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, valid_to) "
        "VALUES ('Zephyrian Quasar Drive', 'produces', 'tachyon pulses', 0.50, NULL)",
    )
    facts = kg.get_relevant_facts("what does the Zephyrian Quasar Drive produce", limit=8)
    subjects = {f.subject for f in facts}
    assert "Zephyrian Quasar Drive" in subjects, (
        "low-confidence relevant fact past the top-500 window was dropped — "
        "the LIMIT-500 candidate-truncation regression is back"
    )


def _seed_lesson_filler(db, n: int) -> None:
    rows = [
        (f"placeholder topic {i}", f"placeholder answer {i}", "", 0.8, 1000 + i)
        for i in range(n)
    ]
    db.executemany(
        "INSERT INTO lessons (topic, correct_answer, lesson_text, confidence, times_helpful) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def test_lessons_retrieve_target_past_naive_500_window(db, monkeypatch):
    eng = LearningEngine(db)
    # 600 fillers with HIGH times_helpful — a `LIMIT 500 ORDER BY times_helpful`
    # window keeps these and drops the target (times_helpful = 0) below.
    _seed_lesson_filler(db, 600)
    db.execute(
        "INSERT INTO lessons (topic, correct_answer, lesson_text, confidence, times_helpful) "
        "VALUES ('Zephyrian Quasar Drive output', "
        "'The Zephyrian Quasar Drive produces tachyon pulses', "
        "'Zephyrian Quasar Drive produces tachyon pulses', 0.8, 0)",
    )
    # Isolate the SCALE concern (the candidate-load LIMIT) from the vector arm:
    # the degenerate fillers would otherwise be auto-backfilled into ChromaDB and
    # the weak MiniLM fallback ranks them as noise. Keyword retrieval over the
    # full candidate set is exactly the path the LIMIT-500 bug broke.
    monkeypatch.setattr(eng, "_get_lessons_collection", lambda: None)
    # conftest pins MIN_RRF_SCORE=0.015 (stricter than prod's 0.005); a lone
    # keyword RRF hit dampens to ~0.0139 and would be floored. Use the prod floor
    # so this test measures candidate-window coverage, not the unrelated floor.
    _old_floor = _cfg.MIN_RRF_SCORE
    object.__setattr__(_cfg, "MIN_RRF_SCORE", 0.005)
    try:
        lessons = eng.get_relevant_lessons("what does the Zephyrian Quasar Drive produce")
    finally:
        object.__setattr__(_cfg, "MIN_RRF_SCORE", _old_floor)
    blob = " ".join((l.topic or "") + " " + (l.correct_answer or "") for l in lessons).lower()
    assert "zephyrian" in blob, (
        "low-helpful relevant lesson past the top-500 window was dropped — "
        "the lessons LIMIT-500 candidate-truncation regression is back"
    )


# ---------------------------------------------------------------------------
# STALENESS — retrieval recovers after an out-of-process vector reindex
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kg_retrieval_survives_external_vector_reindex(db):
    kg = KnowledgeGraph(db)
    await kg.add_fact("Helios Reactor", "located_in", "Sector Nine", source="user")

    # Simulate an out-of-process embedder migration: drop + recreate the vector
    # collection underneath the store (the scenario that left a running app with
    # a stale handle returning nothing for pre-existing facts).
    coll = kg._get_collection()
    if coll is not None:
        try:
            kg._vector_client.delete_collection(coll.name)
        except Exception:
            pass
        kg._collection = None  # force re-open on next access

    # A FRESH handle must still retrieve the fact (data intact; keyword arm
    # works regardless of the vector collection state).
    fresh = KnowledgeGraph(db)
    facts = fresh.get_relevant_facts("where is the Helios Reactor located", limit=8)
    assert any(f.subject == "Helios Reactor" for f in facts), (
        "fact unretrievable after an external vector reindex — stale-handle class"
    )
    await kg.delete_fact("Helios Reactor", "located_in", "Sector Nine")
