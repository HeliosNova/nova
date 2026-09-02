"""KG accumulation regression coverage (audit 2026-09-01).

The live KG was a 5,000-fact ring buffer: `_prune()` retired facts ordered by
`times_retrieved ASC, confidence ASC, created_at ASC`, so a brand-new fact
(0 retrievals) was always the first victim. Measured: 2,036 facts learned in
7 days, 513 still live; 5,514 facts retired at age <1 day with 0.08 average
retrievals; cross_synthesis 83 minted / 1 live. Nova structurally could not
accumulate knowledge past the cap.

These tests lock in the new contract:
  - facts younger than 14 days are NEVER evicted by the cap;
  - provenance classes that represent distilled understanding (principle,
    cross_synthesis, storyline, curiosity) are never evicted by the cap;
  - among eligible facts, never-retrieved ones go first, then the least
    recently retrieved, then the lowest confidence;
  - the keyword arm of retrieval no longer depends on a candidate window
    bounded by the cap (FTS5 candidates), so a relevant low-confidence fact
    is still found in a store far larger than the cap;
  - the FTS index stays in sync through inserts, updates and retirement.
Deterministic — no LLM in the loop.
"""
from __future__ import annotations

import pytest

from app.config import config as _cfg
from app.core.kg import KnowledgeGraph


def _insert(db, subject, predicate, object_, *, days_old=0, times_retrieved=0,
            confidence=0.8, source="extracted", provenance="", last_retrieved_days=None):
    last = None if last_retrieved_days is None else f"-{last_retrieved_days} days"
    db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, source, provenance, "
        "created_at, valid_from, valid_to, times_retrieved, last_retrieved_at) "
        "VALUES (?, ?, ?, ?, ?, ?, datetime('now', ?), datetime('now', ?), NULL, ?, "
        "CASE WHEN ? IS NULL THEN NULL ELSE datetime('now', ?) END)",
        (subject, predicate, object_, confidence, source, provenance,
         f"-{days_old} days", f"-{days_old} days", times_retrieved, last, last or "-0 days"),
    )


def _live_subjects(db) -> set[str]:
    return {r["subject"] for r in db.fetchall(
        "SELECT subject FROM kg_facts WHERE valid_to IS NULL")}


@pytest.fixture
def small_cap():
    old = _cfg.MAX_KG_FACTS
    _cfg.update(MAX_KG_FACTS=10)
    yield
    _cfg.update(MAX_KG_FACTS=old)


# ---------------------------------------------------------------------------
# Eviction policy
# ---------------------------------------------------------------------------

def test_prune_never_evicts_facts_younger_than_14_days(db, small_cap):
    kg = KnowledgeGraph(db)
    for i in range(8):
        _insert(db, f"OldFact{i}", "is_a", f"old thing {i}", days_old=40)
    for i in range(4):
        _insert(db, f"YoungFact{i}", "is_a", f"new thing {i}", days_old=1)
    kg._prune()  # 12 live > cap 10 → 2 must go
    live = _live_subjects(db)
    assert {f"YoungFact{i}" for i in range(4)} <= live, "a fact under 14 days old was evicted"
    assert len(live) == 10
    assert len({s for s in live if s.startswith("OldFact")}) == 6


def test_prune_evicts_only_eligible_when_excess_is_all_young(db, small_cap):
    kg = KnowledgeGraph(db)
    for i in range(3):
        _insert(db, f"OldFact{i}", "is_a", f"old thing {i}", days_old=40)
    for i in range(12):
        _insert(db, f"YoungFact{i}", "is_a", f"new thing {i}", days_old=2)
    kg._prune()  # 15 live, cap 10, but only 3 eligible → live stays at 12
    live = _live_subjects(db)
    assert len(live) == 12
    assert not any(s.startswith("OldFact") for s in live)


def test_prune_protects_distilled_provenance_classes(db, small_cap):
    kg = KnowledgeGraph(db)
    _insert(db, "PrincipleFact", "is_a", "distilled", days_old=90, source="principle")
    _insert(db, "CrossFact", "is_a", "cross-monitor pattern", days_old=90, source="cross_synthesis")
    _insert(db, "StoryFact", "has_status", "state", days_old=90, source="storyline")
    _insert(db, "CuriosityFact", "is_a", "researched answer", days_old=90,
            provenance="curiosity:item 12")
    for i in range(9):
        _insert(db, f"Plain{i}", "is_a", f"ordinary {i}", days_old=60)
    kg._prune()  # 13 live > 10 → 3 evicted, all from Plain*
    live = _live_subjects(db)
    assert {"PrincipleFact", "CrossFact", "StoryFact", "CuriosityFact"} <= live
    assert len({s for s in live if s.startswith("Plain")}) == 6


def test_prune_prefers_never_retrieved_then_least_recently_retrieved(db, small_cap):
    kg = KnowledgeGraph(db)
    # 12 old facts: 2 never retrieved, 10 retrieved at various recency.
    _insert(db, "NeverA", "is_a", "x", days_old=30, times_retrieved=0)
    _insert(db, "NeverB", "is_a", "x", days_old=30, times_retrieved=0, confidence=0.95)
    for i in range(10):
        # retrieved recently (1 day) except Stale0/Stale1 retrieved 25 days ago
        last = 25 if i < 2 else 1
        _insert(db, f"Stale{i}", "is_a", "x", days_old=30, times_retrieved=5,
                last_retrieved_days=last)
    kg._prune()  # 12 → 10: the two never-retrieved go first
    live = _live_subjects(db)
    assert "NeverA" not in live and "NeverB" not in live
    assert len(live) == 10
    # Push over the cap again with two more young facts (protected) → next
    # victims must be the least recently retrieved old facts.
    _insert(db, "Young0", "is_a", "x", days_old=1)
    _insert(db, "Young1", "is_a", "x", days_old=1)
    kg._prune()
    live = _live_subjects(db)
    assert "Stale0" not in live and "Stale1" not in live
    assert {"Young0", "Young1"} <= live


def test_default_cap_is_no_longer_five_thousand():
    assert int(_cfg.MAX_KG_FACTS) >= 50000


# ---------------------------------------------------------------------------
# Keyword candidates no longer bounded by the cap
# ---------------------------------------------------------------------------

def _seed_filler(db, n: int) -> None:
    rows = [(f"FillerEntity{i}", "is_a", f"placeholder category number {i}", 0.95)
            for i in range(n)]
    db.executemany(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, valid_to) "
        "VALUES (?, ?, ?, ?, NULL)", rows)


def test_keyword_arm_finds_target_far_beyond_candidate_cap(db):
    """With the cap at 10 and 2,000 higher-confidence fillers, the only fact
    sharing tokens with the query must still be retrieved: the keyword arm
    must not be a confidence-ordered window."""
    kg = KnowledgeGraph(db)
    _seed_filler(db, 2000)
    db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, valid_to) "
        "VALUES ('Zephyrian Quasar Drive', 'produces', 'tachyon pulses', 0.30, NULL)")
    old = _cfg.MAX_KG_FACTS
    _cfg.update(MAX_KG_FACTS=10)
    try:
        facts = kg.get_relevant_facts("what does the Zephyrian Quasar Drive produce", limit=8)
    finally:
        _cfg.update(MAX_KG_FACTS=old)
    assert "Zephyrian Quasar Drive" in {f.subject for f in facts}


def test_keyword_arm_matches_stemmed_and_irregular_forms(db):
    kg = KnowledgeGraph(db)
    _seed_filler(db, 50)
    db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, valid_to) "
        "VALUES ('Airbus', 'headquartered_in', 'France', 0.9, NULL)")
    db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, valid_to) "
        "VALUES ('Dassault', 'produces', 'French fighter jets', 0.9, NULL)")
    facts = kg.get_relevant_facts("which companies are headquartered in France", limit=8)
    assert "Airbus" in {f.subject for f in facts}
    facts = kg.get_relevant_facts("who produces French fighter jets", limit=8)
    assert "Dassault" in {f.subject for f in facts}


def test_fts_index_tracks_inserts_updates_and_retirement(db):
    kg = KnowledgeGraph(db)
    _seed_filler(db, 20)
    db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, valid_to) "
        "VALUES ('Orbital Widget Corp', 'launched', 'Nimbus satellite bus', 0.9, NULL)")
    assert "Orbital Widget Corp" in {
        f.subject for f in kg.get_relevant_facts("Orbital Widget Corp launched what", limit=8)}
    # Update the subject → old words no longer match, new words do.
    db.execute("UPDATE kg_facts SET subject='Stellar Gadget Inc' WHERE subject='Orbital Widget Corp'")
    assert "Stellar Gadget Inc" in {
        f.subject for f in kg.get_relevant_facts("Stellar Gadget Inc launched what", limit=8)}
    assert not {f.subject for f in kg.get_relevant_facts("Orbital Widget Corp launched what", limit=8)}
    # Retire it → never retrieved again.
    db.execute("UPDATE kg_facts SET valid_to = CURRENT_TIMESTAMP WHERE subject='Stellar Gadget Inc'")
    assert not {f.subject for f in kg.get_relevant_facts("Stellar Gadget Inc launched what", limit=8)}


def test_quarantined_facts_stay_out_of_keyword_candidates(db):
    kg = KnowledgeGraph(db)
    _seed_filler(db, 20)
    db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, valid_to, quarantined) "
        "VALUES ('Poisoned Widget', 'is_a', 'malicious claim', 0.9, NULL, 1)")
    assert not {f.subject for f in kg.get_relevant_facts("Poisoned Widget malicious claim", limit=8)}
