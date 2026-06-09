"""Tests for PPR (Personalized PageRank) over the KG graph.

Covers:
  - extract_entities: proper-noun phrases + content tokens, stop-word filtering
  - compute_ppr: empty seeds, missing seeds, basic propagation
  - rank_facts_by_ppr: top-k ranking
  - invalidate_cache: clears _ADJ_CACHE so the next call rebuilds
  - is_enabled: respects config flag
"""

from __future__ import annotations

import pytest

from app.core import ppr
from app.database import get_db, _instances


@pytest.fixture(autouse=True)
def _reset_ppr_cache():
    """Force a fresh adjacency build for each test."""
    ppr.invalidate_cache()
    yield
    ppr.invalidate_cache()


@pytest.fixture
def kg_db(tmp_path, monkeypatch):
    """Tiny KG DB with a known fact graph for PPR tests.

    Graph (undirected):
        apple — tim cook
        apple — steve jobs
        apple — ios
        ios — swift
        microsoft — windows
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "ppr.db"))
    _instances.clear()
    db = get_db()
    db.init_schema()
    facts = [
        ("apple", "ceo_is", "tim cook", 0.95),
        ("apple", "founded_by", "steve jobs", 0.92),
        ("apple", "makes", "ios", 0.9),
        ("ios", "developed_in", "swift", 0.85),
        ("microsoft", "makes", "windows", 0.9),
    ]
    for s, p, o, c in facts:
        db.execute(
            "INSERT INTO kg_facts (subject, predicate, object, confidence, source, valid_from) "
            "VALUES (?, ?, ?, ?, 'test', CURRENT_TIMESTAMP)",
            (s, p, o, c),
        )
    yield db
    db.close()
    _instances.clear()


# ---- extract_entities ---------------------------------------------------

def test_extract_entities_proper_noun_phrase():
    seeds = ppr.extract_entities("Who is Tim Cook of Apple?")
    assert seeds  # not empty
    # multi-word phrase first
    assert any("tim cook" == s or "tim cook of apple" == s for s in seeds)


def test_extract_entities_strips_stopwords():
    seeds = ppr.extract_entities("what is the latest about quantum mechanics")
    # "what", "the", "is", "about" must not appear
    for stop in ("what", "the", "is", "about"):
        assert stop not in seeds


def test_extract_entities_empty_query():
    assert ppr.extract_entities("") == []
    assert ppr.extract_entities("   ") == []


def test_extract_entities_caps_at_max_seeds():
    # 10 distinct content tokens but max_seeds=3
    q = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    seeds = ppr.extract_entities(q, max_seeds=3)
    assert len(seeds) == 3


# ---- compute_ppr --------------------------------------------------------

def test_compute_ppr_empty_seeds(kg_db):
    assert ppr.compute_ppr([]) == {}


def test_compute_ppr_unknown_seed(kg_db):
    # "yellowstone" isn't in the graph and has no substring match
    out = ppr.compute_ppr(["yellowstone"])
    assert out == {}


def test_compute_ppr_basic_propagation(kg_db):
    # Seed at "apple" — apple should have high mass; tim cook / steve jobs / ios
    # should all get non-trivial mass; microsoft should get nearly zero
    out = ppr.compute_ppr(["apple"], top_k=10)
    assert out  # non-empty
    assert "apple" in out
    # neighbors get mass
    neighbor_mass = sum(
        out.get(n, 0.0) for n in ("tim cook", "steve jobs", "ios")
    )
    assert neighbor_mass > 0.05
    # microsoft is in a disconnected component — should be 0
    assert out.get("microsoft", 0.0) == 0.0


def test_compute_ppr_substring_fallback(kg_db):
    # query "apple inc" — not in graph as exact node, but "apple" is a substring
    out = ppr.compute_ppr(["apple inc"])
    # substring fallback should let "apple" become the seed
    assert "apple" in out


# ---- rank_facts_by_ppr --------------------------------------------------

def test_rank_facts_by_ppr_returns_top_k(kg_db):
    rows = kg_db.fetchall(
        "SELECT id, subject, predicate, object FROM kg_facts WHERE valid_to IS NULL"
    )
    ranked = ppr.rank_facts_by_ppr(list(rows), seeds=["apple"], top_k=3)
    assert len(ranked) <= 3
    # ranked is descending by score
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    # facts about apple come back; microsoft fact does not (zero score)
    fact_ids = {fid for fid, _ in ranked}
    apple_ids = {
        r["id"] for r in rows if r["subject"] == "apple" or r["object"] == "apple"
    }
    assert apple_ids & fact_ids


def test_rank_facts_by_ppr_empty_inputs(kg_db):
    assert ppr.rank_facts_by_ppr([], seeds=["apple"]) == []
    rows = kg_db.fetchall("SELECT id, subject, object FROM kg_facts LIMIT 1")
    assert ppr.rank_facts_by_ppr(list(rows), seeds=[]) == []


# ---- cache invalidation -------------------------------------------------

def test_invalidate_cache_forces_rebuild(kg_db):
    # Trigger a build
    out_before = ppr.compute_ppr(["apple"])
    # Add a new fact bypassing the public KG path
    kg_db.execute(
        "INSERT INTO kg_facts (subject, predicate, object, confidence, source, valid_from) "
        "VALUES ('apple', 'has_subsidiary', 'beats', 0.8, 'test', CURRENT_TIMESTAMP)"
    )
    # Without invalidation: stale (TTL=300s in test)
    out_stale = ppr.compute_ppr(["apple"])
    assert "beats" not in out_stale
    # After invalidation: fresh build sees beats
    ppr.invalidate_cache()
    out_after = ppr.compute_ppr(["apple"])
    assert "beats" in out_after


# ---- is_enabled flag ----------------------------------------------------

def test_is_enabled_reads_config(monkeypatch):
    from app.config import reset_config
    monkeypatch.setenv("ENABLE_PPR_RETRIEVAL", "false")
    reset_config()
    assert ppr.is_enabled() is False
    monkeypatch.setenv("ENABLE_PPR_RETRIEVAL", "true")
    reset_config()
    assert ppr.is_enabled() is True
