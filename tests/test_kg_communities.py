"""KG community layer (GraphRAG-style global synthesis, 2026-06-13).

Deterministic parts (no LLM, no embedder). detect_communities needs networkx;
on the live container it found 143 themes in 2994 facts. Here we pin the global
-query routing, formatting, and keyword retrieval that wrap it.
"""
import importlib.util
import json

import pytest

from app.core import kg_communities as kc

_HAS_NX = importlib.util.find_spec("networkx") is not None


@pytest.mark.parametrize("q", [
    "what are the big themes across your monitors?",
    "give me an overview of recent trends",
    "summarize the main developments lately",
    "what's happening broadly in the markets",
])
def test_global_queries_detected(q):
    assert kc.is_global_query(q) is True


@pytest.mark.parametrize("q", [
    "what is the price of gold",
    "who is the CEO of Apple",
    "hello there",
])
def test_specific_queries_not_global(q):
    assert kc.is_global_query(q) is False


def test_format_for_prompt():
    assert kc.format_for_prompt([]) == ""
    out = kc.format_for_prompt([{"title": "AI policy", "summary": "Regulation is ramping up.", "entity_count": 12}])
    assert "AI policy" in out and "Regulation" in out


def test_retrieval_keyword_and_fallback(db):
    kc.ensure_schema(db)
    db.execute(
        "INSERT INTO kg_communities (title, summary, entities, entity_count) VALUES (?,?,?,?)",
        ("AI hardware", "Chips and robotics suppliers are scaling.", json.dumps(["AgiBot", "AI Chips"]), 30),
    )
    db.execute(
        "INSERT INTO kg_communities (title, summary, entities, entity_count) VALUES (?,?,?,?)",
        ("EU defense finance", "European military loans and central banks.", json.dumps(["Bank of England"]), 20),
    )
    # Keyword overlap routes to the right theme.
    hits = kc.get_relevant_communities(db, "what's new in AI chips and robotics", limit=2)
    assert hits and hits[0]["title"] == "AI hardware"
    # Bare "themes" with no overlap still returns the largest clusters.
    fallback = kc.get_relevant_communities(db, "themes please", limit=2)
    assert len(fallback) == 2


@pytest.mark.skipif(not _HAS_NX, reason="networkx not installed on this host")
def test_detect_communities_finds_clusters():
    # Two dense clusters joined sparsely -> >=2 communities.
    facts = []
    for a in "ABCD":
        for b in "ABCD":
            if a < b:
                facts.append({"subject": f"x{a}", "predicate": "p", "object": f"x{b}", "confidence": 1.0})
    for a in "EFGH":
        for b in "EFGH":
            if a < b:
                facts.append({"subject": f"y{a}", "predicate": "p", "object": f"y{b}", "confidence": 1.0})
    comms = kc.detect_communities(facts, min_size=3)
    assert len(comms) >= 2
