"""Tests for KG entity canonicalization (casing-variant collapse)."""
from __future__ import annotations

import pytest

from app.core.kg import KnowledgeGraph, normalize_entity, _casing_score


def test_normalize_entity_preserves_casing():
    # No more naive .capitalize() mangling.
    assert normalize_entity("BlackRock") == "BlackRock"
    assert normalize_entity("OpenAI") == "OpenAI"
    assert normalize_entity("iPhone 17e") == "iPhone 17e"
    assert normalize_entity("AMD") == "AMD"
    # still collapses/strips whitespace
    assert normalize_entity("  Acme   Corp  ") == "Acme Corp"
    assert normalize_entity("") == ""


def test_casing_score_ranks_intentional_casing():
    assert _casing_score("BlackRock") > _casing_score("Blackrock")
    assert _casing_score("AMD") > _casing_score("Amd")
    assert _casing_score("OpenAI") > _casing_score("Openai")


def test_canonical_entity_registers_and_resolves(db):
    kg = KnowledgeGraph(db)
    # First sighting registers the canonical form.
    assert kg._canonical_entity("BlackRock", register=True) == "BlackRock"
    # Any casing variant resolves to it (lookup only — no upgrade).
    assert kg._canonical_entity("blackrock") == "BlackRock"
    assert kg._canonical_entity("BLACKROCK") == "BlackRock"
    # Query path must not pollute the registry with a new lowercase form.
    assert kg._canonical_entity("blackrock", register=False) == "BlackRock"


def test_canonical_entity_upgrades_to_richer_casing(db):
    kg = KnowledgeGraph(db)
    # A poor-casing form registers first...
    assert kg._canonical_entity("amd", register=True) == "amd"
    # ...then a richer-cased form on a write path upgrades the canonical.
    assert kg._canonical_entity("AMD", register=True) == "AMD"
    assert kg._canonical_entity("amd") == "AMD"


@pytest.mark.asyncio
async def test_add_fact_collapses_casing_variants(db):
    kg = KnowledgeGraph(db)
    await kg.add_fact("amd", "is_a", "company", confidence=0.8)
    await kg.add_fact("AMD", "is_a", "company", confidence=0.9)
    # One active fact, subject upgraded to the richer casing.
    rows = [r for r in db.fetchall(
        "SELECT subject FROM kg_facts WHERE valid_to IS NULL AND LOWER(subject)='amd'")]
    assert len(rows) == 1
    assert rows[0]["subject"] == "AMD"
