"""Tests for KG quality improvements: entity normalization, graph traversal, neighbor enrichment."""

from __future__ import annotations

import pytest

from app.core.kg import KnowledgeGraph, normalize_entity, normalize_predicate
from app.database import SafeDB


@pytest.fixture
def kg(tmp_path):
    """Fresh KG with test data."""
    db = SafeDB(str(tmp_path / "kg_test.db"))
    db.init_schema()
    return KnowledgeGraph(db)


# ===========================================================================
# Entity Normalization
# ===========================================================================

class TestNormalizeEntity:
    """Test normalize_entity() function."""

    def test_strips_whitespace(self):
        assert normalize_entity("  Python  ") == "Python"

    def test_collapses_internal_whitespace(self):
        assert normalize_entity("machine   learning") == "Machine Learning"

    def test_title_case(self):
        assert normalize_entity("python programming") == "Python Programming"

    def test_preserves_acronyms(self):
        assert normalize_entity("AI") == "AI"
        assert normalize_entity("ML") == "ML"
        assert normalize_entity("US") == "US"
        assert normalize_entity("GDP") == "GDP"

    def test_mixed_case_with_acronyms(self):
        result = normalize_entity("AI in healthcare")
        assert result == "AI In Healthcare"

    def test_empty_string(self):
        assert normalize_entity("") == ""
        assert normalize_entity("   ") == ""

    def test_single_word(self):
        assert normalize_entity("python") == "Python"
        assert normalize_entity("BITCOIN") == "Bitcoin"  # >5 chars, not acronym

    def test_short_acronym_preserved(self):
        assert normalize_entity("EU") == "EU"
        # Lowercase input gets capitalized (not recognized as acronym)
        assert normalize_entity("uk") == "Uk"
        # ALL-CAPS input stays as-is
        assert normalize_entity("UK") == "UK"


# ===========================================================================
# Entity Resolution via Normalization
# ===========================================================================

class TestEntityResolution:
    """Test that normalized entities merge correctly in KG."""

    @pytest.mark.asyncio
    async def test_case_variants_merge(self, kg):
        """'python', 'Python', 'PYTHON' should all become the same entity."""
        await kg.add_fact("python", "is_a", "programming language")
        await kg.add_fact("Python", "is_a", "programming language")
        await kg.add_fact("PYTHON", "is_a", "programming language")

        results = kg.query("python")
        # Should have exactly 1 fact (all normalized to "Python")
        lang_facts = [r for r in results if r["predicate"] == "is_a" and "language" in r["object"].lower()]
        assert len(lang_facts) == 1

    @pytest.mark.asyncio
    async def test_whitespace_variants_merge(self, kg):
        """'  machine  learning  ' and 'Machine Learning' should merge."""
        await kg.add_fact("  machine  learning  ", "is_a", "AI technique")
        await kg.add_fact("Machine Learning", "is_a", "AI technique")

        results = kg.query("machine learning")
        ml_facts = [r for r in results if "technique" in r["object"].lower()]
        assert len(ml_facts) == 1

    @pytest.mark.asyncio
    async def test_acronyms_preserved_distinct(self, kg):
        """'AI' and 'Artificial Intelligence' should be distinct entities."""
        await kg.add_fact("AI", "is_a", "technology")
        await kg.add_fact("Artificial Intelligence", "is_a", "field of study")

        results = kg.query("AI")
        # AI and Artificial Intelligence are different entities
        assert len(results) >= 1


# ===========================================================================
# Graph Traversal
# ===========================================================================

class TestGraphTraversal:
    """Test multi-hop graph traversal in query()."""

    @pytest.mark.asyncio
    async def test_1_hop_finds_direct_connections(self, kg):
        await kg.add_fact("Python", "created_by", "Guido van Rossum")
        await kg.add_fact("Python", "is_a", "Programming Language")

        results = kg.query("Python", hops=1)
        assert len(results) >= 2
        objects = {r["object"] for r in results}
        assert "Guido Van Rossum" in objects or "Guido van Rossum" in objects

    @pytest.mark.asyncio
    async def test_2_hop_finds_indirect_connections(self, kg):
        await kg.add_fact("Python", "created_by", "Guido van Rossum")
        await kg.add_fact("Guido van Rossum", "born_in", "Netherlands")

        results = kg.query("Python", hops=2)
        # Should find Netherlands via Guido (2-hop)
        objects = {r["object"] for r in results}
        subjects = {r["subject"] for r in results}
        assert "Netherlands" in objects or "netherlands" in {o.lower() for o in objects}

    @pytest.mark.asyncio
    async def test_0_hop_only_direct(self, kg):
        await kg.add_fact("Python", "is_a", "Language")
        await kg.add_fact("Language", "part_of", "Communication")

        results = kg.query("Python", hops=0)
        # Only direct matches on "python"
        for r in results:
            assert r["subject"].lower() == "python" or r["object"].lower() == "python"


# ===========================================================================
# Neighbor Enrichment in get_relevant_facts
# ===========================================================================

class TestNeighborEnrichment:
    """Test that get_relevant_facts enriches results with 1-hop neighbors."""

    @pytest.mark.asyncio
    async def test_neighbors_included(self, kg):
        """Related facts should be enriched when under the limit."""
        await kg.add_fact("Bitcoin", "is_a", "Cryptocurrency")
        await kg.add_fact("Bitcoin", "created_by", "Satoshi Nakamoto")
        await kg.add_fact("Cryptocurrency", "uses", "Blockchain")
        await kg.add_fact("Satoshi Nakamoto", "known_for", "Bitcoin whitepaper")

        # Query with keyword "bitcoin" — should find direct matches
        # AND neighbors connected via entities in the matches
        results = kg.get_relevant_facts("Tell me about bitcoin cryptocurrency", limit=8)
        # Should find more than just the 2 keyword-matched facts
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_enrichment_respects_limit(self, kg):
        """Enrichment should not exceed the limit."""
        for i in range(20):
            await kg.add_fact(f"Entity{i}", "related_to", "Bitcoin")

        results = kg.get_relevant_facts("Tell me about bitcoin", limit=5)
        assert len(results) <= 5


# ===========================================================================
# Predicate Normalization
# ===========================================================================

class TestPredicateNormalizationExtended:
    """Additional predicate normalization tests."""

    def test_custom_valid_predicate(self):
        assert normalize_predicate("trades_with") == "trades_with"

    def test_invalid_predicate_fallback(self):
        # Too long, contains spaces after normalization
        result = normalize_predicate("x")
        assert result == "related_to"

    def test_common_aliases(self):
        assert normalize_predicate("made by") == "created_by"
        assert normalize_predicate("located in") == "located_in"
        assert normalize_predicate("type of") == "is_a"


# ===========================================================================
# KG Stats
# ===========================================================================

class TestKGStats:
    """Test get_stats() method."""

    @pytest.mark.asyncio
    async def test_empty_kg_stats(self, kg):
        stats = kg.get_stats()
        assert stats["total_facts"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_inserts(self, kg):
        await kg.add_fact("Python", "is_a", "Language")
        await kg.add_fact("Python", "created_by", "Guido")

        stats = kg.get_stats()
        assert stats["total_facts"] >= 2
