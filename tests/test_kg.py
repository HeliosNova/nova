"""Tests for the Knowledge Graph module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.kg import KnowledgeGraph, Fact, normalize_predicate, CANONICAL_PREDICATES, is_garbage_triple


# ===========================================================================
# Predicate Normalization
# ===========================================================================

class TestPredicateNormalization:
    def test_canonical_unchanged(self):
        assert normalize_predicate("is_a") == "is_a"
        assert normalize_predicate("located_in") == "located_in"

    def test_spaces_to_underscores(self):
        assert normalize_predicate("is a") == "is_a"
        assert normalize_predicate("located in") == "located_in"
        assert normalize_predicate("created by") == "created_by"

    def test_aliases(self):
        assert normalize_predicate("made by") == "created_by"
        assert normalize_predicate("famous for") == "known_for"
        assert normalize_predicate("is capital of") == "capital_of"
        assert normalize_predicate("authored by") == "written_by"
        assert normalize_predicate("established in") == "founded_in"

    def test_prefix_stripping(self):
        assert normalize_predicate("is_located_in") == "located_in"
        assert normalize_predicate("has_property") == "has_property"

    def test_unknown_falls_back(self):
        assert normalize_predicate("!!!xyz") == "related_to"
        assert normalize_predicate("a") == "related_to"  # too short
        # Well-formed custom predicates are now allowed through
        assert normalize_predicate("dances with") == "dances_with"
        assert normalize_predicate("smells like") == "smells_like"

    def test_case_insensitive(self):
        assert normalize_predicate("IS A") == "is_a"
        assert normalize_predicate("Located In") == "located_in"

    def test_whitespace_trimming(self):
        assert normalize_predicate("  is_a  ") == "is_a"


# ===========================================================================
# KnowledgeGraph Core
# ===========================================================================

class TestKnowledgeGraph:
    @pytest.fixture
    def kg(self, db):
        return KnowledgeGraph(db)

    @pytest.mark.asyncio
    async def test_add_fact(self, kg):
        assert await kg.add_fact("python", "is_a", "programming language") is True
        facts = kg.query("python")
        assert len(facts) == 1
        assert facts[0]["subject"] == "python"
        assert facts[0]["predicate"] == "is_a"
        assert facts[0]["object"] == "programming language"

    @pytest.mark.asyncio
    async def test_add_fact_normalizes(self, kg):
        await kg.add_fact("Python", "IS A", "Language")
        facts = kg.query("python")
        assert facts[0]["subject"] == "Python"  # Preserves original casing
        assert facts[0]["predicate"] == "is_a"
        assert facts[0]["object"] == "Language"  # Preserves original casing

    @pytest.mark.asyncio
    async def test_add_fact_deduplication(self, kg):
        assert await kg.add_fact("python", "is_a", "language", confidence=0.8) is True
        assert await kg.add_fact("python", "is_a", "language", confidence=0.7) is False  # lower confidence
        assert await kg.add_fact("python", "is_a", "language", confidence=0.9) is True   # higher confidence
        stats = kg.get_stats()
        assert stats["total_facts"] == 1

    @pytest.mark.asyncio
    async def test_add_fact_rejects_empty(self, kg):
        assert await kg.add_fact("", "is_a", "thing") is False
        assert await kg.add_fact("thing", "is_a", "") is False

    @pytest.mark.asyncio
    async def test_add_fact_rejects_long(self, kg):
        assert await kg.add_fact("x" * 201, "is_a", "thing") is False

    @pytest.mark.asyncio
    async def test_delete_fact(self, kg):
        await kg.add_fact("python", "is_a", "language")
        assert await kg.delete_fact("python", "is_a", "language") is True
        assert await kg.delete_fact("python", "is_a", "language") is False  # already retired
        # Retired facts still exist in DB but have valid_to set (temporal retirement)
        # Active query should not return them
        active_facts = kg.query("python")
        active_subjects = [f["subject"] for f in active_facts if f.get("subject") == "python"]
        # The retired fact should not appear in active queries
        assert len([f for f in active_facts if f.get("object") == "language"]) == 0

    @pytest.mark.asyncio
    async def test_query_single_hop(self, kg):
        await kg.add_fact("python", "created_by", "guido van rossum")
        await kg.add_fact("guido van rossum", "born_in", "netherlands")
        await kg.add_fact("java", "created_by", "james gosling")

        facts = kg.query("python", hops=1)
        subjects = {f["subject"] for f in facts}
        # Should get python's direct facts + guido's facts (1-hop)
        assert "python" in subjects
        assert "guido van rossum" in subjects
        # java is not connected
        assert "java" not in subjects

    def test_query_empty_entity(self, kg):
        assert kg.query("") == []

    @pytest.mark.asyncio
    async def test_query_unknown_entity(self, kg):
        await kg.add_fact("python", "is_a", "language")
        assert kg.query("unknown_entity") == []

    @pytest.mark.asyncio
    async def test_search(self, kg):
        await kg.add_fact("bitcoin", "is_a", "cryptocurrency")
        await kg.add_fact("ethereum", "is_a", "cryptocurrency")
        await kg.add_fact("python", "is_a", "language")

        results = kg.search("crypto")
        assert len(results) == 2  # bitcoin and ethereum

    def test_search_empty(self, kg):
        assert kg.search("") == []

    @pytest.mark.asyncio
    async def test_get_stats(self, kg):
        await kg.add_fact("python", "is_a", "language")
        await kg.add_fact("python", "created_by", "guido")
        stats = kg.get_stats()
        assert stats["total_facts"] == 2
        assert stats["unique_predicates"] == 2

    @pytest.mark.asyncio
    async def test_get_all_facts(self, kg):
        await kg.add_fact("a", "is_a", "b")
        await kg.add_fact("c", "is_a", "d")
        facts = kg.get_all_facts(limit=10)
        assert len(facts) == 2
        assert all(isinstance(f, Fact) for f in facts)

    @pytest.mark.asyncio
    async def test_get_all_facts_pagination(self, kg):
        for i in range(5):
            await kg.add_fact(f"entity_{i}", "is_a", "thing")
        page1 = kg.get_all_facts(limit=2, offset=0)
        page2 = kg.get_all_facts(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].subject != page2[0].subject


# ===========================================================================
# Relevance Scoring
# ===========================================================================

class TestKGRelevanceBase:
    @pytest.fixture
    async def kg(self, db):
        kg = KnowledgeGraph(db)
        await kg.add_fact("bitcoin", "is_a", "cryptocurrency", confidence=0.9)
        await kg.add_fact("bitcoin", "has_property", "deflationary", confidence=0.8)
        await kg.add_fact("ethereum", "is_a", "cryptocurrency", confidence=0.9)
        await kg.add_fact("python", "created_by", "guido van rossum", confidence=0.95)
        return kg

    def test_relevant_facts_found(self, kg):
        facts = kg.get_relevant_facts("bitcoin cryptocurrency info")
        assert len(facts) >= 1
        # Best match (highest overlap) should be a bitcoin fact
        assert facts[0].subject == "bitcoin" or facts[0].object == "bitcoin"

    def test_irrelevant_query_empty(self, kg):
        facts = kg.get_relevant_facts("what is the weather")
        assert len(facts) == 0

    def test_limit_respected(self, kg):
        facts = kg.get_relevant_facts("bitcoin cryptocurrency", limit=1)
        assert len(facts) <= 1

    def test_format_for_prompt(self, kg):
        facts = kg.get_relevant_facts("bitcoin cryptocurrency")
        text = KnowledgeGraph.format_for_prompt(facts)
        assert "bitcoin" in text
        assert "confidence:" in text  # confidence score

    def test_format_empty(self, kg):
        assert KnowledgeGraph.format_for_prompt([]) == ""


# ===========================================================================
# Decay
# ===========================================================================

class TestKGDecay:
    @pytest.mark.asyncio
    async def test_decay_stale_facts(self, db):
        kg = KnowledgeGraph(db)
        await kg.add_fact("old_entity", "is_a", "thing", confidence=0.8)
        # Force the fact to be old
        db.execute(
            "UPDATE kg_facts SET created_at = datetime('now', '-90 days') WHERE subject = ?",
            ("old_entity",),
        )
        decayed = await kg.decay_stale(days=60, decay_amount=0.1)
        assert decayed == 1
        facts = kg.get_all_facts()
        assert facts[0].confidence == pytest.approx(0.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_decay_respects_floor(self, db):
        kg = KnowledgeGraph(db)
        await kg.add_fact("old_entity", "is_a", "thing", confidence=0.15)
        db.execute(
            "UPDATE kg_facts SET created_at = datetime('now', '-90 days') WHERE subject = ?",
            ("old_entity",),
        )
        await kg.decay_stale(days=60, decay_amount=0.1)
        facts = kg.get_all_facts()
        assert facts[0].confidence >= 0.1  # floor


# ===========================================================================
# Integration: Brain + KG
# ===========================================================================

class TestKGBrainIntegration:
    @pytest.fixture
    async def services_with_kg(self, db):
        from app.core.brain import Services, set_services
        from app.core.memory import ConversationStore, UserFactStore

        kg = KnowledgeGraph(db)
        await kg.add_fact("bitcoin", "is_a", "cryptocurrency", confidence=0.9)

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            kg=kg,
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_kg_facts_in_prompt(self, services_with_kg):
        """KG facts should appear in the system prompt when relevant."""
        from app.core.brain import think
        from app.schema import EventType

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Bitcoin is a cryptocurrency."
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            captured_messages = []
            original_gen = mock_llm.generate_with_tools

            async def capture(msgs, tools, **kwargs):
                captured_messages.extend(msgs)
                return await original_gen(msgs, tools, **kwargs)

            mock_llm.generate_with_tools = AsyncMock(side_effect=capture)

            events = []
            async for event in think("Tell me about bitcoin cryptocurrency"):
                events.append(event)

        # System prompt should contain KG facts
        system_msgs = [m for m in captured_messages if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        system_text = system_msgs[0]["content"]
        assert "bitcoin" in system_text.lower()
        assert "cryptocurrency" in system_text.lower()


# ===========================================================================
# Triple Extraction
# ===========================================================================

class TestKGExtraction:
    @pytest.mark.asyncio
    async def test_extract_triples(self, db):
        """_extract_kg_triples should add facts with LLM-scored confidence."""
        from app.core.brain import _extract_kg_triples

        kg = KnowledgeGraph(db)
        mock_response = json.dumps([
            {"subject": "Python", "predicate": "created_by", "object": "Guido van Rossum", "confidence": 0.9},
            {"subject": "Python", "predicate": "is_a", "object": "programming language", "confidence": 0.85},
        ])

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            await _extract_kg_triples(kg, "Who created Python?", "Python was created by Guido van Rossum.")

        facts = kg.get_all_facts()
        assert len(facts) == 2
        confs = sorted(f.confidence for f in facts)
        assert confs == [0.85, 0.9]  # LLM-scored, not all 0.7

    @pytest.mark.asyncio
    async def test_extract_rejects_garbage(self, db):
        """Extraction should reject invalid triples."""
        from app.core.brain import _extract_kg_triples

        kg = KnowledgeGraph(db)
        mock_response = json.dumps([
            {"subject": "", "predicate": "is_a", "object": "thing"},  # empty subject
            {"subject": "x" * 101, "predicate": "is_a", "object": "thing"},  # too long
            {"subject": "valid", "predicate": "is_a", "object": "thing"},  # OK
        ])

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            await _extract_kg_triples(kg, "test query for extraction", "test answer for extraction")

        facts = kg.get_all_facts()
        assert len(facts) == 1
        assert facts[0].subject == "valid"

    @pytest.mark.asyncio
    async def test_extract_handles_failure(self, db):
        """Extraction failure should not raise."""
        from app.core.brain import _extract_kg_triples

        kg = KnowledgeGraph(db)

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value="not json")
            # Should not raise
            await _extract_kg_triples(kg, "test query for extraction", "test answer")

        assert kg.get_stats()["total_facts"] == 0

    @pytest.mark.asyncio
    async def test_extract_fallback_confidence(self, db):
        """Without LLM confidence, facts should get source-tiered confidence."""
        from app.core.brain import _extract_kg_triples

        kg = KnowledgeGraph(db)
        mock_response = json.dumps([
            {"subject": "quantum", "predicate": "is_a", "object": "physics branch"},
        ])

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            await _extract_kg_triples(kg, "What is quantum?", "Quantum is a branch of physics.", source_name="Domain Study: Science")

        facts = kg.get_all_facts()
        assert len(facts) == 1
        assert facts[0].confidence == 0.75  # Science tier, not 0.7

    @pytest.mark.asyncio
    async def test_extract_clamps_confidence(self, db):
        """LLM confidence outside [0.3, 0.95] should be clamped."""
        from app.core.brain import _extract_kg_triples

        kg = KnowledgeGraph(db)
        mock_response = json.dumps([
            {"subject": "fact_high", "predicate": "is_a", "object": "thing", "confidence": 1.5},
            {"subject": "fact_low", "predicate": "is_a", "object": "thing", "confidence": 0.1},
        ])

        with patch("app.core.brain.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=mock_response)
            await _extract_kg_triples(kg, "test", "test answer")

        facts = kg.get_all_facts()
        high = next(f for f in facts if f.subject == "fact_high")
        low = next(f for f in facts if f.subject == "fact_low")
        assert high.confidence == 0.95
        assert low.confidence == 0.3


# ===========================================================================
# Garbage Triple Detection
# ===========================================================================

class TestGarbageTripleFilter:
    def test_self_referential(self):
        assert is_garbage_triple("python", "is_a", "python") is True

    def test_too_short(self):
        assert is_garbage_triple("x", "is_a", "thing") is True
        assert is_garbage_triple("thing", "is_a", "y") is True

    def test_math_expression(self):
        assert is_garbage_triple("12 + 5", "has_property", "17") is True
        assert is_garbage_triple("296", "has_property", "even") is True

    def test_file_path(self):
        assert is_garbage_triple("/data/test.txt", "is_a", "file") is True
        assert is_garbage_triple("config", "located_in", "/etc/config.yml") is True

    def test_test_artifacts(self):
        assert is_garbage_triple("testuser", "is_a", "person") is True
        assert is_garbage_triple("pizza", "is_a", "food") is False  # pizza removed from _GARBAGE_VALUES

    def test_valid_triple_passes(self):
        assert is_garbage_triple("python", "created_by", "guido van rossum") is False
        assert is_garbage_triple("bitcoin", "is_a", "cryptocurrency") is False


# ===========================================================================
# Contradiction Detection
# ===========================================================================

class TestContradictionDetection:
    @pytest.mark.asyncio
    async def test_no_contradiction(self, db):
        kg = KnowledgeGraph(db)
        await kg.add_fact("python", "created_by", "guido van rossum")
        result = await kg.check_and_resolve_contradictions(
            "java", "created_by", "james gosling"
        )
        assert result is True  # Different subject, no conflict

    @pytest.mark.asyncio
    async def test_contradiction_keeps_new(self, db):
        from app.core import llm as llm_mod
        kg = KnowledgeGraph(db)
        await kg.add_fact("france", "capital_of", "lyon")  # wrong fact

        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke, \
             patch.object(llm_mod, "extract_json_object") as mock_extract:
            mock_invoke.return_value = '{"keep": "B"}'
            mock_extract.side_effect = lambda raw: json.loads(raw)
            result = await kg.check_and_resolve_contradictions(
                "france", "capital_of", "paris"
            )

        assert result is True
        # Old fact should be deleted
        facts = kg.query("france")
        assert all(f["object"] != "lyon" for f in facts)

    @pytest.mark.asyncio
    async def test_contradiction_keeps_old(self, db):
        from app.core import llm as llm_mod
        kg = KnowledgeGraph(db)
        await kg.add_fact("france", "capital_of", "paris")  # correct fact

        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke, \
             patch.object(llm_mod, "extract_json_object") as mock_extract:
            mock_invoke.return_value = '{"keep": "A"}'
            mock_extract.side_effect = lambda raw: json.loads(raw)
            result = await kg.check_and_resolve_contradictions(
                "france", "capital_of", "lyon"
            )

        assert result is False  # New fact rejected

    @pytest.mark.asyncio
    async def test_contradiction_llm_failure_allows_both(self, db):
        from app.core import llm as llm_mod
        kg = KnowledgeGraph(db)
        await kg.add_fact("france", "capital_of", "paris")

        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = Exception("LLM down")
            result = await kg.check_and_resolve_contradictions(
                "france", "capital_of", "lyon"
            )

        assert result is True  # Fails open


# ===========================================================================
# KG Curation
# ===========================================================================

class TestKGCuration:
    @pytest.mark.asyncio
    async def test_heuristic_curation(self, db):
        from app.core import llm as llm_mod
        kg = KnowledgeGraph(db)
        await kg.add_fact("python", "created_by", "guido van rossum")  # valid

        # Force add garbage via raw SQL (bypasses add_fact validation)
        db.execute(
            "INSERT INTO kg_facts (subject, predicate, object, confidence) VALUES (?, ?, ?, ?)",
            ("foo", "is_a", "foo", 0.5),
        )

        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke, \
             patch.object(llm_mod, "extract_json_object") as mock_extract:
            mock_invoke.return_value = '{"results": []}'
            mock_extract.side_effect = lambda raw: json.loads(raw)
            result = await kg.curate(sample_size=5)

        assert result["heuristic"] >= 1  # Self-referential garbage removed
        # python fact survives
        facts = kg.get_all_facts()
        assert any(f.subject == "python" for f in facts)

    @pytest.mark.asyncio
    async def test_curate_empty_kg(self, db):
        kg = KnowledgeGraph(db)
        result = await kg.curate()
        assert result["heuristic"] == 0
        assert result["llm"] == 0


# ===========================================================================
# KG Relevance -- Audit Fix 2 (from test_audit_fixes2)
# ===========================================================================

class TestKGRelevanceSingleWord:
    """KG relevance uses overlap >= 2 to avoid overly loose matches."""

    @pytest.fixture
    def kg(self, db):
        return KnowledgeGraph(db)

    @pytest.mark.asyncio
    async def test_single_word_no_match(self, kg):
        """Single-word queries require overlap >= 2, so one word alone doesn't match."""
        await kg.add_fact("bitcoin", "is_a", "cryptocurrency", confidence=0.9)
        facts = kg.get_relevant_facts("bitcoin")
        # Single word overlap (1) is below threshold (2), so no match
        assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_two_word_match(self, kg):
        """Two matching words should match facts (overlap >= 2)."""
        await kg.add_fact("bitcoin", "is_a", "cryptocurrency", confidence=0.9)
        facts = kg.get_relevant_facts("bitcoin cryptocurrency")
        assert len(facts) >= 1

    @pytest.mark.asyncio
    async def test_multi_word_still_works(self, kg):
        """Multi-word overlap should still rank higher than single-word."""
        await kg.add_fact("bitcoin", "is_a", "cryptocurrency", confidence=0.9)
        await kg.add_fact("bitcoin", "created_by", "satoshi nakamoto", confidence=0.85)
        facts = kg.get_relevant_facts("bitcoin cryptocurrency")
        assert len(facts) >= 1
        assert facts[0].subject == "bitcoin"
        assert facts[0].object == "cryptocurrency"

    @pytest.mark.asyncio
    async def test_no_match_still_empty(self, kg):
        """Completely unrelated queries should still return no facts."""
        await kg.add_fact("bitcoin", "is_a", "cryptocurrency", confidence=0.9)
        facts = kg.get_relevant_facts("quantum physics")
        assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_empty_query_returns_nothing(self, kg):
        """Empty or stopword-only queries return nothing."""
        await kg.add_fact("bitcoin", "is_a", "cryptocurrency", confidence=0.9)
        facts = kg.get_relevant_facts("")
        assert len(facts) == 0
