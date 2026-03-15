"""Breaking Point Tests — Find Where Nova_ Actually Breaks.

Unlike battle tests (which verify defenses hold), these tests target
specific unguarded code paths identified through source analysis:
- Zero-division crashes in Jaccard similarity
- NaN/infinity/negative confidence propagation
- ReDoS in skill trigger patterns
- Training data file corruption under concurrent rotation
- None dereference in token estimation
- Unbounded resource growth
- Malformed LLM responses crashing the pipeline
- Race conditions in concurrent supersession
- Out-of-bounds quality scores stored in DB
- Empty collections causing index errors

These are the kind of tests that find REAL bugs.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import signal
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import Services, set_services
from app.core.memory import ConversationStore, UserFactStore


# ===========================================================================
# 1. ZERO-DIVISION CRASHES
# ===========================================================================

class TestZeroDivision:
    """Target every Jaccard/ratio calculation with empty sets."""

    def test_kg_jaccard_empty_words(self, db):
        """KG dedup with two facts that have zero words should not crash."""
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        # Single-char entities produce empty word sets after normalization
        kg.add_fact("x", "is_a", "y", confidence=0.8)
        # Adding another similar fact triggers dedup logic
        result = kg.add_fact("x", "is_a", "z", confidence=0.9)
        # Should not crash with zero-division — just supersede or add
        assert isinstance(result, bool)

    def test_reflexion_find_recurring_empty_query(self, db):
        """find_recurring_failures with empty/whitespace query should not crash."""
        from app.core.reflexion import ReflexionStore
        store = ReflexionStore(db)
        store.store("some task", "failure", "some reflection", quality_score=0.3)

        # Empty query → empty word set → potential zero-division in Jaccard
        result = store.find_recurring_failures("")
        assert isinstance(result, list)

        result2 = store.find_recurring_failures("   ")
        assert isinstance(result2, list)

    def test_learning_relevant_lessons_stopwords_only(self, db):
        """Query with only stop words should not crash get_relevant_lessons."""
        from app.core.learning import LearningEngine
        engine = LearningEngine(db)

        # Add a lesson first
        db.execute(
            """INSERT INTO lessons (topic, wrong_answer, correct_answer, context, confidence, lesson_text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("test topic here", "wrong answer text", "right answer text", "context info", 0.8, "lesson content"),
        )

        # Query with words that might all be filtered out
        lessons = engine.get_relevant_lessons("the a an is are")
        assert isinstance(lessons, list)

    def test_retriever_entity_filter_empty_query(self, db):
        """_entity_relevance_filter with empty query words should not crash."""
        from app.core.retriever import _entity_relevance_filter, Chunk

        chunks = [Chunk(chunk_id="c1", document_id="d1", content="test content", score=0.5)]
        # Empty query → empty word set → ratio = overlap / len(query_words) → zero-division
        # This tests the module-level function
        result = _entity_relevance_filter("", chunks)
        assert isinstance(result, list)


# ===========================================================================
# 2. NaN / INFINITY / NEGATIVE CONFIDENCE
# ===========================================================================

class TestBadConfidenceValues:
    """Pump NaN, Infinity, and negative values through confidence fields."""

    def test_kg_nan_confidence(self, db):
        """NaN confidence in KG should not silently propagate."""
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        # NaN should either be rejected or clamped
        result = kg.add_fact("test", "is_a", "thing", confidence=float('nan'))
        # Check: either rejected (False) or stored with sane value
        if result:
            facts = kg.query("test")
            for f in facts:
                assert not math.isnan(f["confidence"]), "NaN confidence stored in KG!"

    def test_kg_infinity_confidence(self, db):
        """Infinity confidence should not corrupt comparisons."""
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        result = kg.add_fact("test_inf", "is_a", "thing", confidence=float('inf'))
        if result:
            facts = kg.query("test_inf")
            for f in facts:
                assert not math.isinf(f["confidence"]), "Infinity confidence stored in KG!"

    def test_kg_negative_confidence(self, db):
        """Negative confidence should be clamped to 0, not stored raw."""
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        kg.add_fact("neg_test", "is_a", "A", confidence=0.8)
        # Negative confidence should be clamped to 0.0
        kg.add_fact("neg_test", "is_a", "B", confidence=-1.0)
        facts = kg.query("neg_test")
        # Verify no negative confidence values stored
        for f in facts:
            assert f["confidence"] >= 0.0, f"Negative confidence stored: {f['confidence']}"

    def test_reflexion_nan_quality_score(self, db):
        """NaN quality_score should not corrupt stats."""
        from app.core.reflexion import ReflexionStore
        store = ReflexionStore(db)
        store.store("nan task", "failure", "nan reflection", quality_score=float('nan'))
        stats = store.get_stats()
        # avg_quality should not be NaN
        assert not math.isnan(stats["avg_quality"]), "NaN quality propagated to stats!"

    def test_reflexion_negative_quality(self, db):
        """Negative quality_score stored and retrieved."""
        from app.core.reflexion import ReflexionStore
        store = ReflexionStore(db)
        store.store("neg task", "failure", "neg reflection", quality_score=-5.0)
        stats = store.get_stats()
        # Should handle gracefully — either reject or store as-is
        assert isinstance(stats["avg_quality"], (int, float))

    def test_skill_nan_success_rate(self, db):
        """NaN success_rate should not corrupt skill matching."""
        from app.core.skills import SkillStore
        store = SkillStore(db)
        skill_id = store.create_skill(
            name="nan_skill", trigger_pattern=r"\bnan_test\b",
            steps=json.dumps([{"tool": "calculator", "args": {"expression": "1+1"}}]),
        )
        # Manually corrupt success_rate to NaN
        db.execute("UPDATE skills SET success_rate = ? WHERE id = ?", (float('nan'), skill_id))

        # Matching should still work
        match = store.get_matching_skill("nan_test query")
        # Should either find it or skip it, not crash
        assert match is None or hasattr(match, 'name')


# ===========================================================================
# 3. ReDoS — REGEX DENIAL OF SERVICE
# ===========================================================================

class TestReDoS:
    """Craft regex patterns that cause catastrophic backtracking."""

    def test_redos_in_skill_trigger(self, db):
        """A ReDoS trigger pattern should be detected and blocked."""
        from app.core.skills import SkillStore
        store = SkillStore(db)

        # Classic ReDoS pattern: (a+)+b
        redos_pattern = r"(a+)+b"

        # create_skill should reject ReDoS-prone patterns
        skill_id = store.create_skill(
            name="redos_skill", trigger_pattern=redos_pattern,
            steps=json.dumps([]),
        )
        assert skill_id is None, "ReDoS pattern should be rejected by create_skill!"

        # Even if directly inserted into DB, get_matching_skill should catch it
        db.execute(
            "INSERT INTO skills (name, trigger_pattern, steps, enabled) VALUES (?, ?, ?, ?)",
            ("direct_redos", redos_pattern, "[]", 1),
        )

        adversarial_input = "a" * 30 + "!"

        result = [None]
        def run_match():
            result[0] = store.get_matching_skill(adversarial_input)

        t = threading.Thread(target=run_match)
        t.start()
        t.join(timeout=5)

        if t.is_alive():
            pytest.fail("ReDoS: skill matching hung for >5 seconds on adversarial input!")

        # The skill should have been auto-disabled
        row = db.fetchone("SELECT enabled FROM skills WHERE name = 'direct_redos'")
        assert row["enabled"] == 0, "ReDoS skill should be auto-disabled!"

    def test_catastrophic_nested_quantifiers(self, db):
        """Nested quantifiers with no anchors should be caught or time-limited."""
        from app.core.skills import SkillStore
        store = SkillStore(db)

        # Even worse ReDoS: nested groups with overlapping matches
        evil_pattern = r"(\w+\s*)+$"
        store.create_skill(
            name="evil_regex", trigger_pattern=evil_pattern,
            steps=json.dumps([]),
        )

        # Adversarial: long string of words with trailing space
        adversarial = "word " * 20 + "!"

        result = [None]
        def run_match():
            result[0] = store.get_matching_skill(adversarial)

        t = threading.Thread(target=run_match)
        t.start()
        t.join(timeout=5)

        if t.is_alive():
            pytest.fail("ReDoS: nested quantifier regex hung for >5 seconds!")


# ===========================================================================
# 4. CONCURRENT SUPERSESSION RACE
# ===========================================================================

class TestConcurrentSupersession:
    """Two threads superseding the same KG fact simultaneously."""

    def test_concurrent_supersession_same_fact(self, db):
        """Two threads updating the same fact should not corrupt temporal chain."""
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)

        # Create initial fact
        kg.add_fact("Bitcoin", "price_is", "$50000", confidence=0.8)

        errors = []

        def update_price(thread_id, price):
            try:
                kg.add_fact("Bitcoin", "price_is", price, confidence=0.9)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Two threads superseding the same fact at the same time
        t1 = threading.Thread(target=update_price, args=(1, "$60000"))
        t2 = threading.Thread(target=update_price, args=(2, "$70000"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(errors) == 0, f"Concurrent supersession errors: {errors}"

        # Verify integrity: exactly one current fact
        history = kg.get_fact_history("Bitcoin", "price_is")
        current = [h for h in history if h.get("valid_to") is None]
        assert len(current) == 1, f"Expected 1 current fact, got {len(current)}: {current}"


# ===========================================================================
# 5. TRAINING DATA ROTATION RACE
# ===========================================================================

class TestTrainingDataRotation:
    """Race conditions during concurrent training pair saves + rotation."""

    @pytest.mark.asyncio
    async def test_concurrent_training_pair_writes(self, db, tmp_path):
        """20 concurrent save_training_pair calls should not lose data."""
        from app.core.learning import LearningEngine
        engine = LearningEngine(db)
        data_file = str(tmp_path / "training.jsonl")

        async def save_pair(i):
            with patch("app.core.learning.config") as cfg:
                cfg.TRAINING_DATA_PATH = data_file
                cfg.TRAINING_DATA_CHANNELS = "api"
                cfg.TRAINING_DATA_MAX_PAIRS = 10000
                cfg.MAX_TRAINING_PAIRS = 10000
                await engine.save_training_pair(
                    query=f"Question number {i} about a specific topic here",
                    bad_answer=f"Wrong answer number {i} that was incorrect and unhelpful",
                    good_answer=f"Correct answer number {i} that was accurate and very helpful indeed",
                    channel="api",
                    confidence=1.0,
                )

        # Fire 20 concurrent writes
        await asyncio.gather(*[save_pair(i) for i in range(20)])

        # Verify file integrity — all lines should be valid JSON
        assert os.path.exists(data_file)
        with open(data_file) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        valid_count = 0
        corrupt_count = 0
        for line in lines:
            try:
                entry = json.loads(line)
                assert "query" in entry
                assert "chosen" in entry
                valid_count += 1
            except json.JSONDecodeError:
                corrupt_count += 1

        assert corrupt_count == 0, f"Data corruption: {corrupt_count}/{len(lines)} lines invalid!"
        assert valid_count == 20, f"Expected 20 valid entries, got {valid_count}"


# ===========================================================================
# 6. MALFORMED LLM RESPONSES
# ===========================================================================

class TestMalformedLLMResponses:
    """Feed garbage LLM responses through the pipeline."""

    @pytest.mark.asyncio
    async def test_llm_returns_none_content(self, db):
        """LLM returning None content should not crash think()."""
        from app.core.brain import think

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = None  # LLM returns None
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            svc = Services(
                conversations=ConversationStore(db),
                user_facts=UserFactStore(db),
            )
            set_services(svc)

            events = []
            try:
                async for event in think("test query"):
                    events.append(event)
            except (TypeError, AttributeError) as e:
                pytest.fail(f"None content from LLM crashed pipeline: {e}")

    @pytest.mark.asyncio
    async def test_llm_returns_empty_string(self, db):
        """LLM returning empty string should handle gracefully."""
        from app.core.brain import think

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = ""
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            svc = Services(
                conversations=ConversationStore(db),
                user_facts=UserFactStore(db),
            )
            set_services(svc)

            events = []
            async for event in think("test query"):
                events.append(event)
            # Should complete without crash — empty response is valid

    @pytest.mark.asyncio
    async def test_llm_returns_partial_json_tool_call(self, db):
        """LLM returning broken JSON tool call should not crash."""
        from app.core.brain import think

        call_count = 0
        with patch("app.core.brain.llm") as mock_llm:
            async def broken_then_good(msgs, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                result = AsyncMock()
                if call_count == 1:
                    # Broken JSON that looks like a tool call
                    result.content = '{"tool": "calculator", "args": {"expression": "1+1'  # truncated
                else:
                    result.content = "The answer is 2."
                result.tool_call = None
                return result

            mock_llm.generate_with_tools = AsyncMock(side_effect=broken_then_good)

            svc = Services(
                conversations=ConversationStore(db),
                user_facts=UserFactStore(db),
            )
            set_services(svc)

            events = []
            async for event in think("What is 1+1?"):
                events.append(event)
            # Should recover from broken JSON without crashing

    @pytest.mark.asyncio
    async def test_llm_returns_nonexistent_tool(self, db):
        """LLM calling a tool that doesn't exist should not crash."""
        from app.core.brain import think

        call_count = 0
        with patch("app.core.brain.llm") as mock_llm:
            async def fake_tool_then_answer(msgs, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                result = AsyncMock()
                if call_count == 1:
                    result.content = '{"tool": "nonexistent_tool_xyz", "args": {"query": "test"}}'
                else:
                    result.content = "Here's my answer."
                result.tool_call = None
                return result

            mock_llm.generate_with_tools = AsyncMock(side_effect=fake_tool_then_answer)

            svc = Services(
                conversations=ConversationStore(db),
                user_facts=UserFactStore(db),
            )
            set_services(svc)

            events = []
            async for event in think("Use that tool"):
                events.append(event)

    @pytest.mark.asyncio
    async def test_llm_returns_megabyte_response(self, db):
        """LLM returning 1MB of text should not OOM."""
        from app.core.brain import think

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "A" * (1024 * 1024)  # 1MB response
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            svc = Services(
                conversations=ConversationStore(db),
                user_facts=UserFactStore(db),
            )
            set_services(svc)

            events = []
            async for event in think("Give me a long response"):
                events.append(event)
            assert len(events) > 0


# ===========================================================================
# 7. TOKEN ESTIMATION EDGE CASES
# ===========================================================================

class TestTokenEstimation:
    """Edge cases in _estimate_tokens."""

    def test_estimate_tokens_none(self):
        """_estimate_tokens with None should not crash."""
        from app.core.brain import _estimate_tokens
        try:
            result = _estimate_tokens(None)
            assert isinstance(result, int)
        except TypeError:
            # This IS a bug — None should be handled
            pytest.fail("_estimate_tokens crashed on None input!")

    def test_estimate_tokens_empty(self):
        """_estimate_tokens with empty string should return 0."""
        from app.core.brain import _estimate_tokens
        result = _estimate_tokens("")
        assert result == 0

    def test_estimate_tokens_only_cjk(self):
        """Pure CJK text should estimate at ~1.5 chars/token."""
        from app.core.brain import _estimate_tokens
        text = "日本語テスト" * 100  # 600 CJK chars
        result = _estimate_tokens(text)
        assert result > 300  # At least 300 tokens for 600 CJK chars

    def test_estimate_tokens_huge_string(self):
        """10MB string should not take forever."""
        from app.core.brain import _estimate_tokens
        text = "word " * 2_000_000  # ~10MB
        start = time.time()
        result = _estimate_tokens(text)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Token estimation took {elapsed:.1f}s on 10MB string!"
        assert result > 0


# ===========================================================================
# 8. RETRIEVER EDGE CASES
# ===========================================================================

class TestRetrieverEdgeCases:
    @pytest.fixture
    def retriever(self, db):
        from app.core.retriever import Retriever
        return Retriever(db)

    @pytest.mark.asyncio
    async def test_ingest_whitespace_only(self, retriever):
        """Ingesting whitespace-only text should not create broken chunks."""
        doc_id, chunks = await retriever.ingest("   \n\n\t\t   ", doc_id="whitespace-doc")
        assert chunks == 0

    @pytest.mark.asyncio
    async def test_search_empty_query(self, retriever):
        """Empty search query should return empty results, not crash."""
        await retriever.ingest("Some real content here for testing.", doc_id="real-doc")
        results = await retriever.search("")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_very_long_query(self, retriever):
        """100KB search query should not crash FTS5."""
        await retriever.ingest("Content about AI and machine learning.", doc_id="ai-doc")
        long_query = "artificial intelligence " * 5000
        results = await retriever.search(long_query)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_ingest_single_character(self, retriever):
        """Single character doc should handle gracefully."""
        doc_id, chunks = await retriever.ingest("x", doc_id="tiny-doc")
        # Should either create 1 chunk or 0 (below threshold)
        assert chunks >= 0

    @pytest.mark.asyncio
    async def test_search_fts5_special_chars(self, retriever):
        """FTS5 special characters in query should be escaped."""
        await retriever.ingest("Normal document content here.", doc_id="normal-doc")
        # FTS5 special chars that could crash unescaped queries
        dangerous_queries = [
            'test AND OR NOT',
            'test*',
            '"unmatched quote',
            'column:value',
            'NEAR(a b)',
            '(a OR b) AND c',
        ]
        for q in dangerous_queries:
            try:
                results = await retriever.search(q)
                assert isinstance(results, list), f"Bad result for query: {q}"
            except Exception as e:
                pytest.fail(f"FTS5 query crashed on '{q}': {e}")


# ===========================================================================
# 9. KG STATS WITH EMPTY / CORRUPTED DB
# ===========================================================================

class TestKGStatsEdgeCases:
    def test_stats_empty_kg(self, db):
        """get_stats on empty KG should not crash."""
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        stats = kg.get_stats()
        assert stats["total_facts"] == 0
        assert stats["current_facts"] == 0

    def test_format_for_prompt_none_confidence(self, db):
        """Facts with NULL confidence should not crash format_for_prompt."""
        from app.core.kg import KnowledgeGraph, Fact
        kg = KnowledgeGraph(db)

        # Insert a fact with NULL confidence directly
        db.execute(
            "INSERT INTO kg_facts (subject, predicate, object, confidence, source) VALUES (?, ?, ?, ?, ?)",
            ("test", "is_a", "thing", None, "test"),
        )

        # format_for_prompt should handle None confidence
        facts = kg.get_all_facts()
        try:
            text = kg.format_for_prompt(facts)
            assert isinstance(text, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"NULL confidence crashed format_for_prompt: {e}")

    def test_decay_stale_empty_kg(self, db):
        """decay_stale on empty KG should return 0, not crash."""
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        decayed = kg.decay_stale(days=30, decay_amount=0.1)
        assert decayed == 0


# ===========================================================================
# 10. SKILL RECORD_USE AFTER DELETE
# ===========================================================================

class TestSkillRaceConditions:
    def test_record_use_deleted_skill(self, db):
        """record_use on a skill that was deleted should not crash."""
        from app.core.skills import SkillStore
        store = SkillStore(db)

        skill_id = store.create_skill(
            name="ephemeral", trigger_pattern=r"\bephemeral\b",
            steps=json.dumps([]),
        )

        # Delete the skill
        db.execute("DELETE FROM skills WHERE id = ?", (skill_id,))

        # Now try to record use — should handle gracefully
        try:
            store.record_use(skill_id, success=True)
        except (AttributeError, TypeError) as e:
            pytest.fail(f"record_use on deleted skill crashed: {e}")

    def test_match_skill_with_invalid_regex_in_db(self, db):
        """Corrupt regex in DB should not crash skill matching."""
        from app.core.skills import SkillStore
        store = SkillStore(db)

        # Insert a skill with invalid regex directly into DB
        db.execute(
            "INSERT INTO skills (name, trigger_pattern, steps, enabled) VALUES (?, ?, ?, ?)",
            ("bad_regex_skill", "[invalid(regex", "[]", 1),
        )

        # Matching should handle the bad regex gracefully
        try:
            result = store.get_matching_skill("any query here")
            # Should skip the bad skill, not crash
        except re.error:
            pytest.fail("Invalid regex in DB crashed skill matching!")


# ===========================================================================
# 11. PROMPT BUDGET OVERFLOW
# ===========================================================================

class TestPromptBudgetOverflow:
    def test_all_blocks_exceed_budget(self, db):
        """When all prompt blocks exceed MAX_SYSTEM_TOKENS, should not crash."""
        from app.core.prompt import build_system_prompt

        # Create massive context that exceeds budget
        massive_facts = "User fact: " + "x" * 10000
        massive_lessons = "Lesson: " + "y" * 10000
        massive_skills = "Skill: " + "z" * 10000
        massive_context = "Context: " + "w" * 10000

        try:
            prompt = build_system_prompt(
                user_facts_text=massive_facts,
                lessons_text=massive_lessons,
                skills_text=massive_skills,
                retrieved_context=massive_context,
            )
            assert isinstance(prompt, str)
            assert len(prompt) > 0  # Should still produce something
        except Exception as e:
            pytest.fail(f"Prompt builder crashed on oversized blocks: {e}")


# ===========================================================================
# 12. DATABASE TRANSACTION FAILURE
# ===========================================================================

class TestDatabaseTransactionEdgeCases:
    def test_transaction_rollback_on_constraint_violation(self, db):
        """Failed INSERT inside transaction should rollback ALL changes."""
        # Insert a conversation to test with
        db.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", ("txn-test", "test"))

        try:
            with db.transaction() as tx:
                tx.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    ("txn-test", "user", "message 1"),
                )
                # This should fail — invalid conversation_id (foreign key)
                tx.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    ("nonexistent_conv", "user", "message 2"),
                )
        except Exception:
            pass  # Expected

        # First message should NOT be committed (rollback)
        rows = db.fetchall(
            "SELECT * FROM messages WHERE conversation_id = ?", ("txn-test",)
        )
        assert len(rows) == 0, "Transaction rollback failed — partial data committed!"

    def test_nested_concurrent_transactions(self, db):
        """Multiple threads using transactions should not deadlock."""
        errors = []

        def transactional_write(thread_id):
            try:
                db.execute(
                    "INSERT INTO conversations (id, title) VALUES (?, ?)",
                    (f"txn-thread-{thread_id}", f"Thread {thread_id}"),
                )
                for i in range(5):
                    db.execute(
                        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                        (f"txn-thread-{thread_id}", "user", f"msg-{i}"),
                    )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=transactional_write, args=(t,)) for t in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Transaction errors: {errors[:5]}"


# ===========================================================================
# 13. MEMORY STORE EDGE CASES
# ===========================================================================

class TestMemoryEdgeCases:
    def test_user_fact_very_long_value(self, db):
        """Storing a 100KB value should not crash."""
        from app.core.memory import UserFactStore
        store = UserFactStore(db)
        long_value = "x" * 100_000
        store.set("huge_fact", long_value)
        result = store.get("huge_fact")
        assert result is not None
        assert len(result.value) == 100_000

    def test_user_fact_unicode_key(self, db):
        """Unicode keys should work."""
        from app.core.memory import UserFactStore
        store = UserFactStore(db)
        store.set("用户名", "张三")
        result = store.get("用户名")
        assert result is not None
        assert result.value == "张三"

    def test_conversation_store_empty_history(self, db):
        """Getting context from empty conversation should not crash."""
        from app.core.memory import ConversationStore
        store = ConversationStore(db)
        conv_id = store.create_conversation()
        history = store.get_history(conv_id)
        assert history == []


# ===========================================================================
# 14. EXTREME CONCURRENCY — FIND THE ACTUAL LIMIT
# ===========================================================================

class TestExtremeConcurrency:
    @pytest.mark.asyncio
    async def test_100_concurrent_thinks(self, db):
        """100 parallel think() calls — find if there's a breaking point."""
        from app.core.brain import think

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Quick response."
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            svc = Services(
                conversations=ConversationStore(db),
                user_facts=UserFactStore(db),
            )
            set_services(svc)

            async def run_think(i):
                events = []
                async for event in think(f"Q{i}", conversation_id=f"extreme-{i}"):
                    events.append(event)
                return len(events)

            results = await asyncio.gather(
                *[run_think(i) for i in range(100)],
                return_exceptions=True,
            )

        exceptions = [r for r in results if isinstance(r, Exception)]
        success = [r for r in results if isinstance(r, int)]

        if exceptions:
            # Report what broke, don't just fail
            error_types = {}
            for e in exceptions:
                t = type(e).__name__
                error_types[t] = error_types.get(t, 0) + 1
            pytest.fail(
                f"100 concurrent thinks: {len(success)} succeeded, "
                f"{len(exceptions)} failed. Error types: {error_types}"
            )

    def test_200_thread_kg_writes(self, db):
        """200 threads writing to KG simultaneously."""
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        errors = []

        def write(tid):
            try:
                for i in range(5):
                    kg.add_fact(f"t{tid}_e{i}", "is_a", f"thing_{i}", confidence=0.8)
            except Exception as e:
                errors.append(f"Thread {tid}: {e}")

        threads = [threading.Thread(target=write, args=(t,)) for t in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        if errors:
            pytest.fail(f"200-thread KG stress: {len(errors)} errors: {errors[:5]}")


# ===========================================================================
# 15. INJECTION THROUGH UNEXPECTED FIELDS
# ===========================================================================

class TestInjectionUnexpectedFields:
    """Inject through conversation IDs, tool names, and fact keys."""

    @pytest.mark.asyncio
    async def test_injection_in_conversation_id(self, db):
        """SQL injection in conversation_id should be parameterized."""
        from app.core.brain import think

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Response."
            mock_result.tool_call = None
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            svc = Services(
                conversations=ConversationStore(db),
                user_facts=UserFactStore(db),
            )
            set_services(svc)

            # SQL injection in conversation_id
            events = []
            async for event in think(
                "test",
                conversation_id="'; DROP TABLE conversations; --",
            ):
                events.append(event)

        # Conversations table should still exist
        row = db.fetchone("SELECT COUNT(*) as cnt FROM conversations")
        assert row is not None

    def test_injection_in_fact_key(self, db):
        """SQL injection in fact key should be parameterized."""
        from app.core.memory import UserFactStore
        store = UserFactStore(db)
        store.set("'; DROP TABLE user_facts; --", "test value")

        # Table should still exist
        row = db.fetchone("SELECT COUNT(*) as cnt FROM user_facts")
        assert row is not None

    def test_injection_in_kg_entity(self, db):
        """SQL injection in KG entity should be parameterized."""
        from app.core.kg import KnowledgeGraph
        kg = KnowledgeGraph(db)
        kg.add_fact(
            "'; DROP TABLE kg_facts; --",
            "is_a",
            "test",
            confidence=0.8,
        )
        # Table should still exist
        row = db.fetchone("SELECT COUNT(*) as cnt FROM kg_facts")
        assert row is not None
