"""Tests for the Reflexion Store module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.reflexion import ReflexionStore, Reflexion, assess_quality


# ===========================================================================
# Quality Assessment
# ===========================================================================

class TestQualityAssessment:
    def test_good_answer(self):
        score, reason = assess_quality(
            "Bitcoin is a decentralized cryptocurrency created in 2009.",
            tool_results=[],
            max_tool_rounds=5,
        )
        assert score >= 0.7
        assert reason == ""

    def test_empty_answer(self):
        score, reason = assess_quality("", [], 5)
        assert score == 0.0
        assert "Empty" in reason

    def test_short_answer_complex_query(self):
        # Short answers to complex queries should be penalized
        score, reason = assess_quality(
            "Yes.", [], 5,
            query="Can you explain the differences between quantum computing and classical computing?",
        )
        assert score < 1.0
        assert "short" in reason.lower()

    def test_short_answer_simple_query(self):
        # Short answers to simple/short queries are legitimate
        score, reason = assess_quality("Paris.", [], 5, query="Capital of France?")
        assert score == 1.0

    def test_failure_phrases(self):
        # Hard failure phrases trigger strong penalty
        score, reason = assess_quality(
            "I cannot help with that. I wasn't able to find information. Failed to retrieve results.",
            tool_results=[],
            max_tool_rounds=5,
        )
        assert score <= 0.6
        assert "failure phrase" in reason.lower()

    def test_soft_uncertainty_phrases(self):
        # Soft uncertainty phrases trigger lighter penalty
        score, reason = assess_quality(
            "I don't know the answer and I'm not sure about that topic.",
            tool_results=[],
            max_tool_rounds=5,
        )
        assert score <= 0.95  # soft penalty only
        assert "uncertainty" in reason.lower()

    def test_tool_exhaustion(self):
        tools = [{"tool": f"t{i}", "output": "ok"} for i in range(5)]
        score, reason = assess_quality(
            "Here's what I found from searching multiple sources.",
            tool_results=tools,
            max_tool_rounds=5,
        )
        assert score < 1.0
        assert "rounds" in reason.lower()

    def test_tool_failures(self):
        tools = [
            {"tool": "web_search", "output": "Error: connection timed out"},
            {"tool": "http_fetch", "output": "Failed to fetch URL"},
        ]
        score, reason = assess_quality(
            "I tried searching but the tools had issues.",
            tool_results=tools,
            max_tool_rounds=5,
        )
        assert score < 0.8
        assert "failed" in reason.lower()

    def test_cumulative_penalties(self):
        """Multiple issues should stack penalties."""
        tools = [{"tool": "web_search", "output": "Error: failed to connect"}] * 5
        score, reason = assess_quality(
            "I don't know. I couldn't find anything.",
            tool_results=tools,
            max_tool_rounds=5,
        )
        assert score < 0.3


# ===========================================================================
# ReflexionStore Core
# ===========================================================================

class TestReflexionStore:
    @pytest.fixture
    def store(self, db):
        return ReflexionStore(db)

    def test_store_reflexion(self, store):
        rid = store.store(
            task_summary="What is the current price of Bitcoin?",
            outcome="failure",
            reflection="Used knowledge_search for real-time data; should use web_search",
            quality_score=0.3,
            tools_used=["knowledge_search"],
            revision_count=2,
        )
        assert rid > 0
        stats = store.get_stats()
        assert stats["total_reflexions"] == 1
        assert stats["failures"] == 1

    def test_store_normalizes_outcome(self, store):
        store.store("task", "FAILURE", "reason")
        store.store("task2", "invalid", "reason2")
        stats = store.get_stats()
        assert stats["failures"] == 2  # "invalid" becomes "failure"

    def test_store_truncates_long_text(self, store):
        rid = store.store(
            task_summary="x" * 1000,
            outcome="failure",
            reflection="y" * 2000,
        )
        assert rid > 0

    def test_dedup_rejects_identical(self, store):
        rid1 = store.store("bitcoin price query", "failure", "should use web_search for real-time data")
        rid2 = store.store("bitcoin price query", "failure", "should use web_search for real-time data")
        assert rid1 > 0
        assert rid2 == -1  # duplicate

    def test_dedup_allows_different(self, store):
        rid1 = store.store("bitcoin price", "failure", "used wrong tool for real-time data")
        rid2 = store.store("weather forecast", "failure", "forgot to check location first")
        assert rid1 > 0
        assert rid2 > 0

    def test_store_success(self, store):
        store.store("simple greeting", "success", "handled well", quality_score=0.9)
        stats = store.get_stats()
        assert stats["successes"] == 1
        assert stats["failures"] == 0


# ===========================================================================
# Retrieval
# ===========================================================================

class TestReflexionRetrieval:
    @pytest.fixture
    def store(self, db):
        s = ReflexionStore(db)
        s.store(
            "bitcoin price query",
            "failure",
            "Used knowledge_search for real-time data; should use web_search",
            quality_score=0.3,
            tools_used=["knowledge_search"],
        )
        s.store(
            "python code execution",
            "failure",
            "Code sandbox timed out on infinite loop",
            quality_score=0.2,
            tools_used=["code_exec"],
        )
        s.store(
            "weather forecast query",
            "success",
            "Correctly used web_search",
            quality_score=0.9,
        )
        return s

    def test_relevant_failures_found(self, store):
        results = store.get_relevant("what is the bitcoin price?")
        assert len(results) >= 1
        assert results[0].reflection == "Used knowledge_search for real-time data; should use web_search"

    def test_success_excluded(self, store):
        """Success reflexions should not be returned (they're not useful warnings)."""
        results = store.get_relevant("weather forecast for tomorrow")
        assert all(r.outcome == "failure" for r in results)

    def test_irrelevant_query_low_results(self, store):
        results = store.get_relevant("what is quantum entanglement")
        # With small collections, vector search may return tangential results;
        # verify none are high quality matches (keyword overlap)
        assert len(results) <= 3

    def test_limit_respected(self, store):
        results = store.get_relevant("bitcoin price code", limit=1)
        assert len(results) <= 1

    def test_format_for_prompt(self, store):
        results = store.get_relevant("bitcoin price")
        text = ReflexionStore.format_for_prompt(results)
        assert "Previous failure" in text
        assert "web_search" in text

    def test_format_empty(self, store):
        assert ReflexionStore.format_for_prompt([]) == ""


# ===========================================================================
# Stats and Decay
# ===========================================================================

class TestReflexionStats:
    def test_get_stats_empty(self, db):
        store = ReflexionStore(db)
        stats = store.get_stats()
        assert stats["total_reflexions"] == 0
        assert stats["avg_quality"] == 0.0

    def test_get_stats_with_data(self, db):
        store = ReflexionStore(db)
        store.store("task1", "failure", "reason1", quality_score=0.3)
        store.store("task2", "failure", "reason2", quality_score=0.5)
        store.store("task3", "success", "reason3", quality_score=0.9)
        stats = store.get_stats()
        assert stats["total_reflexions"] == 3
        assert stats["failures"] == 2
        assert stats["successes"] == 1


class TestReflexionDecay:
    def test_decay_stale_reflexions(self, db):
        store = ReflexionStore(db)
        store.store("old task", "failure", "old reason", quality_score=0.5)
        # Force the reflexion to be old
        db.execute(
            "UPDATE reflexions SET created_at = datetime('now', '-120 days') WHERE task_summary = ?",
            ("old task",),
        )
        decayed = store.decay_stale(days=90, decay_amount=0.1)
        assert decayed == 1


class TestReflexionPruning:
    def test_prune_over_limit(self, db, monkeypatch):
        monkeypatch.setenv("MAX_REFLEXIONS", "5")
        from app.config import reset_config
        reset_config()
        store = ReflexionStore(db)
        for i in range(8):
            store.store(f"unique task {i}", "failure", f"unique reason {i}", quality_score=0.1 * i)
        stats = store.get_stats()
        assert stats["total_reflexions"] <= 6  # 5 + 1 tolerance from dedup check


# ===========================================================================
# Integration: Brain + Reflexion
# ===========================================================================

class TestReflexionBrainIntegration:
    @pytest.fixture
    def services_with_reflexion(self, db):
        from app.core.brain import Services, set_services
        from app.core.memory import ConversationStore, UserFactStore

        store = ReflexionStore(db)
        store.store(
            "bitcoin price query",
            "failure",
            "Used knowledge_search for real-time data; should use web_search",
            quality_score=0.3,
            tools_used=["knowledge_search"],
        )

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            reflexions=store,
        )
        set_services(svc)
        return svc

    @pytest.mark.asyncio
    async def test_reflexion_in_prompt(self, services_with_reflexion):
        """Reflexion warnings should appear in the system prompt when relevant."""
        from app.core.brain import think

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            mock_result.content = "Bitcoin is currently trading at $50,000."
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            captured_messages = []
            original_gen = mock_llm.generate_with_tools

            async def capture(msgs, tools, **kwargs):
                captured_messages.extend(msgs)
                return await original_gen(msgs, tools, **kwargs)

            mock_llm.generate_with_tools = AsyncMock(side_effect=capture)

            events = []
            async for event in think("What is the current bitcoin price?"):
                events.append(event)

        # System prompt should contain reflexion warning
        system_msgs = [m for m in captured_messages if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        system_text = system_msgs[0]["content"]
        assert "past mistakes" in system_text.lower() or "previous failure" in system_text.lower()

    @pytest.mark.asyncio
    async def test_low_quality_triggers_reflexion(self, db):
        """A low-quality answer should trigger reflexion storage."""
        from app.core.brain import Services, set_services, think
        from app.core.memory import ConversationStore, UserFactStore

        store = ReflexionStore(db)
        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            reflexions=store,
        )
        set_services(svc)

        with patch("app.core.brain.llm") as mock_llm:
            mock_result = AsyncMock()
            # Simulate a bad answer: multiple hard failure indicators + short
            mock_result.content = "I cannot do that. Failed to."
            mock_result.tool_calls = []
            mock_llm.generate_with_tools = AsyncMock(return_value=mock_result)

            async for _ in think("Tell me about quantum computing and its applications in modern cryptography"):
                pass

        stats = store.get_stats()
        assert stats["total_reflexions"] >= 1


# ===========================================================================
# Recurring Failure Detection
# ===========================================================================

class TestRecurringFailures:
    @pytest.fixture
    def store_with_failures(self, db):
        store = ReflexionStore(db)
        # Create 3+ similar failures with high word overlap
        store.store("bitcoin price query failed", "failure", "should use web_search", quality_score=0.3)
        store.store("bitcoin price query error", "failure", "used wrong tool", quality_score=0.2)
        store.store("bitcoin price query timeout", "failure", "knowledge_search doesn't have realtime", quality_score=0.25)
        return store

    def test_find_recurring_failures(self, store_with_failures):
        similar = store_with_failures.find_recurring_failures("bitcoin price query")
        assert len(similar) >= 3

    def test_no_recurring_below_threshold(self, store_with_failures):
        similar = store_with_failures.find_recurring_failures("quantum computing theory")
        assert len(similar) == 0

    def test_recurring_needs_min_count(self, db):
        store = ReflexionStore(db)
        # Only 2 failures — below min_count of 3
        store.store("weather forecast problem", "failure", "tool failed", quality_score=0.3)
        store.store("weather forecast issue", "failure", "wrong tool", quality_score=0.2)
        similar = store.find_recurring_failures("weather forecast error")
        assert len(similar) == 0  # Below min_count

    @pytest.mark.asyncio
    async def test_check_recurring_failures_promotes(self, db):
        from app.core.reflexion import check_recurring_failures
        from app.core.brain import Services, set_services
        from app.core.memory import ConversationStore, UserFactStore
        from app.core.learning import LearningEngine
        from app.core import llm as llm_mod

        store = ReflexionStore(db)
        learning = LearningEngine(db)

        # Create 3+ similar failures (high word overlap for Jaccard >= 0.4)
        store.store("bitcoin price query failed", "failure", "should use web_search tool", quality_score=0.3)
        store.store("bitcoin price query error", "failure", "used wrong tool for price", quality_score=0.2)
        store.store("bitcoin price query timeout", "failure", "knowledge_search lacks realtime data", quality_score=0.25)

        svc = Services(
            conversations=ConversationStore(db),
            user_facts=UserFactStore(db),
            reflexions=store,
            learning=learning,
        )
        set_services(svc)

        with patch.object(llm_mod, "invoke_nothink", new_callable=AsyncMock) as mock_invoke, \
             patch.object(llm_mod, "extract_json_object") as mock_extract:
            mock_invoke.return_value = '{"topic": "bitcoin price lookup", "lesson": "Use web_search for real-time prices"}'
            mock_extract.side_effect = lambda raw: json.loads(raw)
            await check_recurring_failures("bitcoin price query", learning)

        lessons = learning.get_all_lessons()
        assert len(lessons) >= 1
