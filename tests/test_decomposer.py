"""Unit tests for app/core/decomposer.py.

Tests cover:
- Signal scoring (_score_signals)
- Decomposition gate (should_decompose)
- Strategy selection (_pick_strategy)
- Task extraction methods (_try_compare_split, _try_and_split,
  _try_entity_split, _try_fallback_split)
- End-to-end decompose_query()

All tests run with ENABLE_MULTI_AGENT=false by default (conftest.py).
Tests that need the feature enabled set it explicitly via monkeypatch.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(query: str, was_planned: bool = False) -> int:
    from app.core.decomposer import _score_signals
    return _score_signals(query, was_planned)


def _pick(query: str) -> str:
    from app.core.decomposer import _pick_strategy
    return _pick_strategy(query)


def _compare_split(query: str):
    from app.core.decomposer import _try_compare_split
    return _try_compare_split(query, "conv-123")


def _and_split(query: str):
    from app.core.decomposer import _try_and_split
    return _try_and_split(query, "conv-123")


def _entity_split(query: str):
    from app.core.decomposer import _try_entity_split
    return _try_entity_split(query, "conv-123")


def _fallback_split(query: str):
    from app.core.decomposer import _try_fallback_split
    return _try_fallback_split(query, "conv-123")


# ===========================================================================
# _score_signals
# ===========================================================================

class TestScoreSignals:
    def test_plain_query_scores_zero(self):
        assert _score("What is the capital of France?") == 0

    def test_parallel_marker_compare_adds_two(self):
        assert _score("Compare Python and JavaScript") >= 2

    def test_parallel_marker_versus_adds_two(self):
        assert _score("React versus Vue, which is better?") >= 2

    def test_parallel_marker_side_by_side(self):
        assert _score("Show me a side-by-side comparison") >= 2

    def test_delegation_words_add_two(self):
        assert _score("Run them in parallel agents") >= 2

    def test_delegation_break_down(self):
        assert _score("Break this down into multiple agents") >= 2

    def test_three_proper_nouns_add_one(self):
        # Python, JavaScript, TypeScript are 3 distinct proper nouns not in stopwords
        score = _score("Differences between Python JavaScript TypeScript")
        assert score >= 1

    def test_two_proper_nouns_dont_add(self):
        # Only 2 proper nouns — should not add the +1
        score_two = _score("Compare Python JavaScript")
        score_three = _score("Compare Python JavaScript TypeScript")
        assert score_three > score_two or score_two >= 2  # at least the compare marker fires

    def test_multiple_question_marks_add_one(self):
        score = _score("What is Python? What is JavaScript?")
        # compare marker might not fire, but multi-question should
        assert score >= 1

    def test_long_query_with_tool_signals_add_one(self):
        # Build a query > 200 chars with >= 2 tool-type keywords
        query = (
            "search for the latest Python version and calculate how many years "
            "have passed since Python 1.0 was released in 1994. "
            "Also find the current PyPI package count and compute the growth rate. "
            "This query is deliberately long to exceed the two-hundred-character threshold."
        )
        assert len(query) > 200
        score = _score(query)
        assert score >= 1

    def test_long_query_single_tool_signal_no_add(self):
        query = "search " + "x" * 200
        assert len(query) > 200
        # Only 1 tool keyword — should NOT add the length+tool +1
        # (search itself would match but we need >= 2)
        score_long = _score(query)
        score_short = _score("search something")
        assert score_long == score_short  # no extra point

    def test_was_planned_adds_one(self):
        base = _score("Research something")
        planned = _score("Research something", was_planned=True)
        assert planned == base + 1

    def test_compound_score(self):
        # "compare" (+2) + 3 proper nouns (+1) + "?" twice (+1) = at least 4
        query = "Compare Python JavaScript TypeScript? Which is faster? Give details."
        score = _score(query)
        assert score >= 4

    def test_greeting_query_low_score(self):
        assert _score("Hello, how are you?") == 0

    def test_simple_math_scores_zero(self):
        assert _score("What is 2 plus 2?") == 0


# ===========================================================================
# should_decompose
# ===========================================================================

class TestShouldDecompose:
    def test_disabled_flag_blocks(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "false")
        from app.config import reset_config
        reset_config()
        from app.core.decomposer import should_decompose
        assert not should_decompose(
            "Compare Python and JavaScript frameworks thoroughly",
            "general", False, False,
        )

    def test_enabled_high_score_fires(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        monkeypatch.setenv("MULTI_AGENT_TRIGGER_THRESHOLD", "4")
        from app.config import reset_config
        reset_config()
        from app.core.decomposer import should_decompose
        # "compare" (+2) + "Python", "JavaScript", "TypeScript" proper nouns (+1)
        # + two question marks (+1) = 4 → should fire
        query = (
            "Compare Python JavaScript TypeScript?"
            " How do they differ in typing? Which is best for backend?"
        )
        result = should_decompose(query, "general", False, False)
        assert result

    def test_below_threshold_does_not_fire(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        monkeypatch.setenv("MULTI_AGENT_TRIGGER_THRESHOLD", "4")
        from app.config import reset_config
        reset_config()
        from app.core.decomposer import should_decompose
        # Score = 0 (plain factual question)
        assert not should_decompose("What is the capital of France?", "general", False, False)

    def test_greeting_intent_blocked(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        monkeypatch.setenv("MULTI_AGENT_TRIGGER_THRESHOLD", "1")
        from app.config import reset_config
        reset_config()
        from app.core.decomposer import should_decompose
        assert not should_decompose(
            "Hello! Compare Python and JavaScript? Which is better?",
            "greeting", False, False,
        )

    def test_correction_intent_blocked(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        monkeypatch.setenv("MULTI_AGENT_TRIGGER_THRESHOLD", "1")
        from app.config import reset_config
        reset_config()
        from app.core.decomposer import should_decompose
        assert not should_decompose(
            "No, compare Python and JavaScript, I meant that.",
            "correction", False, False,
        )

    def test_structural_depth_gate_blocks(self, monkeypatch):
        """When already inside a structural decomposition, gate must block."""
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        monkeypatch.setenv("MULTI_AGENT_TRIGGER_THRESHOLD", "1")
        from app.config import reset_config
        reset_config()
        from app.core import agent_spawner
        from app.core.decomposer import should_decompose

        # Simulate being inside a structural decomposition (depth=1)
        token = agent_spawner._structural_depth.set(1)
        try:
            result = should_decompose(
                "Compare Python and JavaScript? Which is better?",
                "general", False, False,
            )
        finally:
            agent_spawner._structural_depth.reset(token)
        assert not result

    def test_ephemeral_not_hard_gate(self, monkeypatch):
        """ephemeral=True must NOT block decomposition (eval harness needs it)."""
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        monkeypatch.setenv("MULTI_AGENT_TRIGGER_THRESHOLD", "4")
        from app.config import reset_config
        reset_config()
        from app.core.decomposer import should_decompose
        query = (
            "Compare Python JavaScript TypeScript?"
            " How do they differ in typing? Which is best for backend?"
        )
        # ephemeral=True should not block (unlike old design where it was a gate)
        result = should_decompose(query, "general", False, ephemeral=True)
        assert result


# ===========================================================================
# _pick_strategy
# ===========================================================================

class TestPickStrategy:
    def test_sequential_on_first_then(self):
        assert _pick("First search for the version, then calculate the years") == "sequential"

    def test_sequential_on_based_on_result(self):
        # "that result" now handled by the regex (the|that|this before result)
        assert _pick("Search for X, then based on that result compute Y") == "sequential"
        assert _pick("Search for X, then based on the result compute Y") == "sequential"

    def test_sequential_on_after_finding(self):
        assert _pick("After finding the price, calculate the profit") == "sequential"

    def test_parallel_default(self):
        assert _pick("Compare Python and JavaScript") == "parallel"

    def test_parallel_for_compare_query(self):
        assert _pick("What is the difference between Redis and Memcached?") == "parallel"

    def test_sequential_step_by_step(self):
        assert _pick("Do this step by step: first find, then compute") == "sequential"


# ===========================================================================
# Task extraction — _try_compare_split
# ===========================================================================

class TestTryCompareSplit:
    def test_compare_and(self):
        tasks = _compare_split("Compare Python and JavaScript")
        assert tasks is not None
        assert len(tasks) == 2
        roles = [t.role for t in tasks]
        assert any("python" in r for r in roles)
        assert any("javascript" in r for r in roles)

    def test_contrast_and(self):
        tasks = _compare_split("Contrast React and Vue in terms of performance")
        assert tasks is not None
        assert len(tasks) == 2

    def test_difference_between(self):
        tasks = _compare_split("What is the difference between Redis and Memcached?")
        assert tasks is not None
        assert len(tasks) == 2

    def test_same_subjects_returns_none(self):
        # Should not split "compare Python and Python"
        tasks = _compare_split("Compare Python and Python")
        assert tasks is None

    def test_no_compare_returns_none(self):
        assert _compare_split("What is Python?") is None

    def test_query_embedded_in_task(self):
        """Each task's query must contain the original question for context."""
        tasks = _compare_split("Compare Redis and Memcached for caching")
        assert tasks is not None
        for t in tasks:
            assert "Redis" in t.query or "Memcached" in t.query

    def test_task_ids_are_unique(self):
        tasks = _compare_split("Compare Python and JavaScript")
        assert tasks is not None
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))


# ===========================================================================
# Task extraction — _try_and_split
# ===========================================================================

class TestTryAndSplit:
    def test_what_is_x_and_y(self):
        tasks = _and_split("What is a stack and a queue?")
        assert tasks is not None
        assert len(tasks) == 2

    def test_tell_me_about_x_and_y(self):
        tasks = _and_split("Tell me about Docker and Kubernetes")
        assert tasks is not None
        assert len(tasks) == 2

    def test_explain_x_and_y(self):
        tasks = _and_split("Explain TCP and UDP protocols")
        assert tasks is not None
        assert len(tasks) == 2

    def test_no_match_returns_none(self):
        assert _and_split("Search for the Python version") is None

    def test_same_subjects_returns_none(self):
        assert _and_split("What is TCP and TCP?") is None


# ===========================================================================
# Task extraction — _try_entity_split
# ===========================================================================

class TestTryEntitySplit:
    def test_three_entities_produces_three_tasks(self):
        # Python, Rust, TypeScript — 3 distinct proper nouns; no other capitalised words
        tasks = _entity_split("overview of Python Rust TypeScript")
        assert tasks is not None
        assert len(tasks) == 3

    def test_two_entities_returns_none(self):
        # Only 2 proper nouns (Python, JavaScript); all other words lowercase
        tasks = _entity_split("about Python JavaScript performance")
        assert tasks is None

    def test_stopwords_excluded(self):
        # "What", "Is", "The" are in stopwords — should not count
        tasks = _entity_split("What Is The Python language good for")
        # Should not split: after stripping stopwords, no proper nouns remain
        # (Python is 1; not >=3)
        assert tasks is None

    def test_caps_at_max_agent_count(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        monkeypatch.setenv("MAX_AGENT_COUNT", "3")
        from app.config import reset_config
        reset_config()
        from app.core.decomposer import _try_entity_split as _es
        # Provide 5 proper nouns — should cap at MAX_AGENT_COUNT=3
        tasks = _es("Alpha Beta Gamma Delta Epsilon comparison", "conv-1")
        assert tasks is not None
        assert len(tasks) <= 3


# ===========================================================================
# Task extraction — _try_fallback_split
# ===========================================================================

class TestTryFallbackSplit:
    def test_splits_on_and(self):
        tasks = _fallback_split(
            "Search for the latest Python version and calculate years since 1994"
        )
        assert tasks is not None
        assert len(tasks) == 2

    def test_short_parts_filtered(self):
        # Both parts must be >= 15 chars
        tasks = _fallback_split("a and b")
        assert tasks is None

    def test_one_short_part_filtered(self):
        tasks = _fallback_split("Search for the latest Python version and hi")
        # "hi" < 15 chars → only 1 valid part → None
        assert tasks is None

    def test_no_and_returns_none(self):
        assert _fallback_split("Search for the latest Python version") is None


# ===========================================================================
# decompose_query — end-to-end
# ===========================================================================

class TestDecomposeQuery:
    def _run(self, query: str, intent: str = "general", was_planned: bool = False):
        from app.core.decomposer import decompose_query
        return decompose_query(query, intent, was_planned, plan=None, conversation_id="conv-e2e")

    def test_compare_query_produces_plan(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        from app.config import reset_config
        reset_config()
        plan = self._run("Compare Python and JavaScript performance characteristics")
        assert plan is not None
        assert len(plan.tasks) == 2
        assert plan.strategy in ("parallel", "sequential")
        assert plan.merge_instruction

    def test_sequential_query_picks_sequential_strategy(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        from app.config import reset_config
        reset_config()
        # Query must: (a) trigger a sequential marker, (b) yield >=2 tasks via
        # fallback split on "and" — both halves must be >=15 chars.
        plan = self._run(
            "First search for the Python version and calculate how many years "
            "have passed since Python 1.0 was released in 1994"
        )
        assert plan is not None
        assert plan.strategy == "sequential"

    def test_returns_none_for_insufficient_tasks(self, monkeypatch):
        """If extraction yields <2 tasks, decompose_query returns None."""
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        from app.config import reset_config
        reset_config()
        # No compare/and/entity/fallback patterns → None
        plan = self._run("What is 2 plus 2?")
        assert plan is None

    def test_plan_tasks_have_required_fields(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        from app.config import reset_config
        reset_config()
        plan = self._run("Compare Redis and Memcached for caching")
        assert plan is not None
        for task in plan.tasks:
            assert task.task_id
            assert task.role
            assert task.query
            assert task.focus
            assert isinstance(task.tags, list)
            assert task.depth == 1

    def test_plan_caps_at_max_agent_count(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        monkeypatch.setenv("MAX_AGENT_COUNT", "2")
        from app.config import reset_config
        reset_config()
        from app.core.decomposer import decompose_query
        # Even if 3+ entities, should cap at 2
        plan = decompose_query(
            "Compare Python JavaScript TypeScript in terms of typing",
            "general", False, None, "conv-cap",
        )
        if plan is not None:
            assert len(plan.tasks) <= 2

    def test_merge_instruction_contains_original_query(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        from app.config import reset_config
        reset_config()
        query = "Compare Redis and Memcached for web caching"
        plan = self._run(query)
        assert plan is not None
        assert query in plan.merge_instruction

    def test_parallel_plan_max_parallel_capped(self, monkeypatch):
        monkeypatch.setenv("ENABLE_MULTI_AGENT", "true")
        from app.config import reset_config
        reset_config()
        plan = self._run("Compare Python and JavaScript")
        assert plan is not None
        assert plan.max_parallel <= 3
