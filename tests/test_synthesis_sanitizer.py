"""Tests for app.core.agent_loop.sanitize_synthesis — strips deliberation
scaffolding leaks from final answers; and _is_deliberation_query detector."""

from __future__ import annotations

from app.core.agent_loop import sanitize_synthesis, _is_deliberation_query


# ---- empty / no-op -----------------------------------------------------

def test_empty_input():
    out, n = sanitize_synthesis("")
    assert out == "" and n == 0


def test_clean_text_unchanged():
    text = (
        "A distributed cache for 50,000 reads per second uses sharded Redis "
        "with consistent hashing, master-slave replication for high "
        "availability, and a 5% memory headroom for eviction. Sub-millisecond "
        "latency requires keeping all hot keys in RAM."
    )
    out, n = sanitize_synthesis(text)
    assert out == text
    assert n == 0


# ---- sentence-level deletion -------------------------------------------

def test_strips_step_id_scenario_sentence():
    text = (
        "Use consistent hashing for shard distribution. "
        "However, specific details regarding which consistency model best "
        "fits this exact 26/174 scenario remain unverified."
    )
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "26/174" not in out
    assert "consistent hashing" in out


def test_strips_search_logs_indicated_sentence():
    text = (
        "Token bucket is a strong choice. We cannot specify exact thresholds "
        "as indicated by failed retrieval attempts on rate limiter specifics "
        "in the search logs."
    )
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "search logs" not in out.lower()
    assert "Token bucket is a strong choice" in out


def test_strips_based_on_analysis_plan_sentence():
    text = (
        "Redis is the standard choice. Below is the breakdown based on "
        "your completed analysis plan."
    )
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "analysis plan" not in out.lower()
    assert "Redis is the standard choice" in out


def test_strips_marked_as_requiring_live_config():
    text = (
        "Use sharded Redis. The exact replica count is marked as requiring "
        "live configuration based on actual node health checks."
    )
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "marked as requiring" not in out.lower()
    assert "Use sharded Redis" in out


def test_strips_failed_retrieval_attempts_sentence():
    text = (
        "Consistent hashing handles distribution. Failed retrieval attempts "
        "on supercomputer storage protocols leave one parameter undetermined."
    )
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "failed retrieval" not in out.lower()


def test_strips_search_logs_with_quantifier_qualifier():
    """Catches 'in available search logs' / 'in our search logs' variants."""
    text = (
        "Use sharding. Specific hardware details were not explicitly verified "
        "in available search logs; however industry practice dictates SSDs."
    )
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "search logs" not in out.lower()
    assert "Use sharding" in out


def test_strips_parenthetical_search_logs_note():
    text = (
        "The architecture works. *(Note: Specific details about hardware were "
        "not explicitly verified in available search logs; standard practice "
        "dictates SSDs.)*"
    )
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "search logs" not in out.lower()
    assert "The architecture works" in out


# ---- phrase-level redaction --------------------------------------------

def test_strips_inline_step_paren():
    text = "Use consistent hashing (step 26/174) for shard distribution."
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "26/174" not in out
    assert "consistent hashing" in out
    assert "for shard distribution" in out


def test_strips_inline_analysis_plan_paren():
    text = "Use Redis (based on the completed analysis plan from step 3) for caching."
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "analysis plan" not in out.lower()
    assert "Use Redis" in out
    assert "for caching" in out


def test_strips_in_search_results_phrase():
    text = "Bitcoin trades at $78,000 in the provided search results today."
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "search results" not in out.lower()
    assert "$78,000" in out


def test_strips_not_specified_here_paren():
    text = "Replica count is 3 (not specified here in the requirements) by default."
    out, n = sanitize_synthesis(text)
    assert n >= 1
    assert "not specified" not in out.lower()
    assert "Replica count is 3" in out


# ---- preserves legitimate references -----------------------------------

def test_keeps_legitimate_search_word():
    """'Search' on its own (not 'search logs' / 'search results') is fine."""
    text = "Use binary search to find the median in O(log n)."
    out, _ = sanitize_synthesis(text)
    assert "binary search" in out


def test_keeps_legitimate_step_count_in_user_speech():
    """Numbers like '5 steps' in sentences not framed as plan IDs stay."""
    text = "The algorithm has 5 steps: initialize, hash, distribute, replicate, return."
    out, _ = sanitize_synthesis(text)
    assert "5 steps" in out


def test_keeps_legitimate_plan_word():
    text = "Your deployment plan should include monitoring and rollback."
    out, _ = sanitize_synthesis(text)
    assert "deployment plan" in out


# ---- multiple patterns combine -----------------------------------------

def test_strips_multiple_patterns_in_one_text():
    text = (
        "Designing a rate limiter requires sharding. "
        "However, specific configuration details for this 26/174 scenario "
        "are not fully specified in the search logs. "
        "Use a token bucket algorithm. "
        "Marked as requiring live configuration based on node health checks."
    )
    out, n = sanitize_synthesis(text)
    assert n >= 2  # at least two sentence-level scrubs
    assert "26/174" not in out
    assert "search logs" not in out.lower()
    assert "marked as requiring" not in out.lower()
    assert "Designing a rate limiter requires sharding" in out
    assert "token bucket algorithm" in out


# ---- whitespace cleanup -------------------------------------------------

def test_collapses_double_spaces_after_redaction():
    text = "Use Redis  (based on the completed analysis plan)  for caching."
    out, _ = sanitize_synthesis(text)
    assert "  " not in out


def test_strips_orphan_punctuation():
    text = "Use Redis . . for caching."
    out, _ = sanitize_synthesis(text)
    assert ". ." not in out
    assert "Use Redis" in out


# ---- _is_deliberation_query detector -----------------------------------

def test_deliberation_query_step_by_step():
    assert _is_deliberation_query(
        "Walk me through how you would design a rate limiter step by step"
    ) is True


def test_deliberation_query_design_a():
    assert _is_deliberation_query(
        "Design a distributed cache for 50,000 reads per second"
    ) is True


def test_deliberation_query_architect():
    assert _is_deliberation_query(
        "Architect a microservices system for an e-commerce platform"
    ) is True


def test_deliberation_query_how_would_you():
    assert _is_deliberation_query(
        "How would you build a service mesh that handles 100k RPS?"
    ) is True


def test_deliberation_query_tradeoffs():
    assert _is_deliberation_query(
        "What are the trade-offs between gRPC and REST for inter-service calls?"
    ) is True


def test_simple_lookup_not_deliberation():
    assert _is_deliberation_query(
        "What is 2 + 2?"
    ) is False


def test_factual_question_not_deliberation():
    assert _is_deliberation_query(
        "Who is the CEO of Apple?"
    ) is False


def test_definition_not_deliberation():
    assert _is_deliberation_query(
        "Define eventual consistency"
    ) is False


def test_empty_query_not_deliberation():
    assert _is_deliberation_query("") is False
    assert _is_deliberation_query(None) is False
