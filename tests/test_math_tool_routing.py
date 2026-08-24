"""Forced calculator/code_exec routing for math-classified queries (model
program item 2, 2026-08-24).

Field data (2026): tool-executed arithmetic beats mental arithmetic on every
competition-math benchmark. Nova already forces tools on EXPLICIT requests
("use the calculator"); this extends the same forcing to IMPLICIT math —
arithmetic expressions and constraint word problems — via a [MATH TOOL
ROUTING] system block. The classifier must stay precise: dates, version
strings, and ordinary prose with digits are NOT math queries.
"""

from app.core.brain import _is_computational_query


class TestComputationalDetection:
    # --- must fire ---
    def test_percent_of(self):
        assert _is_computational_query("What is 15% of 240?") is True

    def test_spaced_arithmetic_expression(self):
        assert _is_computational_query("Compute 847 * 293 - 1200") is True

    def test_sqrt_phrasing(self):
        assert _is_computational_query("What's the square root of 7744?") is True

    def test_divided_by(self):
        assert _is_computational_query("What is 9840 divided by 12?") is True

    def test_word_problem_with_constraints(self):
        # Same class as the bat-and-ball trap: relational words + >=2 numbers.
        assert _is_computational_query(
            "A bat and a ball cost $1.10 total. The bat costs $1.00 more than "
            "the ball. How much does the ball cost?"
        ) is True

    def test_compound_interest_word_problem(self):
        assert _is_computational_query(
            "If I invest 5000 at 4% annual interest, how much do I have after 10 years?"
        ) is True

    # --- must NOT fire ---
    def test_date_is_not_math(self):
        assert _is_computational_query("What happened on 2026-08-24 in tech news?") is False

    def test_version_string_is_not_math(self):
        assert _is_computational_query("What changed in Ollama 0.32.15?") is False

    def test_prose_with_one_number_is_not_math(self):
        assert _is_computational_query("Tell me about the Apollo 11 mission") is False

    def test_greeting_is_not_math(self):
        assert _is_computational_query("good morning, what's new today?") is False

    def test_year_range_is_not_math(self):
        assert _is_computational_query("Summarize AI progress from 2020-2026") is False

    def test_empty_and_short(self):
        assert _is_computational_query("") is False
        assert _is_computational_query("hi") is False
