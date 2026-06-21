"""Hard-reasoning detection broadened to math/logic word problems (2026-06-13).

A live probe showed the 9B failing the classic bat-and-ball trap because the
detector only fired on explicit verbs (analyze/derive/why) and missed word
problems — so they never got extended thinking. These pin the broadened gate.
"""
import pytest

from app.core.agent_loop import _is_hard_reasoning_query, _is_math_logic_word_problem


WORD_PROBLEMS = [
    "A bat and ball cost 1.10 total. The bat costs 1.00 more than the ball. What is the new price difference?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
    "Tom is twice as old as his sister. In 6 years he will be 32. How old is the sister now?",
    "A train travels 60 miles in 1.5 hours. How far does it go in 4 hours?",
]
EXPLICIT_REASONING = [
    "compare React and Vue for a large application",
    "explain why the sky is blue",
]
NOT_HARD = [
    "what is 17 times 23?",          # bare arithmetic — calculator handles it
    "hello how are you today",
    "what is the capital of France",
    "what's the weather like at 3pm today",
]


@pytest.mark.parametrize("q", WORD_PROBLEMS)
def test_word_problems_flagged(q):
    assert _is_math_logic_word_problem(q) is True
    assert _is_hard_reasoning_query(q) is True


@pytest.mark.parametrize("q", EXPLICIT_REASONING)
def test_explicit_reasoning_still_flagged(q):
    assert _is_hard_reasoning_query(q) is True


@pytest.mark.parametrize("q", NOT_HARD)
def test_simple_queries_not_flagged(q):
    assert _is_hard_reasoning_query(q) is False
