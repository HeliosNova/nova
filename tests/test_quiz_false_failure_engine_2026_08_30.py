"""The Lesson Quiz false-failure engine — regression lock.

MECHANISM (measured on the live store 2026-08-30):
The quiz treats a lesson's `correct_answer` as GROUND TRUTH. 41 of 51 lessons
stored a PROCESS INSTRUCTION there instead of a fact. So the quiz would ask a
real question, Nova would answer it CORRECTLY, and the grader would score a
failure because the answer did not match the instruction:

    Quiz: "Which Italian Renaissance painter invented sfumato?"
    Expected (stored correct_answer): "Use web_search to answer direct factual
                                       queries about art history."
    Nova answered: "Leonardo da Vinci..."   -> marked FAILURE

That false failure then (a) writes a 0.2 reflexion, and (b) feeds the 3-strikes
recurring-failure promotion path, which mints ANOTHER instruction-shaped
lesson. A closed loop feeding on its own output.

EVIDENCE: 9 near-identical art-history lessons and 4 calculator lessons minted
June->August; reflexion quality by week went 0.632 avg / 1.0 max (W32) ->
0.362 (W33) -> 0.240 avg / 0.40 max (W34), with 7 of 10 recent rows pinned at
exactly 0.2 and every promotion path (threshold 0.85) starved.

_UNQUIZZABLE_ANSWER_RE v1 (2026-08-27) caught 31 of 41 and missed 10. The
strings below are the REAL stored answers it missed.
"""

from __future__ import annotations

import pytest

from app.monitors.heartbeat_loop import _UNQUIZZABLE_ANSWER_RE


def blocked(answer: str) -> bool:
    return bool(_UNQUIZZABLE_ANSWER_RE.search(answer[:200]))


# Verbatim correct_answer values from lessons the v1 guard let through.
MISSED_BY_V1 = [
    ("lesson 24, Bat and Ball Problem",
     "When solving logic puzzles involving differences in value, always verify "
     "your answer by checking if both conditions hold"),
    ("lesson 26, Rate of Work Problems",
     "When determining production time, focus on individual rates rather than "
     "total quantities; since each machine takes the same time"),
    ("lesson 51, Riddles and Language Precision",
     "When solving word problems, carefully analyze phrasing like 'all but X' "
     "to distinguish between mathematical operations"),
    ("lesson 138, art history external verification",
     "Always verify art historical claims by cross-referencing authoritative "
     "sources before answering quiz questions"),
    ("lesson 147, Use calculators for math problems",
     "Verify solutions with a calculator."),
    ("lesson 372, Use tools for math problems",
     "Apply a calculator to solve arithmetic puzzles."),
    ("lesson 373, Use calculators for solving math problems",
     "Verify solutions with a calculator."),
    ("lesson 374, Art History Fact Verification",
     "Always verify historical claims by cross-referencing authoritative "
     "sources, especially when questions contain"),
    ("lesson 375, Algebraic Verification",
     "Always explicitly calculate and state the final numerical value of x "
     "before using a calculator, ensuring you check"),
    ("lesson 376, Art History Fact Verification",
     "Always verify historical claims by cross-referencing authoritative "
     "sources, especially when questions contain"),
]

# Answers the v1 guard already caught — must STAY caught.
CAUGHT_BY_V1 = [
    "Use web_search to answer direct factual queries about art history.",
    "Use web_search to find factual information.",
    "Use a calculator to check your work after solving math problems.",
    "Use web_search for definitive factual information.",
    "Always use a calculator to solve arithmetic word problems.",
    "Use web_search when the answer requires current or specific factual "
    "information.",
]

# Genuine FACT answers — these are gradable and must remain quizzable.
# Over-blocking these would silently disable the whole self-testing loop,
# which is a worse failure than the one being fixed.
REAL_FACTS = [
    "Leonardo da Vinci",
    "Raphael included a self-portrait as Aristotle in the lower right corner.",
    "x = 8",
    "The answer is 553.",
    "Building scalable high-throughput systems requires combining distributed "
    "state management with atomic token buckets and request coalescing.",
    "Python 3.12 was released in October 2023.",
    "9 sheep are left.",
    "The Strait of Hormuz carries roughly 20% of global oil consumption.",
    "bge-m3 produces 1024-dimensional embeddings.",
    "Ollama 0.32.15 is the pinned production version.",
]


class TestGuardCatchesTheEngine:
    @pytest.mark.parametrize("label,answer", MISSED_BY_V1,
                             ids=[m[0] for m in MISSED_BY_V1])
    def test_v1_misses_are_now_blocked(self, label, answer):
        assert blocked(answer), (
            f"{label}: this instruction-shaped answer would be quizzed as if "
            f"it were a fact, so a CORRECT response scores as a failure and "
            f"mints another duplicate lesson"
        )

    @pytest.mark.parametrize("answer", CAUGHT_BY_V1)
    def test_v1_catches_stay_caught(self, answer):
        assert blocked(answer), "widening the pattern must not lose a v1 catch"


class TestGuardDoesNotOverBlock:
    """The dangerous direction. If this over-blocks, Nova stops quizzing
    itself entirely and the failure is silent — no error, just no learning."""

    @pytest.mark.parametrize("answer", REAL_FACTS)
    def test_genuine_facts_remain_quizzable(self, answer):
        assert not blocked(answer), (
            f"{answer!r} is a gradable fact. Blocking it would disable the "
            f"self-testing loop rather than fix it."
        )

    def test_fact_mentioning_a_tool_is_still_quizzable(self):
        """A fact that happens to NAME a tool is not an instruction."""
        assert not blocked(
            "The calculator tool returned 246971 for that expression.")
        assert not blocked(
            "Nova's web_search tool routes through SearXNG on port 8888.")


class TestSeparationIsReal:
    def test_no_overlap_between_the_two_sets(self):
        """Guards against a pattern that trivially passes by matching all or
        nothing — the sets must be separated, not uniformly classified."""
        assert all(blocked(a) for _, a in MISSED_BY_V1)
        assert all(blocked(a) for a in CAUGHT_BY_V1)
        assert not any(blocked(a) for a in REAL_FACTS)
