"""Nova must not be penalised for obeying a request to be brief.

The heuristic scorer docked 0.3 with reason "Very short answer" whenever the
answer was <30 chars and the query was >50 chars. That length proxy reads query
LENGTH but not query INTENT, so a query that explicitly DEMANDS brevity still
tripped it.

Live case (reflexion id=187, 2026-08-26):
    query : "Sanity probe: reply with the single word ALIGNED and nothing else."
    answer: "ALIGNED"                                 <- perfect compliance
    score : 0.3, reason "Very short answer"

These land in the reflexion store as FAILURES, and that store feeds lesson
promotion (threshold 0.85) and success patterns. So the owner's own health
checks were dragging down Nova's self-improvement signal.
"""

from __future__ import annotations

import pytest

from app.core.reflexion import _BREVITY_REQUESTED_RE, assess_quality


def score_of(query: str, answer: str) -> float:
    score, _reason = assess_quality(
        answer=answer, tool_results=[], max_tool_rounds=6, query=query)
    return score


def reason_of(query: str, answer: str) -> str:
    _score, reason = assess_quality(
        answer=answer, tool_results=[], max_tool_rounds=6, query=query)
    return reason or ""


BREVITY_REQUESTS = [
    "Sanity probe: reply with the single word ALIGNED and nothing else.",
    "In one word, are you operational? Please do not elaborate further.",
    "Answer with yes or no only: is the knowledge graph currently healthy?",
    "Respond with only the number, no units and no explanation whatsoever.",
    "Give me just the answer, briefly, without any surrounding commentary.",
    "Reply with the name of the capital city and nothing else at all here.",
]

NO_BREVITY_REQUEST = [
    "Explain in detail how Nova's bitemporal knowledge graph handles the "
    "supersession of contradictory facts over time.",
    "Walk me through the full architecture of the hybrid retrieval pipeline "
    "including RRF fusion and the reranker stage.",
]


class TestBrevityDetection:
    @pytest.mark.parametrize("q", BREVITY_REQUESTS)
    def test_detects_explicit_brevity_request(self, q):
        assert _BREVITY_REQUESTED_RE.search(q), (
            f"{q!r} explicitly asks for a short answer"
        )

    @pytest.mark.parametrize("q", NO_BREVITY_REQUEST)
    def test_does_not_fire_on_open_questions(self, q):
        assert not _BREVITY_REQUESTED_RE.search(q), (
            f"{q!r} invites a full answer — brevity there IS suspicious and "
            f"the penalty must still apply"
        )


class TestPenaltyBehaviour:
    def test_obeying_a_brevity_request_is_not_penalised(self):
        """The exact live case."""
        q = "Sanity probe: reply with the single word ALIGNED and nothing else."
        assert len(q) > 50, "guard precondition: query must exceed the length bar"
        assert len("ALIGNED") < 30, "guard precondition: answer must be short"
        assert "Very short answer" not in reason_of(q, "ALIGNED"), (
            "the brevity penalty must not fire when brevity was requested"
        )
        assert score_of(q, "ALIGNED") >= 0.6, (
            "answering exactly as instructed must not score below baseline"
        )

    def test_short_answer_to_an_open_question_is_still_penalised(self):
        """The rule must keep working where it was right.

        This is the direction that matters: if the fix simply disabled the
        penalty, a genuinely evasive one-word reply to a complex question
        would start scoring as a success.
        """
        q = NO_BREVITY_REQUEST[0]
        assert score_of(q, "Yes.") < 0.6, (
            "a 4-char answer to a detailed architecture question is still a "
            "bad answer — the penalty must survive the fix"
        )

    def test_long_answer_never_triggers_the_rule(self):
        q = NO_BREVITY_REQUEST[1]
        long_answer = (
            "The pipeline fuses BM25 over FTS5 with bge-m3 dense vectors using "
            "reciprocal rank fusion at k=60, then applies a composite reranker "
            "weighting 0.55 vector, 0.30 bm25 and 0.15 coverage."
        )
        assert score_of(q, long_answer) >= 0.6
