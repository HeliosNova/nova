"""Lesson precision (audit 2026-09-01).

Measured: LESSON_VECTOR_MAX_DISTANCE=0.9 admitted every lesson (4-5 injected on
88% of turns; the two CPU lessons and a placeholder injected 313 of 448 times);
demoted lessons (confidence 0.35) still took top-5 slots; paraphrase siblings
sat at answer-Jaccard 0.07-0.41 against a 0.55 gate (five rate-limiter lessons);
the negation veto fired on a rhetorical 'not'; five legacy rows injected the
literal provenance string 'Promoted from success reflexion (quality=0.95)'.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.config import config as _cfg
from app.core import learning as learning_mod
from app.core.learning import LearningEngine, _answers_conflict
from app.core.prompt import format_lessons_for_prompt


def _lesson(db, topic, answer, *, confidence=0.8, lesson_text=None):
    cur = db.execute(
        "INSERT INTO lessons (topic, correct_answer, wrong_answer, lesson_text, confidence, times_helpful) "
        "VALUES (?, ?, '', ?, ?, 0)",
        (topic, answer, lesson_text if lesson_text is not None else answer, confidence))
    return cur.lastrowid


def test_vector_gate_is_paraphrase_grade():
    assert float(_cfg.LESSON_VECTOR_MAX_DISTANCE) <= 0.6


def test_demoted_lessons_do_not_take_retrieval_slots(db, monkeypatch):
    # conftest pins MIN_RRF_SCORE at the pre-fix 0.015 (a lone keyword rank
    # blends to ~0.0115 for a brand-new lesson); production runs 0.005.
    old_floor = _cfg.MIN_RRF_SCORE
    _cfg.update(MIN_RRF_SCORE=0.005)
    monkeypatch.setattr(_cfg, "MIN_RRF_SCORE", 0.005, raising=False)
    eng = LearningEngine(db)
    monkeypatch.setattr(eng, "_get_lessons_collection", lambda: None)
    _lesson(db, "High-throughput rate limiter architecture", "Combine sharded counters with local caching for rate limiting", confidence=0.35)
    keep = _lesson(db, "High-throughput rate limiter design", "Use a sliding window log for strict fairness in rate limiting", confidence=0.8)
    try:
        got = eng.get_relevant_lessons("how should I design a high-throughput rate limiter?", limit=5)
    finally:
        _cfg.update(MIN_RRF_SCORE=old_floor)
    assert [l.id for l in got] == [keep]


def test_rhetorical_negation_is_not_a_conflict():
    assert _answers_conflict(
        "Raw GHz is not a reliable metric for real-world performance because throughput depends on IPC",
        "Real-world CPU performance depends more on IPC efficiency and multi-core scaling than raw GHz") is False
    assert _answers_conflict("Pluto is a planet", "Pluto is not a planet") is True
    assert _answers_conflict("The price is $50k", "The price is $60k") is True


def test_paraphrase_sibling_is_a_vector_duplicate(db, monkeypatch):
    eng = LearningEngine(db)
    existing = _lesson(db, "High-throughput rate limiter architecture",
                       "Building scalable rate limiters requires atomic distributed token buckets with real-time metrics")

    class _Col:
        def count(self):
            return 1

        def query(self, query_texts, n_results, include):
            return {"ids": [[str(existing)]], "distances": [[0.11]]}

    monkeypatch.setattr(eng, "_get_lessons_collection", lambda: _Col())
    dup = eng._find_similar_lesson(
        "High-throughput rate limiting at 100k RPS",
        "Achieving high throughput requires distributed state via Redis with local caching to minimise latency")
    assert dup is not None and dup["id"] == existing


def test_far_vector_neighbour_is_not_a_duplicate(db, monkeypatch):
    eng = LearningEngine(db)
    existing = _lesson(db, "CPU clock speed vs performance", "Raw GHz is not a reliable performance metric")

    class _Col:
        def count(self):
            return 1

        def query(self, query_texts, n_results, include):
            return {"ids": [[str(existing)]], "distances": [[0.45]]}

    monkeypatch.setattr(eng, "_get_lessons_collection", lambda: _Col())
    assert eng._find_similar_lesson("Bank of Japan yield curve control",
                                    "The BoJ kept its YCC framework but widened the band") is None


def test_provenance_strings_never_reach_the_prompt():
    lessons = [SimpleNamespace(topic="Direct factual queries",
                               lesson_text="Promoted from success reflexion (quality=0.95)",
                               correct_answer="Use web_search for high-quality answers to specific questions.",
                               wrong_answer="", confidence=0.9)]
    out = format_lessons_for_prompt(lessons)
    assert "Promoted from" not in out
    assert "Use web_search" in out
