"""Scheduled recurring-failure promotion sweep (audit 2026-08-23).

check_recurring_failures only runs when a NEW failure arrives in live chat
(brain.py post-response) — the same chat-starvation that kept auto-skills at
zero organic skills: with monitor-driven usage, recurring failure clusters sat
unpromoted forever (live probe found an n=9 quiz-failure cluster and 0
auto-lessons ever). sweep_recurring_failures walks recent failures on a
schedule and pushes each corroborated cluster through the SAME promotion path.
Eval-seeded failures (is_eval=1) are synthetic and must never form clusters.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.learning import LearningEngine
from app.core.reflexion import ReflexionStore, sweep_recurring_failures


def _seed_failure(db, summary: str, is_eval: int = 0):
    db.execute(
        "INSERT INTO reflexions (task_summary, outcome, reflection, quality_score, is_eval) "
        "VALUES (?, 'failure', 'missed the correct painter attribution', 0.3, ?)",
        (summary, is_eval),
    )


def _auto_lessons(db):
    return db.fetchall("SELECT id, topic, lesson_text FROM lessons WHERE lesson_text LIKE 'Auto-lesson:%'")


@pytest.mark.asyncio
async def test_cluster_of_three_promotes_lesson(db):
    for i in range(3):
        _seed_failure(db, f"Quiz on Renaissance painter attribution question {i}")
    store = ReflexionStore(db)
    learning = LearningEngine(db)
    with patch("app.core.llm.invoke_nothink", new=AsyncMock(return_value=json.dumps(
            {"topic": "renaissance attribution", "lesson": "Verify painter attributions against period sources"}))):
        promoted = await sweep_recurring_failures(store, learning, max_promotions=2)
    lessons = _auto_lessons(db)
    assert promoted == 1 and len(lessons) == 1, f"expected 1 auto-lesson, got {[dict(r) for r in lessons]}"


@pytest.mark.asyncio
async def test_scattered_failures_do_not_promote(db):
    _seed_failure(db, "Quiz on Renaissance painter attribution")
    _seed_failure(db, "Weather forecast tool timeout for Tokyo")
    _seed_failure(db, "Currency conversion rounding error yen")
    store = ReflexionStore(db)
    learning = LearningEngine(db)
    with patch("app.core.llm.invoke_nothink", new=AsyncMock(return_value=json.dumps(
            {"topic": "x", "lesson": "y"}))) as mock_llm:
        promoted = await sweep_recurring_failures(store, learning, max_promotions=2)
        mock_llm.assert_not_called()
    assert promoted == 0 and len(_auto_lessons(db)) == 0


@pytest.mark.asyncio
async def test_eval_failures_never_cluster(db):
    # 3 identical failures but all synthetic (is_eval=1) — must not promote.
    for i in range(3):
        _seed_failure(db, f"Quiz on Renaissance painter attribution question {i}", is_eval=1)
    store = ReflexionStore(db)
    learning = LearningEngine(db)
    with patch("app.core.llm.invoke_nothink", new=AsyncMock(return_value=json.dumps(
            {"topic": "x", "lesson": "y"}))) as mock_llm:
        promoted = await sweep_recurring_failures(store, learning, max_promotions=2)
        mock_llm.assert_not_called()
    assert promoted == 0 and len(_auto_lessons(db)) == 0


@pytest.mark.asyncio
async def test_one_cluster_promoted_once_not_per_member(db):
    # 4 similar failures = ONE cluster → one LLM call, one lesson, not four.
    for i in range(4):
        _seed_failure(db, f"Quiz on Renaissance painter attribution question {i}")
    store = ReflexionStore(db)
    learning = LearningEngine(db)
    with patch("app.core.llm.invoke_nothink", new=AsyncMock(return_value=json.dumps(
            {"topic": "renaissance attribution", "lesson": "Verify attributions"}))) as mock_llm:
        promoted = await sweep_recurring_failures(store, learning, max_promotions=5)
    assert promoted == 1
    assert mock_llm.call_count == 1, f"cluster promoted per-member: {mock_llm.call_count} LLM calls"
