"""Pair-timeout semantics + seeded-state cleanup guarantees (2026-07-07).

Post num_ctx-fix rules:

- A timed-out BEFORE leg is evidence (before_correct=False) — with the seed
  absent the pipeline tool-hunts the unknown entity and delivers no correct
  answer within the budget. Only an AFTER-leg timeout makes a pair untestable.
  (The old both-legs rule dated from when num_ctx truncation cut the tool
  block and made before-legs unrealistically fast.)
- Seeded state (lessons / KG facts / "Eval: *" skills) must never outlive a
  run — including a run cancelled by a client disconnect. Orphaned eval skills
  hijack real queries via semantic skill matching (observed live 2026-07-07).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.monitors.eval_harness import EvalHarness, EvalTask, TaskResult, _Invocation


def _inv(text: str, timed_out: bool = False) -> _Invocation:
    return _Invocation(
        response_text=text, tools_invoked=[], skill_used=None,
        decomposed=False, max_decomposition_depth=0,
        reflexion_score=0.8 if text else None,
        latency_seconds=180.0 if timed_out else 0.01,
        error="Timeout after 180s" if timed_out else None,
        timed_out=timed_out,
    )


def _mem_task() -> EvalTask:
    return EvalTask(
        id="mem_t2", category="memory-learning",
        query="What is the codename of Nova scheduler?",
        assertions=[{"type": "answer_contains", "value": "Chronos"}],
        seed_lesson={"topic": "t", "correct_answer": "It is Chronos.",
                     "lesson_text": "codename Chronos", "confidence": 0.95},
    )


def _kg_task() -> EvalTask:
    return EvalTask(
        id="kg_t", category="kg-retrieval",
        query="Where is Vorenza based?",
        assertions=[{"type": "answer_contains", "value": "Brindlemark"}],
        seed_fact={"subject": "Vorenza", "predicate": "based_in",
                   "object": "Brindlemark", "confidence": 0.95},
    )


def _svc(learning=None, kg=None, skills=None) -> MagicMock:
    svc = MagicMock()
    svc.learning = learning
    svc.kg = kg
    svc.skills = skills
    return svc


class TestBeforeLegTimeoutIsEvidence:
    @pytest.mark.asyncio
    async def test_memory_before_leg_timeout_is_testable(self):
        """before timed out with no answer + after correct → caused_fix=True."""
        learning = MagicMock()
        learning.add_knowledge_lesson.return_value = 1
        learning._db.fetchall.return_value = []
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=_svc(learning=learning)):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(side_effect=[_inv("", timed_out=True),
                                           _inv("It is Chronos.")]),
            ):
                result = await harness._run_memory_task(_mem_task())
        assert result.timed_out is False
        assert result.passed is True
        assert result.memory_before_correct is False
        assert result.memory_caused_fix is True
        assert any("before leg timed out" in a for a in result.failed_assertions)

    @pytest.mark.asyncio
    async def test_kg_before_leg_timeout_is_testable(self):
        kg = MagicMock()
        kg.add_fact = AsyncMock()
        kg.delete_fact = AsyncMock()
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=_svc(kg=kg)):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(side_effect=[_inv("", timed_out=True),
                                           _inv("Vorenza is based in Brindlemark.")]),
            ):
                result = await harness._run_kg_task(_kg_task())
        assert result.timed_out is False
        assert result.passed is True
        assert result.memory_before_correct is False
        assert result.memory_caused_fix is True

    @pytest.mark.asyncio
    async def test_kg_after_leg_timeout_is_untestable(self):
        kg = MagicMock()
        kg.add_fact = AsyncMock()
        kg.delete_fact = AsyncMock()
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=_svc(kg=kg)):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(side_effect=[_inv("No idea."),
                                           _inv("", timed_out=True)]),
            ):
                result = await harness._run_kg_task(_kg_task())
        assert result.timed_out is True
        assert result.memory_before_correct is None
        assert result.memory_caused_fix is None


class TestSeededStateNeverOutlivesRun:
    @pytest.mark.asyncio
    async def test_memory_cancelled_run_still_purges_lesson(self):
        """CancelledError mid-after-leg must not orphan the seeded lesson."""
        learning = MagicMock()
        learning.add_knowledge_lesson.return_value = 1
        learning._db.fetchall.return_value = [{"id": 1}]
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=_svc(learning=learning)):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(side_effect=[_inv("No idea."),
                                           asyncio.CancelledError()]),
            ):
                with pytest.raises(asyncio.CancelledError):
                    await harness._run_memory_task(_mem_task())
        # pre-clean + finally-clean both ran
        assert learning._db.fetchall.call_count == 2
        learning.delete_lesson.assert_called()

    @pytest.mark.asyncio
    async def test_kg_cancelled_run_still_retires_fact(self):
        """CancelledError mid-after-leg must not leave the fictional triple live."""
        kg = MagicMock()
        kg.add_fact = AsyncMock()
        kg.delete_fact = AsyncMock()
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=_svc(kg=kg)):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(side_effect=[_inv("No idea."),
                                           asyncio.CancelledError()]),
            ):
                with pytest.raises(asyncio.CancelledError):
                    await harness._run_kg_task(_kg_task())
        # pre-clean + finally-clean both ran
        assert kg.delete_fact.await_count == 2

    @pytest.mark.asyncio
    async def test_run_all_cleans_seeded_skills_on_success(self):
        skills = MagicMock()
        skills._db.fetchall.return_value = [{"id": 7}, {"id": 8}]
        harness = EvalHarness()
        task = EvalTask(id="r1", category="reasoning", query="2+2?",
                        assertions=[{"type": "answer_contains", "value": "4"}])
        ok = TaskResult(
            task_id="r1", category="reasoning", query="2+2?", passed=True,
            response_text="4", tools_invoked=[], skill_used=None,
            reflexion_score=0.8, latency_seconds=0.1, failed_assertions=[],
        )
        with patch("app.core.brain.get_services", return_value=_svc(skills=skills)):
            with patch.object(EvalHarness, "_seed_skills"), \
                 patch.object(EvalHarness, "_seed_documents", new=AsyncMock()), \
                 patch.object(EvalHarness, "run_task", new=AsyncMock(return_value=ok)):
                await harness.run_all([task])
        skills.delete_skill.assert_any_call(7)
        skills.delete_skill.assert_any_call(8)
        sql = skills._db.fetchall.call_args[0][0]
        assert "LIKE 'Eval:%'" in sql

    @pytest.mark.asyncio
    async def test_run_all_cleans_seeded_skills_on_cancellation(self):
        """Client disconnect (task cancellation) must not orphan 'Eval: *' skills."""
        skills = MagicMock()
        skills._db.fetchall.return_value = [{"id": 9}]
        harness = EvalHarness()
        task = EvalTask(id="r1", category="reasoning", query="2+2?",
                        assertions=[{"type": "answer_contains", "value": "4"}])
        with patch("app.core.brain.get_services", return_value=_svc(skills=skills)):
            with patch.object(EvalHarness, "_seed_skills"), \
                 patch.object(EvalHarness, "_seed_documents", new=AsyncMock()), \
                 patch.object(EvalHarness, "run_task",
                              new=AsyncMock(side_effect=asyncio.CancelledError())):
                with pytest.raises(asyncio.CancelledError):
                    await harness.run_all([task])
        skills.delete_skill.assert_any_call(9)
