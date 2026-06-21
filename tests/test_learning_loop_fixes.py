"""Learning-loop killer fixes (audit 2026-06-12).

1. detect_correction LLM-rescue: a strict-regex MISS with a soft reactive
   signal + a prior answer still gets the LLM confirmation (recovers the ~12%
   of creatively-phrased corrections the regex can't match).
2. response_pushes_back: a clear concession ("you're right", "my mistake")
   vetoes pushback, so genuinely-ACCEPTED corrections are no longer suppressed.
3. _find_similar_lesson: contradictory answers (polarity flip / different
   numbers) are never merged as duplicates.
4. KG get_relevant_facts: a strong (tightly distance-gated) vector hit can
   drive retrieval with no keyword/PPR support.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core.learning import (
    Correction,
    LearningEngine,
    _answers_conflict,
    has_soft_correction_signal,
    is_likely_correction,
    response_concedes,
    response_pushes_back,
)


# ---------------------------------------------------------------------------
# 1. Correction LLM-rescue on a strict-regex miss
# ---------------------------------------------------------------------------

class TestCorrectionRescue:
    @pytest.mark.asyncio
    async def test_regex_miss_with_soft_signal_runs_llm(self, db):
        eng = LearningEngine(db)
        # A correction the strict patterns miss but that has a soft signal.
        msg = "Pretty sure the launch slipped to Q3, not Q2."
        assert is_likely_correction(msg) is False          # strict miss
        assert has_soft_correction_signal(msg) is True      # soft hit

        payload = (
            '{"is_correction": true, "topic": "launch date", '
            '"wrong_answer": "launch is Q2", "correct_answer": "launch is Q3", '
            '"lesson_text": "Launch slipped to Q3, not Q2"}'
        )
        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value=payload)
            mock_llm.extract_json_object = lambda x: __import__("json").loads(x)
            corr = await eng.detect_correction(msg, previous_answer="The launch is in Q2.")
        assert corr is not None
        assert "Q3" in (corr.correct_answer or corr.lesson_text)

    @pytest.mark.asyncio
    async def test_regex_miss_without_prior_answer_skips_llm(self, db):
        eng = LearningEngine(db)
        msg = "Pretty sure the launch slipped to Q3, not Q2."  # strict miss, soft hit
        assert is_likely_correction(msg) is False
        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"is_correction": true}')
            mock_llm.extract_json_object = lambda x: {"is_correction": True}
            corr = await eng.detect_correction(msg, previous_answer="")
        assert corr is None
        mock_llm.invoke_nothink.assert_not_called()  # no prior answer => no spend

    @pytest.mark.asyncio
    async def test_no_signal_long_message_skips_llm(self, db):
        eng = LearningEngine(db)
        msg = "Please write a detailed essay about the history of the Roman aqueducts " * 6
        with patch("app.core.learning.llm") as mock_llm:
            mock_llm.invoke_nothink = AsyncMock(return_value='{"is_correction": false}')
            mock_llm.extract_json_object = lambda x: {"is_correction": False}
            corr = await eng.detect_correction(msg, previous_answer="Some prior answer.")
        assert corr is None
        mock_llm.invoke_nothink.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Pushback concession veto
# ---------------------------------------------------------------------------

class TestPushbackConcession:
    def test_accepting_reply_is_not_pushback(self):
        # Trips the broad `actually, the` pushback pattern but ACCEPTS — must not
        # be treated as pushback (this is the suppression bug).
        reply = "Actually, the correct answer is Canberra — thanks for the correction!"
        assert response_concedes(reply) is True
        assert response_pushes_back(reply) is False

    def test_my_mistake_is_not_pushback(self):
        reply = "You're right, my mistake. The capital is Canberra, not Sydney."
        assert response_pushes_back(reply) is False

    def test_genuine_pushback_still_detected(self):
        reply = "I must stand by my original answer — according to my search, it is correct."
        assert response_concedes(reply) is False
        assert response_pushes_back(reply) is True


# ---------------------------------------------------------------------------
# 3. Dedup never merges contradictory answers
# ---------------------------------------------------------------------------

class TestDedupConflictGuard:
    def test_polarity_flip_conflicts(self):
        assert _answers_conflict("Pluto is a planet", "Pluto is not a planet") is True

    def test_numeric_mismatch_conflicts(self):
        assert _answers_conflict("The price is $50,000", "The price is $60,000") is True

    def test_same_polarity_no_conflict(self):
        assert _answers_conflict("Canberra is the capital", "The capital is Canberra") is False

    @pytest.mark.asyncio
    async def test_opposite_lessons_not_merged(self, db):
        eng = LearningEngine(db)
        a = Correction(user_message="no", previous_answer="", topic="Pluto status",
                       wrong_answer="Pluto is a full planet",
                       correct_answer="Pluto is a dwarf planet", lesson_text="Pluto is a dwarf planet")
        b = Correction(user_message="no", previous_answer="", topic="Pluto status",
                       wrong_answer="Pluto is a dwarf planet",
                       correct_answer="Pluto is not a planet", lesson_text="Pluto is not a planet")
        id_a = await __import__("asyncio").to_thread(eng.save_lesson, a)
        id_b = await __import__("asyncio").to_thread(eng.save_lesson, b)
        # Contradictory answers on the same topic must be stored separately.
        assert id_a != id_b
        rows = db.fetchall("SELECT correct_answer FROM lessons WHERE topic='Pluto status'")
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# 4. KG strong vector hit can stand alone (no keyword/PPR support)
# ---------------------------------------------------------------------------

class TestKGVectorStandalone:
    @pytest.mark.asyncio
    async def test_strong_vector_only_retrieves(self, db, monkeypatch):
        from app.core.kg import KnowledgeGraph

        kg = KnowledgeGraph(db)
        await kg.add_fact("Zorbon-9", "produces", "flux capacitors", source="user")
        target = db.fetchone("SELECT id FROM kg_facts WHERE subject='Zorbon-9'")
        tid = str(target["id"])

        # A query with NO token overlap and no graph seed -> keyword/PPR empty.
        # Fake a strong-distance vector hit so only the vector arm fires.
        class _FakeCollection:
            def count(self):
                return 1
            def query(self, query_texts, n_results, include):
                return {"ids": [[tid]], "distances": [[0.20]]}  # well inside strong gate

        monkeypatch.setattr(kg, "_get_collection", lambda: _FakeCollection())
        facts = kg.get_relevant_facts("what does the alien factory make", limit=5)
        assert any(f.subject == "Zorbon-9" for f in facts), \
            "strong vector-only hit should retrieve when keyword/PPR miss"

    @pytest.mark.asyncio
    async def test_weak_vector_only_does_not_retrieve(self, db, monkeypatch):
        from app.core.kg import KnowledgeGraph

        kg = KnowledgeGraph(db)
        await kg.add_fact("Zorbon-9", "produces", "flux capacitors", source="user")
        tid = str(db.fetchone("SELECT id FROM kg_facts WHERE subject='Zorbon-9'")["id"])

        class _FakeCollection:
            def count(self):
                return 1
            def query(self, query_texts, n_results, include):
                return {"ids": [[tid]], "distances": [[0.75]]}  # weak: past the strong gate

        monkeypatch.setattr(kg, "_get_collection", lambda: _FakeCollection())
        facts = kg.get_relevant_facts("what does the alien factory make", limit=5)
        assert not any(f.subject == "Zorbon-9" for f in facts), \
            "weak vector-only hit must NOT retrieve (too noisy without corroboration)"
