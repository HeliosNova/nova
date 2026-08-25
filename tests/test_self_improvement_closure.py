"""Self-improvement loop CLOSURE (2026-08-25 audit).

The 08-23 revival made every loop RUN; none CONVERGED:
- the failure-sweep re-minted near-duplicate lessons daily (#376/#377
  were paraphrases of #374/#375, identical topics, one day apart);
- eval fixtures polluted the curiosity queue (fictional Dr. Ferrand
  burned ≥6 GPU research cycles) and one "provisional" resolution stored
  a tool-session summary as the answer;
- principles had minted ZERO ever — an 08-08 junk fact blocks its
  cluster forever and Path B selects consolidation boilerplate as the
  principle text by confidence;
- skill induction minted semantic twins ('answer_factual_questions' vs
  'factual_question_answering') because the coverage check compares raw
  substrings, and skill matching ran on the '=== System Context ==='
  monitor preamble (constant-0.737 wrong matches).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class TestFailureSweepConvergence:
    def test_repromotion_reinforces_recent_same_topic_lesson(self, db, monkeypatch):
        from app.core import llm as llm_mod
        from app.core.learning import LearningEngine
        from app.core.reflexion import check_recurring_failures

        engine = LearningEngine(db=db)
        first_id = engine.add_knowledge_lesson(
            topic="Art History Fact Verification",
            correct_answer="Always verify historical claims via cross-reference.",
            lesson_text="Auto-lesson: Always verify historical claims.",
        )

        async def _fake_invoke(messages, **kwargs):
            return ('{"topic": "Art History Fact Verification", '
                    '"lesson": "Verify historical claims against sources."}')

        monkeypatch.setattr(llm_mod, "invoke_nothink", _fake_invoke)

        store = SimpleNamespace(find_recurring_failures=lambda _q: [
            SimpleNamespace(id=i, reflection=f"failed art quiz {i}",
                            task_summary="art history quiz")
            for i in range(1, 4)
        ])
        before = db.fetchone("SELECT COUNT(*) AS c FROM lessons")["c"]
        asyncio.run(check_recurring_failures(
            "art history quiz", engine, store=store))
        after = db.fetchone("SELECT COUNT(*) AS c FROM lessons")["c"]

        assert after == before, "re-promotion must reinforce, not re-mint"
        row = db.fetchone("SELECT times_helpful FROM lessons WHERE id = ?",
                          (first_id,))
        assert row["times_helpful"] >= 1


class TestCuriosityEphemeralGate:
    def test_add_suppressed_when_ephemeral_flag_set(self, db):
        from app.core import curiosity as cur

        q = cur.CuriosityQueue(db)
        token = cur.set_suppress_organic_minting(True)
        try:
            q.add(topic="Dr. Ferrand photonics group Lyon",
                  source="search_zero_result", urgency=0.4)
        finally:
            cur.reset_suppress_organic_minting(token)
        row = db.fetchone(
            "SELECT COUNT(*) AS c FROM curiosity_queue WHERE topic LIKE '%Ferrand%'")
        assert row["c"] == 0

    def test_add_works_when_flag_clear(self, db):
        from app.core import curiosity as cur

        q = cur.CuriosityQueue(db)
        q.add(topic="real organic curiosity topic",
              source="search_zero_result", urgency=0.4)
        row = db.fetchone(
            "SELECT COUNT(*) AS c FROM curiosity_queue WHERE topic LIKE '%organic%'")
        assert row["c"] == 1


class TestProvisionalResolutionSanity:
    def test_session_summary_rejected(self):
        from app.monitors.heartbeat_loop import _provisional_acceptable

        garbage = ("Based on the tools executed in this session, here is a "
                   "summary of your recent activities: ### ✅ Completed")
        assert not _provisional_acceptable(
            "Can Broadcom absorb Lumentum's photonics supply?", garbage)

    def test_hedged_topical_answer_accepted(self):
        from app.monitors.heartbeat_loop import _provisional_acceptable

        answer = ("Broadcom's capacity to absorb Lumentum's photonics supply "
                  "depends on packaging throughput; analysts are split, and "
                  "the integration timeline suggests 2027 at the earliest.")
        assert _provisional_acceptable(
            "Can Broadcom absorb Lumentum's photonics supply?", answer)


class TestPrincipleTextSelection:
    def test_consolidation_boilerplate_never_becomes_principle_text(self, db):
        from app.core.principles import distill_principles

        # Three clustered lessons (identical top-2 topic keywords); the
        # highest-confidence one is a consolidation stub whose lesson_text
        # is pure boilerplate.
        rows = [
            ("Quantum Claims", "Procedural-consolidation: merged 3 lessons", 0.95, 12),
            ("Quantum Claims Review", "Check qubit coherence times before claims.", 0.90, 8),
            ("Quantum Claims Verify", "Verify vendor qubit counts independently.", 0.85, 5),
        ]
        for topic, text, conf, helpful in rows:
            db.execute(
                "INSERT INTO lessons (topic, correct_answer, lesson_text, "
                "confidence, times_helpful) VALUES (?, ?, ?, ?, ?)",
                (topic, "answer", text, conf, helpful),
            )

        added: list[dict] = []

        class _KG:
            async def add_fact(self, **kwargs):
                added.append(kwargs)
                return True

        asyncio.run(distill_principles(db, _KG()))
        cluster_facts = [a for a in added
                         if a.get("predicate") == "principle_consensus"]
        assert cluster_facts, "cluster should distill a principle"
        assert "Procedural-consolidation" not in cluster_facts[0]["object_"]
        # Path A (solo high-helpful) must skip boilerplate too — the stub
        # clears its bars (helpful 12, conf 0.95) but is not a principle.
        assert not any("Procedural-consolidation" in a.get("object_", "")
                       for a in added)


class TestSkillInductionCoverage:
    def test_stemmed_cluster_coverage_catches_twins(self):
        from app.core.auto_skills import _cluster_covered

        existing = "answer_factual_questions evaluate_cpu_performance"
        assert _cluster_covered(frozenset({"answering", "factual"}), existing)
        assert _cluster_covered(frozenset({"question", "factual"}), existing)
        assert not _cluster_covered(
            frozenset({"quantum", "entanglement"}), existing)


class TestSkillMatchPreambleStrip:
    def test_match_runs_on_real_prompt_not_context_block(self, db):
        from app.core.skills import SkillStore

        store = SkillStore(db)
        store.create_skill(
            name="cpu_bench",
            trigger_pattern=r"cpu performance",
            steps=[],
        )
        store.create_skill(
            name="boilerplate_trap",
            trigger_pattern=r"TODAY IS",
            steps=[],
        )
        enriched = ("=== System Context ===\nTODAY IS: Monday, August 25, 2026 "
                    "(UTC). All secondary lines.\n=== End Context ===\n\n"
                    "Evaluate the cpu performance of the new node.")
        hit = store.get_matching_skill(enriched)
        assert hit is not None and hit.name == "cpu_bench"
