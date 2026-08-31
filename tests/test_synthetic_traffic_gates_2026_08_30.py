"""Synthetic traffic must not feed the self-improvement collectors.

Measured 2026-08-30 (fresh-eyes sweep, after the owner pushed past the task
list): ephemeral eval-harness runs were writing into every persistent
self-improvement input the ephemeral flag never reached —

  * auto_tool_candidates: 22 of 100 rows were eval-fixture queries ("In the
    Nova project, what does the codename 'ZQX' refer to?", the fictional
    Aurora-7 fusion plant, Skylance X9), and SEVERAL had triggered=1 — eval
    probes drove real LLM tool-writing. custom_tools #3 (28 uses) traces to
    this class.
  * capability_gaps: newest rows were quiz-family questions, later mined by
    goal derivation — the 8 historical goals (ALL failed) chased exactly such
    phantoms ("Re-research: Factual Art History Questions").
  * reflexion get_relevant: the store held 114 is_eval=1 failures vs 15 real
    ones, and real-chat failure injection drew from that 88%-synthetic pool
    with no filter.

think(ephemeral=True) documents "keeps eval traffic out of every persistent
store". These are the remaining side channels after web_search's curiosity
mint was closed earlier tonight.
"""

from __future__ import annotations

import pathlib

import pytest

from app.tools.base import EPHEMERAL_REQUEST

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def reset_ephemeral():
    token = EPHEMERAL_REQUEST.set(False)
    yield
    EPHEMERAL_REQUEST.reset(token)


class TestToolCandidateStore:
    def test_ephemeral_record_writes_nothing(self, db):
        from app.core.tool_triggers import ToolCandidateStore
        store = ToolCandidateStore(db=db)
        before = db.fetchone("SELECT COUNT(*) c FROM auto_tool_candidates")["c"]
        EPHEMERAL_REQUEST.set(True)
        rid = store.record("What does codename ZQX refer to?",
                           ["memory_search", "web_search"])
        assert rid == 0
        after = db.fetchone("SELECT COUNT(*) c FROM auto_tool_candidates")["c"]
        assert after == before, "ephemeral traffic minted a tool candidate"

    def test_real_record_still_writes(self, db):
        from app.core.tool_triggers import ToolCandidateStore
        store = ToolCandidateStore(db=db)
        rid = store.record("compare GPU prices across vendors",
                           ["web_search", "calculator"])
        assert rid > 0
        row = db.fetchone("SELECT query FROM auto_tool_candidates WHERE id=?",
                          (rid,))
        assert row is not None


class TestReflexionInjectionScoping:
    def _seed(self, db):
        from app.core.reflexion import ReflexionStore
        store = ReflexionStore(db=db)
        store.store(task_summary="lookup the Vorenza headquarters location",
                    outcome="failure", reflection="fictional eval probe",
                    quality_score=0.2, tools_used="web_search",
                    is_eval=True)
        store.store(task_summary="lookup the Anthropic headquarters location",
                    outcome="failure", reflection="real failure",
                    quality_score=0.3, tools_used="web_search",
                    is_eval=False)
        return store

    def test_real_chat_excludes_eval_failures(self, db):
        store = self._seed(db)
        got = store.get_relevant("lookup the headquarters location", limit=5)
        summaries = [r.task_summary for r in got]
        assert any("Anthropic" in s for s in summaries), \
            "the real failure must still inject"
        assert not any("Vorenza" in s for s in summaries), (
            "real chat injected a fictional eval probe as a 'past failure' — "
            "the 88%-synthetic-pool leak"
        )

    def test_ephemeral_still_sees_eval_rows(self, db):
        """The eval harness's memory-learning tasks seed failures and MUST
        retrieve them — that is why is_eval rows are retrievable at all.
        Scoping must not break the harness."""
        store = self._seed(db)
        EPHEMERAL_REQUEST.set(True)
        got = store.get_relevant("lookup the headquarters location", limit=5)
        assert any("Vorenza" in r.task_summary for r in got), \
            "ephemeral (eval) requests must keep seeing seeded eval rows"


class TestGatePresenceAtRemainingSites:
    """agent_loop's record-keeping and brain's no-skill gap writer run deep in
    machinery a unit test can't cheaply drive; assert the wiring exists so a
    refactor can't silently drop it."""

    def test_agent_loop_record_keeping_is_gated(self):
        src = (REPO / "app" / "core" / "agent_loop.py").read_text(encoding="utf-8")
        i = src.index("INSERT INTO auto_tool_candidates")
        guard = src.rindex("EPHEMERAL_REQUEST.get()", 0, i)
        assert i - guard < 1500, (
            "the ephemeral gate must sit directly above agent_loop's collector "
            "inserts (auto_tool_candidates / capability_gaps / curiosity)"
        )

    def test_brain_capability_gap_is_gated(self):
        src = (REPO / "app" / "core" / "brain.py").read_text(encoding="utf-8")
        i = src.index("INSERT INTO capability_gaps")
        guard = src.rindex("EPHEMERAL_REQUEST.get()", 0, i)
        assert i - guard < 900, (
            "brain's no-skill capability-gap writer must check the ephemeral "
            "flag before inserting"
        )
