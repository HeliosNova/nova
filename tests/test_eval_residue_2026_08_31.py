"""Eval fixtures must not persist in production stores after a run.

Measured 2026-08-31 (fresh-eyes pass 5):
  * The document store's ONLY content was eval fixtures — 3 of 3 documents
    were source='eval-seed', re-seeded by every nightly run and never
    removed. Real-chat document retrieval searched over nothing but test
    material. The 2026-08-12 stable-id fix stopped ACCUMULATION (46 dups at
    worst) but nothing ever cleaned up; the deliberate skills-cleanup
    pattern (_cleanup_seeded_skills, in `finally`) was never extended to
    documents.
  * An ORPHAN chunk (parent document row long deleted) was actively serving
    an injection CONTENT-WARNING banner into live knowledge_search results
    (observed 03:55 and 05:17 the same day).
  * agent_workspace held scratchpads for the FICTIONAL Aurora-7 plant and a
    "codename prometheus-9" probe — workspaces persist across restarts and
    prime future runs of "the same" query.
"""

from __future__ import annotations

import pathlib

import pytest

from app.tools.base import EPHEMERAL_REQUEST

REPO = pathlib.Path(__file__).resolve().parent.parent
HARNESS = (REPO / "app" / "monitors" / "eval_harness.py").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def reset_ephemeral():
    token = EPHEMERAL_REQUEST.set(False)
    yield
    EPHEMERAL_REQUEST.reset(token)


class TestHarnessCleansSeededDocuments:
    def test_cleanup_method_exists_and_uses_retriever_delete(self):
        i = HARNESS.index("def _cleanup_seeded_documents")
        body = HARNESS[i:i + 1800]
        assert "delete_document" in body, (
            "cleanup must go through retriever.delete_document — a raw row "
            "delete leaves orphan chunks that knowledge_search still serves "
            "(one was observed injecting a CONTENT-WARNING banner live)"
        )
        assert "eval-seed-" in body

    def test_cleanup_runs_in_the_finally_block(self):
        # 2026-08-31: the skills cleanup moved off the event loop
        # (asyncio.to_thread(self._cleanup_seeded_skills)) — match without
        # the call parens so the wrapper doesn't break the invariant check.
        i = HARNESS.index("self._cleanup_seeded_skills")
        after = HARNESS[i:i + 200]
        assert "self._cleanup_seeded_documents()" in after, (
            "document cleanup must sit beside the skills cleanup in `finally` "
            "so cancellation/crash paths are covered too"
        )


class TestWorkspaceGate:
    def test_ephemeral_save_writes_nothing(self, db):
        from app.core.agent_workspace import save_workspace

        db.execute(
            "CREATE TABLE IF NOT EXISTS agent_workspace ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, query_signature TEXT, "
            "last_query TEXT, findings_json TEXT, last_answer TEXT, "
            "run_count INTEGER DEFAULT 0, success_count INTEGER DEFAULT 0, "
            "fail_count INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT)")
        EPHEMERAL_REQUEST.set(True)
        rid = save_workspace(
            db, query="What do you know about the Aurora-7 fusion pilot plant?",
            findings={"step1": "fiction"}, answer="a confident hallucination",
            success=True)
        assert rid is None
        n = db.fetchone("SELECT COUNT(*) c FROM agent_workspace")["c"]
        assert n == 0, "ephemeral traffic persisted a scratchpad"

    def test_real_save_still_writes(self, db):
        from app.core.agent_workspace import save_workspace

        db.execute(
            "CREATE TABLE IF NOT EXISTS agent_workspace ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, query_signature TEXT, "
            "last_query TEXT, findings_json TEXT, last_answer TEXT, "
            "run_count INTEGER DEFAULT 0, success_count INTEGER DEFAULT 0, "
            "fail_count INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT)")
        rid = save_workspace(
            db, query="compare GPU prices across the major cloud vendors",
            findings={"step1": "real work"}, answer="a real answer",
            success=True)
        assert rid is not None
        n = db.fetchone("SELECT COUNT(*) c FROM agent_workspace")["c"]
        assert n == 1
