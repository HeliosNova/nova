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
        # The skills cleanup that used to sit beside it went with the skill
        # layer (2026-10-10); the document cleanup must still be in `finally`.
        i = HARNESS.index("Seeded eval documents must never outlive the run")
        before = HARNESS[max(0, i - 200):i]
        after = HARNESS[i:i + 400]
        assert "finally:" in before, "document cleanup must run from a `finally` block"
        assert "self._cleanup_seeded_documents()" in after, (
            "document cleanup must sit in `finally` so cancellation/crash paths are covered too"
        )
