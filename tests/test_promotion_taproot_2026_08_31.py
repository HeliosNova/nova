"""The REM promotion taproot — the actual origin of the lesson-churn families.

Archaeology (2026-08-31, live store): reflexion #7 ("who painted the Mona
Lisa?", 2026-06-20, q=0.95) was promoted into the first art-history lesson on
JUNE 20 — and promoted AGAIN on August 31 at 00:59, hours after a manual
family merge deleted 17 near-identical descendants. Two defects compounded:

  1. dream's gather comment said "High-quality reflexions never promoted to
     lessons" but the query never filtered for it — no consumed-mark existed,
     so the same top reflexions re-qualified every cycle.
  2. the promoted lesson's lesson_text was provenance boilerplate ("Promoted
     from success reflexion (quality=0.95)"), the provenance-as-content class
     — which ALSO blinded lesson dedup: every promoted lesson's text was
     near-identical boilerplate, so family duplicates scored ~0 jaccard
     against real members (dedup_decisions logged 0.0 for the re-mints).

Every earlier fix (quiz guard, family merges, curiosity drain) removed
DESCENDANTS; this pair of defects was the mint. Fix under test: a
promoted_at consumed-mark (schema + gather filter + consumer update) and
lesson_text that carries the lesson itself.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
DREAM = (REPO / "app" / "core" / "dream.py").read_text(encoding="utf-8")


class TestSchemaHasConsumedMark:
    def test_reflexions_gain_promoted_at(self, db):
        from app.core.reflexion import ReflexionStore
        ReflexionStore(db=db)  # ensure-columns runs in ctor
        cols = [c[1] for c in db.execute("PRAGMA table_info(reflexions)").fetchall()] \
            if hasattr(db.execute("PRAGMA table_info(reflexions)"), "fetchall") else None
        rows = db.fetchall("PRAGMA table_info(reflexions)")
        names = {r["name"] for r in rows}
        assert "promoted_at" in names, (
            "without a consumed-mark the same reflexion re-promotes every "
            "dream cycle — #7 re-minted the art family from June to August"
        )


class TestGatherEnforcesUnpromoted:
    def test_query_filters_promoted_at(self):
        m = re.search(
            r"high_quality_unpromoted\s*=", DREAM)
        assert m, "gather signal renamed — update this test"
        window = DREAM[max(0, m.start() - 900):m.start()]
        assert "promoted_at IS NULL" in window, (
            'the gather query must ENFORCE "never promoted", not just say it '
            "in a comment"
        )


class TestConsumerMarksAndWritesContent:
    def test_promotion_marks_source_consumed(self):
        i = DREAM.index("def _promote_reflexions")
        body = DREAM[i:i + 4000]
        assert "SET promoted_at" in body, (
            "promotion must stamp its source — dedup alone cannot stop the "
            "re-mint because boilerplate lesson_text blinded it"
        )

    def test_lesson_text_is_not_provenance_boilerplate(self):
        i = DREAM.index("def _promote_reflexions")
        body = DREAM[i:i + 4000]
        assert 'lesson_text=f"Promoted from success reflexion' not in body, (
            "provenance-as-content: identical boilerplate text made every "
            "promoted lesson invisible to dedup and useless to retrieval"
        )
        # the actual lesson content must flow into lesson_text
        assert re.search(r"lesson_text=\(\s*f\"\{lesson\}", body), (
            "lesson_text should carry the lesson itself"
        )
