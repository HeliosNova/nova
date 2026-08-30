"""times_helpful must never exceed times_retrieved.

A lesson cannot have helped more often than it was retrieved. A DB invariant
check found two rows violating this — both in the art-history churn family, i.e.
the lessons that recur most were inflating their own rank.

Cause: reflexion's recurring-failure path REINFORCED an existing lesson by
bumping times_helpful, but that path never retrieves the lesson. It used a
retrieval-quality metric to carry a dedup signal.

Why it matters beyond tidiness:
  - get_relevant_lessons orders its candidate pool by `times_helpful DESC`
  - mark_lesson_helpful scales its confidence boost by 1/(1+times_helpful)
so a padded counter both promotes the lesson AND shrinks its future real boosts.
"""

from __future__ import annotations

import inspect
import re

from app.core import reflexion


def test_reflexion_reinforcement_does_not_touch_times_helpful():
    """The reinforce-instead-of-duplicate path must adjust confidence only."""
    src = inspect.getsource(reflexion)
    # locate the reinforcement UPDATE
    m = re.search(r'"UPDATE lessons SET[^"]*"\s*\n?\s*"?[^"]*"?\s*WHERE id = \?',
                  src)
    block = src[src.index("REINFORCED") - 900: src.index("REINFORCED")] \
        if "REINFORCED" in src else src
    assert "times_helpful = times_helpful + 1" not in block, (
        "reflexion's recurring-failure path still increments times_helpful; "
        "that path does not retrieve the lesson, so it corrupts a "
        "retrieval-quality metric with a dedup signal"
    )


def test_only_the_retrieval_path_marks_helpful():
    """learning.mark_lesson_helpful stays the single writer of times_helpful."""
    from app.core import learning
    lsrc = inspect.getsource(learning)
    assert lsrc.count("times_helpful = times_helpful + 1") == 1, (
        "times_helpful should have exactly one incrementing writer "
        "(mark_lesson_helpful)"
    )


def test_invariant_is_expressible(db):
    """Guard the invariant itself so a future writer is caught by data, not luck."""
    from app.core.learning import LearningEngine
    eng = LearningEngine(db)
    lid = eng.add_knowledge_lesson(
        topic="counter invariant probe",
        correct_answer="A lesson cannot help more often than it is retrieved.",
        lesson_text="invariant probe",
        context="test",
        confidence=0.9,
    )
    assert lid and lid > 0
    row = db.fetchone(
        "SELECT COALESCE(times_retrieved,0) r, COALESCE(times_helpful,0) h "
        "FROM lessons WHERE id = ?", (lid,))
    assert row["h"] <= row["r"] or row["h"] == 0, (
        "a freshly created lesson must not start with helpful > retrieved"
    )
