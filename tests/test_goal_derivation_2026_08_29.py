"""The goals loop was dead: 8 goals ever, 7 failed, nothing new in 12 days.

Two independent defects, both found by measuring the live store rather than
reading the code:

1. ORDERING. _execute_capability_review marked every gap `reviewed = 1` and
   THEN called derive_goals(), which selects `WHERE reviewed = 0`. The review
   consumed its own input, so the capability_gap source could never mint a goal.
   Live evidence: 18 gaps, ALL reviewed=1, 9 created in the last 7 days, 0
   unreviewed — and the only capability-gap goals dated 2026-06-20.

2. CLUSTER KEY BY POSITION. goal_deriver sliced `words[:3]` and filtered stop
   words afterwards, so the window was spent on whatever opened the query.
   Research queries all open with an imperative, which is why the store holds
   "recurring capability gap: does", "…: higher", "…: clock" — three goals from
   one question about CPU clock speed, all failed.
"""

from __future__ import annotations

import inspect
import re

from app.core import goal_deriver
from app.monitors import heartbeat_loop


class TestCapabilityReviewOrdering:
    def test_derive_runs_before_gaps_are_marked_reviewed(self):
        """derive_goals() must be called BEFORE the reviewed=1 update."""
        src = inspect.getsource(heartbeat_loop)
        derive = src.find("derive_goals(db, max_new_goals=3)")
        mark = src.find("SET reviewed = 1")
        assert derive != -1, "derive_goals call not found"
        assert mark != -1, "reviewed=1 update not found"
        assert derive < mark, (
            "capability review marks gaps reviewed BEFORE deriving goals; "
            "derive_goals selects WHERE reviewed = 0, so its input is emptied "
            "first and the capability_gap source can never mint a goal"
        )


class TestGapClusterKeying:
    """The cluster key must come from content, not from position."""

    @staticmethod
    def _cluster(queries):
        from collections import Counter
        clusters: Counter = Counter()
        for q in queries:
            words = [
                w for w in re.findall(r"\b[a-z][a-z0-9_-]{3,}\b", q.lower())
                if w not in goal_deriver.STOP_WORDS
                and w not in goal_deriver._GOAL_KEYWORD_JUNK
            ]
            for w in set(words[:8]):
                clusters[w] += 1
        return clusters

    def test_imperative_openers_do_not_become_the_key(self):
        """Three CPU questions must cluster on CPU terms, not on 'does'/'explain'."""
        qs = [
            "Why does a higher CPU clock speed not always improve performance?",
            "Explain the relationship between CPU clock speed and performance factors",
            "Detail how clock speed interacts with IPC and cache architecture",
        ]
        c = self._cluster(qs)
        top = [kw for kw, _ in c.most_common(4)]
        for bad in ("does", "explain", "detail", "identify"):
            assert bad not in top, f"cluster keyed on the imperative opener {bad!r}"
        assert any(k in c for k in ("clock", "speed", "performance")), (
            f"no substantive CPU term in the cluster keys: {top}"
        )

    def test_one_query_cannot_stack_its_own_counter(self):
        """set() per query — a verbose query must contribute each term once."""
        c = self._cluster(["throughput throughput throughput throughput limiter"])
        assert c["throughput"] == 1, "a single query inflated its own key"

    def test_source_filters_before_slicing(self):
        """Guard the actual implementation, not just this local copy."""
        src = inspect.getsource(goal_deriver)
        assert "for w in set(words[:8])" in src, (
            "expected filter-then-slice keying in goal_deriver"
        )
        assert "for w in words[:3]" not in src, (
            "positional words[:3] keying is back — it clusters on imperatives"
        )
