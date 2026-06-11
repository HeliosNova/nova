"""Principle distillation — surface load-bearing facts from clusters of lessons.

A "principle" is a high-confidence claim that:
  1. Multiple lessons (3+) agree on, OR
  2. Has been retrieved many times AND helped many times, OR
  3. Was promoted from a sustained pattern of success reflexions

Principles get written as KG facts with `provenance='principle'` so they
survive lesson decay and pruning. They become the load-bearing core of
Nova's beliefs — the things he should never have to re-derive.

This module is called from the daily maintenance cycle.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "for", "and", "or", "but", "with", "by", "as", "from", "this", "that",
    "what", "which", "when", "where", "how", "you", "your", "i", "me", "we",
})


def _topic_keywords(topic: str) -> frozenset[str]:
    if not topic:
        return frozenset()
    words = re.findall(r"\b[a-z][a-z0-9]{2,}\b", topic.lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


async def distill_principles(db, kg, *, min_helpful: int = 5, min_cluster: int = 3) -> int:
    """Find clusters of agreeing lessons and write each as a principle KG fact.

    Returns the count of principles distilled this run.
    """
    distilled = 0

    # --- Path A: very-high-helpful single lessons → promote directly ---
    rows = db.fetchall(
        "SELECT id, topic, lesson_text, confidence, times_helpful "
        "FROM lessons "
        "WHERE times_helpful >= ? AND confidence >= 0.85 "
        "AND lesson_text NOT LIKE '%Promoted from success reflexion%' "
        "ORDER BY times_helpful DESC LIMIT 20",
        (min_helpful * 2,),  # very high bar for solo promotion
    )
    for r in rows:
        topic = r["topic"] or ""
        text = (r["lesson_text"] or "")[:300]
        if not topic or not text:
            continue
        # Skip if already a principle for this topic
        existing = db.fetchone(
            "SELECT id FROM kg_facts WHERE provenance='principle' AND subject=? LIMIT 1",
            (topic[:200],),
        )
        if existing:
            continue
        try:
            ok = await kg.add_fact(
                subject=topic[:200],
                predicate="principle_says",
                object_=text[:200],
                confidence=min(0.95, r["confidence"]),
                source="principle",
                provenance=f"principle:lesson_{r['id']}",
            )
            if ok:
                distilled += 1
                logger.info("principle distilled (solo high-helpful): %s", topic[:80])
        except Exception as e:
            logger.warning("principle add failed: %s", e)

    # --- Path B: cluster lessons by topic-keyword overlap, distill consensus ---
    candidate_rows = db.fetchall(
        "SELECT id, topic, lesson_text, confidence, times_helpful "
        "FROM lessons WHERE confidence >= 0.6 AND times_helpful >= 2 "
        "ORDER BY times_helpful DESC LIMIT 200"
    )
    clusters: dict[frozenset, list[dict]] = defaultdict(list)
    for r in candidate_rows:
        kws = _topic_keywords(r["topic"] or "")
        if len(kws) >= 2:
            # Use top-2 substantive keywords as cluster key
            key = frozenset(sorted(kws)[:2])
            clusters[key].append(r)

    for key, members in clusters.items():
        if len(members) < min_cluster:
            continue
        # Compose a principle statement: pick the highest-confidence member's text
        members.sort(key=lambda m: (m["confidence"], m["times_helpful"]), reverse=True)
        best = members[0]
        topic_label = " + ".join(sorted(key))
        text = (best["lesson_text"] or "")[:200]
        if not text:
            continue
        # Dedupe
        existing = db.fetchone(
            "SELECT id FROM kg_facts WHERE provenance='principle' AND subject=? LIMIT 1",
            (topic_label[:200],),
        )
        if existing:
            continue
        try:
            ok = await kg.add_fact(
                subject=topic_label[:200],
                predicate="principle_consensus",
                object_=text,
                confidence=0.9,
                source="principle",
                provenance=f"principle:cluster:{len(members)}",
            )
            if ok:
                distilled += 1
                logger.info(
                    "principle distilled (cluster of %d): %s",
                    len(members), topic_label[:80],
                )
        except Exception as e:
            logger.warning("principle add (cluster) failed: %s", e)

    return distilled
