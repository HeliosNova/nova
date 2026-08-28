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

import asyncio
import logging
import re
from collections import Counter

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


# Lesson-machinery boilerplate that must never become principle TEXT
# (2026-08-25): procedural-consolidation stubs carry conf 0.95 and the
# literal lesson_text "Procedural-consolidation: merged 3 lessons" — picked
# by confidence, that string became the ONLY principle ever minted (08-08)
# and its subject then blocked the cluster from re-minting forever.
_BOILERPLATE_TEXT_RE = re.compile(
    r"(?i)^\s*(?:procedural-consolidation\b|auto-merged\b|merged\s+\d+\s+lessons)")


def _is_principle_text(text: str) -> bool:
    return bool(text) and not _BOILERPLATE_TEXT_RE.search(text)


def _cluster_by_overlap(rows) -> list[tuple[frozenset, list]]:
    """Greedy keyword-overlap grouping: lessons sharing ≥2 substantive topic
    tokens join the same cluster.

    Replaces the alphabetical-first-2-keywords key (2026-08-27), which
    FRAGMENTED natural families: "Factual Art History Questions" keyed
    {art, factual}, "Historical Art Facts" keyed {art, facts}, "Famous Art
    History Questions" keyed {art, famous} — the same family scattered
    across distinct keys, so no cluster ever reached min_cluster=3 and
    Path B never minted a principle in the system's LIFETIME (verified
    against the live corpus: old keying max cluster = 2).

    Returns [(label_key, members)] where label_key is the 2 most common
    tokens across the cluster's member topics (a stable, meaningful label).
    """
    groups: list[list] = []  # [set_of_kws, [rows]]
    for r in rows:
        kws = set(_topic_keywords(r["topic"] or ""))
        if len(kws) < 2:
            continue
        for g in groups:
            if len(g[0] & kws) >= 2:
                g[1].append(r)
                g[0] |= kws
                break
        else:
            groups.append([kws, [r]])
    out: list[tuple[frozenset, list]] = []
    for gkws, members in groups:
        token_counts = Counter(
            t for m in members for t in _topic_keywords(m["topic"] or ""))
        label = frozenset(t for t, _n in token_counts.most_common(2)) or frozenset(gkws)
        out.append((label, members))
    return out


async def distill_principles(db, kg, *, min_helpful: int = 5, min_cluster: int = 3) -> int:
    """Find clusters of agreeing lessons and write each as a principle KG fact.

    Returns the count of principles distilled this run.
    """
    distilled = 0

    # --- Path A: very-high-helpful single lessons → promote directly ---
    rows = await asyncio.to_thread(
        db.fetchall,
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
        if not topic or not _is_principle_text(text):
            continue
        # Skip if already a principle for this topic
        existing = await asyncio.to_thread(
            db.fetchone,
            "SELECT id FROM kg_facts WHERE source='principle' AND subject=? LIMIT 1",
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
    candidate_rows = await asyncio.to_thread(
        db.fetchall,
        "SELECT id, topic, lesson_text, confidence, times_helpful "
        "FROM lessons WHERE confidence >= 0.6 AND times_helpful >= 2 "
        "ORDER BY times_helpful DESC LIMIT 200"
    )
    clusters = _cluster_by_overlap(candidate_rows)

    for key, members in clusters:
        if len(members) < min_cluster:
            continue
        # Compose a principle statement: pick the highest-confidence member
        # whose text is a REAL lesson (consolidation stubs outrank real
        # lessons on confidence but their text is machinery boilerplate).
        members.sort(key=lambda m: (m["confidence"], m["times_helpful"]), reverse=True)
        best = next((m for m in members
                     if _is_principle_text((m["lesson_text"] or ""))), None)
        if best is None:
            continue
        topic_label = " + ".join(sorted(key))
        text = (best["lesson_text"] or "")[:200]
        # Dedupe
        existing = await asyncio.to_thread(
            db.fetchone,
            "SELECT id FROM kg_facts WHERE source='principle' AND subject=? LIMIT 1",
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
