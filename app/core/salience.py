"""Personalized salience (Monitor Intelligence v2, Phase C).

Scores monitor output by how much it likely matters to the OWNER, so the digest
leads with signal and drops noise — instead of posting every domain's everything.

Signal sources (all cheap, no LLM, off the GPU):
  - owner interest : overlap with topics the owner actually queries (topic_frequency,
                     the TopicTracker substrate) — what they keep asking about.
  - corroboration  : how many outlets confirmed the story (digest "N outlets").
  - learned weight : salience_weights, nudged by 👍/👎 ratings over time.

Learning closes the loop the rating button already opens: a 👍 raises that topic's
weight, 👎 lowers it. Bounded; degrades to a neutral score on any error.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]{3,}")
_STOP = frozenset({
    "this", "that", "with", "from", "have", "will", "their", "about", "which",
    "there", "these", "those", "than", "them", "they", "been", "were", "would",
    "could", "after", "amid", "over", "into", "more", "most", "also", "such",
    "monitor", "update", "report", "today", "news", "outlets", "confirmed", "source",
})


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP}


def _owner_topics(db) -> dict[str, int]:
    """Topics the owner queries, weighted by frequency (TopicTracker substrate)."""
    try:
        rows = db.fetchall(
            "SELECT topic, query_count FROM topic_frequency "
            "WHERE last_seen > datetime('now', '-45 days') ORDER BY query_count DESC LIMIT 60"
        )
    except Exception:
        return {}
    out: dict[str, int] = {}
    for r in rows:
        for tok in _tokens(r["topic"]):
            out[tok] = out.get(tok, 0) + int(r["query_count"] or 1)
    return out


def _learned_weights(db) -> dict[str, float]:
    try:
        rows = db.fetchall("SELECT topic, weight FROM salience_weights")
    except Exception:
        return {}
    return {r["topic"]: float(r["weight"] or 0.0) for r in rows}


def score_text(db, text: str, *, monitors: str = "") -> float:
    """Return a 0..1 salience score. Higher = more likely to matter to the owner."""
    toks = _tokens(text)
    if not toks:
        return 0.3

    # 1) Owner-interest: fraction of this item's tokens the owner queries about.
    owner = _owner_topics(db)
    if owner:
        hits = sum(owner[t] for t in toks if t in owner)
        max_possible = max(owner.values()) * 3  # normalize against a few strong hits
        interest = min(1.0, hits / max_possible) if max_possible else 0.0
    else:
        interest = 0.0  # cold start — no owner signal yet

    # 2) Corroboration: "confirmed by N outlets" → stronger story.
    m = re.search(r"(\d+)\s*outlets?", (text or "").lower())
    corrob = min(1.0, (int(m.group(1)) / 8.0)) if m else 0.0

    # 3) Learned weight: max over matching topics, squashed to 0..1.
    learned = _learned_weights(db)
    lw = 0.0
    if learned:
        best = max((learned.get(t, 0.0) for t in toks), default=0.0)
        lw = max(0.0, min(1.0, 0.5 + best / 4.0))  # 0 weight → 0.5 neutral

    # Weighted blend. Owner interest dominates; corroboration + learning refine.
    # Cold-start (no owner/learned signal) lands near 0.4–0.5 so nothing is
    # wrongly dropped before Nova has learned anything.
    score = 0.5 * interest + 0.3 * corrob + 0.2 * (lw if learned else 0.5)
    base = 0.4  # floor so a brand-new system doesn't nuke its own digest
    return round(max(base, min(1.0, base + score)), 3) if (owner or learned) else round(0.45 + 0.3 * corrob, 3)


def learn_from_rating(db, text: str, rating: int) -> None:
    """Nudge salience_weights for this item's topics from a 👍(+1)/👎(-1) rating."""
    if rating not in (-1, 1):
        return
    toks = list(_tokens(text))[:12]
    if not toks:
        return
    delta = 0.5 * rating
    try:
        for tok in toks:
            db.execute(
                "INSERT INTO salience_weights (topic, weight, updated_at) "
                "VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(topic) DO UPDATE SET "
                "  weight = max(-3.0, min(3.0, weight + ?)), updated_at = datetime('now')",
                (tok, delta, delta),
            )
    except Exception as e:
        logger.warning("[Salience] learn_from_rating failed: %s", e)


def rank_digest_items(db, items: list[tuple[str, str]], *, floor: float = 0.4) -> list[tuple[str, str]]:
    """Order digest items by salience (high first); drop sub-floor items only when
    the digest is large enough that dropping won't blank it."""
    if len(items) <= 2:
        return items  # never thin out a tiny digest
    scored = [(name, msg, score_text(db, msg, monitors=name)) for name, msg in items]
    scored.sort(key=lambda x: x[2], reverse=True)
    kept = [(n, m) for n, m, s in scored if s >= floor]
    # Always keep at least the top half so a harsh floor can't gut the briefing.
    if len(kept) < max(2, len(items) // 2):
        kept = [(n, m) for n, m, s in scored[: max(2, len(items) // 2)]]
    return kept
