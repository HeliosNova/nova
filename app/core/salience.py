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
import statistics

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


def score_text(db, text: str, *, owner: dict | None = None,
               learned: dict | None = None) -> float:
    """Return a 0..1 salience score. Higher = more likely to matter to the owner.

    `owner`/`learned` may be precomputed (see rank_digest_items) to avoid a SQL
    read per item; if None they're fetched. A genuinely irrelevant item (no owner
    interest, no corroboration) scores LOW so it can fall below the drop floor —
    the previous version floored everything at 0.4 == the drop threshold, making
    the noise-drop inert.
    """
    toks = _tokens(text)
    if not toks:
        return 0.2

    if owner is None:
        owner = _owner_topics(db)
    if learned is None:
        learned = _learned_weights(db)

    # 1) Owner-interest: fraction of this item's tokens the owner queries about.
    if owner:
        hits = sum(owner[t] for t in toks if t in owner)
        max_possible = max(owner.values()) * 3  # normalize against a few strong hits
        interest = min(1.0, hits / max_possible) if max_possible else 0.0
    else:
        interest = 0.0  # cold start — no owner signal yet

    # 2) Corroboration: "confirmed by N outlets" → stronger story.
    m = re.search(r"(\d+)\s*outlets?", (text or "").lower())
    corrob = min(1.0, (int(m.group(1)) / 8.0)) if m else 0.0

    # 3) Learned weight: best matching topic weight, squashed to 0..1 (0.5 neutral).
    lw = 0.5
    if learned:
        best = max((learned.get(t, 0.0) for t in toks), default=0.0)
        lw = max(0.0, min(1.0, 0.5 + best / 4.0))

    # COLD START (no owner queries AND no learned weights): we know nothing about
    # what matters, so don't drop anything — score mid/high, corroboration only.
    if not owner and not learned:
        return round(0.5 + 0.3 * corrob, 3)

    # Informed: convex blend in [0,1]; low items CAN fall below the drop floor.
    score = 0.5 * interest + 0.3 * corrob + 0.2 * lw
    return round(min(1.0, score), 3)


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


def rank_digest_items(db, items: list[tuple[str, str]], *, floor: float = 0.15) -> list[tuple[str, str]]:
    """Order digest items by salience (high first) and drop the genuinely weak tail.

    The cut is RELATIVE to this digest's own score distribution (below mean − 0.75σ),
    not an absolute threshold. The convex blend in `score_text` isn't calibrated to a
    fixed scale across digests, so an absolute floor either dropped nothing or — as in
    practice — dropped everything and collapsed to a blind top-half. A distribution-
    relative cut trims the real tail while always leading with the strongest signal.
    `floor` remains only as a hard noise backstop for near-zero-signal items.
    """
    if len(items) <= 2:
        return items  # never thin out a tiny digest
    # Hoist the owner/learned reads ONCE (was 2 SQL reads per item before).
    owner, learned = _owner_topics(db), _learned_weights(db)
    scored = [(name, msg, score_text(db, msg, owner=owner, learned=learned)) for name, msg in items]
    scored.sort(key=lambda x: x[2], reverse=True)
    scores = [s for _, _, s in scored]
    mean, std = statistics.fmean(scores), statistics.pstdev(scores)
    cut = max(floor, mean - 0.75 * std)
    kept = [(n, m) for n, m, s in scored if s >= cut]
    # Always keep at least the top half so a harsh cut can't gut the briefing.
    keep_min = max(2, len(items) // 2)
    if len(kept) < keep_min:
        kept = [(n, m) for n, m, s in scored[:keep_min]]
    return kept
