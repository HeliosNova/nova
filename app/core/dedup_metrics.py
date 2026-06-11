"""Dedup decision instrumentation.

Records every Jaccard-based dedup comparison so the operator can
empirically revisit the hand-tuned thresholds (0.55 / 0.6 / 0.85) that
were initially set by symptom rather than measurement. Each `add()`-style
call in the lesson, curiosity, and reflexion stores produces ONE row:
either a `merged` decision with the actual Jaccard score that triggered
the merge, or an `inserted_new` decision with the MAX score observed
across the candidate set (so the operator can see how close near-misses
got to the threshold).

Storage: `dedup_decisions` table (migration 21). Schema:
    id INTEGER PK,
    entity_type TEXT       -- 'lesson' | 'reflexion' | 'curiosity'
    jaccard_score REAL,
    threshold REAL,
    decision TEXT          -- 'merged' | 'inserted_new'
    created_at TIMESTAMP

Recording is fire-and-forget — DB failures degrade silently so the
production dedup path is never blocked by instrumentation.

Analysis:
    from app.core import dedup_metrics
    dedup_metrics.summarize()                       # all entities, all-time
    dedup_metrics.summarize("lesson", "-7 days")    # last 7 days, lessons only
"""
from __future__ import annotations

import logging
from typing import Any

from app.database import get_db

logger = logging.getLogger(__name__)


_ALLOWED_ENTITIES = frozenset({"lesson", "reflexion", "curiosity"})
_ALLOWED_DECISIONS = frozenset({"merged", "inserted_new"})


def record_decision(
    entity_type: str,
    jaccard_score: float,
    threshold: float,
    decision: str,
) -> bool:
    """Store one dedup decision row. Returns True on insert.

    Drops silently on bad input (unknown entity/decision, NaN scores).
    Drops silently on DB error — never raises into the calling dedup loop.
    """
    if entity_type not in _ALLOWED_ENTITIES:
        logger.debug("[dedup_metrics] unknown entity_type=%r — dropping", entity_type)
        return False
    if decision not in _ALLOWED_DECISIONS:
        logger.debug("[dedup_metrics] unknown decision=%r — dropping", decision)
        return False
    try:
        s = float(jaccard_score)
        t = float(threshold)
    except (TypeError, ValueError):
        return False
    if s != s or t != t:  # NaN guard
        return False
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    try:
        get_db().execute(
            "INSERT INTO dedup_decisions (entity_type, jaccard_score, threshold, decision) "
            "VALUES (?, ?, ?, ?)",
            (entity_type, s, t, decision),
        )
        return True
    except Exception as e:
        logger.debug("[dedup_metrics] record_decision failed: %s", e)
        return False


def summarize(
    entity_type: str | None = None,
    since_relative: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-(entity_type, decision) summary stats.

    Args:
      entity_type:   filter to a single entity, or None for all.
      since_relative: SQLite relative timespec like "-7 days" or "-24 hours",
                     or None for all-time.

    Returns: {entity_type: {decision: {"n": int, "mean": float, "min": float,
              "max": float, "near_threshold_pct": float}}}
              where `near_threshold_pct` = fraction of rows whose
              jaccard_score lies within ±0.05 of the threshold (a proxy
              for "decisions that would have flipped under a small
              threshold change" — useful for sensitivity analysis).
    """
    clauses: list[str] = []
    params: list[Any] = []
    if entity_type is not None:
        if entity_type not in _ALLOWED_ENTITIES:
            return {}
        clauses.append("entity_type = ?")
        params.append(entity_type)
    if since_relative is not None:
        clauses.append("created_at >= datetime('now', ?)")
        params.append(since_relative)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    try:
        rows = get_db().fetchall(
            f"SELECT entity_type, decision, jaccard_score, threshold FROM dedup_decisions{where}",
            tuple(params),
        )
    except Exception as e:
        logger.warning("[dedup_metrics] summarize failed: %s", e)
        return {}

    buckets: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in rows:
        key = (r["entity_type"], r["decision"])
        buckets.setdefault(key, []).append((float(r["jaccard_score"]), float(r["threshold"])))

    out: dict[str, dict[str, dict[str, float]]] = {}
    for (etype, dec), pairs in buckets.items():
        scores = [s for s, _ in pairs]
        near = sum(1 for s, t in pairs if abs(s - t) <= 0.05)
        n = len(scores)
        out.setdefault(etype, {})[dec] = {
            "n": float(n),
            "mean": round(sum(scores) / n, 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "near_threshold_pct": round(near / n, 4),
        }
    return out
