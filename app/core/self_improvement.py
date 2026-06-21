"""Self-improvement loop health + dead-cruft pruning.

The loop continuously emits lessons, skills, auto-tools and reflexions, but
nothing measured whether that OUTPUT is useful. Skills that never match,
auto-tools that never run, and lessons that never help just accumulate (93
skills, 29 auto-tools, 174 lessons at audit time — usefulness unknown).

`compute_health` is a cheap snapshot of usage + quality per artifact type, so
the operator (and the dashboard) can SEE whether the machinery produces value.
`prune_dead_artifacts` disables the never-used, aged artifacts that the existing
success-based gate (_disable_weak_skills, which needs failures) can't catch — a
skill that never fires has no failures, so it lived forever.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# A never-used artifact older than this is considered dead cruft.
_DEAD_AGE_DAYS = 21


def _scalar(db, sql: str, params: tuple = ()) -> float:
    try:
        row = db.fetchone(sql, params)
        return (row[0] if row and row[0] is not None else 0)
    except Exception:
        return 0


def compute_health(db) -> dict:
    """Cheap usage+quality snapshot of the self-improvement loop's output."""
    def ratio(used, total):
        return round(used / total, 3) if total else None

    skills_total = int(_scalar(db, "SELECT COUNT(*) FROM skills WHERE enabled=1"))
    skills_used = int(_scalar(db, "SELECT COUNT(*) FROM skills WHERE enabled=1 AND times_used>0"))
    skills_dead = int(_scalar(
        db,
        "SELECT COUNT(*) FROM skills WHERE enabled=1 AND times_used=0 "
        "AND created_at < datetime('now', ?)", (f"-{_DEAD_AGE_DAYS} days",),
    ))
    skills_succ = _scalar(db, "SELECT AVG(success_rate) FROM skills WHERE enabled=1 AND times_used>0")

    tools_total = int(_scalar(db, "SELECT COUNT(*) FROM custom_tools WHERE enabled=1"))
    tools_used = int(_scalar(db, "SELECT COUNT(*) FROM custom_tools WHERE enabled=1 AND times_used>0"))
    tools_dead = int(_scalar(
        db,
        "SELECT COUNT(*) FROM custom_tools WHERE enabled=1 AND times_used=0 "
        "AND created_at < datetime('now', ?)", (f"-{_DEAD_AGE_DAYS} days",),
    ))

    lessons_total = int(_scalar(db, "SELECT COUNT(*) FROM lessons"))
    lessons_helpful = int(_scalar(db, "SELECT COUNT(*) FROM lessons WHERE times_helpful>0"))
    lessons_retrieved_unhelpful = int(_scalar(
        db, "SELECT COUNT(*) FROM lessons WHERE times_retrieved>2 AND times_helpful=0"))

    reflex_total = int(_scalar(db, "SELECT COUNT(*) FROM reflexions"))
    reflex_injected = int(_scalar(db, "SELECT COUNT(*) FROM reflexions WHERE times_injected>0"))

    return {
        "skills": {
            "active": skills_total, "used": skills_used,
            "used_rate": ratio(skills_used, skills_total),
            "dead_unused_aged": skills_dead,
            "avg_success_rate": round(skills_succ, 3) if skills_succ else None,
        },
        "auto_tools": {
            "active": tools_total, "used": tools_used,
            "used_rate": ratio(tools_used, tools_total),
            "dead_unused_aged": tools_dead,
        },
        "lessons": {
            "total": lessons_total, "helpful": lessons_helpful,
            "helpful_rate": ratio(lessons_helpful, lessons_total),
            "retrieved_but_never_helpful": lessons_retrieved_unhelpful,
        },
        "reflexions": {
            "total": reflex_total, "injected": reflex_injected,
            "injected_rate": ratio(reflex_injected, reflex_total),
        },
    }


def prune_dead_artifacts(db, *, max_age_days: int = _DEAD_AGE_DAYS) -> dict:
    """Disable never-used artifacts older than max_age_days. These have no
    failures (never ran), so the success-based gate can't see them. Disabling
    (not deleting) keeps them recoverable and out of matching/retrieval."""
    cutoff = f"-{max_age_days} days"
    pruned = {"skills": 0, "auto_tools": 0}
    try:
        cur = db.execute(
            "UPDATE skills SET enabled=0 WHERE enabled=1 AND times_used=0 "
            "AND created_at < datetime('now', ?)", (cutoff,),
        )
        pruned["skills"] = cur.rowcount if cur and cur.rowcount and cur.rowcount > 0 else 0
    except Exception as e:
        logger.warning("[self-improvement] skill prune failed: %s", e)
    try:
        cur = db.execute(
            "UPDATE custom_tools SET enabled=0 WHERE enabled=1 AND times_used=0 "
            "AND created_at < datetime('now', ?)", (cutoff,),
        )
        pruned["auto_tools"] = cur.rowcount if cur and cur.rowcount and cur.rowcount > 0 else 0
    except Exception as e:
        logger.warning("[self-improvement] auto-tool prune failed: %s", e)
    if pruned["skills"] or pruned["auto_tools"]:
        logger.info("[self-improvement] pruned dead artifacts: %s", pruned)
    return pruned
