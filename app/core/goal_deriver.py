"""Goal derivation — Nova decides what to work on.

The original KAIROS architecture had a `goals` table and an executor, but
nothing generating goals from state. Goals only landed via human seeding,
and only 3 ever existed in project history.

This module mines Nova's actual operational state for what to do next:
  - **Capability gaps**: failures Nova logged about himself. If 3+ similar
    gaps cluster, propose a goal to fix the underlying capability.
  - **Curiosity queue**: pending research items. If a topic is recurring,
    promote to a goal.
  - **Recurring failures**: the same skill failing N times → goal to either
    repair the skill or replace it.
  - **Trust score regressions**: if a tool's trust score dropped > 20 in a
    week, goal to investigate and either fix or stop using it.
  - **Stale lessons with low retrieval**: lessons that haven't been useful
    in 60+ days → goal to either re-validate or prune.

Each derived goal goes in the `goals` table with `source='derived'` and a
context payload. The KAIROS executor picks them up on its tick.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


_GOAL_TEMPLATES = {
    "capability_gap_cluster": (
        "Investigate and fix the recurring capability gap: {pattern}. "
        "{count} queries failed with similar reasons. "
        "Either acquire the data needed, build a skill, or write a custom tool."
    ),
    "recurring_curiosity": (
        "Resolve the recurring curiosity item: {topic}. "
        "It has been queued {count} times — the topic is important enough to research deeply."
    ),
    "skill_repair": (
        "The skill '{skill_name}' has failed {count} times. "
        "Inspect its trigger pattern and steps; repair or retire it."
    ),
    "tool_trust_regression": (
        "Tool '{tool_name}' lost {drop} trust points in the last week "
        "({recent_failures} failures). Diagnose the failure mode and either "
        "fix the call pattern or stop relying on this tool."
    ),
    "stale_lesson_review": (
        "{count} lessons haven't been retrieved in 60+ days. "
        "Re-validate them through the quiz monitor or prune the dead ones."
    ),
}


async def derive_goals(db, *, max_new_goals: int = 5) -> list[dict]:
    """Inspect state and create new goals. Returns list of created goal dicts.

    The body is pure sequential DB work — it runs in a worker thread so the
    event loop never waits on the SQLite lock (the function had zero awaits;
    it was async-in-name-only while blocking the loop for its whole run).
    """
    import asyncio
    return await asyncio.to_thread(_derive_goals_sync, db, max_new_goals=max_new_goals)


def _derive_goals_sync(db, *, max_new_goals: int = 5) -> list[dict]:
    """Sync body of derive_goals — see wrapper above.

    Conservative: caps new goals per run so we don't spam the queue. Skips
    creating duplicates of recent active/pending goals on the same theme.
    """
    derived: list[dict] = []

    # --- Skip themes already in flight OR recently dispatched ---
    # Originally only 'pending'+'in_progress' were checked, which let goals
    # respawn the same theme as soon as they were marked completed (even if
    # the underlying state hadn't changed — common for trust regressions
    # where the rolling 7-day failure window is sticky). Include completed
    # within 24h so we don't churn.
    active_goals_text = " ".join(
        row["goal"].lower()
        for row in db.fetchall(
            "SELECT goal FROM goals WHERE "
            "(status IN ('pending','in_progress') AND created_at > datetime('now', '-7 days')) "
            "OR (status='completed' AND COALESCE(completed_at, updated_at) > datetime('now', '-1 day'))"
        )
    )

    def _theme_active(keyword: str) -> bool:
        return keyword.lower() in active_goals_text

    # Also build a set of recent dedup signatures (source_kind + key field)
    # so dynamic numbers in goal text don't defeat dedupe.
    active_signatures: set[str] = set()
    for row in db.fetchall(
        "SELECT context FROM goals WHERE "
        "(status IN ('pending','in_progress') AND created_at > datetime('now', '-7 days')) "
        "OR (status IN ('completed','failed') AND COALESCE(completed_at, updated_at) > datetime('now', '-1 day'))"
    ):
        ctx_raw = row["context"]
        if not ctx_raw:
            continue
        try:
            ctx = json.loads(ctx_raw)
        except Exception:
            continue
        src = ctx.get("source")
        key = (
            ctx.get("tool_name") or ctx.get("skill_name") or ctx.get("keyword")
            or ctx.get("topic") or ""
        )
        if src and key:
            active_signatures.add(f"{src}:{key}".lower()[:200])

    # --- Source 1: capability_gap clustering ---
    gap_rows = db.fetchall(
        "SELECT id, query, reason FROM capability_gaps "
        "WHERE reviewed = 0 AND created_at > datetime('now', '-30 days')"
    )
    if gap_rows:
        # Cluster by simple keyword extraction — pull the longest noun-phrase-ish token
        clusters: Counter[str] = Counter()
        for r in gap_rows:
            q = (r["query"] or "").lower()
            # Pull the top 1-2 substantive words
            words = re.findall(r"\b[a-z][a-z0-9_-]{3,}\b", q)
            for w in words[:3]:
                if w not in {"what", "where", "when", "which", "show", "tell", "find", "search", "calculate", "compute"}:
                    clusters[w] += 1
        for keyword, count in clusters.most_common(3):
            if count < 3:
                continue
            if _theme_active(keyword):
                continue
            goal_text = _GOAL_TEMPLATES["capability_gap_cluster"].format(
                pattern=keyword, count=count
            )
            ctx = {"source": "capability_gap_cluster", "keyword": keyword, "gap_count": count}
            row_id = _insert_goal(db, goal_text, priority=0.7, source="derived", context=ctx)
            if row_id:
                derived.append({"id": row_id, "goal": goal_text, "source_kind": "capability_gap_cluster"})
                if len(derived) >= max_new_goals:
                    return derived

    # --- Source 2: curiosity queue duplicates ---
    curiosity_rows = db.fetchall(
        "SELECT topic, COUNT(*) as n FROM curiosity_queue "
        "WHERE status='pending' AND created_at > datetime('now', '-14 days') "
        "GROUP BY topic HAVING n >= 2 ORDER BY n DESC LIMIT 3"
    )
    for r in curiosity_rows:
        topic = r["topic"]
        if _theme_active(topic[:30]):
            continue
        goal_text = _GOAL_TEMPLATES["recurring_curiosity"].format(topic=topic[:120], count=r["n"])
        ctx = {"source": "recurring_curiosity", "topic": topic, "count": r["n"]}
        row_id = _insert_goal(db, goal_text, priority=0.6, source="derived", context=ctx)
        if row_id:
            derived.append({"id": row_id, "goal": goal_text, "source_kind": "recurring_curiosity"})
            if len(derived) >= max_new_goals:
                return derived

    # --- Source 3: failing skills ---
    skill_rows = db.fetchall(
        "SELECT name, consecutive_failures FROM skills "
        "WHERE enabled=1 AND consecutive_failures >= 3 "
        "ORDER BY consecutive_failures DESC LIMIT 3"
    )
    for r in skill_rows:
        if _theme_active(r["name"]):
            continue
        goal_text = _GOAL_TEMPLATES["skill_repair"].format(
            skill_name=r["name"], count=r["consecutive_failures"]
        )
        ctx = {"source": "skill_repair", "skill_name": r["name"], "failures": r["consecutive_failures"]}
        row_id = _insert_goal(db, goal_text, priority=0.5, source="derived", context=ctx)
        if row_id:
            derived.append({"id": row_id, "goal": goal_text, "source_kind": "skill_repair"})
            if len(derived) >= max_new_goals:
                return derived

    # --- Source 4: trust score regressions ---
    # DISABLED 2026-05-06: this source was causing a recursive failure loop.
    # When code_exec / shell_exec accumulated failures (often from external
    # causes like a broken auto-monitor that's since been disabled), this
    # spawned "Diagnose tool 'X'" goals. KAIROS WILL-MODULE then *pursued*
    # the goal by running the same broken tool to query the DB, hallucinated
    # the schema, failed, dropped trust further → respawn. Verified runtime:
    # 130+ near-duplicate goals had stacked, code_exec ran 80%+ failure rate
    # with traces all originating from WILL-MODULE diagnosis attempts.
    #
    # Tool-failure diagnosis isn't a useful autonomous task — it needs a
    # human to look at the underlying call patterns. Keep the rest of the
    # deriver active; just stop minting these self-recursive goals.
    pass  # tool_trust_regression source intentionally retired — see above

    # --- Source 5: stale lessons (only one goal per cycle for this) ---
    if not _theme_active("stale lesson"):
        try:
            stale_count = db.fetchone(
                "SELECT COUNT(*) as n FROM lessons "
                "WHERE confidence > 0.4 AND (last_retrieved_at IS NULL "
                "OR last_retrieved_at < datetime('now', '-60 days'))"
            )["n"]
            if stale_count >= 10:
                goal_text = _GOAL_TEMPLATES["stale_lesson_review"].format(count=stale_count)
                ctx = {"source": "stale_lesson_review", "count": stale_count}
                row_id = _insert_goal(db, goal_text, priority=0.3, source="derived", context=ctx)
                if row_id:
                    derived.append({"id": row_id, "goal": goal_text, "source_kind": "stale_lesson_review"})
        except Exception as e:
            logger.warning("stale lesson check failed: %s", e)

    return derived


def _insert_goal(db, goal_text: str, *, priority: float, source: str, context: dict) -> int | None:
    """Insert a goal row. Dedupe across ALL statuses in the last 7 days —
    if the same goal was already completed and the underlying gap recurs,
    re-minting a new goal won't solve it (the prior 'completion' was
    cosmetic). Only mint when this goal text hasn't been seen recently."""
    existing = db.fetchone(
        "SELECT id, status FROM goals WHERE goal = ? "
        "AND created_at > datetime('now', '-7 days')",
        (goal_text,),
    )
    if existing:
        return None
    cursor = db.execute(
        "INSERT INTO goals (goal, priority, status, source, context, created_at, updated_at) "
        "VALUES (?, ?, 'pending', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (goal_text, priority, source, json.dumps(context)),
    )
    return cursor.lastrowid


async def derive_and_log(db) -> str:
    """Top-level entry — derive goals, return summary string suitable for a monitor result."""
    try:
        derived = await derive_goals(db)
    except Exception as e:
        logger.exception("derive_goals failed")
        return f"GOAL DERIVATION ERROR: {e}"

    if not derived:
        return "GOAL DERIVATION | no new goals derived (state quiet or themes already covered)"
    summary_lines = [f"GOAL DERIVATION | {len(derived)} new goals:"]
    for g in derived:
        summary_lines.append(f"  - [{g['id']}] ({g['source_kind']}) {g['goal'][:140]}")
    return "\n".join(summary_lines)
