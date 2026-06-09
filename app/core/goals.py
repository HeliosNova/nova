"""Goals + will-module — self-directed goal pursuit.

The `goals` table holds desired outcomes (seeded by Phase-0 migration or derived
at runtime). `GoalStore` manages CRUD. `execute_goal()` pulls a goal, passes its
text through `brain.think()`, and logs the outcome. `derive_goals_from_state()`
mints new goals from system state (stub for now).

The daemon orchestrator picks a pending goal each tick (see daemon.py
`pursue_goal` action) and calls `execute_goal`. Output lands in `daemon_log`
with category=`goal_execution` so it's observable from the dashboard / logs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Goal:
    id: int
    goal: str
    priority: float
    status: str
    source: str
    context: dict
    created_at: str


class GoalStore:
    """CRUD wrapper around the `goals` table."""

    def __init__(self, db):
        self._db = db

    def _row_to_goal(self, row) -> Goal:
        ctx = {}
        raw_ctx = row["context"] if row["context"] else ""
        if raw_ctx:
            try:
                ctx = json.loads(raw_ctx)
            except (ValueError, TypeError):
                ctx = {}
        return Goal(
            id=row["id"],
            goal=row["goal"],
            priority=float(row["priority"] or 0.0),
            status=row["status"],
            source=row["source"] or "",
            context=ctx,
            created_at=str(row["created_at"]),
        )

    def get_next_pending(self) -> Goal | None:
        row = self._db.fetchone(
            "SELECT id, goal, priority, status, source, context, created_at "
            "FROM goals WHERE status='pending' "
            "ORDER BY priority DESC, created_at ASC LIMIT 1"
        )
        return self._row_to_goal(row) if row else None

    def get(self, goal_id: int) -> Goal | None:
        row = self._db.fetchone(
            "SELECT id, goal, priority, status, source, context, created_at "
            "FROM goals WHERE id=?",
            (goal_id,),
        )
        return self._row_to_goal(row) if row else None

    def count_pending(self) -> int:
        row = self._db.fetchone("SELECT COUNT(*) AS c FROM goals WHERE status='pending'")
        return int(row["c"]) if row else 0

    def add(self, goal: str, priority: float = 0.5, source: str = "derived",
            context: dict | None = None) -> int:
        ctx_json = json.dumps(context or {})
        self._db.execute(
            "INSERT INTO goals (goal, priority, status, source, context, created_at, updated_at) "
            "VALUES (?, ?, 'pending', ?, ?, datetime('now'), datetime('now'))",
            (goal[:2000], priority, source, ctx_json),
        )
        row = self._db.fetchone("SELECT last_insert_rowid() AS id")
        return int(row["id"]) if row else -1

    def mark_in_progress(self, goal_id: int) -> None:
        self._db.execute(
            "UPDATE goals SET status='in_progress', updated_at=datetime('now') WHERE id=?",
            (goal_id,),
        )

    def mark_completed(self, goal_id: int) -> None:
        self._db.execute(
            "UPDATE goals SET status='completed', completed_at=datetime('now'), "
            "updated_at=datetime('now') WHERE id=?",
            (goal_id,),
        )

    def mark_failed(self, goal_id: int, reason: str = "") -> None:
        self._db.execute(
            "UPDATE goals SET status='failed', updated_at=datetime('now') WHERE id=?",
            (goal_id,),
        )
        if reason:
            logger.warning("[Goals] Goal %d failed: %s", goal_id, reason[:200])

    def attach_output(self, goal_id: int, output: str) -> None:
        """Write the full execution output into goals.context for later review."""
        row = self._db.fetchone("SELECT context FROM goals WHERE id=?", (goal_id,))
        if not row:
            return
        ctx: dict = {}
        raw = row["context"] or ""
        if raw:
            try:
                ctx = json.loads(raw)
            except (ValueError, TypeError):
                ctx = {}
        ctx["last_output"] = output[:8000]
        ctx["last_output_at"] = datetime.utcnow().isoformat()
        self._db.execute(
            "UPDATE goals SET context=?, updated_at=datetime('now') WHERE id=?",
            (json.dumps(ctx), goal_id),
        )


def derive_goals_from_state(db, services) -> list[dict]:
    """Derive new pending goals from current system state.

    Sync wrapper around `app.core.goal_deriver.derive_goals` — full
    implementation lives there. Mines:
      - capability_gap clusters (3+ similar failures)
      - recurring curiosity items
      - repeatedly-failing skills
      - tool trust regressions (>20pt drop in a week)
      - stale-lesson-review backlog
    """
    import asyncio
    from app.core.goal_deriver import derive_goals

    try:
        # derive_goals is async; the daemon caller may not be in an event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Cannot block in a running loop — schedule and return what we have
            task = asyncio.ensure_future(derive_goals(db))
            # Best-effort wait briefly; if it doesn't complete, return empty
            return []
        return asyncio.run(derive_goals(db))
    except Exception as e:
        logger.warning("derive_goals_from_state failed: %s", e)
        return []


async def execute_goal(goal: Goal) -> tuple[bool, str]:
    """Pursue a goal by feeding its text through brain.think() ephemerally.

    Returns (success, output). Caller is responsible for updating goal status
    in the store. Uses ephemeral mode so the goal attempt does not pollute
    conversation history.
    """
    import asyncio
    import re as _re
    from app.config import config as _cfg
    from app.core.brain import think
    from app.schema import EventType

    framed = (
        "=== WILL-MODULE TASK ===\n"
        "You are pursuing an internal self-improvement goal. Your output will be "
        "stored and graded. Follow these rules strictly:\n"
        "- Use tools (web_search, knowledge_search, etc.) if the goal needs external facts.\n"
        "- ALWAYS end with a natural-language final answer — never stop after a tool call.\n"
        "- Do NOT emit raw tool-call JSON or </tool_call> tags in the final answer.\n"
        "- Do NOT ask clarifying questions. If the goal is ambiguous, make a reasonable "
        "interpretation and state it up front.\n"
        "- Produce a concrete, self-contained result that satisfies the goal.\n"
        "=== END TASK ===\n\n"
        f"GOAL: {goal.goal}"
    )

    chunks: list[str] = []
    try:
        async with asyncio.timeout(_cfg.GENERATION_TIMEOUT):
            async for event in think(query=framed, ephemeral=True):
                if event.type == EventType.TOKEN:
                    text = event.data.get("text", "")
                    if text:
                        chunks.append(text)
    except asyncio.TimeoutError:
        return False, "[goals] think() timed out"
    except Exception as e:
        logger.warning("[Goals] execute_goal(%d) raised: %s", goal.id, e)
        return False, f"[goals] think() raised: {e}"

    output = "".join(chunks).strip()
    # Strip stray tool-call tags — brain strips the JSON body but the bare tag
    # can still leak when the tool loop terminates without final synthesis.
    output = _re.sub(r"</?tool_call>", "", output).strip()
    if not output or output.startswith("[error") or len(output) < 40:
        return False, output or "[goals] empty output"
    return True, output
