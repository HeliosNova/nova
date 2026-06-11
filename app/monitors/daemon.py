"""Daemon Orchestrator — proactive agent with LLM tick reasoning.

Inspired by Claude Code's KAIROS feature. Sits above the heartbeat loop and
makes strategic decisions about what to investigate, when to trigger dream
consolidation, and when to proactively alert the user.

Runs every DAEMON_TICK (default 5min). Most ticks result in "no action."
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta

from app.config import config

logger = logging.getLogger(__name__)

# Blocking budget levels
BUDGET_BRIEF = "brief"     # Observe only, no LLM actions
BUDGET_LIGHT = "light"     # Quick investigations, alerts
BUDGET_FULL = "full"       # Dream mode, heavy research, bulk operations

# Default tick interval (seconds)
DAEMON_TICK = 300  # 5 minutes


def _snapshot_goal_metric(db, ctx: dict) -> tuple[int | None, str]:
    """Read the underlying metric a goal is supposed to fix.

    Returns (count_or_none, human_label). count=None means we don't know how
    to verify this goal kind — caller will fall back to the old behavior
    (any non-empty output = success).

    Verifiable goal kinds today:
      - capability_gap_cluster: count unreviewed capability_gaps for the keyword
      - skill_repair: count of consecutive_failures on the named skill
      - tool_trust_regression: trust score (negative = bad) for the tool
      - recurring_curiosity: count of pending curiosity items for the topic
    """
    if not isinstance(ctx, dict):
        return None, ""
    src = ctx.get("source") or ""
    try:
        if src == "capability_gap_cluster":
            kw = (ctx.get("keyword") or "").strip().lower()
            if not kw:
                return None, ""
            row = db.fetchone(
                "SELECT COUNT(*) AS c FROM capability_gaps "
                "WHERE LOWER(query) LIKE ? AND reviewed = 0",
                (f"%{kw}%",),
            )
            return (int(row["c"]) if row else 0), f"unreviewed capability_gaps matching '{kw}'"
        if src == "skill_repair":
            sn = ctx.get("skill_name") or ""
            row = db.fetchone(
                "SELECT consecutive_failures FROM skills WHERE name = ?", (sn,)
            )
            return (int(row["consecutive_failures"]) if row else 0), f"consecutive_failures for skill '{sn}'"
        if src == "recurring_curiosity":
            topic = ctx.get("topic") or ""
            row = db.fetchone(
                "SELECT COUNT(*) AS c FROM curiosity_queue "
                "WHERE topic = ? AND status = 'pending'",
                (topic,),
            )
            return (int(row["c"]) if row else 0), f"pending curiosity items for '{topic[:40]}'"
        # tool_trust_regression: hard to verify quickly (trust scores update lazily) — skip.
    except Exception as e:
        logger.debug("[Goals] _snapshot_goal_metric failed: %s", e)
    return None, ""


class DaemonOrchestrator:
    """Proactive daemon that decides what to do on each tick."""

    def __init__(self, db):
        self._db = db  # SafeDB (sync)
        self._running = False
        self._dream_running = False
        self._task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Start the daemon loop as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("[Daemon] Orchestrator started (tick=%ds)", DAEMON_TICK)

    async def stop(self):
        """Stop the daemon loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Daemon] Orchestrator stopped")

    async def _loop(self):
        """Main daemon loop — evaluate every DAEMON_TICK seconds."""
        await asyncio.sleep(30)  # Let heartbeat and services initialize first
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("[Daemon] Tick failed: %s", e, exc_info=True)
                self._log("error", f"Tick failed: {e}", "daemon")
            await asyncio.sleep(DAEMON_TICK)

    # ── Core tick ─────────────────────────────────────────────────────────

    async def _tick(self):
        """Single daemon tick: gather context, determine budget, decide action."""
        context = await self._gather_context()
        budget = self._determine_budget(context)

        # If brief budget and nothing pending, skip LLM entirely
        if budget == BUDGET_BRIEF and not context["pending_events"] and not context["alerts_unsent"]:
            return

        decision = await self._decide(context, budget)
        if decision:
            await self._execute_decision(decision, context)

    async def _gather_context(self) -> dict:
        """Collect current state for daemon reasoning."""
        now = datetime.utcnow()

        # Last user activity
        idle_minutes = None
        row = self._db.fetchone("SELECT value FROM system_state WHERE key='last_user_activity'")
        if row and row["value"]:
            try:
                last = datetime.fromisoformat(row["value"])
                idle_minutes = (now - last).total_seconds() / 60
            except (ValueError, TypeError):
                idle_minutes = 999

        # Last dream
        hours_since_dream = None
        row = self._db.fetchone("SELECT value FROM system_state WHERE key='last_dream_at'")
        if row and row["value"]:
            try:
                last_dream = datetime.fromisoformat(row["value"])
                hours_since_dream = (now - last_dream).total_seconds() / 3600
            except (ValueError, TypeError):
                hours_since_dream = 999

        # Pending events
        row = self._db.fetchone("SELECT COUNT(*) as c FROM event_queue WHERE status='pending'")
        pending_events = row["c"] if row else 0

        # Recent daemon log (last 6h, max 10 entries)
        cutoff = (now - timedelta(hours=6)).isoformat()
        log_rows = self._db.fetchall(
            "SELECT category, content, created_at FROM daemon_log "
            "WHERE created_at > ? ORDER BY created_at DESC LIMIT 10",
            (cutoff,),
        )
        recent_log = [
            f"[{r['category']}] {r['content']}" for r in log_rows
        ]

        # Monitor health — any recent failures?
        cutoff_1h = (now - timedelta(hours=1)).isoformat()
        failure_rows = self._db.fetchall(
            "SELECT COUNT(*) as c FROM monitor_results WHERE status='error' AND created_at > ?",
            (cutoff_1h,),
        )
        recent_failures = failure_rows[0]["c"] if failure_rows else 0

        # Pending curiosity items
        row = self._db.fetchone(
            "SELECT COUNT(*) as c FROM curiosity_queue WHERE status='pending' AND urgency >= 0.7"
        )
        critical_curiosity = row["c"] if row else 0

        # Pending goals (will-module)
        row = self._db.fetchone(
            "SELECT COUNT(*) as c FROM goals WHERE status='pending'"
        )
        pending_goals = row["c"] if row else 0

        # Unsent alerts (events with high priority)
        alerts_unsent = 0
        if pending_events:
            row = self._db.fetchone(
                "SELECT COUNT(*) as c FROM event_queue WHERE status='pending' AND priority >= 0.8"
            )
            alerts_unsent = row["c"] if row else 0

        return {
            "idle_minutes": idle_minutes,
            "hours_since_dream": hours_since_dream,
            "pending_events": pending_events,
            "alerts_unsent": alerts_unsent,
            "recent_log": recent_log,
            "recent_failures": recent_failures,
            "critical_curiosity": critical_curiosity,
            "pending_goals": pending_goals,
            "now": now.isoformat(),
        }

    def _determine_budget(self, context: dict) -> str:
        """Determine blocking budget based on user idle time."""
        idle = context.get("idle_minutes")
        if idle is None:
            return BUDGET_FULL  # No activity tracking yet — assume idle
        if idle < 5:
            return BUDGET_BRIEF
        if idle < 30:
            return BUDGET_LIGHT
        return BUDGET_FULL

    # ── Decision engine ───────────────────────────────────────────────────

    async def _decide(self, context: dict, budget: str) -> dict | None:
        """Use heuristics + optional LLM to decide what to do.

        Returns a decision dict or None (no action).
        Heuristic-first to minimize LLM calls — most ticks need no LLM.
        """
        idle = context.get("idle_minutes") or 0
        hours_since_dream = context.get("hours_since_dream")

        # High-priority alerts — always send regardless of budget
        if context["alerts_unsent"]:
            return {"action": "send_alerts"}

        # Brief budget — observe only
        if budget == BUDGET_BRIEF:
            return None

        # Dream trigger — idle 30+ min and dream overdue (12h+) or never dreamed
        if (budget == BUDGET_FULL
                and (hours_since_dream is None or hours_since_dream >= 12)
                and idle >= 30
                and not self._dream_running):
            return {"action": "dream"}

        # Critical curiosity research — idle and have urgent items
        if budget in (BUDGET_LIGHT, BUDGET_FULL) and context["critical_curiosity"] > 0:
            return {"action": "research_curiosity"}

        # Process pending events
        if context["pending_events"] > 0 and budget != BUDGET_BRIEF:
            return {"action": "process_events"}

        # Pursue next goal — only when idle with budget for it (avoid competing with user).
        if budget == BUDGET_FULL and context.get("pending_goals", 0) > 0:
            return {"action": "pursue_goal"}

        # Monitor degradation — log observation
        if context["recent_failures"] >= 3:
            return {
                "action": "observe",
                "content": f"{context['recent_failures']} monitor failures in last hour. Backoff active.",
            }

        # Nothing to do (most common outcome)
        return None

    # ── Execution ─────────────────────────────────────────────────────────

    async def _execute_decision(self, decision: dict, context: dict):
        """Execute a daemon decision."""
        action = decision.get("action")
        logger.info("[Daemon] Executing: %s", action)

        if action == "dream":
            await self._trigger_dream()

        elif action == "send_alerts":
            await self._send_pending_alerts()

        elif action == "research_curiosity":
            await self._research_curiosity()

        elif action == "process_events":
            await self._process_events()

        elif action == "pursue_goal":
            await self._pursue_goal()

        elif action == "observe":
            self._log("observation", decision.get("content", ""), "daemon")

    async def _trigger_dream(self):
        """Trigger dream consolidation."""
        from app.core.dream import DreamConsolidator
        from app.database import AsyncSafeDB, SafeDB

        if self._dream_running:
            logger.info("[Daemon] Dream already running, skipping")
            return

        self._dream_running = True
        self._log("decision", "Triggering dream consolidation (user idle, overdue)", "daemon")
        try:
            async_db = AsyncSafeDB(self._db) if isinstance(self._db, SafeDB) else self._db
            consolidator = DreamConsolidator(async_db)
            digest = await consolidator.run()
            self._log("action", f"Dream complete: {digest}", "dream")
        except Exception as e:
            self._log("error", f"Dream failed: {e}", "daemon")
            logger.error("[Daemon] Dream trigger failed: %s", e)
        finally:
            self._dream_running = False

    async def _send_pending_alerts(self):
        """Send high-priority events as alerts to channels."""
        rows = self._db.fetchall(
            "SELECT id, event_type, payload, priority FROM event_queue "
            "WHERE status='pending' AND priority >= 0.8 "
            "ORDER BY priority DESC LIMIT 5"
        )
        if not rows:
            return

        from app.core.brain import get_services
        svc = get_services()

        for row in rows:
            payload = json.loads(row["payload"]) if row["payload"] else {}
            message = f"**{row['event_type']}** (priority {row['priority']})\n{json.dumps(payload, indent=2)[:500]}"

            # Attempt delivery; require at least one channel to confirm receipt
            delivered = False
            if svc.heartbeat:
                try:
                    delivered = await svc.heartbeat._send_alert_to_channels(message)
                except Exception as e:
                    logger.warning("[Daemon] Alert delivery raised: %s", e)
            else:
                logger.warning("[Daemon] HeartbeatLoop unavailable — cannot deliver alert for event %d", row["id"])

            if delivered:
                self._db.execute(
                    "UPDATE event_queue SET status='processed', processed_at=datetime('now') WHERE id=?",
                    (row["id"],),
                )
                self._log("action", f"Alert sent: {row['event_type']}", "daemon")
            else:
                self._db.execute(
                    "UPDATE event_queue SET status='failed', processed_at=datetime('now') WHERE id=?",
                    (row["id"],),
                )
                logger.error(
                    "[Daemon] Event '%s' (id=%d) marked failed — no channel delivered the alert",
                    row["event_type"], row["id"],
                )
                self._log("error", f"Alert delivery failed: {row['event_type']}", "daemon")

    async def _pursue_goal(self):
        """Pick the next pending goal and execute it via the will-module."""
        from app.core.goals import GoalStore, execute_goal

        store = GoalStore(self._db)
        goal = store.get_next_pending()
        if not goal:
            return

        self._log("decision", f"Pursuing goal #{goal.id}: {goal.goal[:120]}", "daemon")
        store.mark_in_progress(goal.id)

        # Snapshot the underlying metric so we can verify the goal actually
        # moved the needle — not just produced text. Without this, every goal
        # was being marked "completed" because Nova returned >40 chars,
        # regardless of whether the gap/skill/tool problem was actually fixed.
        ctx = goal.context if isinstance(goal.context, dict) else {}
        before_metric, metric_query = _snapshot_goal_metric(self._db, ctx)

        try:
            ok, output = await execute_goal(goal)
        except Exception as e:
            store.mark_failed(goal.id, f"unhandled: {e}")
            self._log("error", f"Goal #{goal.id} crashed: {e}", "daemon")
            return

        store.attach_output(goal.id, output)

        if not ok:
            store.mark_failed(goal.id, output[:200])
            self._log(
                "goal_execution",
                f"Goal #{goal.id} failed: {output[:400]}",
                "will_module",
            )
            return

        # Verify the action actually moved the underlying metric. If the goal
        # was about a capability gap and the gap count didn't decrease, this
        # was cosmetic — mark failed so it doesn't pollute the "completed"
        # bucket and so the deriver can re-attempt with a better strategy.
        after_metric, _ = _snapshot_goal_metric(self._db, ctx)
        if before_metric is not None and after_metric is not None and after_metric >= before_metric:
            reason = (
                f"verification: {metric_query} did not improve "
                f"(before={before_metric} after={after_metric})"
            )
            store.mark_failed(goal.id, reason)
            self._log(
                "goal_execution",
                f"Goal #{goal.id} produced output but no metric improvement. {reason}",
                "will_module",
            )
            return

        store.mark_completed(goal.id)
        improvement = ""
        if before_metric is not None and after_metric is not None:
            improvement = f" (metric: {before_metric} → {after_metric})"
        self._log(
            "goal_execution",
            f"Goal #{goal.id} completed{improvement}. Output: {output[:300]}",
            "will_module",
        )

    async def _research_curiosity(self):
        """Research top critical curiosity items."""
        from app.core.brain import get_services
        svc = get_services()
        if not svc.curiosity or not svc.heartbeat:
            return

        self._log("decision", "Researching critical curiosity items", "daemon")
        try:
            # Delegate to the existing curiosity research monitor handler
            result = await svc.heartbeat._execute_curiosity_research({})
            self._log("action", f"Curiosity research: {result[:200]}", "daemon")
        except Exception as e:
            self._log("error", f"Curiosity research failed: {e}", "daemon")
            logger.warning("[Daemon] Curiosity research failed: %s", e)

    async def _process_events(self):
        """Process pending events from the queue."""
        rows = self._db.fetchall(
            "SELECT id, event_type, payload, priority FROM event_queue "
            "WHERE status='pending' ORDER BY priority DESC LIMIT 10"
        )
        processed = 0
        for row in rows:
            self._log(
                "observation",
                f"Event: {row['event_type']} (priority={row['priority']})",
                "event_queue",
            )
            self._db.execute(
                "UPDATE event_queue SET status='processed', processed_at=datetime('now') WHERE id=?",
                (row["id"],),
            )
            processed += 1

        if processed:
            self._log("action", f"Processed {processed} events from queue", "daemon")

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, category: str, content: str, source: str = ""):
        """Write to daemon_log table."""
        try:
            self._db.execute(
                "INSERT INTO daemon_log (category, content, source) VALUES (?, ?, ?)",
                (category, content[:2000], source),
            )
        except Exception as e:
            logger.warning("[Daemon] Failed to write log: %s", e)
