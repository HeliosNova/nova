"""Daemon Orchestrator — proactive agent with LLM tick reasoning.

Sits above the heartbeat loop and makes strategic decisions about what to
investigate, when to trigger dream consolidation, and when to proactively
alert the user.

Runs every DAEMON_TICK (default 5min). Most ticks result in "no action."
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta

from app.config import config

logger = logging.getLogger(__name__)

# Blocking budget levels
BUDGET_BRIEF = "brief"     # Observe only, no LLM actions
BUDGET_LIGHT = "light"     # Quick investigations, alerts
BUDGET_FULL = "full"       # Dream mode, heavy research, bulk operations

# Default tick interval (seconds)
DAEMON_TICK = 300  # 5 minutes


class DaemonOrchestrator:
    """Proactive daemon that decides what to do on each tick."""

    def __init__(self, db):
        self._db = db  # SafeDB (sync)
        self._running = False
        self._dream_running = False
        self._task: asyncio.Task | None = None
        # Last daemon-initiated curiosity research (monotonic seconds). A
        # pending critical item that repeatedly fails its closure check stays
        # pending — without a cooldown the daemon re-researched the SAME item
        # every 5-minute tick (observed live 2026-08-26: 3 full GPU research
        # runs in 20 min, all discarded). The hourly Curiosity Research monitor
        # remains the steady cadence; the daemon only supplements.
        # None = never ran. The old 0.0 sentinel compared against
        # time.monotonic() suppressed research for the first 30 minutes of
        # every process — every CI runner and every post-boot half hour.
        self._last_curiosity_research: float | None = None

    def _curiosity_cooldown_elapsed(self) -> bool:
        last = getattr(self, "_last_curiosity_research", None)
        return last is None or time.monotonic() - last >= 1800

    def _lane_busy(self) -> bool:
        """The heartbeat's LLM lane holds the card for one residency class; an
        opportunistic run of the other class evicts it. Live 2026-09-22 08:35
        UTC: curiosity think() (9B) fired twenty seconds into a three-wide 27B
        digest batch and the card swapped six times in five minutes. The
        monitors are class-gated; the daemon supplements, so it defers."""
        try:
            from app.core.brain import get_services
            hb = get_services().heartbeat
            return bool(hb is not None and hb.lane_busy())
        except Exception:
            return False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Start the daemon loop as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        self._task.add_done_callback(self._on_task_done)
        logger.info("[Daemon] Orchestrator started (tick=%ds)", DAEMON_TICK)

    def _on_task_done(self, task: "asyncio.Task") -> None:
        """The loop is meant to run until stop(). If it ends while we still
        expect it up, it died on an unhandled exception (the per-tick try/except
        only guards _tick) — nothing else watches the daemon (the KG dead-man
        switches live in the heartbeat loop), so proactive dream/curiosity/goal
        work would silently stop. Log loudly and respawn; _loop's leading 30s
        sleep bounds any crash-loop to one respawn / 30s (audit 2026-08-22)."""
        if task.cancelled() or not self._running:
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[Daemon] loop died unhandled: %s — respawning", exc, exc_info=exc)
        else:
            logger.error("[Daemon] loop exited unexpectedly while running — respawning")
        try:
            self._task = asyncio.get_running_loop().create_task(self._loop())
            self._task.add_done_callback(self._on_task_done)
        except Exception as e:  # pragma: no cover - respawn is best-effort
            logger.error("[Daemon] respawn failed: %s", e)

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
        """Collect current state for daemon reasoning.

        Pure sequential DB reads — runs once per tick via to_thread so the
        event loop never waits on the SQLite lock (2026-06-11 incident class).
        """
        return await asyncio.to_thread(self._gather_context_sync)

    def _gather_context_sync(self) -> dict:
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
        # strftime, not isoformat (2026-08-29): created_at is SQLite's
        # "YYYY-MM-DD HH:MM:SS" while .isoformat() emits a "T". Compared as
        # STRINGS, ' ' (0x20) sorts before 'T' (0x54) — and with `>` the bug
        # INVERTS: rows dated on the cutoff day were judged "not newer" and
        # EXCLUDED, hiding recent entries from the daemon's own situational
        # awareness. Same root cause as the conversation-prune bug in memory.py.
        cutoff = (now - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
        log_rows = self._db.fetchall(
            "SELECT category, content, created_at FROM daemon_log "
            "WHERE created_at > ? ORDER BY created_at DESC LIMIT 10",
            (cutoff,),
        )
        recent_log = [
            f"[{r['category']}] {r['content']}" for r in log_rows
        ]

        # Monitor health — any recent failures?
        # Same T-vs-space fix: this one silently under-counted recent errors,
        # so the daemon's health signal read cleaner than reality.
        cutoff_1h = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
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
                and not self._dream_running
                and not self._lane_busy()):
            return {"action": "dream"}

        # Critical curiosity research — idle and have urgent items. 30-min
        # cooldown between daemon-initiated runs (see __init__ note).
        if (budget in (BUDGET_LIGHT, BUDGET_FULL)
                and context["critical_curiosity"] > 0
                and self._curiosity_cooldown_elapsed()
                and not self._lane_busy()):
            return {"action": "research_curiosity"}

        # Process pending events
        if context["pending_events"] > 0 and budget != BUDGET_BRIEF:
            return {"action": "process_events"}

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
        rows = await asyncio.to_thread(
            self._db.fetchall,
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
                await asyncio.to_thread(
                    self._db.execute,
                    "UPDATE event_queue SET status='processed', processed_at=datetime('now') WHERE id=?",
                    (row["id"],),
                )
                self._log("action", f"Alert sent: {row['event_type']}", "daemon")
            else:
                await asyncio.to_thread(
                    self._db.execute,
                    "UPDATE event_queue SET status='failed', processed_at=datetime('now') WHERE id=?",
                    (row["id"],),
                )
                logger.error(
                    "[Daemon] Event '%s' (id=%d) marked failed — no channel delivered the alert",
                    row["event_type"], row["id"],
                )
                self._log("error", f"Alert delivery failed: {row['event_type']}", "daemon")

    async def _research_curiosity(self):
        """Ask the heartbeat to run Curiosity Research early.

        The daemon used to run the research itself, from outside the tick's
        residency gate: it fired into 27B digest batches three times on
        2026-09-22 (07:48, 08:35, 16:13), five or six model swaps each, once
        for a single item that produced nothing. It now only makes the
        monitor due; the tick runs it under the gate in class order.
        """
        from app.core.brain import get_services
        svc = get_services()
        if not svc.curiosity or not svc.heartbeat:
            return

        self._last_curiosity_research = time.monotonic()
        try:
            ok = await asyncio.to_thread(svc.heartbeat.request_early_run, "Curiosity Research")
            self._log("decision",
                      "Curiosity Research made due for the heartbeat lane" if ok
                      else "Curiosity Research monitor unavailable — nothing requested",
                      "daemon")
        except Exception as e:
            self._log("error", f"Curiosity hand-off failed: {e}", "daemon")
            logger.warning("[Daemon] Curiosity hand-off failed: %s", e)

    async def _process_events(self):
        """Process pending events from the queue."""

        # Pure DB loop — single thread hop instead of blocking the loop per row.
        def _drain() -> int:
            rows = self._db.fetchall(
                "SELECT id, event_type, payload, priority FROM event_queue "
                "WHERE status='pending' ORDER BY priority DESC LIMIT 10"
            )
            n = 0
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
                n += 1
            return n

        processed = await asyncio.to_thread(_drain)

        if processed:
            self._log("action", f"Processed {processed} events from queue", "daemon")

    # ── Logging ───────────────────────────────────────────────────────────

    def _log(self, category: str, content: str, source: str = ""):
        """Write to daemon_log table.

        Callable from both sync helpers (worker threads) and async code. When
        a loop is running, the INSERT is handed to the default executor
        fire-and-forget so the loop never waits on the DB lock for a log row;
        ordering across rows is best-effort, which is fine for a log.
        """
        def _write() -> None:
            try:
                self._db.execute(
                    "INSERT INTO daemon_log (category, content, source) VALUES (?, ?, ?)",
                    (category, content[:2000], source),
                )
            except Exception as e:
                logger.warning("[Daemon] Failed to write log: %s", e)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _write()
            return
        loop.run_in_executor(None, _write)
