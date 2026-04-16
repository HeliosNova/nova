"""Event-Driven Trigger System — fires monitors on external/internal events.

Watches the event_queue table for pending events, matches them against
monitors that have trigger_events configured, and fires matching monitors
immediately (bypassing schedule check).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class EventTrigger:
    """Watches event_queue and fires matching monitors."""

    def __init__(self, store, heartbeat, db):
        """
        Args:
            store: MonitorStore instance
            heartbeat: HeartbeatLoop instance
            db: SafeDB instance
        """
        self.store = store
        self.heartbeat = heartbeat
        self._db = db
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> asyncio.Task:
        """Start the event trigger loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="event-trigger")
        logger.info("EventTrigger started")
        return self._task

    def stop(self):
        """Stop the event trigger loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("EventTrigger stopped")

    async def _loop(self):
        """Main loop — poll event_queue every 5 seconds."""
        await asyncio.sleep(5)  # Let services initialize
        while self._running:
            try:
                await self._process_pending_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[EventTrigger] Loop error: %s", e)
            await asyncio.sleep(5)

    async def _process_pending_events(self):
        """Fetch pending events and fire matching monitors."""
        rows = self._db.fetchall(
            "SELECT * FROM event_queue WHERE status = 'pending' "
            "ORDER BY priority DESC, created_at ASC LIMIT 10"
        )
        if not rows:
            return

        for row in rows:
            event_type = row["event_type"]
            try:
                payload = json.loads(row["payload"]) if row["payload"] else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}

            # Mark as processing
            self._db.execute(
                "UPDATE event_queue SET status = 'processing' WHERE id = ?",
                (row["id"],),
            )

            # Find and fire matching monitors
            monitors = self.store.get_event_monitors(event_type)
            fired = 0
            for monitor in monitors:
                # Check cooldown — prevent alert storms
                if monitor.last_alert_at and monitor.cooldown_minutes:
                    try:
                        last_alert = datetime.fromisoformat(monitor.last_alert_at)
                        elapsed = (datetime.now(timezone.utc) - last_alert).total_seconds()
                        if elapsed < monitor.cooldown_minutes * 60:
                            logger.debug(
                                "[EventTrigger] Monitor '%s' in cooldown (%ds remaining)",
                                monitor.name,
                                monitor.cooldown_minutes * 60 - elapsed,
                            )
                            continue
                    except (ValueError, TypeError):
                        pass

                try:
                    logger.info(
                        "[EventTrigger] Firing monitor '%s' for event '%s'",
                        monitor.name, event_type,
                    )
                    await self.heartbeat._check_monitor(monitor)
                    fired += 1
                except Exception as e:
                    logger.error(
                        "[EventTrigger] Monitor '%s' failed for event '%s': %s",
                        monitor.name, event_type, e,
                    )

            # Mark as processed
            self._db.execute(
                "UPDATE event_queue SET status = 'processed', processed_at = datetime('now') WHERE id = ?",
                (row["id"],),
            )
            if fired:
                logger.info("[EventTrigger] Event '%s' fired %d monitor(s)", event_type, fired)

    def emit_event(self, event_type: str, payload: dict | None = None, priority: float = 0.5):
        """Insert an event into the queue (for internal use).

        Args:
            event_type: Namespaced event type (e.g. "internal:lesson_saved", "webhook:github_push")
            payload: Optional JSON-serializable payload
            priority: 0.0–1.0 (higher = processed first)
        """
        self._db.execute(
            "INSERT INTO event_queue (event_type, payload, priority) VALUES (?, ?, ?)",
            (event_type, json.dumps(payload or {}), priority),
        )


# ---------------------------------------------------------------------------
# Module-level helper for internal event emission
# ---------------------------------------------------------------------------

_event_trigger: EventTrigger | None = None


def set_event_trigger(trigger: EventTrigger | None) -> None:
    """Set the module-level event trigger instance (called at startup)."""
    global _event_trigger
    _event_trigger = trigger


def emit_event(event_type: str, payload: dict | None = None, priority: float = 0.5) -> None:
    """Emit an event if the event trigger system is active.

    Safe to call even if event triggers are disabled — silently no-ops.
    """
    if _event_trigger is not None:
        try:
            _event_trigger.emit_event(event_type, payload, priority)
        except Exception as e:
            logger.debug("Event emission failed (non-blocking): %s", e)
