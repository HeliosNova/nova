"""Daemon API — event ingestion, daemon log, dream trigger."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import require_auth
from app.config import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["daemon"], dependencies=[Depends(require_auth)])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EventIn(BaseModel):
    event_type: str = Field(..., min_length=1, max_length=100)
    payload: dict = Field(default_factory=dict)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class DreamTrigger(BaseModel):
    force: bool = False  # Skip idle/cooldown checks


# ---------------------------------------------------------------------------
# Event queue endpoints
# ---------------------------------------------------------------------------

@router.post("/daemon/events")
async def ingest_event(event: EventIn):
    """Ingest an external event into the daemon event queue."""
    from app.core.brain import get_services

    svc = get_services()
    if not svc.monitor_store:
        raise HTTPException(503, "Monitor store not initialized")

    db = svc.monitor_store.db
    db.execute(
        "INSERT INTO event_queue (event_type, payload, priority) VALUES (?, ?, ?)",
        (event.event_type, json.dumps(event.payload), event.priority),
    )
    return {"status": "queued", "event_type": event.event_type, "priority": event.priority}


@router.get("/daemon/events")
async def list_events(status: str = "pending", limit: int = 50):
    """List events in the queue."""
    from app.core.brain import get_services

    svc = get_services()
    if not svc.monitor_store:
        raise HTTPException(503, "Monitor store not initialized")

    db = svc.monitor_store.db
    rows = db.fetchall(
        "SELECT id, event_type, payload, priority, status, created_at "
        "FROM event_queue WHERE status=? ORDER BY priority DESC, created_at ASC LIMIT ?",
        (status, limit),
    )
    return [
        {
            "id": r["id"],
            "event_type": r["event_type"],
            "payload": json.loads(r["payload"]) if r["payload"] else {},
            "priority": r["priority"],
            "status": r["status"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Daemon log endpoints
# ---------------------------------------------------------------------------

@router.get("/daemon/log")
async def get_daemon_log(hours: int = 24, category: str | None = None, limit: int = 100):
    """Get recent daemon log entries."""
    from app.core.brain import get_services

    svc = get_services()
    if not svc.monitor_store:
        raise HTTPException(503, "Monitor store not initialized")

    db = svc.monitor_store.db
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    if category:
        rows = db.fetchall(
            "SELECT id, category, content, source, created_at "
            "FROM daemon_log WHERE created_at > ? AND category = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (cutoff, category, limit),
        )
    else:
        rows = db.fetchall(
            "SELECT id, category, content, source, created_at "
            "FROM daemon_log WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
            (cutoff, limit),
        )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Dream trigger
# ---------------------------------------------------------------------------

@router.post("/daemon/dream")
async def trigger_dream(req: DreamTrigger | None = None):
    """Manually trigger a dream consolidation cycle."""
    from app.core.brain import get_services
    from app.core.dream import DreamConsolidator
    from app.database import AsyncSafeDB, SafeDB

    svc = get_services()
    if not svc.monitor_store:
        raise HTTPException(503, "Monitor store not initialized")

    db = svc.monitor_store.db
    force = req.force if req else False

    if not force:
        # Check cooldown
        row = db.fetchone("SELECT value FROM system_state WHERE key='last_dream_at'")
        if row and row["value"]:
            try:
                last = datetime.fromisoformat(row["value"])
                hours = (datetime.utcnow() - last).total_seconds() / 3600
                if hours < 1:
                    return {"status": "skipped", "reason": f"Last dream was {hours:.1f}h ago (min 1h)"}
            except (ValueError, TypeError):
                pass

    async_db = AsyncSafeDB(db) if isinstance(db, SafeDB) else db
    consolidator = DreamConsolidator(async_db)
    digest = await consolidator.run()
    return {"status": "completed", "digest": digest}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/daemon/status")
async def daemon_status():
    """Get daemon status: last activity, last dream, pending events."""
    from app.core.brain import get_services

    svc = get_services()
    if not svc.monitor_store:
        raise HTTPException(503, "Monitor store not initialized")

    db = svc.monitor_store.db

    last_activity = db.fetchone("SELECT value FROM system_state WHERE key='last_user_activity'")
    last_dream = db.fetchone("SELECT value FROM system_state WHERE key='last_dream_at'")
    pending_events = db.fetchone("SELECT COUNT(*) as c FROM event_queue WHERE status='pending'")
    log_count_24h = db.fetchone(
        "SELECT COUNT(*) as c FROM daemon_log WHERE created_at > ?",
        ((datetime.utcnow() - timedelta(hours=24)).isoformat(),),
    )

    idle_minutes = None
    if last_activity and last_activity["value"]:
        try:
            last = datetime.fromisoformat(last_activity["value"])
            idle_minutes = round((datetime.utcnow() - last).total_seconds() / 60, 1)
        except (ValueError, TypeError):
            pass

    return {
        "last_user_activity": last_activity["value"] if last_activity else None,
        "idle_minutes": idle_minutes,
        "last_dream_at": last_dream["value"] if last_dream else None,
        "pending_events": pending_events["c"] if pending_events else 0,
        "log_entries_24h": log_count_24h["c"] if log_count_24h else 0,
    }
