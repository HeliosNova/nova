"""Event trigger API — external webhook endpoint for pushing events."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.auth import require_auth
from app.database import get_db

logger = logging.getLogger(__name__)
# require_auth: this endpoint queues events that background processing acts on —
# it was the only mutating router without auth (audit 2026-07-06). No-ops when
# REQUIRE_AUTH=false, same as every other router.
router = APIRouter(prefix="/events", tags=["events"], dependencies=[Depends(require_auth)])

_EVENT_TYPE_RE = re.compile(r"^[a-zA-Z0-9_]+:[a-zA-Z0-9_.]+$")


class TriggerEventRequest(BaseModel):
    event_type: str
    payload: dict = {}
    priority: float = 0.5


@router.post("/trigger")
async def trigger_event(req: TriggerEventRequest):
    """Push an event into the event queue (for external webhooks).

    Event type must be in namespace:name format (e.g. "webhook:github_push").
    """
    if not _EVENT_TYPE_RE.match(req.event_type):
        raise HTTPException(
            400,
            "event_type must be in 'namespace:name' format "
            "(alphanumeric + underscores, e.g. 'webhook:github_push')",
        )

    if not 0.0 <= req.priority <= 1.0:
        raise HTTPException(400, "priority must be between 0.0 and 1.0")

    db = get_db()
    # to_thread: writer-lock DB call — blocks every coroutine if run on the
    # event-loop thread while the lock is contended (54h-freeze bug class).
    await asyncio.to_thread(
        db.execute,
        "INSERT INTO event_queue (event_type, payload, priority) VALUES (?, ?, ?)",
        (req.event_type, json.dumps(req.payload), req.priority),
    )

    logger.info("Event queued: %s (priority=%.1f)", req.event_type, req.priority)
    return {"status": "queued", "event_type": req.event_type}


@router.get("/pending")
async def list_pending_events(limit: int = Query(20, ge=1, le=100)):
    """List pending events in the queue."""
    db = get_db()
    rows = db.fetchall(
        "SELECT id, event_type, payload, priority, status, created_at "
        "FROM event_queue WHERE status = 'pending' "
        "ORDER BY priority DESC, created_at ASC LIMIT ?",
        (limit,),
    )
    return {"count": len(rows), "events": [dict(r) for r in (rows or [])]}


@router.get("/recent")
async def list_recent_events(limit: int = Query(50, ge=1, le=200)):
    """List recently processed events."""
    db = get_db()
    rows = db.fetchall(
        "SELECT id, event_type, payload, priority, status, processed_at, created_at "
        "FROM event_queue ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    return {"count": len(rows), "events": [dict(r) for r in (rows or [])]}
