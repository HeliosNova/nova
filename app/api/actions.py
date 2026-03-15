"""Action log API — audit trail for all action tool executions."""

from __future__ import annotations

import re
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_auth
from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["actions"], dependencies=[Depends(require_auth)])


def _row_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "action_type": row["action_type"],
        "params": row["params"],
        "result": row["result"],
        "success": bool(row["success"]),
        "created_at": row["created_at"],
    }


@router.get("/actions")
async def list_actions(
    hours: int = Query(default=24, ge=1, le=720),
    limit: int = Query(default=50, ge=1, le=500),
    action_type: str = Query(default="", max_length=50),
):
    """List recent actions from the audit log."""
    # Resolve Query defaults for direct calls outside FastAPI
    if not isinstance(hours, int):
        hours = 24
    if not isinstance(limit, int):
        limit = 50
    if not isinstance(action_type, str):
        action_type = ""
    # Validate action_type contains only safe characters (used in SQL query)
    if action_type and not re.match(r"^[a-zA-Z0-9_\-]{1,50}$", action_type):
        raise HTTPException(status_code=400, detail="Invalid action_type format")
    db = get_db()
    if action_type:
        rows = db.fetchall(
            "SELECT * FROM action_log WHERE action_type = ? AND created_at > datetime('now', ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (action_type, f"-{hours} hours", limit),
        )
    else:
        rows = db.fetchall(
            "SELECT * FROM action_log WHERE created_at > datetime('now', ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (f"-{hours} hours", limit),
        )
    return {"actions": [_row_to_dict(r) for r in rows], "count": len(rows)}


@router.get("/actions/{action_id}")
async def get_action(action_id: int):
    """Get a specific action log entry."""
    db = get_db()
    row = db.fetchone("SELECT * FROM action_log WHERE id = ?", (action_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Action not found")
    return _row_to_dict(row)
