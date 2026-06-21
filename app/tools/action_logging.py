"""Shared action logging utility for tool modules."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.database import get_db

logger = logging.getLogger(__name__)

# Field names whose values should be masked before logging
_SENSITIVE_FIELD_RE = re.compile(
    r"^(password|token|secret|key|api_key|access_token|client_secret)$",
    re.IGNORECASE,
)


def _mask_sensitive_fields(params: dict) -> dict:
    """Return a copy of params with sensitive field values replaced by '***'."""
    masked = {}
    for k, v in params.items():
        if _SENSITIVE_FIELD_RE.match(k):
            masked[k] = "***"
        elif isinstance(v, dict):
            masked[k] = _mask_sensitive_fields(v)
        else:
            masked[k] = v
    return masked


def log_action(action_type: str, params: dict, result: str, success: bool) -> None:
    """Log an action to the action_log table. Sensitive fields are masked.

    Called from the async tool-execution path (the post-execute audit hook
    fires on EVERY tool call). The INSERT is a synchronous SQLite write that
    blocks the event loop until commit — under disk stress that commit's
    fsync can hang, freezing the whole loop (2026-06-12 production incident:
    a wedged WSL2 write path stalled here and took the API down). So when a
    loop is running we snapshot the row inline and hand the write to the
    default executor fire-and-forget; off-loop callers (if any) write inline.
    Audit-log ordering is best-effort, which is fine for an audit trail.
    """
    try:
        db = get_db()
        safe_params = _mask_sensitive_fields(params)
        row = (
            action_type,
            json.dumps(safe_params, default=str),
            result[:2000],
            1 if success else 0,
        )
    except Exception as e:
        logger.warning("Failed to prepare action log: %s", e)
        return

    def _write() -> None:
        try:
            db.execute(
                "INSERT INTO action_log (action_type, params, result, success) VALUES (?, ?, ?, ?)",
                row,
            )
        except Exception as e:
            logger.warning("Failed to log action: %s", e)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _write()
        return
    loop.run_in_executor(None, _write)
