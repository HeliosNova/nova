"""Monitor API — CRUD for monitors + manual trigger."""

from __future__ import annotations

import asyncio
import re
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.auth import require_auth
from app.config import config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitors"], dependencies=[Depends(require_auth)])

# Serialize on-demand eval runs. Each run fires real brain.think() generations on
# the single GPU; without this a click-storm or overlapping POSTs would queue
# dozens of runs ahead of live chat (multi-minute TTFT). A plain flag (not a
# Lock) so the check-and-set is atomic on the single-threaded event loop — no
# await between the read and the set, so two racing requests can't both pass
# (a Lock + .locked() pre-check has that race). The finetune path has its own
# lock; eval had none (audit 2026-08-22).
_eval_running = {"active": False}


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

_USER_CHECK_TYPES = {"url", "search", "command", "query"}
_ALL_CHECK_TYPES = {"url", "search", "command", "query", "system_health", "quiz", "skill_validation", "kg_curate", "curiosity", "finetune_check", "auto_monitor"}
_VALID_NOTIFY_CONDITIONS = {"on_change", "always", "on_error", "on_threshold"}


class MonitorCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    check_type: str = Field("search", max_length=50)
    check_config: dict = {}
    schedule_seconds: int = Field(300, ge=10, le=604_800)  # 10s to 7 days
    cooldown_minutes: int = Field(60, ge=0, le=10_080)      # 0 to 7 days
    notify_condition: str = Field("on_change", max_length=50)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9 _\-:.()]{1,200}$", v):
            raise ValueError("Monitor name contains invalid characters")
        return v.strip()

    @field_validator("check_type")
    @classmethod
    def validate_check_type(cls, v: str) -> str:
        if v not in _USER_CHECK_TYPES:
            raise ValueError(f"check_type must be one of {_USER_CHECK_TYPES}")
        return v

    @field_validator("notify_condition")
    @classmethod
    def validate_notify_condition(cls, v: str) -> str:
        if v not in _VALID_NOTIFY_CONDITIONS:
            raise ValueError(f"notify_condition must be one of {_VALID_NOTIFY_CONDITIONS}")
        return v


class MonitorUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    check_type: str | None = Field(None, max_length=50)
    check_config: dict | None = None
    schedule_seconds: int | None = Field(None, ge=10, le=604_800)
    cooldown_minutes: int | None = Field(None, ge=0, le=10_080)
    notify_condition: str | None = Field(None, max_length=50)
    enabled: bool | None = None
    # CSV of channel names: "discord,signal" → only those channels.
    # Empty string clears override (falls back to category default).
    channels: str | None = Field(None, max_length=200)

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: str | None) -> str | None:
        if v is None:
            return v
        valid = {"discord", "telegram", "whatsapp", "signal"}
        if v.strip() == "":
            return ""  # explicit empty = clear override
        parts = [p.strip().lower() for p in v.split(",") if p.strip()]
        bad = [p for p in parts if p not in valid]
        if bad:
            raise ValueError(f"Unknown channel(s) {bad}. Valid: {sorted(valid)}")
        return ",".join(parts)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.match(r"^[a-zA-Z0-9 _\-:.()]{1,200}$", v):
            raise ValueError("Monitor name contains invalid characters")
        return v.strip()

    @field_validator("check_type")
    @classmethod
    def validate_check_type(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v not in _USER_CHECK_TYPES:
            raise ValueError(f"check_type must be one of {_USER_CHECK_TYPES}")
        return v

    @field_validator("notify_condition")
    @classmethod
    def validate_notify_condition(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v not in _VALID_NOTIFY_CONDITIONS:
            raise ValueError(f"notify_condition must be one of {_VALID_NOTIFY_CONDITIONS}")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Allowed keys per check_type for check_config validation
_ALLOWED_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "search": frozenset({"query"}),
    "url": frozenset({"url", "match"}),
    "command": frozenset({"command"}),
    "api": frozenset({"url", "method", "headers"}),
    "query": frozenset({"query"}),
}


def _validate_check_config(check_type: str, check_config: dict) -> str | None:
    """Validate check_config keys against whitelist for the given check_type.

    Returns an error message string if invalid, None if valid.
    """
    allowed = _ALLOWED_CONFIG_KEYS.get(check_type)
    if allowed is None:
        # Internal check types (system_health, quiz, etc.) — no user-facing validation
        return None
    unknown = set(check_config.keys()) - allowed
    if unknown:
        return f"Unknown check_config keys for '{check_type}': {', '.join(sorted(unknown))}. Allowed: {', '.join(sorted(allowed))}"
    return None


def _get_store():
    """Get the MonitorStore from services."""
    from app.core.brain import get_services
    svc = get_services()
    if not hasattr(svc, "monitor_store") or svc.monitor_store is None:
        raise HTTPException(status_code=503, detail="Monitor system not initialized")
    return svc.monitor_store


def _get_heartbeat():
    """Get the HeartbeatLoop from services."""
    from app.core.brain import get_services
    svc = get_services()
    if not hasattr(svc, "heartbeat") or svc.heartbeat is None:
        raise HTTPException(status_code=503, detail="Heartbeat system not initialized")
    return svc.heartbeat


def _monitor_to_dict(m) -> dict:
    return {
        "id": m.id,
        "name": m.name,
        "check_type": m.check_type,
        "check_config": m.check_config,
        "schedule_seconds": m.schedule_seconds,
        "enabled": m.enabled,
        "cooldown_minutes": m.cooldown_minutes,
        "notify_condition": m.notify_condition,
        "last_check_at": m.last_check_at,
        "last_alert_at": m.last_alert_at,
        "last_result": m.last_result,
        "created_at": m.created_at,
        "category": getattr(m, "category", "content"),
        "channels": getattr(m, "channels", None),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/monitors")
async def list_monitors():
    store = _get_store()
    monitors = store.list_all()
    return {"monitors": [_monitor_to_dict(m) for m in monitors], "count": len(monitors)}


@router.post("/monitors", status_code=201)
async def create_monitor(body: MonitorCreate):
    # Validate check_config keys against whitelist
    config_err = _validate_check_config(body.check_type, body.check_config)
    if config_err:
        raise HTTPException(status_code=422, detail=config_err)
    store = _get_store()
    monitor_id = store.create(
        name=body.name,
        check_type=body.check_type,
        check_config=body.check_config,
        schedule_seconds=body.schedule_seconds,
        cooldown_minutes=body.cooldown_minutes,
        notify_condition=body.notify_condition,
    )
    if monitor_id < 0:
        raise HTTPException(status_code=409, detail=f"Monitor '{body.name}' already exists or creation failed")
    monitor = store.get(monitor_id)
    return _monitor_to_dict(monitor)


# IMPORTANT: literal path /monitors/results/recent must be registered BEFORE
# the parameterized /monitors/{monitor_id} to avoid FastAPI matching "results" as an ID.
@router.get("/monitors/results/recent")
async def recent_results(hours: int = Query(default=24, ge=1, le=720), limit: int = Query(default=50, ge=1, le=500)):
    store = _get_store()
    results = store.get_recent_results(hours=hours, limit=limit)
    return {
        "results": [
            {
                "id": r.id,
                "monitor_id": r.monitor_id,
                "status": r.status,
                "value": r.value,
                "message": r.message,
                "created_at": r.created_at,
                "user_rating": r.user_rating,
            }
            for r in results
        ],
        "count": len(results),
    }


@router.get("/monitors/results/search")
async def search_results(
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=8, ge=1, le=50),
):
    """Search monitor results by text content."""
    store = _get_store()
    db = store.db
    rows = db.fetchall(
        "SELECT mr.id, mr.monitor_id, m.name as monitor_name, mr.status, "
        "mr.message, mr.value, mr.created_at "
        "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
        "WHERE mr.message LIKE ? OR mr.value LIKE ? "
        "ORDER BY mr.created_at DESC LIMIT ?",
        (f"%{q}%", f"%{q}%", limit),
    )
    return [
        {
            "id": r["id"],
            "monitor_name": r["monitor_name"],
            "status": r["status"],
            "content": (r["message"] or r["value"] or "")[:500],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@router.get("/monitors/{monitor_id}")
async def get_monitor(monitor_id: int):
    store = _get_store()
    monitor = store.get(monitor_id)
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")
    results = store.get_results(monitor_id, limit=20)
    return {
        **_monitor_to_dict(monitor),
        "results": [
            {
                "id": r.id,
                "status": r.status,
                "value": r.value,
                "message": r.message,
                "created_at": r.created_at,
            }
            for r in results
        ],
    }


@router.put("/monitors/{monitor_id}")
async def update_monitor(monitor_id: int, body: MonitorUpdate):
    store = _get_store()
    monitor = store.get(monitor_id)
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")

    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Validate check_config keys if both check_type and check_config are provided
    check_type = updates.get("check_type", monitor.check_type)
    if "check_config" in updates:
        config_err = _validate_check_config(check_type, updates["check_config"])
        if config_err:
            raise HTTPException(status_code=422, detail=config_err)

    store.update(monitor_id, **updates)
    return _monitor_to_dict(store.get(monitor_id))


@router.delete("/monitors/{monitor_id}")
async def delete_monitor(monitor_id: int):
    store = _get_store()
    monitor = store.get(monitor_id)
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")
    store.delete(monitor_id)
    return {"deleted": True, "id": monitor_id, "name": monitor.name}


# Convenience endpoints for per-monitor channel routing
@router.get("/monitors/{monitor_id}/channels")
async def get_monitor_channels(monitor_id: int):
    store = _get_store()
    monitor = store.get(monitor_id)
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")
    return {
        "id": monitor_id,
        "name": monitor.name,
        "category": monitor.category,
        "channels": monitor.channels,  # None or CSV
        "effective": (
            [c.strip() for c in monitor.channels.split(",") if c.strip()]
            if monitor.channels
            else (["telegram"] if monitor.category == "system" else ["discord", "telegram", "whatsapp", "signal"])
        ),
    }


@router.put("/monitors/{monitor_id}/channels")
async def set_monitor_channels(monitor_id: int, body: MonitorUpdate):
    """Set per-monitor channel routing. Body: {"channels": "discord,signal"} or empty string to clear."""
    if body.channels is None:
        raise HTTPException(status_code=400, detail="Body must include 'channels' field")
    store = _get_store()
    monitor = store.get(monitor_id)
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")
    # Empty string → clear override (revert to category default)
    new_channels = body.channels if body.channels else None
    store.update(monitor_id, channels=new_channels)
    refreshed = store.get(monitor_id)
    return _monitor_to_dict(refreshed)


@router.put("/monitors/channels/bulk")
async def bulk_set_channels(body: dict):
    """Set channels for many monitors at once.

    Body: {"updates": [{"id": 5, "channels": "signal"}, {"id": 35, "channels": "discord"}]}
    Or by-pattern: {"name_match": "Domain Study:", "channels": "discord"}
    """
    store = _get_store()
    valid = {"discord", "telegram", "whatsapp", "signal"}
    updated = []
    if "updates" in body:
        for u in body["updates"]:
            mid = u.get("id")
            ch = u.get("channels")
            if mid is None or ch is None:
                continue
            ch = ch.strip()
            if ch:
                parts = [p.strip().lower() for p in ch.split(",") if p.strip()]
                if any(p not in valid for p in parts):
                    raise HTTPException(status_code=422, detail=f"Invalid channel in: {ch}")
                ch = ",".join(parts)
            else:
                ch = None
            if store.get(mid):
                store.update(mid, channels=ch)
                updated.append(mid)
    elif "name_match" in body and "channels" in body:
        ch = body["channels"].strip()
        if ch:
            parts = [p.strip().lower() for p in ch.split(",") if p.strip()]
            if any(p not in valid for p in parts):
                raise HTTPException(status_code=422, detail=f"Invalid channel in: {ch}")
            ch = ",".join(parts)
        else:
            ch = None
        pattern = body["name_match"]
        from app.database import get_db
        db = get_db()
        rows = db.fetchall(
            "SELECT id FROM monitors WHERE name LIKE ?",
            (f"%{pattern}%",),
        )
        for row in rows:
            store.update(row["id"], channels=ch)
            updated.append(row["id"])
    else:
        raise HTTPException(status_code=400, detail="Body must have 'updates' or 'name_match'+'channels'")
    return {"updated_count": len(updated), "ids": updated}


@router.post("/monitors/{monitor_id}/trigger")
async def trigger_monitor(monitor_id: int):
    store = _get_store()
    monitor = store.get(monitor_id)
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")
    heartbeat = _get_heartbeat()
    result = await heartbeat.trigger_monitor(monitor_id)
    return result


@router.post("/eval/run")
async def run_eval_categories(categories: str = Query("memory-learning,kg-retrieval")):
    """Run selected eval categories on demand, in-process (full service fidelity).

    Faster than triggering the full Quality Eval Harness monitor when you only
    need the causal-fix categories. Returns per-category causal_fix_rate plus
    per-task before/after correctness.
    """
    from app.monitors.eval_harness import EvalHarness
    cats = {c.strip() for c in categories.split(",") if c.strip()}
    harness = EvalHarness()
    tasks = [t for t in harness.load_suite() if t.category in cats]
    if not tasks:
        raise HTTPException(status_code=400, detail=f"No tasks for categories {cats}")
    if _eval_running["active"]:
        raise HTTPException(
            status_code=409,
            detail="An eval run is already in progress; retry once it finishes.",
        )
    _eval_running["active"] = True
    try:
        report = await harness.run_all(tasks)
    finally:
        _eval_running["active"] = False
    return {
        "categories": {
            cat: {
                "pass_rate": cm.pass_rate,
                "causal_fix_rate": cm.memory_causal_fix_rate,
                "testable": cm.memory_testable,
                "total": cm.total,
            }
            for cat, cm in (report.categories or {}).items()
        },
        "tasks": [
            {"id": r.task_id, "passed": r.passed,
             "before_correct": r.memory_before_correct,
             "after_correct": r.memory_after_correct}
            for r in report.task_results
        ],
    }


# Internal QA/meta annotations that get appended to storyline_events but are
# NOT narrative events — hide them from the user-facing timeline (2026-08-20
# visual pass; screenshots showed a "Fresh-check could NOT be confirmed" note
# rendered as a story beat). Definition moved to core/storylines.py
# 2026-08-31 so dossier consolidation applies the same exclusion.
from app.core.storylines import EVENT_META_EXCL_SQL as _EVENT_META_EXCL


@router.get("/storylines")
async def list_storylines(
    status: str = Query(default="active", pattern="^(active|closed|all)$"),
    limit: int = Query(default=40, ge=1, le=200),
):
    """Tracked narrative threads (2026-08-20): ongoing multi-update stories Nova
    maintains across monitors — 'where does X stand'. Had no UI despite being a
    headline capability. List carries each thread's summary + event count;
    detail (/{id}) carries the full event timeline."""
    from app.database import get_db
    db = get_db()

    def _sync():
        where = "" if status == "all" else "WHERE s.status = ?"
        params = () if status == "all" else (status,)
        rows = db.fetchall(
            f"SELECT s.id, s.title, s.status, s.summary, s.monitors_csv, "
            f"       s.first_seen, s.last_updated, s.update_count, "
            f"       (SELECT COUNT(*) FROM storyline_events e WHERE e.storyline_id = s.id "
            f"        {_EVENT_META_EXCL}) AS event_count "
            f"FROM storylines s {where} "
            f"ORDER BY CASE s.status WHEN 'active' THEN 0 ELSE 1 END, s.last_updated DESC "
            f"LIMIT ?", (*params, limit))
        counts = {r["status"]: r["c"] for r in db.fetchall(
            "SELECT status, COUNT(*) c FROM storylines GROUP BY status")}
        out = []
        for r in rows:
            d = dict(r)
            d["monitors"] = [m.strip() for m in (r["monitors_csv"] or "").split(",") if m.strip()]
            d.pop("monitors_csv", None)
            out.append(d)
        return {"stats": {"active": counts.get("active", 0), "closed": counts.get("closed", 0)},
                "storylines": out}

    return await asyncio.to_thread(_sync)


@router.get("/storylines/{storyline_id}")
async def get_storyline(storyline_id: int):
    """One storyline with its full event timeline (newest first)."""
    from app.database import get_db
    db = get_db()

    def _sync():
        s = db.fetchone("SELECT * FROM storylines WHERE id = ?", (storyline_id,))
        if s is None:
            return None
        events = [
            {"id": e["id"], "summary": e["summary"], "source": e["source_monitor"],
             "url": e["item_url"] or None, "is_new": bool(e["is_new"]),
             "published": e["published"], "created_at": e["created_at"]}
            for e in db.fetchall(
                f"SELECT * FROM storyline_events WHERE storyline_id = ? "
                f"{_EVENT_META_EXCL} ORDER BY created_at DESC LIMIT 60", (storyline_id,))
        ]
        d = dict(s)
        d["monitors"] = [m.strip() for m in (s["monitors_csv"] or "").split(",") if m.strip()]
        d.pop("monitors_csv", None)
        d["events"] = events
        return d

    result = await asyncio.to_thread(_sync)
    if result is None:
        raise HTTPException(status_code=404, detail="Storyline not found")
    return result


@router.get("/forecasts")
async def list_forecasts(limit: int = Query(default=40, ge=1, le=200)):
    """Nova's self-grading forecasts (2026-08-20): open predictions + recently
    graded ones, with the calibration record. This apparatus was invisible in
    the UI — Nova mints falsifiable predictions from consolidation and grades
    them at resolution, but nothing surfaced them."""
    from app.database import get_db
    db = get_db()

    def _sync():
        counts = {r["status"]: r["c"] for r in db.fetchall(
            "SELECT status, COUNT(*) c FROM forecasts GROUP BY status")}
        n_hit, n_miss = counts.get("hit", 0), counts.get("miss", 0)
        graded = n_hit + n_miss
        # Brier-style skill would need per-forecast outcomes; the resolved
        # hit-rate is the honest headline number here.
        accuracy = (n_hit / graded) if graded else None

        def _row(r):
            return {"id": r["id"], "claim": r["claim"], "confidence": r["confidence"],
                    "status": r["status"], "created_at": r["created_at"],
                    "resolves_at": r["resolves_at"], "resolved_at": r["resolved_at"],
                    "resolution": r["resolution"], "source": r["source_monitor"]}

        open_rows = [_row(r) for r in db.fetchall(
            "SELECT * FROM forecasts WHERE status='open' "
            "ORDER BY resolves_at ASC LIMIT ?", (limit,))]
        resolved_rows = [_row(r) for r in db.fetchall(
            "SELECT * FROM forecasts WHERE status IN ('hit','miss') "
            "ORDER BY resolved_at DESC LIMIT ?", (limit,))]
        return {
            "stats": {"open": counts.get("open", 0), "hit": n_hit, "miss": n_miss,
                      "graded": graded, "accuracy": accuracy},
            "open": open_rows,
            "resolved": resolved_rows,
        }

    return await asyncio.to_thread(_sync)


class InstructionCreate(BaseModel):
    instruction: str = Field(min_length=1, max_length=5_000)
    schedule_seconds: int = Field(3600, ge=60, le=604_800)
    notify_channels: str = Field("discord,telegram", max_length=200)


class InstructionUpdate(BaseModel):
    instruction: str | None = Field(None, min_length=1, max_length=5_000)
    schedule_seconds: int | None = Field(None, ge=60, le=604_800)
    enabled: bool | None = None
    notify_channels: str | None = Field(None, max_length=200)


class RatingBody(BaseModel):
    rating: int  # -1, 0, or 1


@router.post("/monitors/results/{result_id}/rate")
async def rate_result(result_id: int, body: RatingBody):
    if body.rating not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="Rating must be -1, 0, or 1")
    store = _get_store()
    ok = store.rate_result(result_id, body.rating)
    if not ok:
        raise HTTPException(status_code=404, detail="Result not found or invalid rating")

    # Check for auto-adaptation
    # Find the monitor_id for this result
    from app.database import get_db
    db = get_db()
    row = db.fetchone("SELECT monitor_id, value FROM monitor_results WHERE id = ?", (result_id,))
    adapted = None
    if row:
        adapted = store.adapt_cooldown(row["monitor_id"])
        # Salience learning: nudge this content's topic weights from the rating
        # (closes the loop the rating button opens — see app/core/salience.py).
        if config.ENABLE_SALIENCE_FILTER and body.rating in (-1, 1) and row["value"]:
            try:
                from app.core.salience import learn_from_rating
                learn_from_rating(db, row["value"], body.rating)
            except Exception:
                pass

    return {
        "rated": True,
        "result_id": result_id,
        "rating": body.rating,
        "cooldown_adapted": adapted,
    }


# ---------------------------------------------------------------------------
# Heartbeat Instructions CRUD
# ---------------------------------------------------------------------------

def _instruction_to_dict(inst) -> dict:
    return {
        "id": inst.id,
        "instruction": inst.instruction,
        "schedule_seconds": inst.schedule_seconds,
        "enabled": inst.enabled,
        "last_run_at": inst.last_run_at,
        "notify_channels": inst.notify_channels,
        "created_at": inst.created_at,
    }


@router.get("/heartbeat/instructions")
async def list_instructions():
    store = _get_store()
    instructions = store.list_instructions()
    return {"instructions": [_instruction_to_dict(i) for i in instructions], "count": len(instructions)}


@router.post("/heartbeat/instructions", status_code=201)
async def create_instruction(body: InstructionCreate):
    store = _get_store()
    inst_id = store.create_instruction(
        instruction=body.instruction,
        schedule_seconds=body.schedule_seconds,
        notify_channels=body.notify_channels,
    )
    inst = store.get_instruction(inst_id)
    return _instruction_to_dict(inst)


@router.get("/heartbeat/instructions/{instruction_id}")
async def get_instruction(instruction_id: int):
    store = _get_store()
    inst = store.get_instruction(instruction_id)
    if not inst:
        raise HTTPException(status_code=404, detail="Instruction not found")
    return _instruction_to_dict(inst)


@router.put("/heartbeat/instructions/{instruction_id}")
async def update_instruction(instruction_id: int, body: InstructionUpdate):
    store = _get_store()
    inst = store.get_instruction(instruction_id)
    if not inst:
        raise HTTPException(status_code=404, detail="Instruction not found")
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    store.update_instruction(instruction_id, **updates)
    return _instruction_to_dict(store.get_instruction(instruction_id))


@router.delete("/heartbeat/instructions/{instruction_id}")
async def delete_instruction(instruction_id: int):
    store = _get_store()
    inst = store.get_instruction(instruction_id)
    if not inst:
        raise HTTPException(status_code=404, detail="Instruction not found")
    store.delete_instruction(instruction_id)
    return {"deleted": True, "id": instruction_id}
