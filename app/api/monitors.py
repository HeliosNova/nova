"""Monitor API — CRUD for monitors + manual trigger."""

from __future__ import annotations

import re
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.auth import require_auth

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitors"], dependencies=[Depends(require_auth)])


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


@router.post("/monitors/{monitor_id}/trigger")
async def trigger_monitor(monitor_id: int):
    store = _get_store()
    monitor = store.get(monitor_id)
    if not monitor:
        raise HTTPException(status_code=404, detail="Monitor not found")
    heartbeat = _get_heartbeat()
    result = await heartbeat.trigger_monitor(monitor_id)
    return result


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
    row = db.fetchone("SELECT monitor_id FROM monitor_results WHERE id = ?", (result_id,))
    adapted = None
    if row:
        adapted = store.adapt_cooldown(row["monitor_id"])

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
