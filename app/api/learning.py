"""Learning API — metrics, lessons, skills management, fine-tune readiness.

GET  /api/learning/metrics — Learning system metrics
GET  /api/learning/lessons — List all lessons
DELETE /api/learning/lessons/{id} — Delete a lesson
GET  /api/learning/skills — List all skills
POST /api/learning/skills/{id}/toggle — Enable/disable a skill
DELETE /api/learning/skills/{id} — Delete a skill
GET  /api/learning/finetune/status — Fine-tuning readiness check
POST /api/learning/finetune/trigger — Trigger automated fine-tuning
GET  /api/learning/finetune/history — List past fine-tuning runs
GET  /api/learning/training-data — View training data entries
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_auth
from app.config import config
from app.core.brain import get_services

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learning", tags=["learning"], dependencies=[Depends(require_auth)])


@router.get("/metrics")
async def get_metrics():
    """Get learning system metrics."""
    svc = get_services()
    if not svc.learning:
        return {
            "total_lessons": 0,
            "total_skills": 0,
            "total_corrections": 0,
            "training_examples": 0,
            "last_correction_date": None,
        }
    return svc.learning.get_metrics()


@router.get("/lessons")
async def list_lessons(limit: int = Query(default=100, ge=1, le=500)):
    """List all learned lessons."""
    svc = get_services()
    if not svc.learning:
        return []
    lessons = svc.learning.get_all_lessons(limit=limit)
    return [
        {
            "id": l.id,
            "topic": l.topic,
            "wrong_answer": l.wrong_answer,
            "correct_answer": l.correct_answer,
            "lesson_text": l.lesson_text,
            "confidence": l.confidence,
            "times_retrieved": l.times_retrieved,
            "times_helpful": l.times_helpful,
            "created_at": l.created_at,
        }
        for l in lessons
    ]


@router.delete("/lessons/{lesson_id}")
async def delete_lesson(lesson_id: int):
    """Delete a lesson."""
    svc = get_services()
    if not svc.learning:
        raise HTTPException(status_code=503, detail="Learning engine not initialized")
    deleted = svc.learning.delete_lesson(lesson_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Lesson not found")
    return {"status": "deleted", "lesson_id": lesson_id}


@router.get("/skills")
async def list_skills(limit: int = Query(default=50, ge=1, le=500)):
    """List all learned skills."""
    svc = get_services()
    if not svc.skills:
        return []
    skills = svc.skills.get_all_skills(limit=limit)
    return [
        {
            "id": s.id,
            "name": s.name,
            "trigger_pattern": s.trigger_pattern,
            "steps": s.steps,
            "answer_template": s.answer_template,
            "times_used": s.times_used,
            "success_rate": round(s.success_rate, 3),
            "enabled": s.enabled,
            "created_at": s.created_at,
        }
        for s in skills
    ]


@router.post("/skills/{skill_id}/toggle")
async def toggle_skill(skill_id: int, enabled: bool = True):
    """Enable or disable a skill."""
    svc = get_services()
    if not svc.skills:
        raise HTTPException(status_code=503, detail="Skills not initialized")
    toggled = svc.skills.toggle_skill(skill_id, enabled)
    if not toggled:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"status": "ok", "skill_id": skill_id, "enabled": enabled}


@router.delete("/skills/{skill_id}")
async def delete_skill(skill_id: int):
    """Delete a skill."""
    svc = get_services()
    if not svc.skills:
        raise HTTPException(status_code=503, detail="Skills not initialized")
    deleted = svc.skills.delete_skill(skill_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"status": "deleted", "skill_id": skill_id}


# ---------------------------------------------------------------------------
# Fine-tuning readiness
# ---------------------------------------------------------------------------

@router.get("/finetune/status")
async def finetune_status():
    """Check fine-tuning readiness — training data count, quality, recommendation."""
    path = Path(config.TRAINING_DATA_PATH)

    entries = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        # Validate
                        if entry.get("query", "").strip() and entry.get("chosen", "").strip():
                            entries.append(entry)
                    except json.JSONDecodeError:
                        pass

    total = len(entries)
    valid = sum(1 for e in entries if len(e.get("chosen", "")) > 10)
    has_rejected = sum(1 for e in entries if len(e.get("rejected", "")) > 10)

    min_recommended = 10
    good_count = 50
    ready = total >= min_recommended

    if total == 0:
        recommendation = "No training data yet. Keep correcting Nova to build the dataset."
    elif total < min_recommended:
        recommendation = f"Only {total} pairs. Need at least {min_recommended} for meaningful training. Keep correcting!"
    elif total < good_count:
        recommendation = f"{total} pairs — enough to start. Run: python scripts/finetune.py"
    else:
        recommendation = f"{total} pairs — great dataset! Ready for fine-tuning."

    return {
        "ready": ready,
        "total_pairs": total,
        "valid_pairs": valid,
        "pairs_with_rejected": has_rejected,
        "min_recommended": min_recommended,
        "recommendation": recommendation,
        "data_path": str(path),
        "command": "docker compose stop ollama && python scripts/finetune.py && docker compose start ollama",
    }


# ---------------------------------------------------------------------------
# Fine-tuning automation — trigger + history
# ---------------------------------------------------------------------------

# Track active fine-tuning jobs (in-memory, only one at a time)
_active_finetune_job: dict | None = None
_finetune_lock = asyncio.Lock()


@router.post("/finetune/trigger")
async def finetune_trigger(force: bool = False):
    """Trigger automated fine-tuning pipeline (async, returns job ID).

    The fine-tuning runs as a background task. Check status via
    GET /api/learning/finetune/history for results.

    NOTE: Fine-tuning runs on the HOST (not in Docker) because it
    needs direct GPU access. This endpoint records the trigger and
    returns a job ID. The actual training must be started externally
    via `python scripts/finetune_auto.py` or will be picked up by
    the heartbeat system.
    """
    async with _finetune_lock:
        return await _finetune_trigger_inner(force)


async def _finetune_trigger_inner(force: bool = False):
    global _active_finetune_job

    # Check if a job is already running
    if _active_finetune_job and _active_finetune_job.get("status") == "running":
        return {
            "status": "already_running",
            "job_id": _active_finetune_job["job_id"],
            "message": "A fine-tuning job is already in progress.",
        }

    # Check readiness
    data_path = Path(config.TRAINING_DATA_PATH)
    output_dir = config.FINETUNE_OUTPUT_DIR
    min_pairs = config.FINETUNE_MIN_NEW_PAIRS

    total = 0
    if data_path.exists():
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if entry.get("query", "").strip() and entry.get("chosen", "").strip():
                            total += 1
                    except json.JSONDecodeError:
                        pass

    # Check last training count
    history_path = Path(output_dir) / "run_history.json"
    last_count = 0
    if history_path.exists():
        try:
            with open(history_path, encoding="utf-8") as f:
                history = json.load(f)
            if history:
                last_count = history[-1].get("training_pairs", 0)
        except (json.JSONDecodeError, OSError):
            pass

    new_pairs = total - last_count

    if not force and new_pairs < min_pairs:
        return {
            "status": "not_ready",
            "total_pairs": total,
            "new_pairs": new_pairs,
            "min_required": min_pairs,
            "message": f"Only {new_pairs} new pairs (need {min_pairs}). Use force=true to override.",
        }

    # Create job record
    job_id = str(uuid.uuid4())[:8]
    from datetime import datetime, timezone
    _active_finetune_job = {
        "job_id": job_id,
        "status": "pending",
        "total_pairs": total,
        "new_pairs": new_pairs,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "message": (
            f"Fine-tuning triggered with {total} total pairs ({new_pairs} new). "
            f"Run on host: python scripts/finetune_auto.py"
        ),
    }

    logger.info(
        "[Finetune] Job %s triggered: %d total pairs, %d new",
        job_id, total, new_pairs,
    )

    return {
        "status": "triggered",
        "job_id": job_id,
        "total_pairs": total,
        "new_pairs": new_pairs,
        "message": (
            f"Fine-tuning job {job_id} created. "
            f"Run on host: python scripts/finetune_auto.py"
        ),
        "command": "python scripts/finetune_auto.py" + (" --force" if force else ""),
    }


@router.get("/finetune/history")
async def finetune_history():
    """List past fine-tuning runs with evaluation results."""
    output_dir = config.FINETUNE_OUTPUT_DIR
    history_path = Path(output_dir) / "run_history.json"

    if not history_path.exists():
        return {"runs": [], "total": 0}

    try:
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"runs": [], "total": 0}

    # Return most recent first, with summary info
    runs = []
    for run in reversed(history):
        entry = {
            "started_at": run.get("started_at"),
            "completed_at": run.get("completed_at"),
            "status": run.get("status"),
            "training_pairs": run.get("training_pairs", 0),
            "new_pairs": run.get("new_pairs", 0),
            "base_model": run.get("base_model"),
            "ft_model": run.get("ft_model"),
        }
        if run.get("eval_results"):
            er = run["eval_results"]
            entry["eval"] = {
                "win_rate": er.get("win_rate"),
                "avg_preference": er.get("avg_preference"),
                "candidate_wins": er.get("candidate_wins"),
                "base_wins": er.get("base_wins"),
                "candidate_is_better": er.get("candidate_is_better"),
            }
        if run.get("reason"):
            entry["reason"] = run["reason"]
        runs.append(entry)

    return {"runs": runs, "total": len(runs)}


@router.get("/training-data/stats")
async def training_data_stats():
    """Return statistics about the collected training data."""
    path = Path(config.TRAINING_DATA_PATH)
    if not path.exists():
        return {"total_pairs": 0, "valid_pairs": 0, "topics": [], "date_range": None, "avg_chosen_length": 0, "avg_rejected_length": 0}

    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not entries:
        return {"total_pairs": 0, "valid_pairs": 0, "topics": [], "date_range": None, "avg_chosen_length": 0, "avg_rejected_length": 0}

    valid = [e for e in entries if e.get("query", "").strip() and e.get("chosen", "").strip()]
    timestamps = [e.get("timestamp", "") for e in entries if e.get("timestamp")]
    topics = list({e.get("query", "")[:80] for e in valid})[:20]

    chosen_lengths = [len(e.get("chosen", "")) for e in valid]
    rejected_lengths = [len(e.get("rejected", "")) for e in valid if e.get("rejected")]

    return {
        "total_pairs": len(entries),
        "valid_pairs": len(valid),
        "topics": topics,
        "date_range": {
            "earliest": min(timestamps) if timestamps else None,
            "latest": max(timestamps) if timestamps else None,
        },
        "avg_chosen_length": round(sum(chosen_lengths) / len(chosen_lengths)) if chosen_lengths else 0,
        "avg_rejected_length": round(sum(rejected_lengths) / len(rejected_lengths)) if rejected_lengths else 0,
    }


@router.post("/training-data/export")
async def export_training_data():
    """Export training data in DPO format ready for fine-tuning frameworks (Unsloth/TRL)."""
    from app.core.prompt import IDENTITY_AND_REASONING

    path = Path(config.TRAINING_DATA_PATH)
    if not path.exists():
        return {"format": "dpo", "count": 0, "data": []}

    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if entry.get("query", "").strip() and entry.get("chosen", "").strip():
                        entries.append(entry)
                except json.JSONDecodeError:
                    pass

    # Truncate system prompt for context (first 500 chars)
    system_context = IDENTITY_AND_REASONING[:500].strip()

    data = []
    for e in entries:
        data.append({
            "system": system_context,
            "prompt": e.get("query", ""),
            "chosen": e.get("chosen", ""),
            "rejected": e.get("rejected", ""),
            "timestamp": e.get("timestamp", ""),
        })

    return {"format": "dpo", "count": len(data), "data": data}


@router.get("/reflexions")
async def list_reflexions(limit: int = Query(default=50, ge=1, le=200)):
    """List recent reflexion entries."""
    svc = get_services()
    if not svc.reflexions:
        return []
    reflexions = svc.reflexions.get_recent(limit=limit)
    return [
        {
            "id": r.id,
            "task_summary": r.task_summary,
            "outcome": r.outcome,
            "reflection": r.reflection,
            "quality_score": round(r.quality_score, 3),
            "tools_used": r.tools_used,
            "created_at": r.created_at,
        }
        for r in reflexions
    ]


@router.get("/training-data")
async def list_training_data(limit: int = Query(default=50, ge=1, le=500)):
    """View training data entries."""
    path = Path(config.TRAINING_DATA_PATH)
    if not path.exists():
        return []

    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entries.append({
                        "query": entry.get("query", "")[:200],
                        "chosen": entry.get("chosen", "")[:200],
                        "rejected": entry.get("rejected", "")[:200],
                        "timestamp": entry.get("timestamp", ""),
                    })
                except json.JSONDecodeError:
                    pass

    return entries[-limit:]  # Return most recent
