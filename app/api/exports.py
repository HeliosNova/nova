"""Export/Import API — lessons, KG facts, skills, bundles with HMAC signing."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_auth
from app.database import get_db

logger = logging.getLogger(__name__)
# require_auth: exports the ENTIRE knowledge base (lessons/KG/skills) and accepts
# imports — was unauthenticated (audit 2026-07-06). No-ops when REQUIRE_AUTH=false.
router = APIRouter(prefix="/exports", tags=["exports"], dependencies=[Depends(require_auth)])


class ExportRequest(BaseModel):
    key_path: str | None = None


class ImportLessonRequest(BaseModel):
    data: dict
    verify_key_path: str | None = None


class ImportKGFactRequest(BaseModel):
    data: dict
    verify_key_path: str | None = None


class ImportBundleRequest(BaseModel):
    data: dict
    verify_key_path: str | None = None


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------

@router.post("/lessons")
async def export_lessons(req: ExportRequest = ExportRequest()):
    """Export all lessons, optionally signed."""
    from app.core.data_export import export_all_lessons
    db = get_db()
    lessons = await asyncio.to_thread(export_all_lessons, db, req.key_path)
    return {"count": len(lessons), "lessons": lessons}


@router.post("/kg-facts")
async def export_kg_facts(req: ExportRequest = ExportRequest()):
    """Export all current KG facts, optionally signed."""
    from app.core.data_export import export_all_kg_facts
    db = get_db()
    facts = await asyncio.to_thread(export_all_kg_facts, db, req.key_path)
    return {"count": len(facts), "kg_facts": facts}


@router.post("/bundle")
async def export_bundle(req: ExportRequest = ExportRequest()):
    """Export everything (skills + lessons + KG facts) in one signed envelope."""
    from app.core.data_export import export_bundle as _export_bundle
    db = get_db()
    bundle = await asyncio.to_thread(_export_bundle, db, req.key_path)
    return bundle


# ---------------------------------------------------------------------------
# Import endpoints
# ---------------------------------------------------------------------------

@router.post("/import/lessons")
async def import_lessons(req: ImportLessonRequest):
    """Import a lesson dict with optional signature verification."""
    from app.core.data_export import import_lesson, SignatureError
    db = get_db()
    try:
        result = await asyncio.to_thread(import_lesson, req.data, db, req.verify_key_path)
        if result > 0:
            return {"status": "imported", "lesson_id": result}
        return {"status": "skipped", "reason": "duplicate"}
    except SignatureError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/import/kg-facts")
async def import_kg_fact(req: ImportKGFactRequest):
    """Import a KG fact dict with optional signature verification."""
    from app.core.data_export import import_kg_fact as _import_kg_fact, SignatureError
    db = get_db()
    try:
        result = await asyncio.to_thread(_import_kg_fact, req.data, db, req.verify_key_path)
        if result > 0:
            return {"status": "imported", "fact_id": result}
        return {"status": "skipped", "reason": "duplicate"}
    except SignatureError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/import/bundle")
async def import_bundle(req: ImportBundleRequest):
    """Import a full bundle (skills + lessons + KG facts)."""
    from app.core.data_export import import_bundle as _import_bundle, SignatureError
    db = get_db()
    try:
        counts = await asyncio.to_thread(_import_bundle, req.data, db, req.verify_key_path)
        return {"status": "imported", "counts": counts}
    except SignatureError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
