"""Dossier API — read the knowing tier (2026-08-12).

Read-only surface over `dossiers` + `dossier_revisions`: browse Nova's
standing understanding (domain / entity / storyline / the State-of-the-World
meta capstone) and time-travel prior revisions ("what did Nova understand
about X before this consolidation"). Writes happen ONLY through the
Knowledge Consolidation cycle — there is deliberately no PUT/POST here;
understanding is earned from evidence, not edited by hand.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_auth
from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dossiers"], dependencies=[Depends(require_auth)])

_KINDS = {"domain", "entity", "storyline", "meta"}


@router.get("/dossiers")
async def list_dossiers(kind: str | None = Query(None, max_length=20)):
    """All dossiers (bodies omitted — list stays light). Meta first, then
    freshest understanding first."""
    if kind is not None and kind not in _KINDS:
        raise HTTPException(status_code=422, detail=f"kind must be one of {sorted(_KINDS)}")
    db = get_db()
    where = "WHERE kind = ?" if kind else ""
    rows = db.fetchall(
        f"SELECT id, kind, dkey, title, changed_note, update_count, "
        f"       LENGTH(body) AS body_chars, created_at, updated_at "
        f"FROM dossiers {where} "
        f"ORDER BY CASE kind WHEN 'meta' THEN 0 ELSE 1 END, updated_at DESC",
        (kind,) if kind else (),
    )
    return {"dossiers": [dict(r) for r in rows]}


@router.get("/dossiers/{dossier_id}")
async def get_dossier_detail(dossier_id: int):
    """One dossier with its full body + how many prior revisions exist."""
    db = get_db()
    row = db.fetchone("SELECT * FROM dossiers WHERE id = ?", (dossier_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Dossier not found")
    n_rev = db.fetchone(
        "SELECT COUNT(*) n FROM dossier_revisions WHERE dossier_id = ?", (dossier_id,)
    )["n"]
    out = dict(row)
    out["revision_count"] = n_rev
    return out


@router.get("/dossiers/{dossier_id}/revisions")
async def list_dossier_revisions(dossier_id: int):
    """Revision index (bodies omitted), newest first — the trail of what Nova
    used to understand."""
    db = get_db()
    if db.fetchone("SELECT id FROM dossiers WHERE id = ?", (dossier_id,)) is None:
        raise HTTPException(status_code=404, detail="Dossier not found")
    rows = db.fetchall(
        "SELECT id, valid_from, valid_to, LENGTH(body) AS body_chars "
        "FROM dossier_revisions WHERE dossier_id = ? ORDER BY valid_to DESC",
        (dossier_id,),
    )
    return {"revisions": [dict(r) for r in rows]}


@router.get("/dossiers/{dossier_id}/revisions/{revision_id}")
async def get_dossier_revision(dossier_id: int, revision_id: int):
    """A prior body — time-travel reading of superseded understanding."""
    db = get_db()
    row = db.fetchone(
        "SELECT * FROM dossier_revisions WHERE id = ? AND dossier_id = ?",
        (revision_id, dossier_id),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Revision not found")
    return dict(row)
