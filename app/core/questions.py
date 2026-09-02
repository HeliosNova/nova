"""Open-questions ledger + belief revisions (2026-09-02, Phase 4.4).

A dossier's ``## Open questions`` section is Nova's stated epistemic frontier,
but until now it existed only as prose inside the body: consolidation rewrote
it every cycle, curiosity picked one question per cycle, and nothing recorded
whether a question was ever researched, answered, or quietly dropped. The
ledger makes the frontier a first-class, queryable object:

``dossier_questions`` — one row per (dossier, normalized question)
    open        the current body still asks it
    queued      handed to the curiosity queue (``curiosity_id`` set)
    researched  curiosity resolved it (``resolution`` stored)
    retired     the body no longer asks it (answered by consolidation, or dropped)

``belief_revisions`` — one row per ``REVISED:`` line a consolidation emitted:
what was believed → what is now understood → why. This is the audit trail of
Nova changing its mind, which the revision bodies bury in prose.

All functions are synchronous DB work — call them through ``asyncio.to_thread``
from the event loop.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

STATUSES = ("open", "queued", "researched", "retired")

_REVISED_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?REVISED:\s*(.+?)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def question_key(question: str) -> str:
    """Normalized dedup key: lowercase alphanumerics, single-spaced."""
    return _NON_ALNUM_RE.sub(" ", (question or "").lower()).strip()[:200]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def extract_questions(body: str, *, limit: int = 8) -> list[str]:
    """The researchable questions a dossier body currently asks (future-shaped
    'will X happen' lines are forecast material, not questions)."""
    from app.core.dossiers import _extract_open_questions, _is_future_question
    out: list[str] = []
    for q in _extract_open_questions(body, limit=limit):
        if _is_future_question(q) or q.lower().startswith("watch for"):
            continue
        out.append(q)
    return out


def sync_questions(db, dossier_id: int, dkey: str, body: str) -> dict:
    """Reconcile the ledger with the questions the body asks NOW.

    New questions open; questions still asked keep their status (a retired
    one that comes back reopens); open/queued questions the body dropped are
    retired — consolidation either answered them or stopped caring, and
    either way they are no longer the frontier.
    """
    now = _now()
    asked: dict[str, str] = {}
    for q in extract_questions(body):
        k = question_key(q)
        if k and k not in asked:
            asked[k] = q
    existing = {
        r["qkey"]: dict(r)
        for r in db.fetchall(
            "SELECT id, qkey, status FROM dossier_questions WHERE dkey = ?", (dkey,))
    }
    new = kept = retired = 0
    for k, q in asked.items():
        row = existing.get(k)
        if row is None:
            db.execute(
                "INSERT OR IGNORE INTO dossier_questions "
                "(dossier_id, dkey, question, qkey, status, first_seen_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, 'open', ?, ?)",
                (dossier_id, dkey, q[:300], k, now, now),
            )
            new += 1
        else:
            if row["status"] == "retired":
                db.execute(
                    "UPDATE dossier_questions SET status = 'open', resolved_at = NULL, "
                    "last_seen_at = ?, question = ? WHERE id = ?",
                    (now, q[:300], row["id"]),
                )
            else:
                db.execute(
                    "UPDATE dossier_questions SET last_seen_at = ?, question = ? WHERE id = ?",
                    (now, q[:300], row["id"]),
                )
            kept += 1
    for k, row in existing.items():
        if k not in asked and row["status"] in ("open", "queued"):
            db.execute(
                "UPDATE dossier_questions SET status = 'retired', resolved_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            retired += 1
    return {"new": new, "kept": kept, "retired": retired}


def record_revisions(db, dossier_id: int, dkey: str, text: str) -> int:
    """Persist every REVISED: line in `text` (once per 30 days per dossier)."""
    n = 0
    for m in _REVISED_RE.finditer(text or ""):
        line = m.group(1).strip()[:600]
        if len(line) < 12:
            continue
        dup = db.fetchone(
            "SELECT 1 FROM belief_revisions WHERE dkey = ? AND revised = ? "
            "AND created_at > datetime('now', '-30 days')",
            (dkey, line),
        )
        if dup:
            continue
        db.execute(
            "INSERT INTO belief_revisions (dossier_id, dkey, revised) VALUES (?, ?, ?)",
            (dossier_id, dkey, line),
        )
        n += 1
    return n


def sync_after_consolidation(db, kind: str, dkey: str) -> dict:
    """Called after a dossier body is (re)written: reconcile questions and
    record any REVISED: lines. Returns the sync stats (empty if no dossier)."""
    row = db.fetchone("SELECT id, body FROM dossiers WHERE kind = ? AND dkey = ?", (kind, dkey))
    if not row:
        return {}
    body = row["body"] or ""
    stats = sync_questions(db, row["id"], dkey, body)
    stats["revisions"] = record_revisions(db, row["id"], dkey, body)
    if stats["new"] or stats["retired"] or stats["revisions"]:
        logger.info("[Knowing] question ledger %s: +%d new, %d retired, %d revisions",
                    dkey, stats["new"], stats["retired"], stats["revisions"])
    return stats


def mark_queued(db, dkey: str, question: str, curiosity_id: int | None) -> bool:
    """The question was handed to the curiosity queue."""
    if not curiosity_id:
        return False
    cur = db.execute(
        "UPDATE dossier_questions SET status = 'queued', curiosity_id = ? "
        "WHERE dkey = ? AND qkey = ? AND status IN ('open', 'queued')",
        (int(curiosity_id), dkey, question_key(question)),
    )
    return bool(getattr(cur, "rowcount", 0))


def mark_researched(db, curiosity_id: int | None, resolution: str) -> bool:
    """Curiosity resolved the queued question (any status but researched)."""
    if not curiosity_id:
        return False
    cur = db.execute(
        "UPDATE dossier_questions SET status = 'researched', resolution = ?, resolved_at = ? "
        "WHERE curiosity_id = ? AND status != 'researched'",
        ((resolution or "")[:600], _now(), int(curiosity_id)),
    )
    return bool(getattr(cur, "rowcount", 0))


def retire_orphaned(db) -> int:
    """Retire open/queued questions whose storyline thread has closed.

    A closed storyline's dossier is never rewritten, so its questions would
    sit 'open' forever and inflate the frontier (18 such dossiers on the live
    install when the ledger was backfilled, 2026-09-02).
    """
    cur = db.execute(
        "UPDATE dossier_questions SET status = 'retired', resolved_at = ?, "
        "resolution = COALESCE(resolution, 'storyline closed') "
        "WHERE status IN ('open', 'queued') AND dossier_id IN ("
        "  SELECT d.id FROM dossiers d JOIN storylines s ON s.story_key = d.dkey "
        "  WHERE d.kind = 'storyline' AND s.status != 'active')",
        (_now(),),
    )
    return int(getattr(cur, "rowcount", 0) or 0)


def frontier(db, dkey: str | None = None) -> dict:
    """Counts by status (+ total) — the size and disposition of the frontier."""
    where = "WHERE dkey = ?" if dkey else ""
    rows = db.fetchall(
        f"SELECT status, COUNT(*) AS n FROM dossier_questions {where} GROUP BY status",
        (dkey,) if dkey else (),
    )
    out = {s: 0 for s in STATUSES}
    for r in rows:
        out[r["status"]] = int(r["n"])
    out["total"] = sum(out[s] for s in STATUSES)
    return out


def list_questions(db, *, status: str | None = None, dkey: str | None = None,
                   limit: int = 50) -> list[dict]:
    clauses, params = [], []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if dkey:
        clauses.append("dkey = ?")
        params.append(dkey)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = db.fetchall(
        f"SELECT id, dossier_id, dkey, question, status, curiosity_id, resolution, "
        f"first_seen_at, last_seen_at, resolved_at FROM dossier_questions {where} "
        f"ORDER BY last_seen_at DESC, id DESC LIMIT ?",
        (*params, int(limit)),
    )
    return [dict(r) for r in rows]


def list_revisions(db, *, dkey: str | None = None, limit: int = 50) -> list[dict]:
    where = "WHERE dkey = ?" if dkey else ""
    rows = db.fetchall(
        f"SELECT id, dossier_id, dkey, revised, created_at FROM belief_revisions {where} "
        f"ORDER BY id DESC LIMIT ?",
        (dkey, int(limit)) if dkey else (int(limit),),
    )
    return [dict(r) for r in rows]
