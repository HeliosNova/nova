"""Entity timelines — the dated spine of what Nova learned about a subject.

The knowledge graph is bitemporal: every fact carries `created_at` (when Nova
recorded it) and, once replaced, `superseded_at` plus `superseded_by` (what
replaced it). That trail is the single most interesting thing the KG holds —
it is Nova changing its mind, dated — and nothing read it.

The gap this closes: the entity-dossier prompt asks for "5-10 dated bullets of
the major shifts, oldest→newest", while `_entity_sources` fed it a flat list of
currently-live facts with no dates and no supersessions at all. The model was
being asked for a history from material that had none, which is the same shape
as every other "output judged against a structure it never had" defect in this
codebase. A timeline gives it the dates it was already required to produce.

Deliberately NOT wired into digest synthesis. A paired A/B on 16 topics
(2026-09-03) measured injecting prior understanding into the synthesis prompt
as no gain and a cost to fact grounding — see `_synthesize_from_evidence`.
Timelines belong where prior knowledge is the product (dossier consolidation,
chat, the API), not where today's evidence is supposed to be the only
admissible source.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Event kinds, in the order a reader cares about them.
LEARNED = "learned"
REVISED = "revised"
STORYLINE = "storyline"
BELIEF = "belief"


def _day(value) -> str:
    return str(value or "")[:10]


def entity_timeline(db, subject: str, *, days: int = 180, limit: int = 40) -> list[dict]:
    """Dated events for `subject`, oldest first.

    Sources, all already stored:
      learned    a KG fact was recorded (created_at)
      revised    a KG fact was superseded (superseded_at) — with what replaced it
      storyline  a story thread event naming the subject
      belief     a consolidation emitted a REVISED: line naming the subject

    Returns [{when, kind, text}]. Never raises: a timeline is an enrichment,
    and a missing one must not take a consolidation cycle down with it.
    """
    subject = (subject or "").strip()
    if not subject:
        return []
    since = f"-{int(days)} days"
    out: list[dict] = []

    try:
        for r in db.fetchall(
            "SELECT predicate, object, created_at FROM kg_facts "
            "WHERE subject = ? COLLATE NOCASE AND superseded_at IS NULL "
            "  AND created_at > datetime('now', ?) "
            "ORDER BY created_at LIMIT ?", (subject, since, limit)):
            out.append({"when": _day(r["created_at"]), "kind": LEARNED,
                        "text": f"{subject} {r['predicate']} {r['object']}"})
    except Exception as e:
        logger.debug("[Timeline] live facts unreadable for %r: %s", subject, e)

    # The belief-change trail: what a fact was, what replaced it, and when.
    try:
        for r in db.fetchall(
            "SELECT old.predicate AS predicate, old.object AS was, new.object AS now_is, "
            "       old.superseded_at AS when_ "
            "FROM kg_facts old LEFT JOIN kg_facts new ON new.id = old.superseded_by "
            "WHERE old.subject = ? COLLATE NOCASE AND old.superseded_at IS NOT NULL "
            "  AND old.superseded_at > datetime('now', ?) "
            "ORDER BY old.superseded_at LIMIT ?", (subject, since, limit)):
            if r["now_is"]:
                text = f"{subject} {r['predicate']}: {r['was']} → {r['now_is']}"
            else:
                text = f"{subject} {r['predicate']} {r['was']} no longer holds"
            out.append({"when": _day(r["when_"]), "kind": REVISED, "text": text})
    except Exception as e:
        logger.debug("[Timeline] supersessions unreadable for %r: %s", subject, e)

    try:
        for r in db.fetchall(
            "SELECT se.summary AS summary, se.created_at AS created_at, s.title AS title "
            "FROM storyline_events se JOIN storylines s ON s.id = se.storyline_id "
            "WHERE se.summary LIKE ? COLLATE NOCASE AND se.created_at > datetime('now', ?) "
            "ORDER BY se.created_at LIMIT ?", (f"%{subject}%", since, limit)):
            out.append({"when": _day(r["created_at"]), "kind": STORYLINE,
                        "text": f"[{r['title']}] {r['summary']}"})
    except Exception as e:
        logger.debug("[Timeline] storyline events unreadable for %r: %s", subject, e)

    try:
        for r in db.fetchall(
            "SELECT revised, created_at FROM belief_revisions "
            "WHERE revised LIKE ? COLLATE NOCASE AND created_at > datetime('now', ?) "
            "ORDER BY created_at LIMIT ?", (f"%{subject}%", since, limit)):
            out.append({"when": _day(r["created_at"]), "kind": BELIEF, "text": r["revised"]})
    except Exception as e:
        logger.debug("[Timeline] belief revisions unreadable for %r: %s", subject, e)

    out.sort(key=lambda e: (e["when"], e["kind"]))
    return out[-limit:] if len(out) > limit else out


def format_timeline(events: list[dict], *, cap: int = 3000) -> str:
    """Render a timeline as dated bullets, oldest first, bounded.

    Revisions are marked so the reader can see where understanding CHANGED
    rather than merely accumulated — that distinction is the point.
    """
    if not events:
        return ""
    lines = ["TIMELINE (what Nova learned about this subject, oldest first — "
             "'changed:' marks a belief that was revised):"]
    for e in events:
        mark = "changed: " if e["kind"] in (REVISED, BELIEF) else ""
        lines.append(f"- {e['when']}: {mark}{e['text']}")
    out = "\n".join(lines)
    if len(out) <= cap:
        return out
    kept = [lines[0]]
    for line in lines[1:]:
        if sum(len(x) + 1 for x in kept) + len(line) > cap:
            break
        kept.append(line)
    return "\n".join(kept)


def timeline_block(db, subject: str, *, days: int = 180, limit: int = 40,
                   cap: int = 3000) -> str:
    """Convenience: fetch and render in one call (empty string when there is
    nothing dated to say)."""
    try:
        return format_timeline(entity_timeline(db, subject, days=days, limit=limit), cap=cap)
    except Exception as e:
        logger.debug("[Timeline] block failed for %r: %s", subject, e)
        return ""
