"""Living dossiers — the KNOWING tier (2026-08-12).

Nova gathers elite intelligence (digests) and stores verified facts (bitemporal
KG), but its richest thinking — the analyses, connections, "why it matters" —
lives in `monitor_results` under a 30-day retention and evaporates. Each digest
is written from a 48h window: comprehension without accumulation.

This module closes that gap. A dossier is DURABLE, REVISABLE understanding:
one per watched domain and one per mature storyline, distilled from the
already-8-layer-verified digests BEFORE they expire, superseded-not-deleted
(`dossier_revisions` keeps every prior body → "what did Nova understand about
X on date D" is queryable, same philosophy as kg_facts bitemporality).

Consumers:
  - heartbeat "Knowledge Consolidation" monitor (check_type="consolidation",
    daily) → `consolidate_dossiers` — the distillation cycle, bounded.
  - deep_research `_synthesize_from_evidence` → `get_domain_dossier` — digests
    are written AGAINST prior understanding (lead with what's NEW, flag
    contradictions) instead of amnesiac re-reporting.
  - brain chat context → `get_relevant_dossiers` — questions get answered FROM
    understanding (dossier prose retrieves into the prompt where raw triples
    were denied by the 9B, forward-queue 2026-07-07).

Reuses llm.invoke_nothink (27B via MONITOR_SYNTHESIS_MODEL when set — this is
background/latency-tolerant, quality wins). Every LLM call passes an explicit
num_ctx (num_ctx discipline, CLAUDE.md 2026-08-11).
"""

from __future__ import annotations

import logging
import re

from app.core import llm

logger = logging.getLogger(__name__)

# Bounded cycle: at most this many dossier updates per consolidation run.
_MAX_UPDATES_PER_CYCLE = 8
# Dossier body ceiling (chars) — bounded at a sentence boundary, never mid-thought.
_BODY_CAP = 9000
# New-material budget per update (chars of recent digests fed to the LLM).
_SOURCE_CAP = 18000
# Prior-body budget fed back for revision.
_PRIOR_CAP = 8000
# Storylines graduate to a dossier once they've moved this many times.
_STORYLINE_MIN_UPDATES = 3

_DOMAIN_PREFIX = "Domain Study: "

# Function words excluded from the ≥2-char retrieval tokens (get_relevant_dossiers).
_SHORT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "had", "has", "how", "i", "if", "in", "is", "it", "its", "no",
    "not", "now", "of", "on", "or", "our", "out", "per", "so", "the", "their",
    "they", "this", "that", "to", "up", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
})


def _slug(text: str) -> str:
    """Stable slug key (mirrors storylines._story_key)."""
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:80]


def _bound(text: str, cap: int = _BODY_CAP) -> str:
    """Trim to cap at the last sentence boundary — a dossier must never end
    mid-thought (the truncation lesson, 2026-08-11)."""
    text = (text or "").strip()
    if len(text) <= cap:
        return text
    cut = text[:cap]
    # Prefer the last sentence end; fall back to last newline; then hard cut.
    m = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("!\n"), cut.rfind("?\n"))
    if m > cap // 2:
        return cut[: m + 1].rstrip()
    nl = cut.rfind("\n")
    return (cut[:nl] if nl > cap // 2 else cut).rstrip()


_UPDATE_PROMPT = (
    "You maintain Nova's PERMANENT UNDERSTANDING of: {title}\n"
    "Below is the PRIOR DOSSIER (what was understood before), then NEW MATERIAL "
    "(verified intelligence digests written since the last consolidation — more recent).\n\n"
    "Rewrite the dossier to reflect the CURRENT state of understanding. Rules:\n"
    "- NEW MATERIAL wins where it conflicts with the prior dossier: describe the "
    "latest reality; when a prior belief is OVERTAKEN, replace it and record the "
    "shift in 'How we got here'.\n"
    "- If new material CONTRADICTS prior understanding, flag it explicitly with a "
    "line starting 'REVISED:' (what was believed → what is now understood → why).\n"
    "- Keep only what has DURABLE value — drop day-trivia and routine scheduling. "
    "Keep EXACT numbers, names, dates, and inline (outlet.com) citations exactly as "
    "they appear in the material. Invent NOTHING beyond the prior dossier + material.\n"
    "- Structure (markdown, keep these exact headings):\n"
    "## Current understanding\n"
    "(400-600 words — the state of things and why it matters)\n"
    "## How we got here\n"
    "(5-10 dated bullets of the major shifts, oldest→newest)\n"
    "## Key facts & figures\n"
    "(bullets, each with its citation)\n"
    "## Open questions\n"
    "(3-6 things genuinely not yet known)\n"
    "- End with exactly one line 'CHANGED: <what moved in this consolidation>' "
    "(or 'CHANGED: initial dossier' for a first version).\n"
    "Write the COMPLETE dossier and finish cleanly — no preamble, no meta-commentary.\n\n"
    "PRIOR DOSSIER:\n{prior}\n\n"
    "NEW MATERIAL:\n{sources}"
)

_CHANGED_RE = re.compile(r"(?im)^CHANGED:\s*(.+?)\s*$")


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

def get_dossier(db, kind: str, dkey: str):
    """Current dossier row for (kind, dkey) or None."""
    try:
        return db.fetchone(
            "SELECT * FROM dossiers WHERE kind = ? AND dkey = ?", (kind, dkey)
        )
    except Exception:
        return None


def get_domain_dossier(db, label: str):
    """Domain dossier for a deep_research feed label (e.g. 'finance').

    Domain dossiers are keyed by slug of the monitor name minus the
    'Domain Study: ' prefix, which equals slug(label) for domain studies.
    """
    return get_dossier(db, "domain", _slug(label))


def get_relevant_dossiers(db, query: str, *, limit: int = 1) -> list[dict]:
    """Cheap keyword match of a chat query to dossiers (no LLM, no embedder).

    Mirrors storylines.get_relevant_storylines: overlap >= 2 significant tokens;
    returns [{'title', 'excerpt'}] with the excerpt cut from 'Current
    understanding' so chat injects prose the model answers from naturally.
    Returns [] on any miss — the chat path must never break on this.
    """
    # ≥2-char tokens with a function-word stoplist: the ≥4 floor used elsewhere
    # made the 'AI and ML' title contribute ZERO tokens (ai/ml are 2 chars) —
    # short-acronym dossiers were unmatchable (seen live 2026-08-12).
    _stop = _SHORT_STOPWORDS
    def _toks(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]{2,}", (text or "").lower()) if t not in _stop}
    q_toks = _toks(query)
    if not q_toks:
        return []
    try:
        rows = db.fetchall(
            "SELECT title, body FROM dossiers WHERE body != '' "
            "ORDER BY updated_at DESC LIMIT 60"
        )
    except Exception:
        return []
    scored = []
    for r in rows:
        title_toks = _toks(r["title"])
        body_toks = _toks(r["body"][:2000])
        overlap = len(q_toks & (title_toks | body_toks))
        # A single TITLE hit qualifies (titles are curated + specific: 'biotech
        # advances lately?' must match 'Biotech and Genetics'); body-only matches
        # still need >=2 tokens so incidental mentions don't retrieve.
        if overlap >= 2 or (q_toks & title_toks):
            # Title hits weigh 3× — "what changed at DeepMind?" must prefer the
            # 'AI and ML' dossier over a broad one that merely mentions AI
            # (mis-rank seen live on the first retrieval check, 2026-08-12).
            scored.append((overlap + 2 * len(q_toks & title_toks), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, r in scored[:limit]:
        body = r["body"]
        m = re.search(r"(?is)## Current understanding\s*(.+?)(?=\n## |\Z)", body)
        excerpt = _bound((m.group(1) if m else body).strip(), 900)
        out.append({"title": r["title"], "excerpt": excerpt})
    return out


# ---------------------------------------------------------------------------
# Consolidation cycle
# ---------------------------------------------------------------------------

def _domains_needing_update(db) -> list[dict]:
    """Domain-study monitors with digests newer than their dossier (or with no
    dossier yet). Staleness-first: the longest-unconsolidated domain goes first,
    so partial cycles still rotate coverage instead of starving the tail."""
    try:
        rows = db.fetchall(
            "SELECT m.name AS name, MAX(mr.created_at) AS latest "
            "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
            "WHERE m.name LIKE ? AND mr.status IN ('ok','changed','alert') "
            "  AND mr.value IS NOT NULL AND length(mr.value) > 400 "
            "GROUP BY m.name",
            (_DOMAIN_PREFIX + "%",),
        )
    except Exception:
        return []
    out = []
    for r in rows:
        label = r["name"][len(_DOMAIN_PREFIX):]
        dkey = _slug(label)
        d = get_dossier(db, "domain", dkey)
        if d is None or (d["updated_at"] or "") < (r["latest"] or ""):
            out.append({
                "kind": "domain", "dkey": dkey, "title": label,
                "monitor_name": r["name"],
                "since": d["updated_at"] if d else None,
                "staleness": d["updated_at"] if d else "",   # '' sorts first = never consolidated
            })
    out.sort(key=lambda x: x["staleness"])
    return out


def _storylines_needing_update(db) -> list[dict]:
    """Mature active storylines (moved >= _STORYLINE_MIN_UPDATES times) whose
    events outrun their dossier."""
    try:
        rows = db.fetchall(
            "SELECT s.id, s.story_key, s.title, s.last_updated "
            "FROM storylines s WHERE s.status = 'active' AND s.update_count >= ? "
            "ORDER BY s.last_updated DESC LIMIT 40",
            (_STORYLINE_MIN_UPDATES,),
        )
    except Exception:
        return []
    out = []
    for r in rows:
        d = get_dossier(db, "storyline", r["story_key"])
        if d is None or (d["updated_at"] or "") < (r["last_updated"] or ""):
            out.append({
                "kind": "storyline", "dkey": r["story_key"], "title": r["title"],
                "storyline_id": r["id"],
                "since": d["updated_at"] if d else None,
                "staleness": d["updated_at"] if d else "",
            })
    out.sort(key=lambda x: x["staleness"])
    return out


def _domain_sources(db, monitor_name: str, since: str | None) -> str:
    """New digests for a domain since the last consolidation (newest last so the
    model reads chronologically), bounded to _SOURCE_CAP."""
    try:
        if since:
            rows = db.fetchall(
                "SELECT mr.value AS value, mr.created_at AS created_at "
                "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
                "WHERE m.name = ? AND mr.created_at > ? AND mr.value IS NOT NULL "
                "  AND length(mr.value) > 400 ORDER BY mr.created_at DESC LIMIT 3",
                (monitor_name, since),
            )
        else:
            rows = db.fetchall(
                "SELECT mr.value AS value, mr.created_at AS created_at "
                "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
                "WHERE m.name = ? AND mr.value IS NOT NULL AND length(mr.value) > 400 "
                "ORDER BY mr.created_at DESC LIMIT 3",
                (monitor_name,),
            )
    except Exception:
        return ""
    parts = [f"[digest {r['created_at']}]\n{r['value']}" for r in reversed(rows)]
    return "\n\n".join(parts)[:_SOURCE_CAP]


def _storyline_sources(db, storyline_id: int, since: str | None) -> str:
    """Storyline's current summary + its events since the last consolidation."""
    try:
        s = db.fetchone("SELECT title, summary FROM storylines WHERE id = ?", (storyline_id,))
        if since:
            evs = db.fetchall(
                "SELECT summary, created_at FROM storyline_events "
                "WHERE storyline_id = ? AND created_at > ? ORDER BY created_at LIMIT 40",
                (storyline_id, since),
            )
        else:
            evs = db.fetchall(
                "SELECT summary, created_at FROM storyline_events "
                "WHERE storyline_id = ? ORDER BY created_at LIMIT 60",
                (storyline_id,),
            )
    except Exception:
        return ""
    lines = [f"[tracked thread state] {s['summary']}"] if s and s["summary"] else []
    lines += [f"- ({e['created_at']}) {e['summary']}" for e in evs]
    return "\n".join(lines)[:_SOURCE_CAP]


async def _update_dossier(db, cand: dict, sources: str, syn_model: str | None) -> dict | None:
    """One consolidation: prior body + new material -> revised understanding.
    Persists the revision trail. Returns {'title','changed'} on change, else None."""
    if len(sources) < 300:
        return None   # nothing substantive to consolidate
    prior_row = get_dossier(db, cand["kind"], cand["dkey"])
    prior = (prior_row["body"] if prior_row else "")[:_PRIOR_CAP] or "(none — first consolidation)"

    prompt = _UPDATE_PROMPT.format(title=cand["title"], prior=prior, sources=sources)
    try:
        out = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            # 2600 (was 1600), 2026-08-12: the first live cycle hit the 1600 cap
            # on 5/8 dossiers (eval=1600, ~4 c/t — clean generation, budget too
            # small; caught by the truncation tripwire built the same day). The
            # 27B writes ~6-7k chars for a full dossier; 2600 gives it room to
            # finish 'Open questions' + the CHANGED line cleanly.
            max_tokens=2600, temperature=0.2, model=syn_model,
            # prior ≤8k + sources ≤18k + instructions ≈ 27.5k chars ≈ ~7.5k tokens
            # + 2.6k generation → 12288 holds with headroom (num_ctx discipline).
            num_ctx=12288,
        )
    except Exception as e:
        logger.warning("[Knowing] dossier LLM failed for %r: %s", cand["title"], e)
        return None
    out = (out or "").strip()
    if len(out) < 200 or "## Current understanding" not in out:
        # Un-silenced: a malformed consolidation must be loud, not a quiet skip.
        logger.warning("[Knowing] dossier update for %r returned malformed body "
                       "(%d chars) — head: %r", cand["title"], len(out), out[:160])
        return None

    m = _CHANGED_RE.search(out)
    changed = m.group(1).strip() if m else ("initial dossier" if not prior_row else "updated")
    # Normalize citations: the model sometimes embellishes bare (outlet.com)
    # cites into markdown links with INVENTED URLs (seen live 2026-08-12:
    # technologyreview.com → a fabricated technologyresearch.com URL). Keep the
    # visible outlet text, drop the un-verifiable link target.
    out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)
    body = _bound(_CHANGED_RE.sub("", out).strip())

    if prior_row:
        db.execute(
            "INSERT INTO dossier_revisions (dossier_id, body, valid_from) VALUES (?, ?, ?)",
            (prior_row["id"], prior_row["body"], prior_row["updated_at"]),
        )
        db.execute(
            "UPDATE dossiers SET body = ?, changed_note = ?, title = ?, "
            "update_count = update_count + 1, updated_at = datetime('now') WHERE id = ?",
            (body, changed, cand["title"], prior_row["id"]),
        )
    else:
        db.execute(
            "INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (cand["kind"], cand["dkey"], cand["title"], body, changed),
        )
    logger.info("[Knowing] %s dossier %r consolidated (%s)",
                cand["kind"], cand["title"], changed[:100])
    return {"title": cand["title"], "changed": changed}


async def consolidate_dossiers(db) -> str:
    """Full cycle: find domains/storylines that outran their dossiers, distill
    the newest material into revised understanding — staleness-first, bounded to
    _MAX_UPDATES_PER_CYCLE. Sequential on purpose: these are big-model calls on
    one GPU; the heartbeat is latency-tolerant."""
    from app.config import config as _cfg
    syn_model = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None

    candidates = _domains_needing_update(db) + _storylines_needing_update(db)
    candidates.sort(key=lambda x: x["staleness"])
    if not candidates:
        return "KNOWING | all dossiers current — nothing to consolidate"

    updated, attempted = [], 0
    for cand in candidates[:_MAX_UPDATES_PER_CYCLE]:
        attempted += 1
        if cand["kind"] == "domain":
            sources = _domain_sources(db, cand["monitor_name"], cand["since"])
        else:
            sources = _storyline_sources(db, cand["storyline_id"], cand["since"])
        try:
            res = await _update_dossier(db, cand, sources, syn_model)
        except Exception as e:
            logger.warning("[Knowing] consolidation failed for %r: %s", cand["title"], e)
            continue
        if res:
            updated.append(res)

    backlog = max(0, len(candidates) - _MAX_UPDATES_PER_CYCLE)
    if not updated:
        return (f"KNOWING | {attempted} candidate(s) checked, no dossier changed"
                + (f" ({backlog} queued)" if backlog else ""))
    lines = [f"## 📚 KNOWING — {len(updated)} dossier(s) consolidated"
             + (f" ({backlog} queued for next cycle)" if backlog else "")]
    for u in updated:
        lines.append(f"- **{u['title']}** — {u['changed']}")
    return "\n".join(lines)
