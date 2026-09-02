"""Storyline tracking + change detection (Monitor Intelligence v2, Phase A).

Turns the per-domain news feed into persistent STORY THREADS with state. Instead
of 50 disconnected headlines, Nova maintains named storylines ("Iran–Hormuz
crisis"), accrues developments, and surfaces ONLY what MOVED — "here's how your
threads changed" — not raw items.

Reuses:
  - cross_monitor._gather_recent_outputs : recent content monitor_results
  - cross_monitor._extract_signals       : cheap entity/phrase pre-cluster
  - kg bitemporal API : each moved thread emits one `<entity> has_status <value>`
    fact through `add_fact`; `has_status` is functional, so a changed value
    SUPERSEDES the prior one — `get_fact_history` then yields a structured,
    queryable state trail and the digest surfaces the deterministic delta
    ("status: X → Y") alongside the narrative CHANGED line.
  - llm.invoke_nothink                    : naming + summary passes

Background-only (runs on the Storyline Tracker monitor schedule). One LLM pass to
cluster+name, one per moved thread to update its summary — bounded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.core import llm
from app.core.cross_monitor import _gather_recent_outputs, _extract_signals

logger = logging.getLogger(__name__)


async def _bg_invoke(*args, **kwargs):
    """llm.invoke_nothink with the GPU-yield checkpoint (parity with
    deep_research._invoke_bg, added 2026-08-19): this module's background
    chains make many sequential 27B calls — without the checkpoint an owner
    chat arriving mid-chain contends for the GPU for the whole run (probe:
    350 tokens at ~2 tok/s during a consolidation cycle)."""
    try:
        from app.core.llm import wait_for_interactive_quiet as _w
        waited = await _w(max_wait_s=240.0)
        if waited:
            logger.info("[gpu-yield] %s yielded to chat for %.0fs", __name__, waited)
    except Exception:
        pass
    return await llm.invoke_nothink(*args, **kwargs)


# How far back to scan monitor outputs for developments.
_WINDOW_HOURS = 48
# Cap stories per cycle so the LLM passes stay bounded on a single GPU.
_MAX_STORIES = 8

# Grammar-constrained array output (Ollama `format`). Replaces the json_prefix="[{"
# prefill that broke on Ollama 0.32.13 (the model returned a bare {object}, the
# provider re-prepend corrupted it to "[{{…", json.loads failed → cluster
# returned [] → "no ongoing stories" every cycle from 2026-08-15). See
# brain_kg._KG_TRIPLES_SCHEMA for the full root-cause note.
_STORIES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["title", "items"],
    },
}


def _story_key(title: str) -> str:
    """Stable slug key for matching a story to an existing storyline."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:80]


_CLUSTER_PROMPT = (
    "You are an intelligence analyst grouping recent monitor items into ONGOING "
    "STORIES (durable threads), not isolated headlines.\n\n"
    "{existing}"
    "From the items below, identify up to {max_stories} distinct ongoing stories. "
    "Merge items about the SAME underlying situation into one story.\n"
    "For each story return: a short stable title (e.g. 'Iran-Hormuz tensions', "
    "'NVIDIA export controls'), and the indices of its items.\n"
    "Ignore one-off trivia that isn't part of an evolving situation.\n\n"
    "Items:\n{items}\n\n"
    'Return JSON only: [{{"title": "...", "items": [0, 3, 7]}}]'
)

# Generic narrative words that must NOT drive fuzzy story matching — only
# specific entities (hormuz, nvidia, colombia) should signal "same story".
_GENERIC_STORY_WORDS = frozenset({
    "tensions", "crisis", "talks", "deal", "war", "conflict", "news", "update",
    "saga", "situation", "threats", "threat", "closure", "election", "summit",
    "story", "developments", "peace", "negotiations", "dispute", "standoff",
})


def _sig_tokens(text: str) -> set[str]:
    """Significant (specific-entity) tokens for fuzzy story matching."""
    return {t for t in re.findall(r"[a-z0-9]{4,}", (text or "").lower())
            if t not in _GENERIC_STORY_WORDS}

_UPDATE_PROMPT = (
    "You track an ongoing story. Below is its PRIOR STATE (written in an earlier "
    "cycle), then NEW DEVELOPMENTS (more recent — what is happening now).\n\n"
    "STORY: {title}\n"
    "PRIOR STATE: {prior}\n\n"
    "NEW DEVELOPMENTS:\n{developments}\n\n"
    "Write the CURRENT state of this story in 2-3 sentences. Rules:\n"
    "- The NEW DEVELOPMENTS are more recent than the prior state. Where they "
    "CONFLICT, the new developments WIN: describe the latest reality and do NOT "
    "repeat a prior claim that has been overtaken (e.g. if a ceasefire later holds, "
    "do not still call it 'shattered'; if a deal is signed, the talks are no longer "
    "'ongoing').\n"
    "- If the developments REVERSE or supersede the prior state, say so plainly.\n"
    "- Describe the situation as it stands NOW — not a blend of stale and current.\n"
    "Then on a new line starting 'CHANGED: ' give ONE sentence on what specifically "
    "moved since the prior state.\n"
    "If the story has a clear CURRENT STATUS that can change over time (a ceasefire "
    "holding, talks pending, an outage ongoing, a price level, a deal signed), add a "
    "line 'STATE: <main entity> | <short current status>' — the status in 2-6 words. "
    "Use the SAME main-entity wording each cycle so the status can be tracked. Omit "
    "the line entirely if there is no clear trackable status.\n"
    "Then end with one final line, always present: 'FORECAST: <specific testable "
    "claim> | resolves YYYY-MM-DD | <0.x confidence>' when there is a clear "
    "falsifiable expectation (a dated event or scheduled decision usually supports "
    "one), else exactly 'FORECAST: none'. The date is when the outcome becomes "
    "knowable (up to a year out); put any deadline inside the claim, make it "
    "settleable by a news search on that date, and never forecast a target, "
    "guidance figure or plan — only a realized outcome.\n"
    "If the new developments add nothing substantive, reply exactly 'NO CHANGE'."
)

# Parses an optional 'STATE: <entity> | <current status>' line for KG state-tracking.
_STATE_RE = re.compile(r"(?im)^\s*STATE:\s*(?P<entity>[^|\n]{2,70}?)\s*\|\s*(?P<status>[^\n|]{2,70}?)\s*$")


def _collect_items(db) -> list[dict]:
    """Flatten recent content monitor outputs into candidate story items."""
    grouped = _gather_recent_outputs(db, hours=_WINDOW_HOURS, max_per_monitor=6)
    items: list[dict] = []
    for monitor_name, values in grouped.items():
        for val in values:
            # Each monitor value is a formatted digest; split into lines and keep
            # substantive ones (a headline/development is usually one line).
            for line in str(val).splitlines():
                line = line.strip()
                # Drop pure formatting/scaffold lines; keep lines with real signal.
                if len(line) < 40 or not _extract_signals(line):
                    continue
                # Word-safe cap (2026-09-01): this [:300] was the ACTUAL
                # mid-word cutter behind the len-300 storyline_events — items
                # arrive at _record already ≤300, so the 08-31 _event_summary
                # bound at the INSERT never fired (6 fresh mid-word rows from
                # the first post-fix Morning Check-in proved it).
                clean = _event_summary(re.sub(r"^[`*#>\d\.\-\s]+", "", line), cap=300)
                if clean:
                    items.append({"text": clean, "monitor": monitor_name})
                if len(items) >= 120:  # hard cap on prompt size
                    return items
    return items


async def _cluster_into_stories(items: list[dict], existing_titles: list[str] | None = None) -> list[dict]:
    """One LLM pass: group items into named ongoing stories.

    `existing_titles` are the currently-active threads; the LLM is told to REUSE
    an existing title verbatim when a cluster continues it — the primary defense
    against the same story fragmenting into multiple threads across cycles.
    """
    if not items:
        return []
    numbered = "\n".join(f"{i}. [{it['monitor']}] {it['text']}" for i, it in enumerate(items[:120]))
    existing_block = ""
    if existing_titles:
        listed = "\n".join(f"  - {t}" for t in existing_titles[:30])
        existing_block = (
            "ALREADY-TRACKED stories — if a cluster CONTINUES one of these, reuse "
            "its EXACT title verbatim (do not invent a new name for the same story):\n"
            f"{listed}\n\n"
        )
    prompt = _CLUSTER_PROMPT.format(max_stories=_MAX_STORIES, items=numbered, existing=existing_block)
    try:
        raw = await _bg_invoke(
            [{"role": "user", "content": prompt}],
            json_mode=True, json_schema=_STORIES_SCHEMA, max_tokens=700, temperature=0.2,
            # num_ctx REQUIRED (2026-08-11): 120 items × ~300 chars ≈ 20-36k chars
            # (~5-9k tokens) — far past the 4096 model default. Without it Ollama
            # silently truncated the prompt and the model returned hallucinated
            # garbage → parse fail → [] → "no ongoing stories identified" every
            # cycle since the 7/6 monitor_store un-truncation fix ballooned the
            # items. Repro'd red (garbage) / green (8 clean stories) 2026-08-11.
            num_ctx=16384,
        )
    except Exception as e:
        logger.warning("[Storyline] cluster LLM failed: %s", e)
        return []
    if not raw:
        logger.warning("[Storyline] cluster returned EMPTY for %d items — storylines will not update this cycle", len(items))
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            data = data.get("stories") or data.get("items") or []
    except Exception:
        data = llm.extract_json_object(raw)
        data = data.get("stories", []) if isinstance(data, dict) else []
    stories: list[dict] = []
    for s in (data or [])[:_MAX_STORIES]:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title", "")).strip()
        idxs = [i for i in (s.get("items") or []) if isinstance(i, int) and 0 <= i < len(items)]
        if title and idxs:
            stories.append({
                "title": title,
                "key": _story_key(title),
                "monitors": sorted({items[i]["monitor"] for i in idxs}),
                "developments": [items[i]["text"] for i in idxs],
            })
    if not stories:
        # Un-silenced (2026-08-11): a zero-story parse on real items is the
        # signature of an upstream failure (prompt truncation, malformed JSON),
        # not a quiet news day — it killed storylines invisibly for 5 weeks.
        logger.warning("[Storyline] cluster parse yielded 0 stories from %d items — raw head: %r",
                       len(items), (raw or "")[:200])
    return stories


def _find_matching_storyline(db, story: dict):
    """Match a story to an existing active thread — exact key first, then a
    deterministic fuzzy fallback on shared SPECIFIC entities, so the same story
    named slightly differently across cycles doesn't fragment into two threads.

    Entities appearing across MANY active threads (e.g. 'trump', 'us') can't
    discriminate one story from another, so they're down-weighted by document
    frequency — only rare, specific shared entities (hormuz, nvidia) signal a
    merge. This is what stops over-merging distinct stories about a common actor.
    """
    row = db.fetchone("SELECT * FROM storylines WHERE story_key = ?", (story["key"],))
    if row:
        return row

    active = db.fetchall(
        "SELECT * FROM storylines WHERE status = 'active' ORDER BY last_updated DESC LIMIT 80"
    )
    if not active:
        return None

    # Document frequency: how many active threads each entity appears in.
    df: dict[str, int] = {}
    cand_tok_map: dict[int, set[str]] = {}
    for r in active:
        toks = _sig_tokens(r["title"]) | _sig_tokens(r["summary"])
        cand_tok_map[r["id"]] = toks
        for t in toks:
            df[t] = df.get(t, 0) + 1
    # An entity in >=3 distinct active threads is too common to discriminate.
    common = {t for t, c in df.items() if c >= 3}

    story_toks = _sig_tokens(story["title"])
    for d in story.get("developments", [])[:5]:
        story_toks |= _sig_tokens(d)
    story_specific = story_toks - common
    if len(story_specific) < 2:
        return None

    best, best_ov = None, 0
    for r in active:
        # Overlap on RARE shared entities only.
        ov = len(story_specific & (cand_tok_map[r["id"]] - common))
        title_ov = len((_sig_tokens(story["title"]) - common) & (_sig_tokens(r["title"]) - common))
        if ov >= 2 and title_ov >= 1 and ov > best_ov:
            best, best_ov = r, ov
    return best


async def _record_state(kg, entity: str, status: str, key: str) -> str:
    """Write the thread's current status as a functional `has_status` KG fact and
    return a deterministic 'status: prior → new' delta when the value changed (else
    ''). `has_status` is functional, so the new value supersedes the old — that IS
    the change-detection, and `get_fact_history` keeps the queryable trail.
    Garbage-gated + provenanced so monitor state can't pollute the user-fact graph."""
    from app.core.kg import is_garbage_triple
    entity, status = (entity or "").strip(), (status or "").strip()
    if not entity or not status or is_garbage_triple(entity, "has_status", status):
        return ""
    prior = ""
    try:
        for h in kg.get_fact_history(entity, "has_status"):  # most-recent first
            if not h.get("superseded_by"):
                prior = (h.get("object") or "").strip()
                break
    except Exception:
        prior = ""
    try:
        await kg.add_fact(entity, "has_status", status, confidence=0.6,
                          source="storyline", provenance=f"storyline_state:{key[:60]}")
    except Exception as e:
        logger.debug("[Storyline] state add_fact failed for %r: %s", entity, e)
        return ""
    if prior and prior.lower() != status.lower():
        return f"{entity} status: {prior} → {status}"
    return ""


async def _update_story(db, story: dict, kg=None) -> dict | None:
    """Match story to an existing storyline, diff new developments, update state.

    Returns a digest-ready dict {title, changed, summary} ONLY if the thread moved,
    else None. When `kg` is provided, the thread's current status is also written as
    a functional `has_status` fact so a changed value supersedes the prior one (a
    structured, queryable state-change trail surfaced as a deterministic delta).
    """
    row = _find_matching_storyline(db, story)
    prior_summary = row["summary"] if row else ""

    # Diff: which developments are NEW vs already-recorded events for this thread.
    seen = set()
    if row:
        for ev in db.fetchall(
            "SELECT summary FROM storyline_events WHERE storyline_id = ? ORDER BY id DESC LIMIT 60",
            (row["id"],),
        ):
            seen.add((ev["summary"] or "")[:120].lower())
    fresh = [d for d in story["developments"] if d[:120].lower() not in seen]
    if row and not fresh:
        return None  # known thread, nothing new — skip

    dev_text = "\n".join(f"- {d}" for d in (fresh or story["developments"])[:10])
    # Global calibration record for the FORECAST tail (2026-09-01): storyline
    # mints never saw any calibration feedback before.
    try:
        from app.core.forecasts import global_calibration_note
        _gnote = await asyncio.to_thread(global_calibration_note, db)
        if _gnote:
            dev_text += "\n\n" + _gnote
    except Exception:
        pass
    new_story = row is None

    # One LLM pass to update the narrative state + name what changed.
    try:
        out = await _bg_invoke(
            [{"role": "user", "content": _UPDATE_PROMPT.format(
                title=story["title"], prior=prior_summary or "(new story — no prior state)",
                developments=dev_text)}],
            # 300 (was 260), 2026-08-13: the FORECAST tail is now mandatory
            # ('FORECAST: none' opt-out — the v1 optional hedge was never taken;
            # zero storyline mints since June 29); headroom so the tail can't
            # truncate away.
            max_tokens=300, temperature=0.2,
        )
    except Exception as e:
        logger.warning("[Storyline] update LLM failed for %r: %s", story["title"], e)
        return None
    out = (out or "").strip()
    # No-change detection (2026-08-31): the guard only caught a "NO CHANGE"
    # PREFIX. When the model rambled its rationale first and ended with the
    # verdict ("The provided new developments contain no information
    # regarding X…\n\nNO CHANGE"), the whole rationale was stored as the
    # thread's summary, overwriting the real one (live: storyline 31).
    # A STANDALONE "NO CHANGE" line anywhere is the verdict; content that
    # merely mentions "no change" mid-sentence (e.g. "Fed made no change to
    # rates") never matches a standalone line.
    _no_change = bool(re.search(r"(?im)^\s*NO CHANGE[.!]?\s*$", out))
    if not out or out.upper().startswith("NO CHANGE") or _no_change:
        # Still record events so we don't re-summarize them next cycle.
        if not new_story:
            # to_thread (2026-08-29): _record does an INSERT-or-UPDATE plus a
            # loop of development inserts, all sync, called from this async
            # function — writes on the event-loop thread (54h-freeze class).
            await asyncio.to_thread(_record, db, row, story, fresh, summary=None)
        return None

    summary, changed = out, ""
    m = re.search(r"(?im)^CHANGED:\s*(.+?)(?=\n|$)", out)
    if m:
        changed = m.group(1).strip()
        summary = out[:m.start()].strip()
    # Strip trailing STATE: / FORECAST: lines so they never leak into the stored/
    # displayed summary (the CHANGED parse above misses them when CHANGED is absent).
    summary = re.sub(r"(?im)^\s*(?:STATE|FORECAST):.*$", "", summary).strip()

    # Effective key: when this story was FUZZY-matched into an existing thread,
    # the forecast must reference the MATCHED thread's key, not the unused new one.
    eff_key = row["story_key"] if row else story["key"]

    # Structured state-change via the KG: a `<entity> has_status <value>` fact whose
    # functional supersession yields the queryable trail + a deterministic delta.
    state_delta = ""
    if kg is not None:
        sm = _STATE_RE.search(out)
        if sm:
            state_delta = await _record_state(kg, sm.group("entity"), sm.group("status"), eff_key)

    # Opportunistic forecast: if the model emitted a falsifiable call, record it
    # for later self-grading (Phase C). Gated; failures are non-fatal.
    try:
        from app.config import config as _cfg
        if getattr(_cfg, "ENABLE_FORECASTS", True):
            from app.core.forecasts import parse_and_store_forecast
            # to_thread (2026-08-31): sync INSERT on the loop (forecasts.py:36
            # tripwire); dossiers.py:701 already threads this same call.
            _fid = await asyncio.to_thread(
                parse_and_store_forecast, db, out, storyline_key=eff_key,
                source_monitor="Storyline Tracker")
            if not _fid and "FORECAST:" in out.upper() and "FORECAST: NONE" not in out.upper():
                logger.warning("[Storyline] FORECAST line present but not stored for %r "
                               "— mint format drift?", eff_key)
    except Exception:
        pass

    sid = await asyncio.to_thread(
        _record, db, row, story, fresh or story["developments"], summary=summary)
    logger.info("[Storyline] %s thread %r (+%d new)", "NEW" if new_story else "moved",
                story["title"], len(fresh or story["developments"]))
    narrative = changed or ("new story" if new_story else "updated")
    return {
        "title": story["title"],
        # Lead with the structured KG delta when the status moved; keep the
        # narrative one-liner alongside it.
        "changed": f"{state_delta} · {narrative}" if state_delta else narrative,
        "summary": summary,
        "new": new_story,
        "storyline_id": sid,
    }


def _record(db, row, story, developments, *, summary):
    """Upsert the storyline + append its new events. Returns storyline id."""
    monitors_csv = ",".join(story["monitors"])
    if row is None:
        cur = db.execute(
            "INSERT INTO storylines (story_key, title, summary, monitors_csv, update_count, last_updated) "
            "VALUES (?, ?, ?, ?, 1, datetime('now'))",
            (story["key"], story["title"], summary or "", monitors_csv),
        )
        sid = cur.lastrowid
    else:
        sid = row["id"]
        new_summary = summary if summary is not None else row["summary"]
        db.execute(
            "UPDATE storylines SET summary = ?, monitors_csv = ?, "
            "update_count = update_count + 1, last_updated = datetime('now'), status = 'active' "
            "WHERE id = ?",
            (new_summary, monitors_csv, sid),
        )
    for dev in developments[:10]:
        db.execute(
            "INSERT INTO storyline_events (storyline_id, summary, source_monitor, is_new) "
            "VALUES (?, ?, ?, 1)",
            (sid, _event_summary(dev), story["monitors"][0] if story["monitors"] else ""),
        )
    return sid


# Internal QA/meta annotations appended to storyline_events that are NOT
# narrative events (fresh-check notes, sourcing notes, read-sources headers).
# Moved here from api/monitors.py 2026-08-31: only the DISPLAY side filtered
# them — dossier consolidation (_storyline_sources) was reading QA notes as
# story beats. `\_%` needs ESCAPE (LIKE `_` is a wildcard).
EVENT_META_EXCL_SQL = (
    "AND summary NOT LIKE '⚠%' "
    "AND summary NOT LIKE '\\_%' ESCAPE '\\' "
    "AND summary NOT LIKE '%Fresh-check%' "
    "AND summary NOT LIKE '%Sourcing note%' "
    "AND summary NOT LIKE '%could NOT be confirmed%' "
    "AND summary NOT LIKE 'read %sources%'"
)


def _event_summary(dev: str, cap: int = 300) -> str:
    """Whole-word bound for stored event summaries (2026-08-31). The old
    `dev[:300]` hard cut left 948 of 1,761 storyline_events at exactly 300
    chars ending mid-word — the timeline the frontend, dossier consolidation
    and next-cycle dedup all read. Prefer the last sentence boundary in the
    tail; fall back to the last whole word + ellipsis."""
    d = (dev or "").strip()
    if len(d) <= cap:
        return d
    cut = d[:cap]
    tail = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if tail >= int(cap * 0.7):
        return cut[:tail + 1]
    ws = cut.rfind(" ")
    return (cut[:ws].rstrip(" ,;:—-") + "…") if ws > 0 else cut


def close_stale(db, days: int = 21) -> int:
    """Close active storylines that haven't moved in `days` days — a thread that
    hasn't developed in three weeks has stopped being 'ongoing'. Closing is NOT
    deletion: the storyline + its events stay queryable, and the next development
    flips status back to 'active' (see _record's UPDATE). Storylines were
    previously insert/update-only with no close side (31/48 active were stale
    >14d, 2026-08-14 audit). Runs from daily maintenance. Returns count closed."""
    try:
        cur = db.execute(
            "UPDATE storylines SET status = 'closed' "
            "WHERE status = 'active' AND last_updated < datetime('now', ?)",
            (f"-{days} days",),
        )
        return cur.rowcount
    except Exception as e:
        logger.warning("[Storylines] close_stale failed: %s", e)
        return 0


def get_relevant_storylines(db, query: str, *, limit: int = 2) -> list[dict]:
    """Cheap keyword match of a chat query to active storylines (no LLM).

    Makes tracked threads INTERROGABLE in chat — "where does the Iran situation
    stand?" pulls the maintained narrative state. Single indexed-ish scan over
    active threads; returns [] on any miss so the chat path never breaks.
    """
    q_toks = {t for t in re.findall(r"[a-z0-9]{4,}", (query or "").lower())}
    if not q_toks:
        return []
    try:
        rows = db.fetchall(
            "SELECT title, summary, last_updated FROM storylines "
            "WHERE status = 'active' AND summary != '' "
            "ORDER BY last_updated DESC LIMIT 80"
        )
    except Exception:
        return []
    scored = []
    for r in rows:
        hay = {t for t in re.findall(r"[a-z0-9]{4,}", f"{r['title']} {r['summary']}".lower())}
        overlap = len(q_toks & hay)
        if overlap >= 2:
            scored.append((overlap, {"title": r["title"], "summary": r["summary"]}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:limit]]


async def track_storylines(db, kg=None) -> str:
    """Full cycle: collect → cluster → diff/update → digest of moved threads.

    Returns a digest string (only moved threads) or a no-change marker. When `kg`
    is provided, each moved thread also records a functional `has_status` fact so
    state changes supersede (queryable trail + deterministic delta in the digest).
    """
    items = _collect_items(db)
    if len(items) < 3:
        return "STORYLINES | not enough recent monitor signal to track"
    # Hand the LLM the active threads so it reuses their titles for continuations
    # (primary anti-fragmentation defense; the fuzzy matcher is the backstop).
    try:
        _active = db.fetchall(
            "SELECT title FROM storylines WHERE status = 'active' ORDER BY last_updated DESC LIMIT 30"
        )
        existing_titles = [r["title"] for r in _active]
    except Exception:
        existing_titles = []
    stories = await _cluster_into_stories(items, existing_titles)
    if not stories:
        return "STORYLINES | no ongoing stories identified"

    moved = []
    for story in stories[:_MAX_STORIES]:
        try:
            upd = await _update_story(db, story, kg=kg)
        except Exception as e:
            logger.warning("[Storyline] update failed for %r: %s", story.get("title"), e)
            continue
        if upd:
            moved.append(upd)

    if not moved:
        return "STORYLINES | tracked, no threads moved this cycle"

    lines = ["## 🧵 STORYLINE UPDATES — what moved"]
    for u in moved:
        tag = "🆕 NEW" if u["new"] else "📍"
        lines.append(f"\n**{tag} {u['title']}**")
        lines.append(f"  ↳ _Changed:_ {u['changed']}")
        if u["summary"]:
            lines.append(f"  {u['summary']}")
    return "\n".join(lines)
