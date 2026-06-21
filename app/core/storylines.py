"""Storyline tracking + change detection (Monitor Intelligence v2, Phase A).

Turns the per-domain news feed into persistent STORY THREADS with state. Instead
of 50 disconnected headlines, Nova maintains named storylines ("Iran–Hormuz
crisis"), accrues developments, and surfaces ONLY what MOVED — "here's how your
threads changed" — not raw items.

Reuses:
  - cross_monitor._gather_recent_outputs : recent content monitor_results
  - cross_monitor._extract_signals       : cheap entity/phrase pre-cluster
  - kg bitemporal API (add_fact supersession) : state-change detection
  - llm.invoke_nothink                    : naming + summary passes

Background-only (runs on the Storyline Tracker monitor schedule). One LLM pass to
cluster+name, one per moved thread to update its summary — bounded.
"""

from __future__ import annotations

import json
import logging
import re

from app.core import llm
from app.core.cross_monitor import _gather_recent_outputs, _extract_signals

logger = logging.getLogger(__name__)

# How far back to scan monitor outputs for developments.
_WINDOW_HOURS = 48
# Cap stories per cycle so the LLM passes stay bounded on a single GPU.
_MAX_STORIES = 8


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
    "You track an ongoing story. Here is its prior state, then NEW developments.\n\n"
    "STORY: {title}\n"
    "PRIOR STATE: {prior}\n\n"
    "NEW DEVELOPMENTS:\n{developments}\n\n"
    "Write an updated state in 2-3 sentences (the current situation, incorporating "
    "what's new). Then on a new line starting 'CHANGED: ' give ONE sentence on what "
    "specifically moved since the prior state.\n"
    "If — and only if — there is a clear, falsifiable near-term expectation, add a "
    "final line 'FORECAST: <specific testable claim> | <N> days | <0.x confidence>'.\n"
    "If the new developments add nothing substantive, reply exactly 'NO CHANGE'."
)


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
                clean = re.sub(r"^[`*#>\d\.\-\s]+", "", line)[:300]
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
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True, json_prefix="[{", max_tokens=700, temperature=0.2,
        )
    except Exception as e:
        logger.warning("[Storyline] cluster LLM failed: %s", e)
        return []
    if not raw:
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


async def _update_story(db, story: dict) -> dict | None:
    """Match story to an existing storyline, diff new developments, update state.

    Returns a digest-ready dict {title, changed, summary} ONLY if the thread moved,
    else None.
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
    new_story = row is None

    # One LLM pass to update the narrative state + name what changed.
    try:
        out = await llm.invoke_nothink(
            [{"role": "user", "content": _UPDATE_PROMPT.format(
                title=story["title"], prior=prior_summary or "(new story — no prior state)",
                developments=dev_text)}],
            max_tokens=260, temperature=0.2,
        )
    except Exception as e:
        logger.warning("[Storyline] update LLM failed for %r: %s", story["title"], e)
        return None
    out = (out or "").strip()
    if not out or out.upper().startswith("NO CHANGE"):
        # Still record events so we don't re-summarize them next cycle.
        if not new_story:
            _record(db, row, story, fresh, summary=None)
        return None

    summary, changed = out, ""
    m = re.search(r"(?im)^CHANGED:\s*(.+?)(?=\n|$)", out)
    if m:
        changed = m.group(1).strip()
        summary = out[:m.start()].strip()

    # Opportunistic forecast: if the model emitted a falsifiable call, record it
    # for later self-grading (Phase C). Gated; failures are non-fatal.
    try:
        from app.config import config as _cfg
        if getattr(_cfg, "ENABLE_FORECASTS", True):
            from app.core.forecasts import parse_and_store_forecast
            parse_and_store_forecast(db, out, storyline_key=story["key"],
                                     source_monitor="Storyline Tracker")
    except Exception:
        pass

    sid = _record(db, row, story, fresh or story["developments"], summary=summary)
    logger.info("[Storyline] %s thread %r (+%d new)", "NEW" if new_story else "moved",
                story["title"], len(fresh or story["developments"]))
    return {
        "title": story["title"],
        "changed": changed or ("new story" if new_story else "updated"),
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
            (sid, dev[:300], story["monitors"][0] if story["monitors"] else ""),
        )
    return sid


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


async def track_storylines(db) -> str:
    """Full cycle: collect → cluster → diff/update → digest of moved threads.

    Returns a digest string (only moved threads) or a no-change marker.
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
            upd = await _update_story(db, story)
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
