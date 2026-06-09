"""GSW (Generative Semantic Workspace) — episodic memory layer.

Builds a rolling per-conversation narrative summary so Nova can pick up where
prior sessions left off. The architecture follows the GSW paper (arxiv 2511.07587):

    Operator   - extracts a space-time-anchored narrative from conversation
                  messages. Returns short summary + key entities + expanded
                  narrative.
    Reconciler - merges a new operator output with the existing summary for
                  the same conversation, keeping the narrative current.

The summary is retrieved at the start of subsequent conversations (via key
entity overlap) so cross-session continuity works without dragging the entire
message history into context. Distinct from:
    - lessons       (correction-derived patterns, query-shaped)
    - kg_facts      (atomic triples about the world)
    - workspace     (per-query findings cache)
This layer captures *what happened in our shared interaction* over time.

DB table: conversation_summaries (see database.py SCHEMA_SQL).

Public surface:
    extract_summary(messages, prior_summary=None) -> dict
    save_summary(db, conversation_id, summary_dict)
    get_relevant_summaries(db, query, limit=3) -> list[dict]
    format_for_prompt(summaries) -> str
    maybe_update_summary(db, conversation_id) -> bool   # background-friendly entry point
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.config import config
from app.database import SafeDB, get_db

logger = logging.getLogger(__name__)


# Tunables
_GSW_MIN_MESSAGES = 4               # below this, conversation isn't worth summarizing
_GSW_RESUMMARIZE_INTERVAL = 6       # re-summarize every N new messages
_GSW_MAX_SUMMARY_TOKENS = 400       # LLM budget per summary
_GSW_NARRATIVE_BUDGET = 800
_GSW_MAX_RETRIEVED = 3              # at most N prior session summaries injected


_OPERATOR_PROMPT = """You are summarizing an ongoing conversation between a user and Nova.
Produce a JSON object with these fields:
  "summary":   a single sentence (max 30 words) capturing what we worked on this session
  "narrative": 2-4 sentences with concrete details (what was decided, key entities, timeline)
  "key_entities": array of 3-8 lowercase strings naming the topics/people/projects discussed

Rules:
- Be concrete: name systems, files, decisions, deadlines.
- Lead with the *outcome* or *current state*, not "we discussed".
- Don't invent details that aren't in the messages.
- "Nova" is the assistant; treat the user as "they" or "the user".

Conversation messages (oldest first):
{messages}

Output: a single JSON object, no preamble."""


_RECONCILER_PROMPT = """You're merging two summaries of the same ongoing conversation. The
"prior" summary is older; the "recent" summary describes new messages. Produce a single
merged JSON object that:
- Preserves what's still current from the prior
- Updates anything the recent summary contradicts
- Adds new entities/decisions from recent
- Drops detail that's been superseded

Output JSON: {{"summary": "...", "narrative": "...", "key_entities": [...]}}.

PRIOR:
{prior}

RECENT:
{recent}

Output: a single JSON object, no preamble."""


def is_enabled() -> bool:
    return bool(getattr(config, "ENABLE_GSW_EPISODIC", False))


# ---------------------------------------------------------------------------
# Operator — extract narrative from a message slice
# ---------------------------------------------------------------------------

async def extract_summary(
    messages: list[dict[str, Any]],
    prior_summary: dict | None = None,
) -> dict | None:
    """Run the GSW Operator: turn message list into structured summary.

    `messages`: ordered (oldest first) dicts with at least 'role' and 'content'.
    `prior_summary`: optional prior summary dict to feed into the Reconciler step.

    Returns dict with keys 'summary', 'narrative', 'key_entities'. Returns None
    on LLM failure or empty inputs.
    """
    if not messages or len(messages) < _GSW_MIN_MESSAGES:
        return None

    # Build a compact message rendering. Cap content to keep prompt size bounded
    # — long single messages get truncated, but ordering is preserved.
    rendered_lines: list[str] = []
    char_budget = 4000
    used = 0
    for msg in messages:
        role = msg.get("role", "?")
        if role not in ("user", "assistant", "system"):
            continue
        content = msg.get("content", "") or ""
        if not content.strip():
            continue
        if role == "system":
            # Skip system messages — they're prompts, not conversation
            continue
        snippet = content[:600]
        line = f"[{role}] {snippet}"
        if used + len(line) > char_budget:
            line = line[: char_budget - used] + "…"
            rendered_lines.append(line)
            break
        rendered_lines.append(line)
        used += len(line)

    if not rendered_lines:
        return None

    rendered = "\n".join(rendered_lines)

    try:
        from app.core import llm
        operator_raw = await asyncio.wait_for(
            llm.invoke_nothink(
                [{"role": "user", "content": _OPERATOR_PROMPT.format(messages=rendered)}],
                json_mode=True,
                json_prefix="{",
                max_tokens=_GSW_MAX_SUMMARY_TOKENS,
                temperature=0.2,
            ),
            timeout=max(float(config.INTERNAL_LLM_TIMEOUT), 60.0),
        )
    except Exception as e:
        logger.warning("[GSW Operator] LLM failed: %s", e)
        return None

    summary = _parse_summary_json(operator_raw)
    if not summary:
        return None

    if not prior_summary:
        return summary

    # Reconciler step
    try:
        recon_raw = await asyncio.wait_for(
            llm.invoke_nothink(
                [{
                    "role": "user",
                    "content": _RECONCILER_PROMPT.format(
                        prior=json.dumps(prior_summary, ensure_ascii=False),
                        recent=json.dumps(summary, ensure_ascii=False),
                    ),
                }],
                json_mode=True,
                json_prefix="{",
                max_tokens=_GSW_MAX_SUMMARY_TOKENS,
                temperature=0.2,
            ),
            timeout=max(float(config.INTERNAL_LLM_TIMEOUT), 60.0),
        )
        merged = _parse_summary_json(recon_raw)
        if merged:
            return merged
    except Exception as e:
        logger.warning("[GSW Reconciler] LLM failed: %s — using fresh summary", e)

    return summary


def _parse_summary_json(raw: str | None) -> dict | None:
    """Parse the operator/reconciler JSON output. Returns None on bad shape."""
    if not raw:
        return None
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        # Try balanced brace extraction
        if not isinstance(raw, str):
            return None
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None

    if not isinstance(obj, dict):
        return None
    s = obj.get("summary")
    n = obj.get("narrative") or s
    e = obj.get("key_entities") or []
    if not isinstance(s, str) or not s.strip():
        return None
    if not isinstance(e, list):
        e = []
    e = [str(x).strip().lower() for x in e if x and len(str(x)) < 80][:12]
    return {
        "summary": s.strip()[:600],
        "narrative": str(n).strip()[:_GSW_NARRATIVE_BUDGET],
        "key_entities": e,
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_summary(
    db: SafeDB,
    conversation_id: str,
    summary_dict: dict,
    last_message_id: str | None = None,
    message_count: int = 0,
) -> int | None:
    """Insert a new summary row, retiring any prior current summary for this conversation.

    Temporal store — we don't update in place; old rows get valid_to set so the
    timeline is recoverable.
    """
    if not summary_dict or not summary_dict.get("summary"):
        return None
    try:
        # Retire prior current summary
        db.execute(
            "UPDATE conversation_summaries SET valid_to = CURRENT_TIMESTAMP "
            "WHERE conversation_id = ? AND valid_to IS NULL",
            (conversation_id,),
        )
        cursor = db.execute(
            "INSERT INTO conversation_summaries "
            "(conversation_id, summary, narrative, key_entities, message_count, last_message_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                summary_dict["summary"],
                summary_dict.get("narrative") or "",
                json.dumps(summary_dict.get("key_entities") or []),
                message_count,
                last_message_id,
            ),
        )
        new_id = cursor.lastrowid
        logger.info(
            "[GSW] saved summary #%d for conv=%s (entities=%s)",
            new_id, conversation_id[:12], summary_dict.get("key_entities", [])[:3],
        )
        return new_id
    except Exception as e:
        logger.warning("[GSW] save_summary failed: %s", e)
        return None


def get_current_summary(db: SafeDB, conversation_id: str) -> dict | None:
    """Return the active summary for a conversation, or None."""
    row = db.fetchone(
        "SELECT summary, narrative, key_entities, message_count, last_message_id "
        "FROM conversation_summaries "
        "WHERE conversation_id = ? AND valid_to IS NULL "
        "ORDER BY id DESC LIMIT 1",
        (conversation_id,),
    )
    if not row:
        return None
    try:
        entities = json.loads(row["key_entities"]) if row["key_entities"] else []
    except Exception:
        entities = []
    return {
        "summary": row["summary"],
        "narrative": row["narrative"] or "",
        "key_entities": entities,
        "message_count": row["message_count"] or 0,
        "last_message_id": row["last_message_id"],
    }


def get_relevant_summaries(db: SafeDB, query: str, limit: int = _GSW_MAX_RETRIEVED) -> list[dict]:
    """Fetch session summaries whose key_entities overlap with the query.

    Used at the start of new conversations to surface "what we worked on
    recently". Returns up to `limit` summaries ordered by recency.
    """
    if not query:
        return []
    # Cheap entity extraction — re-use PPR's
    try:
        from app.core.ppr import extract_entities
        seeds = set(extract_entities(query, max_seeds=8))
    except Exception:
        seeds = set()
    if not seeds:
        return []

    rows = db.fetchall(
        "SELECT conversation_id, summary, narrative, key_entities, created_at "
        "FROM conversation_summaries "
        "WHERE valid_to IS NULL "
        "ORDER BY id DESC LIMIT 200"
    )

    scored: list[tuple[int, dict]] = []
    for row in rows:
        try:
            entities = json.loads(row["key_entities"]) if row["key_entities"] else []
        except Exception:
            entities = []
        ent_set = set(e.lower() for e in entities)
        # Substring match — "apple" matches "apple silicon" etc.
        overlap = 0
        for s in seeds:
            for e in ent_set:
                if s == e or s in e or e in s:
                    overlap += 1
                    break
        if overlap > 0:
            scored.append((overlap, {
                "conversation_id": row["conversation_id"],
                "summary": row["summary"],
                "narrative": row["narrative"] or "",
                "key_entities": entities,
                "created_at": row["created_at"],
                "overlap": overlap,
            }))

    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:limit]]


def format_for_prompt(summaries: list[dict]) -> str:
    """Render relevant prior session summaries as a compact context block."""
    if not summaries:
        return ""
    lines = []
    for s in summaries:
        date = (s.get("created_at") or "").split("T")[0][:10]
        ents = ", ".join(s.get("key_entities", [])[:5])
        text = s.get("narrative") or s.get("summary") or ""
        if not text:
            continue
        prefix = f"[{date}]" if date else "[prior]"
        line = f"- {prefix} {text[:280]}"
        if ents:
            line += f"  (re: {ents[:80]})"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Update entry point
# ---------------------------------------------------------------------------

async def maybe_update_summary(db: SafeDB, conversation_id: str) -> bool:
    """Background-friendly: if the conversation has new messages since last
    summary, re-run the Operator/Reconciler and save.

    Returns True if a new summary was saved.
    """
    if not is_enabled() or not conversation_id:
        return False
    try:
        # Fetch messages
        msg_rows = db.fetchall(
            "SELECT id, role, content FROM messages "
            "WHERE conversation_id = ? "
            "ORDER BY created_at ASC LIMIT 200",
            (conversation_id,),
        )
        messages = [
            {"id": r["id"], "role": r["role"], "content": r["content"] or ""}
            for r in msg_rows
            if r["role"] in ("user", "assistant")
        ]
        if len(messages) < _GSW_MIN_MESSAGES:
            return False

        prior = get_current_summary(db, conversation_id)
        if prior:
            # Skip if message count hasn't grown enough since last summary
            if len(messages) - (prior.get("message_count") or 0) < _GSW_RESUMMARIZE_INTERVAL:
                return False

        # Use only the messages we haven't summarized yet (plus a few of the
        # last summarized for continuity), capped at 30 to bound prompt size.
        slice_start = max(0, len(messages) - 30)
        slice_messages = messages[slice_start:]

        summary = await extract_summary(slice_messages, prior_summary=prior)
        if not summary:
            return False

        last_id = messages[-1]["id"] if messages else None
        save_summary(
            db,
            conversation_id=conversation_id,
            summary_dict=summary,
            last_message_id=last_id,
            message_count=len(messages),
        )
        return True
    except Exception as e:
        logger.warning("[GSW] maybe_update_summary failed: %s", e)
        return False
