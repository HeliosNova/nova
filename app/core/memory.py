"""Memory — conversation history and user facts.

Two tiers:
1. Conversations + messages (SQLite) — chat history, loaded per conversation
2. User facts (SQLite) — key-value pairs about the owner, always in prompt
3. Automatic fact extraction — LLM pulls facts from conversation turns
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core import llm
from app.database import get_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation Store
# ---------------------------------------------------------------------------

@dataclass
class Message:
    id: str
    conversation_id: str
    role: str       # user | assistant | tool
    content: str
    tool_calls: list[dict] | None = None
    tool_name: str | None = None
    sources: list[dict] | None = None
    created_at: str | None = None


class ConversationStore:
    """Manages conversations and messages."""

    def __init__(self, db=None):
        self._db = db or get_db()

    def create_conversation(self, title: str | None = None) -> str:
        """Create a new conversation. Returns its ID."""
        conv_id = str(uuid.uuid4())
        self._db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conv_id, title or "New Chat"),
        )
        return conv_id

    def get_conversation(self, conv_id: str) -> dict | None:
        """Get conversation metadata."""
        row = self._db.fetchone(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        )
        return dict(row) if row else None

    def update_title(self, conv_id: str, title: str) -> None:
        self._db.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (title, conv_id),
        )

    def list_conversations(self, limit: int = 50) -> list[dict]:
        """List recent conversations, newest first."""
        rows = self._db.fetchall(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        tool_calls: list[dict] | None = None,
        tool_name: str | None = None,
        sources: list[dict] | None = None,
    ) -> str:
        """Add a message to a conversation. Returns message ID."""
        msg_id = str(uuid.uuid4())
        self._db.execute(
            """INSERT INTO messages (id, conversation_id, role, content, tool_calls, tool_name, sources)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id,
                conversation_id,
                role,
                content,
                json.dumps(tool_calls) if tool_calls else None,
                tool_name,
                json.dumps(sources) if sources else None,
            ),
        )
        # Touch conversation updated_at
        self._db.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )
        return msg_id

    def get_history(self, conversation_id: str, limit: int = 20) -> list[Message]:
        """Get recent messages for a conversation, oldest first."""
        rows = self._db.fetchall(
            """SELECT * FROM messages
               WHERE conversation_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (conversation_id, limit),
        )
        messages = []
        for row in reversed(rows):  # Reverse to get chronological order
            try:
                tc = json.loads(row["tool_calls"]) if row["tool_calls"] else None
            except json.JSONDecodeError:
                tc = None
            try:
                sr = json.loads(row["sources"]) if row["sources"] else None
            except json.JSONDecodeError:
                sr = None
            messages.append(Message(
                id=row["id"],
                conversation_id=row["conversation_id"],
                role=row["role"],
                content=row["content"],
                tool_calls=tc,
                tool_name=row["tool_name"],
                sources=sr,
                created_at=row["created_at"],
            ))
        return messages

    def get_history_as_dicts(self, conversation_id: str, limit: int = 20) -> list[dict]:
        """Get history formatted as Ollama message dicts (role + content)."""
        messages = self.get_history(conversation_id, limit)
        result = []
        for msg in messages:
            if msg.role in ("user", "assistant"):
                result.append({"role": msg.role, "content": msg.content})
            elif msg.role == "tool":
                # Tool results are injected as system messages
                result.append({
                    "role": "system",
                    "content": f"[Tool result from '{msg.tool_name}']: {msg.content}",
                })
        return result

    def search_messages(self, query: str, limit: int = 20) -> list[dict]:
        """Search across all conversation messages by text content.

        Returns matches with conversation context (title, role, timestamp).
        """
        # Use LIKE with wildcards for substring matching
        # Split query into words and require all to be present
        words = query.lower().split()[:50]
        if not words:
            return []

        # Build WHERE clause: content LIKE '%word1%' AND content LIKE '%word2%'
        conditions = " AND ".join("LOWER(m.content) LIKE ?" for _ in words)
        params = [f"%{w}%" for w in words]
        params.append(limit)

        rows = self._db.fetchall(
            f"""SELECT m.id, m.conversation_id, m.role, m.content, m.created_at,
                       c.title as conversation_title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE {conditions} AND m.role IN ('user', 'assistant')
                ORDER BY m.created_at DESC
                LIMIT ?""",
            tuple(params),
        )
        return [
            {
                "message_id": row["id"],
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "role": row["role"],
                "content": row["content"][:500],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def search_conversations(self, query: str, limit: int = 20) -> list[dict]:
        """Search conversations by title or message content.

        Returns conversations (deduplicated) with match snippets.
        """
        results = self.search_messages(query, limit=limit * 2)

        # Deduplicate by conversation_id, keep best match
        seen = {}
        for r in results:
            cid = r["conversation_id"]
            if cid not in seen:
                seen[cid] = {
                    "conversation_id": cid,
                    "title": r["conversation_title"],
                    "snippet": r["content"][:200],
                    "match_role": r["role"],
                    "created_at": r["created_at"],
                }

        return list(seen.values())[:limit]

    def delete_conversation(self, conv_id: str) -> None:
        """Delete a conversation and all its messages."""
        self._db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        self._db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    def cleanup_old_conversations(self, days: int = 90) -> int:
        """Delete conversations (and their messages) older than N days.

        Returns the number of conversations deleted.
        Uses a transaction to ensure messages and conversations are deleted atomically.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        old_convs = self._db.fetchall(
            "SELECT id FROM conversations WHERE updated_at < ?",
            (cutoff,),
        )
        if not old_convs:
            return 0

        count = len(old_convs)
        ids = [row["id"] for row in old_convs]
        # Process in batches to avoid SQLite variable limit (999)
        for i in range(0, len(ids), 500):
            batch = ids[i:i + 500]
            placeholders = ",".join("?" for _ in batch)
            with self._db.transaction() as tx:
                tx.execute(f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", tuple(batch))
                tx.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", tuple(batch))

        logger.info("Cleaned up %d conversations older than %d days", count, days)
        return count


# ---------------------------------------------------------------------------
# User Fact Store
# ---------------------------------------------------------------------------

@dataclass
class UserFact:
    id: int
    key: str
    value: str
    source: str
    confidence: float
    category: str = "fact"
    updated_at: str | None = None


class UserFactStore:
    """Key-value facts about the user. Always injected into the system prompt."""

    def __init__(self, db=None):
        self._db = db or get_db()

    def get_all(self) -> list[UserFact]:
        """Get all user facts."""
        rows = self._db.fetchall("SELECT * FROM user_facts ORDER BY key")
        result = []
        for r in rows:
            d = dict(r)
            d.setdefault("category", "fact")
            d.pop("last_accessed_at", None)
            d.pop("access_count", None)
            result.append(UserFact(**d))
        return result

    def get(self, key: str) -> UserFact | None:
        """Get a specific user fact by key."""
        row = self._db.fetchone("SELECT * FROM user_facts WHERE key = ?", (key,))
        if not row:
            return None
        d = dict(row)
        d.setdefault("category", "fact")
        d.pop("last_accessed_at", None)
        d.pop("access_count", None)
        return UserFact(**d)

    def set(self, key: str, value: str, source: str = "inferred", confidence: float = 1.0, category: str = "fact") -> None:
        """Set a user fact. Upserts (inserts or updates)."""
        if category not in ("fact", "preference", "instruction"):
            category = "fact"
        existing = self.get(key)
        if existing:
            self._db.execute(
                """UPDATE user_facts
                   SET value = ?, source = ?, confidence = ?, category = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE key = ?""",
                (value, source, confidence, category, key),
            )
        else:
            self._db.execute(
                "INSERT INTO user_facts (key, value, source, confidence, category) VALUES (?, ?, ?, ?, ?)",
                (key, value, source, confidence, category),
            )

    def delete(self, key: str) -> bool:
        """Delete a user fact. Returns True if it existed."""
        cursor = self._db.execute("DELETE FROM user_facts WHERE key = ?", (key,))
        return cursor.rowcount > 0

    def refresh_access(self, keys: list[str]) -> None:
        """Mark facts as accessed (updates last_accessed_at and access_count)."""
        if not keys:
            return
        for key in keys:
            self._db.execute(
                "UPDATE user_facts SET last_accessed_at = CURRENT_TIMESTAMP, "
                "access_count = COALESCE(access_count, 0) + 1 WHERE key = ?",
                (key,),
            )

    def get_stale_facts(self, days: int = 60) -> list[UserFact]:
        """Get facts not accessed in N days (candidates for review)."""
        rows = self._db.fetchall(
            "SELECT * FROM user_facts "
            "WHERE last_accessed_at IS NULL OR last_accessed_at < datetime('now', ?) "
            "ORDER BY last_accessed_at ASC",
            (f"-{days} days",),
        )
        result = []
        for r in rows:
            d = dict(r)
            d.setdefault("category", "fact")
            d.pop("last_accessed_at", None)
            d.pop("access_count", None)
            result.append(UserFact(**d))
        return result

    def format_for_prompt(self) -> str:
        """Format all user facts as a prompt block with separate sections for facts and instructions."""
        facts = self.get_all()
        if not facts:
            return ""

        fact_lines = [f"- {f.key}: {f.value}" for f in facts if f.category == "fact"]
        instruction_lines = [f"- {f.value}" for f in facts if f.category in ("preference", "instruction")]

        sections = []
        if fact_lines:
            sections.append("## What You Know About Your Owner\n\n" + "\n".join(fact_lines))
        if instruction_lines:
            sections.append(
                "## Owner's Standing Instructions\n\n"
                "Follow these directives in EVERY response unless explicitly overridden:\n"
                + "\n".join(instruction_lines)
            )
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Automatic Fact Extraction
# ---------------------------------------------------------------------------

# Quick regex pre-filter: only attempt LLM extraction if the user's message
# contains phrases that suggest personal info
_FACT_HINT_PATTERNS = [
    re.compile(r"(?i)\bmy\s+(?:name|job|work|company|title|role|email|phone|"
               r"birthday|location|city|country|timezone|favorite|preference)\b"),
    re.compile(r"(?i)\bi\s+(?:am|work|live|prefer|like|use|speak|study|"
               r"go\s+to|graduated|majored)\b"),
    re.compile(r"(?i)\bi'?m\s+(?:a|an|the|from|based|located|working|living|"
               r"using|learning|interested)\b"),
    re.compile(r"(?i)\bcall\s+me\b"),
    re.compile(r"(?i)\bi\s+(?:don'?t|do\s+not)\s+(?:like|use|want|eat|drink)\b"),
    re.compile(r"(?i)\bremember\s+(?:that\s+)?(?:i|my)\b"),
    re.compile(r"(?i)\balways\s+(?:include|use|prefer|show|add|give|provide)\b"),
    re.compile(r"(?i)\bnever\s+(?:include|use|show|add|give)\b"),
    re.compile(r"(?i)\b(?:from now on|going forward)\b"),
    # Broader project/workplace patterns
    re.compile(r"(?i)\bwe\s+use\b"),
    re.compile(r"(?i)\bour\s+(?:stack|team|project|company|workflow|setup|infra)\b"),
    re.compile(r"(?i)\bat\s+work\b"),
    re.compile(r"(?i)\b(?:project|app|codebase|repo)\s+uses\b"),
    re.compile(r"(?i)\bmy\s+setup\b"),
]

_EXTRACTION_PROMPT = """Extract personal facts and behavioral preferences from this conversation exchange.

Only extract STABLE facts (name, job, preferences, location, expertise, etc.).
Do NOT extract:
- Temporary states ("I'm tired", "I'm busy today")
- One-time opinions without "always/prefer/favorite" language
- Facts about other people
- The AI's own responses or examples

IMPORTANT: The assistant response is provided for context only. Only extract facts stated
BY THE USER in their message, never from the assistant's examples or references.

Return a JSON object where each key maps to an object with "value" and "category".
Keys should be lowercase_snake_case. Use CANONICAL key names: name, employer, job_title,
location, timezone, preferred_language, etc. Do NOT create synonyms or variations
(e.g. use "preferred_language" not "favorite_programming_language" or "language_preference").
Categories: "fact" (personal info), "preference" (style/format preference), "instruction" (standing directive).

If no personal facts are found, return: {}

Examples:
User: "My name is Alex and I work at Google as a senior engineer"
→ {"name": {"value": "Alex", "category": "fact"}, "employer": {"value": "Google", "category": "fact"}, "job_title": {"value": "Senior Engineer", "category": "fact"}}

User: "Always include code examples when explaining technical concepts"
→ {"pref_code_examples": {"value": "Always include code examples when explaining technical concepts", "category": "instruction"}}

User: "I prefer concise answers without lengthy preambles"
→ {"pref_concise_answers": {"value": "Prefer concise answers without lengthy preambles", "category": "preference"}}

User: "I'm based in Tokyo, timezone is JST"
→ {"location": {"value": "Tokyo", "category": "fact"}, "timezone": {"value": "JST", "category": "fact"}}

User: "What's the weather like tomorrow?"
→ {}"""


def has_fact_signals(text: str) -> bool:
    """Fast check: does the message likely contain personal facts worth extracting?"""
    return any(p.search(text) for p in _FACT_HINT_PATTERNS)


_ERROR_VALUE_PHRASES = [
    "no stable", "not found", "cannot determine", "unable to",
    "error", "failed", "exception", "traceback",
]

_MAX_USER_FACTS = 30


def _has_semantic_overlap(new_key: str, existing_keys: list[str], threshold: float = 0.35) -> str | None:
    """Check if new_key has significant word overlap with any existing key.

    Returns the existing key if overlap >= threshold, else None.
    """
    new_words = set(new_key.split("_"))
    new_words.discard("")
    if not new_words:
        return None
    for ek in existing_keys:
        ek_words = set(ek.split("_"))
        ek_words.discard("")
        if not ek_words:
            continue
        overlap = len(new_words & ek_words)
        union = len(new_words | ek_words)
        # Require at least 2 overlapping words to avoid false positives
        if overlap >= 2 and union > 0 and overlap / union >= threshold:
            return ek
    return None


async def extract_facts_from_message(
    user_message: str,
    assistant_response: str = "",
    *,
    fact_store: UserFactStore | None = None,
) -> dict[str, dict]:
    """Use LLM to extract user facts from a conversation turn.

    Returns dict of {key: {"value": str, "category": str}} facts, or empty dict if none found.
    Supports both new format (dict with value/category) and legacy format (plain string → category "fact").
    Only called when has_fact_signals() returns True.
    """
    _VALID_CATEGORIES = {"fact", "preference", "instruction"}

    # Max facts cap — skip extraction entirely if already at limit
    if fact_store is not None:
        existing_facts = fact_store.get_all()
        if len(existing_facts) >= _MAX_USER_FACTS:
            logger.warning("User facts at %d (limit %d), skipping extraction", len(existing_facts), _MAX_USER_FACTS)
            return {}
        existing_keys = [f.key for f in existing_facts]
    else:
        existing_keys = []

    # Build input: show the exchange so the LLM has context
    exchange = f'User: "{user_message}"'
    if assistant_response:
        exchange += f'\nAssistant: "{assistant_response[:300]}"'

    try:
        result = await llm.invoke_nothink(
            [
                {"role": "system", "content": _EXTRACTION_PROMPT},
                {"role": "user", "content": exchange},
            ],
            json_mode=True,
            json_prefix="{",
            max_tokens=300,
            temperature=0.1,
        )

        obj = llm.extract_json_object(result)
        if not obj or not isinstance(obj, dict):
            return {}

        facts: dict[str, dict] = {}
        for k, v in obj.items():
            k = str(k).strip().lower().replace(" ", "_")
            # Reject keys that look like garbage
            if len(k) < 2 or not re.match(r"^[a-z][a-z0-9_]{1,50}$", k):
                continue

            # Support both old format (plain string) and new format (dict)
            if isinstance(v, dict):
                value = str(v.get("value", "")).strip()
                category = str(v.get("category", "fact")).strip().lower()
            else:
                value = str(v).strip()
                category = "fact"

            if len(value) < 1 or len(value) > 200:
                continue
            if category not in _VALID_CATEGORIES:
                category = "fact"

            # Guard: skip values that look like error messages
            value_lower = value.lower()
            if any(phrase in value_lower for phrase in _ERROR_VALUE_PHRASES):
                logger.info("Skipping error-like fact value: %s=%s", k, value[:60])
                continue

            # Semantic dedup: if a similar key already exists, overwrite it instead
            if existing_keys:
                similar = _has_semantic_overlap(k, existing_keys)
                if similar and similar != k:
                    logger.info("Semantic dedup: '%s' overlaps with existing '%s', using existing key", k, similar)
                    k = similar

            facts[k] = {"value": value, "category": category}

        return facts

    except Exception as e:
        logger.warning("Fact extraction failed: %s", e)
        return {}
