"""Memory — conversation history and user facts.

Two tiers:
1. Conversations + messages (SQLite) — chat history, loaded per conversation
2. User facts (SQLite) — key-value pairs about the owner, manual-write only.
   Writes happen via the active_memory tool or direct corrections; no passive
   inference (removed after test personas polluted production memory).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

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
        with self._db.transaction() as tx:
            tx.execute(
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
            tx.execute(
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
                result.append({
                    "role": "assistant",
                    "content": f"[Tool '{msg.tool_name}' executed successfully]: {msg.content}",
                })
        return result

    @staticmethod
    def _escape_like(s: str) -> str:
        """Escape LIKE wildcards in user input to prevent wildcard expansion."""
        return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

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
        conditions = " AND ".join("LOWER(m.content) LIKE ? ESCAPE '\\'" for _ in words)
        params = [f"%{self._escape_like(w)}%" for w in words]
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
        with self._db.transaction() as tx:
            tx.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            tx.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

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


_SOURCE_AUTHORITY: dict[str, int] = {
    "user": 4,
    "correction": 3,
    "inferred": 2,
    "extracted": 1,
}


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
        """Set a user fact. Upserts (inserts or updates).

        Source authority hierarchy prevents lower-authority sources from
        overwriting higher-authority facts (e.g. extracted won't overwrite user).
        """
        if category not in ("fact", "preference", "instruction"):
            category = "fact"
        existing = self.get(key)
        if existing:
            new_rank = _SOURCE_AUTHORITY.get(source, 0)
            old_rank = _SOURCE_AUTHORITY.get(existing.source, 0)
            # Only overwrite if new source is equally or more authoritative
            # (same authority always overwrites — user correcting their own facts)
            if new_rank < old_rank:
                logger.debug(
                    "Skipping fact overwrite: key=%s, existing source=%s (rank %d), new source=%s (rank %d)",
                    key, existing.source, old_rank, source, new_rank,
                )
                return
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

        fact_lines = [f"- {f.key}: {f.value.replace(chr(10), ' ').strip()}" for f in facts if f.category == "fact"]
        instruction_lines = [f"- {f.value.replace(chr(10), ' ').strip()}" for f in facts if f.category in ("preference", "instruction")]

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


