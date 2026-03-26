"""Curiosity engine — self-directed learning from detected knowledge gaps.

Components:
- detect_gaps(): heuristic gap detection from responses
- CuriosityQueue: SQLite-backed priority queue for research topics
- TopicTracker: frequency tracking for auto-monitor creation
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.config import config
from app.core.text_utils import normalize_words

logger = logging.getLogger(__name__)

_URGENCY_CRITICAL = 0.8
_URGENCY_HIGH = 0.6
_URGENCY_MEDIUM = 0.4
_URGENCY_LOW = 0.3

# ---------------------------------------------------------------------------
# Gap detection — heuristic, no LLM call
# ---------------------------------------------------------------------------

_HEDGING_PATTERNS = [
    re.compile(r"(?i)\b(?:i'?m\s+not\s+(?:sure|certain|confident))\b"),
    re.compile(r"(?i)\b(?:i\s+(?:think|believe|assume)\s+(?:that\s+)?)\b"),
    re.compile(r"(?i)\b(?:(?:may|might|could)\s+be)\b"),
    re.compile(r"(?i)\b(?:approximately|roughly|around|about)\b"),
    re.compile(r"(?i)\b(?:as\s+far\s+as\s+i\s+know)\b"),
    re.compile(r"(?i)\b(?:if\s+i\s+recall\s+correctly)\b"),
]

_ADMISSION_PATTERNS = [
    re.compile(r"(?i)\b(?:i\s+don'?t\s+(?:know|have))\b"),
    re.compile(r"(?i)\b(?:i\s+(?:can'?t|cannot)\s+(?:find|determine|access))\b"),
    re.compile(r"(?i)\b(?:i\s+(?:wasn'?t|was\s+not)\s+able)\b"),
    re.compile(r"(?i)\b(?:no\s+(?:information|data|results)\s+(?:available|found))\b"),
    re.compile(r"(?i)\b(?:beyond\s+my\s+(?:knowledge|training))\b"),
]

_TOOL_FAILURE_MARKERS = ("failed", "timed out", "error", "not available", "[tool")

# Tool command queries should not become curiosity research topics
_TOOL_COMMAND_RE = re.compile(
    r"(?i)(use the browser|fill.*form|httpbin|submit|navigate to|go to https?://)"
)


def detect_gaps(
    query: str,
    answer: str,
    tool_results: list[dict],
    had_lessons: bool,
    had_kg: bool,
    had_docs: bool,
) -> list[dict]:
    """Detect knowledge gaps from a response. Returns list of gap dicts.

    Each gap: {"topic": str, "source": str, "urgency": float}
    """
    gaps: list[dict] = []
    lower = answer.lower()
    topic = TopicTracker._extract_topic(query[:200])

    # Tool commands should never become research topics
    if _TOOL_COMMAND_RE.search(query):
        return []

    # Check for admissions of ignorance (highest urgency)
    admission_hits = sum(1 for p in _ADMISSION_PATTERNS if p.search(answer))
    if admission_hits >= 1:
        gaps.append({"topic": topic, "source": "admission", "urgency": _URGENCY_CRITICAL})
        return gaps  # One gap per query is enough

    # Check for tool failures — but not if the query is a tool command itself
    tool_failures = sum(
        1 for tr in tool_results
        if any(m in str(tr.get("output", "")).lower() for m in _TOOL_FAILURE_MARKERS)
    )
    if tool_failures:
        gaps.append({"topic": topic, "source": "tool_failure", "urgency": _URGENCY_HIGH})
        return gaps

    # Check for hedging language
    hedging_hits = sum(1 for p in _HEDGING_PATTERNS if p.search(answer))
    if hedging_hits >= 2:
        gaps.append({"topic": topic, "source": "hedging", "urgency": _URGENCY_MEDIUM})
        return gaps

    # Check for missing context (no lessons, KG, or docs matched)
    if not had_lessons and not had_kg and not had_docs and len(query) > 30:
        gaps.append({"topic": topic, "source": "context_gap", "urgency": _URGENCY_LOW})

    return gaps


# ---------------------------------------------------------------------------
# CuriosityQueue — SQLite-backed research queue
# ---------------------------------------------------------------------------

class _LazyConfigInt:
    """Proxy that reads from config at access time, not import time.

    Supports int operations so existing code (comparisons, range()) works.
    """
    def __init__(self, attr: str):
        self._attr = attr
    def _val(self):
        return getattr(config, self._attr)
    def __eq__(self, other):
        return self._val() == other
    def __ge__(self, other):
        return self._val() >= other
    def __le__(self, other):
        return self._val() <= other
    def __lt__(self, other):
        return self._val() < other
    def __gt__(self, other):
        return self._val() > other
    def __int__(self):
        return self._val()
    def __index__(self):
        return self._val()
    def __repr__(self):
        return repr(self._val())
    def __hash__(self):
        return hash(self._val())

MAX_PENDING = _LazyConfigInt("MAX_CURIOSITY_PENDING")
MAX_ATTEMPTS = _LazyConfigInt("MAX_CURIOSITY_ATTEMPTS")


@dataclass
class CuriosityItem:
    id: int
    topic: str
    source: str
    urgency: float
    status: str           # 'pending', 'resolved', 'failed', 'dismissed'
    attempts: int
    resolution: str | None
    created_at: str
    resolved_at: str | None


class CuriosityQueue:
    """Priority queue for self-directed research topics."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS curiosity_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        source TEXT DEFAULT 'gap_detection',
        urgency REAL DEFAULT 0.5,
        status TEXT DEFAULT 'pending',
        attempts INTEGER DEFAULT 0,
        resolution TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved_at TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_curiosity_status ON curiosity_queue(status);
    CREATE INDEX IF NOT EXISTS idx_curiosity_urgency ON curiosity_queue(urgency);
    """

    def __init__(self, db):
        self._db = db
        for stmt in self._SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._db.execute(stmt)

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Compute Jaccard word similarity between two strings."""
        words_a = normalize_words(a)
        words_b = normalize_words(b)
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    # Patterns indicating invalid/low-quality curiosity topics
    _INVALID_TOPIC_RE = [
        re.compile(r"(?i)^(hey|hi|hello|yo|sup|what'?s up|how are you)"),
        re.compile(r"^\s*[\d\s\+\-\*\/\.\(\)\^%=]+\s*$"),  # pure math
        re.compile(r"(?i)(ignore|forget|disregard|override|pretend|act as)"),
        re.compile(r"(?i)^(yes|no|ok|sure|thanks|thank you|please|sorry)\b"),
        # Tool/test commands
        re.compile(r"(?i)(use the browser|httpbin|fill.*(form|out)|submit.*(form|it))"),
        re.compile(r"(?i)^(search|go to|visit|navigate|open|click|type)\s"),
        # Conversational/trivial
        re.compile(r"(?i)(tell me .* joke|tell me a joke|what did (i|you) (just )?(say|said)|repeat that)"),
        re.compile(r"(?i)^(test|ping|debug|check|try)\b"),
        # URLs as topics
        re.compile(r"https?://"),
        # Declarative assertions — user stating facts, not asking questions
        # "the capital of france is berlin", "my name is Alex", "actually X is Y"
        re.compile(r"(?i)^(?:the|my|his|her|its|our|their|this|that|it|i|we|you|actually)\b.+\b(?:is|are|was|were|am)\b"),
        # Corrections — these create lessons, not curiosity items
        re.compile(r"(?i)^(?:no|nope|wrong|incorrect|that'?s (?:not|wrong)|actually)"),
        # Calculator-style queries
        re.compile(r"(?i)^(?:what is|calculate|compute)\s+\d"),
        # Very short generic queries
        re.compile(r"(?i)^(?:use|do|make|run|set|get|put|add|try|fix)\s"),
    ]

    @classmethod
    def _is_valid_topic(cls, topic: str) -> bool:
        """Check if a topic is worth researching."""
        t = topic.strip()
        if len(t) < 15 or len(t.split()) < 2:
            return False
        for pat in cls._INVALID_TOPIC_RE:
            if pat.search(t):
                return False
        return True

    def add(self, topic: str, source: str = "gap_detection", urgency: float = 0.5) -> int:
        """Add a topic to the queue. Deduplicates by boosting urgency if already pending."""
        topic = topic.strip()[:500]
        if not topic:
            return -1

        if not self._is_valid_topic(topic):
            logger.debug("Curiosity topic rejected (validation): '%s'", topic[:80])
            return -1

        # Dedup: exact match
        existing = self._db.fetchone(
            "SELECT id, urgency FROM curiosity_queue "
            "WHERE status = 'pending' AND topic = ?",
            (topic,),
        )
        if existing:
            new_urgency = min(1.0, existing["urgency"] + 0.1)
            self._db.execute(
                "UPDATE curiosity_queue SET urgency = ? WHERE id = ?",
                (new_urgency, existing["id"]),
            )
            return existing["id"]

        # Dedup: Jaccard fuzzy match against pending topics
        pending = self._db.fetchall(
            "SELECT id, topic, urgency FROM curiosity_queue WHERE status = 'pending'"
        )
        for row in pending:
            if self._jaccard_similarity(topic, row["topic"]) > 0.6:
                new_urgency = min(1.0, row["urgency"] + 0.1)
                self._db.execute(
                    "UPDATE curiosity_queue SET urgency = ? WHERE id = ?",
                    (new_urgency, row["id"]),
                )
                logger.debug(
                    "Curiosity dedup: '%s' fuzzy-matched existing '%s' (Jaccard > 0.6)",
                    topic[:80], row["topic"][:80],
                )
                return row["id"]

        # Cap pending items
        pending_count = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM curiosity_queue WHERE status = 'pending'"
        )["c"]
        if pending_count >= MAX_PENDING:
            # Remove lowest-urgency pending item
            self._db.execute(
                "DELETE FROM curiosity_queue WHERE id = ("
                "  SELECT id FROM curiosity_queue WHERE status = 'pending' "
                "  ORDER BY urgency ASC LIMIT 1"
                ")"
            )

        cursor = self._db.execute(
            "INSERT INTO curiosity_queue (topic, source, urgency) VALUES (?, ?, ?)",
            (topic, source, urgency),
        )
        return cursor.lastrowid

    def get_next(self) -> CuriosityItem | None:
        """Get the highest-urgency pending item."""
        row = self._db.fetchone(
            "SELECT * FROM curiosity_queue "
            "WHERE status = 'pending' AND attempts < ? "
            "ORDER BY urgency DESC, created_at ASC LIMIT 1",
            (int(MAX_ATTEMPTS),),
        )
        return self._row_to_item(row) if row else None

    def resolve(self, item_id: int, resolution: str) -> None:
        """Mark item as resolved with findings."""
        self._db.execute(
            "UPDATE curiosity_queue SET status = 'resolved', resolution = ?, "
            "resolved_at = CURRENT_TIMESTAMP WHERE id = ?",
            (resolution[:2000], item_id),
        )

    def fail(self, item_id: int) -> None:
        """Record a failed attempt. Auto-fails after MAX_ATTEMPTS."""
        self._db.execute(
            "UPDATE curiosity_queue SET attempts = attempts + 1 WHERE id = ?",
            (item_id,),
        )
        row = self._db.fetchone(
            "SELECT attempts FROM curiosity_queue WHERE id = ?", (item_id,)
        )
        if row and row["attempts"] >= MAX_ATTEMPTS:
            self._db.execute(
                "UPDATE curiosity_queue SET status = 'failed' WHERE id = ?",
                (item_id,),
            )

    def dismiss(self, item_id: int) -> None:
        """Manually dismiss an item."""
        self._db.execute(
            "UPDATE curiosity_queue SET status = 'dismissed' WHERE id = ?",
            (item_id,),
        )

    def get_recent(self, limit: int = 20) -> list[CuriosityItem]:
        """Get recent items across all statuses."""
        rows = self._db.fetchall(
            "SELECT * FROM curiosity_queue ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_item(r) for r in rows]

    def get_stats(self) -> dict:
        """Return queue statistics."""
        total = self._db.fetchone("SELECT COUNT(*) AS c FROM curiosity_queue")["c"]
        pending = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM curiosity_queue WHERE status = 'pending'"
        )["c"]
        resolved = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM curiosity_queue WHERE status = 'resolved'"
        )["c"]
        failed = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM curiosity_queue WHERE status = 'failed'"
        )["c"]
        return {
            "total": total,
            "pending": pending,
            "resolved": resolved,
            "failed": failed,
        }

    def get_resolved_since(self, hours: int = 24) -> list[CuriosityItem]:
        """Get items resolved in the last N hours (for digest)."""
        rows = self._db.fetchall(
            "SELECT * FROM curiosity_queue "
            "WHERE status = 'resolved' AND resolved_at > datetime('now', ?) "
            "ORDER BY resolved_at DESC",
            (f"-{hours} hours",),
        )
        return [self._row_to_item(r) for r in rows]

    def get_failed_exhausted(self, hours: int = 24) -> list[CuriosityItem]:
        """Get items that exhausted all attempts in the last N hours."""
        rows = self._db.fetchall(
            "SELECT * FROM curiosity_queue "
            "WHERE status = 'failed' AND created_at > datetime('now', ?) "
            "ORDER BY created_at DESC",
            (f"-{hours} hours",),
        )
        return [self._row_to_item(r) for r in rows]

    def prune(self, days: int = 30) -> int:
        """Delete old resolved/failed/dismissed items."""
        cursor = self._db.execute(
            "DELETE FROM curiosity_queue WHERE status IN ('resolved', 'failed', 'dismissed') "
            "AND created_at < datetime('now', ?)",
            (f"-{days} days",),
        )
        return cursor.rowcount

    def _row_to_item(self, row) -> CuriosityItem:
        return CuriosityItem(
            id=row["id"],
            topic=row["topic"],
            source=row["source"],
            urgency=row["urgency"],
            status=row["status"],
            attempts=row["attempts"],
            resolution=row["resolution"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )


# ---------------------------------------------------------------------------
# TopicTracker — frequency tracking for auto-monitor creation
# ---------------------------------------------------------------------------

class TopicTracker:
    """Tracks query topics to detect patterns worth monitoring."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS topic_frequency (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        query_count INTEGER DEFAULT 1,
        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_topic_freq_topic ON topic_frequency(topic);
    """

    def __init__(self, db):
        self._db = db
        for stmt in self._SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._db.execute(stmt)

    def record_topic(self, query: str) -> None:
        """Record a query topic with fuzzy matching.

        Uses Jaccard similarity (≥0.6) on normalized word sets to merge
        semantically similar topics like "What is Bitcoin" and "Tell me about Bitcoin".
        """
        topic = self._extract_topic(query)
        if not topic or len(topic) < 3:
            return

        # Exact match first
        existing = self._db.fetchone(
            "SELECT id, query_count FROM topic_frequency WHERE topic = ?",
            (topic,),
        )
        if existing:
            self._record_exact(existing["id"])
            return

        # Fuzzy match against recent topics (last 30 days)
        topic_words = normalize_words(topic)
        if topic_words:
            recent = self._db.fetchall(
                "SELECT id, topic FROM topic_frequency "
                "WHERE last_seen > datetime('now', '-30 days')"
            )
            for row in recent:
                row_words = normalize_words(row["topic"])
                if not row_words:
                    continue
                jaccard = len(topic_words & row_words) / len(topic_words | row_words)
                if jaccard >= 0.6:
                    self._record_exact(row["id"])
                    return

        # No match — insert new
        self._db.execute(
            "INSERT INTO topic_frequency (topic) VALUES (?)",
            (topic,),
        )

        # Auto-cleanup old entries
        self._db.execute(
            "DELETE FROM topic_frequency WHERE last_seen < datetime('now', '-30 days')"
        )

    def _record_exact(self, topic_id: int) -> None:
        """Increment count and update last_seen for an existing topic row."""
        self._db.execute(
            "UPDATE topic_frequency SET query_count = query_count + 1, "
            "last_seen = CURRENT_TIMESTAMP WHERE id = ?",
            (topic_id,),
        )

    def get_monitor_candidates(self, min_count: int = 3, days: int = 7) -> list[dict]:
        """Find topics queried 3+ times in the last N days."""
        rows = self._db.fetchall(
            "SELECT topic, query_count, first_seen, last_seen "
            "FROM topic_frequency "
            "WHERE query_count >= ? AND last_seen > datetime('now', ?) "
            "ORDER BY query_count DESC LIMIT 10",
            (min_count, f"-{days} days"),
        )
        return [dict(r) for r in rows]

    @staticmethod
    def _extract_topic(query: str) -> str:
        """Extract a normalized topic from a query string.

        Strips common question prefixes and normalizes to lowercase.
        """
        q = query.strip().lower()
        # Strip question prefixes
        for prefix in (
            "what is ", "what are ", "who is ", "who are ",
            "how do ", "how does ", "how to ", "how can ",
            "tell me about ", "explain ", "describe ",
            "can you ", "could you ", "please ",
        ):
            if q.startswith(prefix):
                q = q[len(prefix):]
                break
        # Strip trailing punctuation
        q = q.rstrip("?!. ")
        # Cap length
        return q[:100]
