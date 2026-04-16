"""TrustManager — Sovereign-OS inspired earned trust system.

Asymmetric scoring: successes earn +2, failures cost -15.
Tool access gated by trust level thresholds.
Trust decays daily — must be maintained through use, not just accumulated.
Append-only audit trail with proof hashes.

Reference: Sovereign-OS (arXiv 2603.14011)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Tool capability tiers — must earn trust to use higher-tier tools
_TOOL_TIERS: dict[str, int] = {
    # Tier 1 (threshold: 20) — basic information retrieval
    "web_search": 20, "knowledge_search": 20, "memory_search": 20,
    "calculator": 20, "active_memory": 20, "screenshot": 20,
    "context_detail": 0,  # Always allowed (read-only)
    # Tier 2 (threshold: 40) — data modification + browsing
    "file_ops": 40, "monitor": 40, "calendar": 40, "reminder": 40,
    "browser": 40, "http_fetch": 40,
    # Tier 3 (threshold: 60) — external communication + code
    "email_send": 60, "webhook": 60, "integration": 60, "code_exec": 60,
    # Tier 4 (threshold: 80) — system access
    "shell_exec": 80, "desktop": 80,
    # Meta — low threshold (orchestration, not direct action)
    "delegate": 20, "background_task": 20,
}

# Default thresholds — start low, earn access
DEFAULT_STARTING_SCORE = 30.0
DEFAULT_SUCCESS_DELTA = 1.0    # Slow climb — trust is earned gradually
DEFAULT_FAILURE_DELTA = -15.0  # Failures cost 15x more than a success earns
DEFAULT_DECAY_AMOUNT = 2.0     # Per dream cycle (~4x/day = 8 points/day)
TRUST_FLOOR = 20               # Minimum score — calculator + web_search always available


class TrustManager:
    """Manages Nova's earned trust score with asymmetric updates and daily decay."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS trust_scores (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        current_score REAL DEFAULT 30.0,
        total_successes INTEGER DEFAULT 0,
        total_failures INTEGER DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS trust_audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tool_name TEXT NOT NULL,
        action TEXT,
        result TEXT NOT NULL,
        score_delta REAL NOT NULL,
        new_score REAL NOT NULL,
        proof_hash TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    def __init__(self, db, *, starting_score: float = DEFAULT_STARTING_SCORE,
                 success_delta: float = DEFAULT_SUCCESS_DELTA,
                 failure_delta: float = DEFAULT_FAILURE_DELTA):
        self._db = db
        self._success_delta = success_delta
        self._failure_delta = failure_delta

        for stmt in self._SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._db.execute(stmt)

        # Ensure singleton row exists
        row = self._db.fetchone("SELECT current_score, total_successes FROM trust_scores WHERE id = 1")
        if not row:
            self._db.execute(
                "INSERT INTO trust_scores (id, current_score) VALUES (1, ?)",
                (starting_score,),
            )
        else:
            # One-time reset: check if the audit log shows old +5 delta entries.
            # If so, score was built under the inflated regime — reset to starting_score.
            old_deltas = self._db.fetchone(
                "SELECT COUNT(*) as c FROM trust_audit_log WHERE score_delta = 5.0"
            )
            if old_deltas and old_deltas["c"] > 50:
                self._db.execute(
                    "UPDATE trust_scores SET current_score = ? WHERE id = 1",
                    (starting_score,),
                )
                logger.info(
                    "Trust: reset inflated score to %.0f (%d audit entries at old +5 delta)",
                    starting_score, old_deltas["c"],
                )
            logger.info("Trust: reset inflated score (was 100) to %.0f", starting_score)

    def get_score(self) -> float:
        """Get current trust score (0-100)."""
        row = self._db.fetchone("SELECT current_score FROM trust_scores WHERE id = 1")
        return row["current_score"] if row else DEFAULT_STARTING_SCORE

    def can_use(self, tool_name: str) -> bool:
        """Check if current trust level allows using this tool.

        NOTE: Trust is observational, not gating. Nova is a sovereign personal AI
        running on the owner's hardware — blocking tools from the owner's own system
        is counterproductive. Score is tracked for self-awareness, not enforcement.
        """
        return True  # All tools always allowed — trust is tracked, not enforced

    def record_outcome(self, tool_name: str, success: bool, action: str = "") -> float:
        """Record a tool execution outcome. Returns new trust score."""
        delta = self._success_delta if success else self._failure_delta
        result = "success" if success else "failure"

        current = self.get_score()
        new_score = max(TRUST_FLOOR, min(100.0, current + delta))

        now = datetime.now(timezone.utc).isoformat()

        # Update score
        if success:
            self._db.execute(
                "UPDATE trust_scores SET current_score = ?, total_successes = total_successes + 1, "
                "updated_at = ? WHERE id = 1",
                (new_score, now),
            )
        else:
            self._db.execute(
                "UPDATE trust_scores SET current_score = ?, total_failures = total_failures + 1, "
                "updated_at = ? WHERE id = 1",
                (new_score, now),
            )

        # Append-only audit log with proof hash
        proof_data = f"{tool_name}|{action}|{result}|{delta}|{new_score}|{now}"
        proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()[:16]

        self._db.execute(
            "INSERT INTO trust_audit_log (tool_name, action, result, score_delta, new_score, proof_hash, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tool_name, action, result, delta, new_score, proof_hash, now),
        )

        if not success:
            logger.warning(
                "Trust: %s FAILED (score %.0f -> %.0f, threshold %d)",
                tool_name, current, new_score, _TOOL_TIERS.get(tool_name, 0),
            )

        return new_score

    def decay(self, amount: float = DEFAULT_DECAY_AMOUNT) -> float:
        """Daily decay — trust must be maintained through use, not just accumulated.

        Called during dream consolidation so trust erodes during idle periods.
        """
        current = self.get_score()
        new_score = max(TRUST_FLOOR, current - amount)
        if new_score != current:
            now = datetime.now(timezone.utc).isoformat()
            self._db.execute(
                "UPDATE trust_scores SET current_score = ?, updated_at = ? WHERE id = 1",
                (new_score, now),
            )
            logger.info("Trust: daily decay %.0f -> %.0f", current, new_score)
        return new_score

    def get_stats(self) -> dict:
        """Get trust statistics."""
        row = self._db.fetchone("SELECT * FROM trust_scores WHERE id = 1")
        if not row:
            return {"score": DEFAULT_STARTING_SCORE, "successes": 0, "failures": 0}
        return {
            "score": row["current_score"],
            "successes": row["total_successes"],
            "failures": row["total_failures"],
        }

    def get_audit_trail(self, limit: int = 20) -> list[dict]:
        """Get recent audit log entries."""
        rows = self._db.fetchall(
            "SELECT tool_name, action, result, score_delta, new_score, timestamp "
            "FROM trust_audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def get_blocked_tools(self) -> list[str]:
        """Get list of tools blocked by current trust level."""
        score = self.get_score()
        return [name for name, threshold in _TOOL_TIERS.items() if score < threshold]
