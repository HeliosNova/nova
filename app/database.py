"""Thread-safe SQLite wrapper with auto-schema creation.

Ported from Nova's battle-tested SafeDB pattern.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ceiling on waiting for the writer lock. Writes are milliseconds; a wait this
# long means a writer is wedged or the lock leaked (incident 2026-07-03: a
# leaked lock froze every writer INCLUDING the event loop for 54h). Raising
# loudly is strictly better than a silent permanent hang.
_WRITE_LOCK_TIMEOUT = 120.0

_instances: dict[str, "SafeDB"] = {}
_instance_lock = threading.Lock()

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT,
    tool_name TEXT,
    sources TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Memory
CREATE TABLE IF NOT EXISTS user_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    source TEXT DEFAULT 'inferred',
    confidence REAL DEFAULT 1.0,
    category TEXT DEFAULT 'fact',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learning
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    wrong_answer TEXT,
    correct_answer TEXT NOT NULL,
    lesson_text TEXT DEFAULT '',
    context TEXT,
    confidence REAL DEFAULT 0.8,
    times_retrieved INTEGER DEFAULT 0,
    times_helpful INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    trigger_pattern TEXT NOT NULL,
    steps TEXT NOT NULL,
    answer_template TEXT,
    learned_from INTEGER REFERENCES lessons(id),
    times_used INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,
    enabled BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 for BM25 keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id,
    document_id,
    content,
    tokenize='porter unicode61'
);

-- Knowledge Graph
CREATE TABLE IF NOT EXISTS kg_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    source TEXT DEFAULT 'extracted',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    provenance TEXT DEFAULT '',
    superseded_by INTEGER,
    times_retrieved INTEGER DEFAULT 0,
    UNIQUE(subject, predicate, object)
);

-- Reflexions (experiential learning from failures)
CREATE TABLE IF NOT EXISTS reflexions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_summary TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'failure',
    reflection TEXT NOT NULL,
    quality_score REAL DEFAULT 0.5,
    tools_used TEXT DEFAULT '',
    revision_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Custom Tools (dynamic tool creation)
CREATE TABLE IF NOT EXISTS custom_tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    parameters TEXT NOT NULL,
    code TEXT NOT NULL,
    times_used INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 1.0,
    enabled BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Monitors (heartbeat system)
CREATE TABLE IF NOT EXISTS monitors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    check_type TEXT NOT NULL,
    check_config TEXT NOT NULL,
    schedule_seconds INTEGER DEFAULT 300,
    enabled INTEGER DEFAULT 1,
    cooldown_minutes INTEGER DEFAULT 60,
    notify_condition TEXT DEFAULT 'on_change',
    last_check_at TEXT,
    last_alert_at TEXT,
    last_result TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    -- category: 'system' = internal/health (telegram only);
    --          'content' = news/domain feeds (all channels).
    -- See app/monitors/monitor_store.classify_category.
    category TEXT DEFAULT 'content'
);

CREATE TABLE IF NOT EXISTS monitor_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    monitor_id INTEGER REFERENCES monitors(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    value TEXT,
    message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Action log (audit trail for action tools)
CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT NOT NULL,
    params TEXT,
    result TEXT,
    success INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Heartbeat instructions
CREATE TABLE IF NOT EXISTS heartbeat_instructions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instruction TEXT NOT NULL,
    schedule_seconds INTEGER DEFAULT 3600,
    enabled INTEGER DEFAULT 1,
    last_run_at TEXT,
    notify_channels TEXT DEFAULT 'discord,telegram',
    created_at TEXT DEFAULT (datetime('now'))
);

-- System state (key-value persistence for runtime state)
CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Auth lockout persistence (survive restarts)
CREATE TABLE IF NOT EXISTS auth_lockouts (
    ip TEXT PRIMARY KEY,
    failures TEXT DEFAULT '[]',
    locked_until REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Capability gaps (detected when no skill/tool covers a query + low quality)
CREATE TABLE IF NOT EXISTS capability_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    reason TEXT,
    tools_tried TEXT DEFAULT '[]',
    quality_score REAL,
    reviewed INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Prompt self-modification module registry
-- Immutable baseline rows (is_baseline=1) are never touched by the optimizer.
-- Only module_names in _SELF_MOD_ALLOWED_MODULES (prompt_optimizer.py) are writable.
CREATE TABLE IF NOT EXISTS prompt_modules (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    module_name             TEXT NOT NULL,
    version                 INTEGER NOT NULL,
    content                 TEXT NOT NULL,
    is_baseline             INTEGER DEFAULT 0,
    status                  TEXT NOT NULL DEFAULT 'candidate',
    parent_version          INTEGER,
    delta_description       TEXT,
    promoted_at             TEXT,
    promoted_eval_run_id    TEXT,
    rolled_back_at          TEXT,
    quarantined_until       TEXT,
    shadow_eval_metrics     TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_modules_name_version
    ON prompt_modules (module_name, version);
CREATE INDEX IF NOT EXISTS idx_prompt_modules_name_status
    ON prompt_modules (module_name, status);

-- GSW (Generative Semantic Workspace) — episodic memory for cross-session recall.
-- Stores rolling per-conversation narratives anchored to entities + time. Distinct
-- from kg_facts (atomic triples) and lessons (correction-derived patterns) — this
-- is "what happened in our recent sessions" so Nova can pick up where we left off.
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,           -- 1-2 sentence high-level summary
    narrative TEXT,                  -- expanded space-time-anchored narrative
    key_entities TEXT,               -- JSON array of lower-case entity strings
    message_count INTEGER DEFAULT 0,
    last_message_id TEXT,            -- so we can extend without re-summarizing
    valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valid_to TIMESTAMP,              -- NULL = current; populated on supersede
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_reflexions_outcome ON reflexions(outcome);
CREATE INDEX IF NOT EXISTS idx_reflexions_quality ON reflexions(quality_score);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_lessons_topic ON lessons(topic);
-- created_at is the filter column for the lesson/KG decay + age-prune passes
-- run in daily maintenance; without these they full-scan (audit 2026-08-22).
CREATE INDEX IF NOT EXISTS idx_lessons_created ON lessons(created_at);
CREATE INDEX IF NOT EXISTS idx_skills_trigger ON skills(trigger_pattern);
CREATE INDEX IF NOT EXISTS idx_kg_subject ON kg_facts(subject);
CREATE INDEX IF NOT EXISTS idx_kg_object ON kg_facts(object);
CREATE INDEX IF NOT EXISTS idx_kg_valid_from ON kg_facts(valid_from);
CREATE INDEX IF NOT EXISTS idx_kg_created ON kg_facts(created_at);
CREATE INDEX IF NOT EXISTS idx_monitors_enabled ON monitors(enabled);
CREATE INDEX IF NOT EXISTS idx_monitor_results_monitor ON monitor_results(monitor_id, created_at);
CREATE INDEX IF NOT EXISTS idx_action_log_type ON action_log(action_type, created_at);
CREATE INDEX IF NOT EXISTS idx_conv_summaries_conv ON conversation_summaries(conversation_id, valid_to);
"""


class _TransactionCursor:
    """Thin wrapper exposing execute/fetchone/fetchall inside a transaction."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        return self._conn.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Row | None:
        return self._conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchall()


class SafeDB:
    """Thread-safe SQLite wrapper. Singleton per db_path."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        # One connection PER THREAD. A single shared connection behind one global
        # lock serialized every read and write, negating WAL entirely — the root
        # of the recurring event-loop lock-convoy incidents (2026-06-11/-12).
        # With a connection per thread, WAL lets readers run concurrently with
        # each other and with the single writer.
        self._local = threading.local()
        # Writers are still serialized among OUR threads (RLock: re-entrant so a
        # nested write on the same thread can't self-deadlock) so two to_thread
        # workers never collide into SQLITE_BUSY. Reads take no lock at all.
        self._write_lock = threading.RLock()
        # Registry of every per-thread connection so close() can shut them all.
        self._all_conns: list[sqlite3.Connection] = []
        self._all_conns_lock = threading.Lock()
        # Per-instance schema-ensure memo (audit 2026-08-23): stores ran their
        # idempotent __init__ DDL (CREATE IF NOT EXISTS + ALTER-catch) on EVERY
        # construction, and stores are constructed repeatedly in async contexts
        # — live logs showed kg DDL 8x/2h taking the write lock on the
        # event-loop thread (the lock-convoy class). First construction per
        # (SafeDB, tag) runs DDL and marks; later ones skip entirely.
        self._ensured_tags: set[str] = set()
        self._ensured_lock = threading.Lock()

    def schema_ensured(self, tag: str) -> bool:
        """True once mark_schema_ensured(tag) ran on THIS SafeDB instance."""
        with self._ensured_lock:
            return tag in self._ensured_tags

    def mark_schema_ensured(self, tag: str) -> None:
        with self._ensured_lock:
            self._ensured_tags.add(tag)

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
            with self._all_conns_lock:
                self._all_conns.append(conn)
        return conn

    def init_schema(self) -> None:
        """Create all tables. Safe to call multiple times."""
        with self._write_lock:
            conn = self._get_conn()
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            self._run_migrations(conn)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run schema migrations for existing databases.

        Each migration is versioned and wrapped in a transaction.
        Already-applied migrations are skipped via the schema_version table.
        """
        # Create schema_version tracking table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version "
            "(version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.commit()

        # Get already-applied versions
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_version").fetchall()}

        # --- Migration 1: lesson columns ---
        if 1 not in applied:
            conn.execute("BEGIN")
            try:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)").fetchall()}
                if "lesson_text" not in cols:
                    conn.execute("ALTER TABLE lessons ADD COLUMN lesson_text TEXT DEFAULT ''")
                if "last_retrieved_at" not in cols:
                    conn.execute("ALTER TABLE lessons ADD COLUMN last_retrieved_at TIMESTAMP")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (1,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 2: kg_facts columns ---
        if 2 not in applied:
            conn.execute("BEGIN")
            try:
                kg_cols = {row[1] for row in conn.execute("PRAGMA table_info(kg_facts)").fetchall()}
                if "times_retrieved" not in kg_cols:
                    conn.execute("ALTER TABLE kg_facts ADD COLUMN times_retrieved INTEGER DEFAULT 0")
                if "last_retrieved_at" not in kg_cols:
                    conn.execute("ALTER TABLE kg_facts ADD COLUMN last_retrieved_at TEXT")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (2,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 3: user_facts columns ---
        if 3 not in applied:
            conn.execute("BEGIN")
            try:
                uf_cols = {row[1] for row in conn.execute("PRAGMA table_info(user_facts)").fetchall()}
                if "category" not in uf_cols:
                    conn.execute("ALTER TABLE user_facts ADD COLUMN category TEXT DEFAULT 'fact'")
                if "last_accessed_at" not in uf_cols:
                    conn.execute("ALTER TABLE user_facts ADD COLUMN last_accessed_at TIMESTAMP")
                if "access_count" not in uf_cols:
                    conn.execute("ALTER TABLE user_facts ADD COLUMN access_count INTEGER DEFAULT 0")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (3,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 4: monitor_results user_rating ---
        if 4 not in applied:
            conn.execute("BEGIN")
            try:
                mr_cols = {row[1] for row in conn.execute("PRAGMA table_info(monitor_results)").fetchall()}
                if "user_rating" not in mr_cols:
                    conn.execute("ALTER TABLE monitor_results ADD COLUMN user_rating INTEGER DEFAULT 0")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (4,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 5: lessons quiz columns ---
        if 5 not in applied:
            conn.execute("BEGIN")
            try:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)").fetchall()}
                if "last_quizzed_at" not in cols:
                    conn.execute("ALTER TABLE lessons ADD COLUMN last_quizzed_at TIMESTAMP")
                if "quiz_failures" not in cols:
                    conn.execute("ALTER TABLE lessons ADD COLUMN quiz_failures INTEGER DEFAULT 0")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (5,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 6: indexes ---
        if 6 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_predicate ON kg_facts(predicate)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_facts_last_accessed ON user_facts(last_accessed_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_lessons_last_retrieved ON lessons(last_retrieved_at)")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (6,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 7: heartbeat_instructions table ---
        if 7 not in applied:
            conn.execute("BEGIN")
            try:
                hi_tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
                if "heartbeat_instructions" not in hi_tables:
                    conn.execute("""
                        CREATE TABLE heartbeat_instructions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            instruction TEXT NOT NULL,
                            schedule_seconds INTEGER DEFAULT 3600,
                            enabled INTEGER DEFAULT 1,
                            last_run_at TEXT,
                            notify_channels TEXT DEFAULT 'discord,telegram',
                            created_at TEXT DEFAULT (datetime('now'))
                        )
                    """)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (7,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 8: auth_lockouts table ---
        if 8 not in applied:
            conn.execute("BEGIN")
            try:
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
                if "auth_lockouts" not in tables:
                    conn.execute("""
                        CREATE TABLE auth_lockouts (
                            ip TEXT PRIMARY KEY,
                            failures TEXT DEFAULT '[]',
                            locked_until REAL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (8,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 9: skill quality + composition columns ---
        if 9 not in applied:
            conn.execute("BEGIN")
            try:
                skill_cols = {row[1] for row in conn.execute("PRAGMA table_info(skills)").fetchall()}
                if "last_used_at" not in skill_cols:
                    conn.execute("ALTER TABLE skills ADD COLUMN last_used_at TIMESTAMP")
                if "consecutive_failures" not in skill_cols:
                    conn.execute("ALTER TABLE skills ADD COLUMN consecutive_failures INTEGER DEFAULT 0")
                if "source" not in skill_cols:
                    conn.execute("ALTER TABLE skills ADD COLUMN source TEXT DEFAULT 'correction'")
                if "composed_of" not in skill_cols:
                    conn.execute("ALTER TABLE skills ADD COLUMN composed_of TEXT DEFAULT '[]'")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_skills_last_used ON skills(last_used_at)"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (9,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 10: re-apply skill columns for DBs where migration 9 ran empty ---
        # Migration 9 was recorded before the ALTER TABLE statements were added to the code,
        # so existing databases have version=9 but are missing the columns. This migration
        # unconditionally ensures the columns exist.
        if 10 not in applied:
            conn.execute("BEGIN")
            try:
                skill_cols = {row[1] for row in conn.execute("PRAGMA table_info(skills)").fetchall()}
                if "last_used_at" not in skill_cols:
                    conn.execute("ALTER TABLE skills ADD COLUMN last_used_at TIMESTAMP")
                if "consecutive_failures" not in skill_cols:
                    conn.execute("ALTER TABLE skills ADD COLUMN consecutive_failures INTEGER DEFAULT 0")
                if "source" not in skill_cols:
                    conn.execute("ALTER TABLE skills ADD COLUMN source TEXT DEFAULT 'correction'")
                if "composed_of" not in skill_cols:
                    conn.execute("ALTER TABLE skills ADD COLUMN composed_of TEXT DEFAULT '[]'")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_skills_last_used ON skills(last_used_at)"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (10,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 11: capability_gaps table ---
        if 11 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS capability_gaps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        reason TEXT,
                        tools_tried TEXT DEFAULT '[]',
                        quality_score REAL,
                        reviewed INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT (datetime('now'))
                    )"""
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (11,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 12: prompt_modules table ---
        if 12 not in applied:
            conn.execute("BEGIN")
            try:
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
                if "prompt_modules" not in tables:
                    conn.execute("""
                        CREATE TABLE prompt_modules (
                            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                            module_name             TEXT NOT NULL,
                            version                 INTEGER NOT NULL,
                            content                 TEXT NOT NULL,
                            is_baseline             INTEGER DEFAULT 0,
                            status                  TEXT NOT NULL DEFAULT 'candidate',
                            parent_version          INTEGER,
                            delta_description       TEXT,
                            promoted_at             TEXT,
                            promoted_eval_run_id    TEXT,
                            rolled_back_at          TEXT,
                            quarantined_until       TEXT,
                            shadow_eval_metrics     TEXT,
                            created_at              TEXT NOT NULL DEFAULT (datetime('now'))
                        )
                    """)
                    conn.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_prompt_modules_name_version "
                        "ON prompt_modules (module_name, version)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_prompt_modules_name_status "
                        "ON prompt_modules (module_name, status)"
                    )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (12,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 13: monitors.category column for channel routing ---
        if 13 not in applied:
            conn.execute("BEGIN")
            try:
                mcols = {row[1] for row in conn.execute("PRAGMA table_info(monitors)").fetchall()}
                if "category" not in mcols:
                    conn.execute(
                        "ALTER TABLE monitors ADD COLUMN category TEXT DEFAULT 'content'"
                    )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (13,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 14: daemon_log + event_queue tables ---
        # These tables are referenced throughout app/monitors/daemon.py,
        # app/api/daemon.py, app/api/events.py, app/core/dream.py, and
        # app/monitors/event_trigger.py, but were never created by any prior
        # schema step — so the DaemonOrchestrator would silently swallow
        # "no such table" errors on every tick. Phase-0 bootstrap fix.
        if 14 not in applied:
            conn.execute("BEGIN")
            try:
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
                if "daemon_log" not in tables:
                    conn.execute("""
                        CREATE TABLE daemon_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            category TEXT NOT NULL,
                            content TEXT NOT NULL DEFAULT '',
                            source TEXT DEFAULT '',
                            created_at TEXT DEFAULT (datetime('now'))
                        )
                    """)
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_daemon_log_created "
                        "ON daemon_log (created_at)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_daemon_log_category "
                        "ON daemon_log (category, created_at)"
                    )
                if "event_queue" not in tables:
                    conn.execute("""
                        CREATE TABLE event_queue (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            event_type TEXT NOT NULL,
                            payload TEXT DEFAULT '',
                            priority REAL DEFAULT 0.5,
                            status TEXT DEFAULT 'pending',
                            created_at TEXT DEFAULT (datetime('now')),
                            processed_at TEXT
                        )
                    """)
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_event_queue_status "
                        "ON event_queue (status, priority DESC, created_at ASC)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_event_queue_type "
                        "ON event_queue (event_type, status)"
                    )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (14,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 15: goals table + Phase-0 bootstrap seed ---
        # Minimal will-module scaffold so Nova has somewhere to read its
        # intended next action from. The schema is intentionally bare —
        # the bootstrap goal's purpose is to write app/core/goals.py with
        # a proper GoalStore, which will then own this table.
        if 15 not in applied:
            conn.execute("BEGIN")
            try:
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
                if "goals" not in tables:
                    conn.execute("""
                        CREATE TABLE goals (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            goal TEXT NOT NULL,
                            priority REAL DEFAULT 0.5,
                            status TEXT DEFAULT 'pending',
                            source TEXT DEFAULT 'user',
                            context TEXT DEFAULT '{}',
                            created_at TEXT DEFAULT (datetime('now')),
                            updated_at TEXT DEFAULT (datetime('now')),
                            completed_at TEXT
                        )
                    """)
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_goals_status_priority "
                        "ON goals (status, priority DESC, created_at ASC)"
                    )

                # Seed Phase-0 bootstrap goal exactly once. Idempotent on
                # (source='phase_0_bootstrap') so this migration is safe to
                # re-run on DBs that already contain it.
                seed_text = (
                    "write app/core/goals.py with GoalStore, "
                    "derive_goals_from_state(), and execute_goal(); "
                    "wire pursue_goal into DaemonOrchestrator._decide"
                )
                existing = conn.execute(
                    "SELECT id FROM goals WHERE source='phase_0_bootstrap' LIMIT 1"
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO goals (goal, priority, status, source, context) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            seed_text,
                            1.0,                    # priority high
                            "pending",
                            "phase_0_bootstrap",
                            '{"phase": 0, "seeded_by": "phase-0-bootstrap migration"}',
                        ),
                    )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (15,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 16: lessons.retrieval_score (Q-value), monitors.trigger_events ---
        # retrieval_score: blended into RRF ranking (MemRL pattern). Code referenced
        # the column but no migration created it; production DB had it via manual
        # backfill, fresh DBs (tests, new installs) crashed on first use.
        # trigger_events / trigger_mode: EventTrigger uses these to fire monitors
        # on internal events (e.g. internal:lesson_saved) bypassing the schedule.
        if 16 not in applied:
            conn.execute("BEGIN")
            try:
                lesson_cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)").fetchall()}
                if "retrieval_score" not in lesson_cols:
                    conn.execute("ALTER TABLE lessons ADD COLUMN retrieval_score REAL DEFAULT 0.5")
                monitor_cols = {row[1] for row in conn.execute("PRAGMA table_info(monitors)").fetchall()}
                if "trigger_events" not in monitor_cols:
                    conn.execute("ALTER TABLE monitors ADD COLUMN trigger_events TEXT")
                if "trigger_mode" not in monitor_cols:
                    conn.execute("ALTER TABLE monitors ADD COLUMN trigger_mode TEXT DEFAULT 'schedule'")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (16,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 17: monitors.channels (per-monitor channel routing) ---
        # NULL = use category default (system→telegram, content→all). Set to
        # a CSV like "discord,signal" to route only to those channels.
        if 17 not in applied:
            conn.execute("BEGIN")
            try:
                monitor_cols = {row[1] for row in conn.execute("PRAGMA table_info(monitors)").fetchall()}
                if "channels" not in monitor_cols:
                    conn.execute("ALTER TABLE monitors ADD COLUMN channels TEXT")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (17,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 18: agent_workspace (persistent scratchpads) ---
        # Each AgentLoop run keyed by its query signature. Future runs of
        # similar queries inherit prior findings/answer instead of starting
        # fresh. Lets multi-step research compound across sessions.
        if 18 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_workspace (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_signature TEXT NOT NULL UNIQUE,
                        last_query TEXT NOT NULL,
                        findings_json TEXT,
                        last_answer TEXT,
                        run_count INTEGER NOT NULL DEFAULT 0,
                        success_count INTEGER NOT NULL DEFAULT 0,
                        fail_count INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_workspace_sig "
                    "ON agent_workspace(query_signature)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_workspace_updated "
                    "ON agent_workspace(updated_at)"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (18,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 19: RLVR verifiable_signals + procedural_clusters ---
        # verifiable_signals: stores ground-truth-style signals (tool ran clean,
        # JSON parsed, math correct, claim grounded, quiz right) so the next
        # GRPO/RLVR fine-tune cycle has reward data without re-grading.
        # procedural_clusters: tracks lesson clusters dream consolidates each
        # cycle so we don't re-consolidate the same group every run.
        if 19 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS verifiable_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT,
                        query TEXT,
                        response TEXT,
                        signal_type TEXT NOT NULL,
                        signal_value REAL NOT NULL,
                        evidence TEXT,
                        consumed_for_training INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_verifiable_signals_type_created "
                    "ON verifiable_signals (signal_type, created_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_verifiable_signals_unconsumed "
                    "ON verifiable_signals (consumed_for_training, signal_type)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS procedural_clusters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cluster_key TEXT NOT NULL UNIQUE,
                        member_lesson_ids TEXT NOT NULL,
                        canonical_lesson_id INTEGER,
                        member_count INTEGER NOT NULL DEFAULT 0,
                        last_consolidated_at TEXT NOT NULL DEFAULT (datetime('now')),
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_procedural_clusters_key "
                    "ON procedural_clusters (cluster_key)"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (19,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 20: reflexions.is_eval flag ---
        # Stop eval-derived reflexions from being promoted to lessons. The
        # deliberation_chain_of_reasoning task creates a reflexion every run;
        # those got promoted by dream's _promote_reflexions and accumulated as
        # lessons that contaminated retrieval, biasing future eval runs into
        # summary-shaped answers (root cause of the 0.20 ↔ 0.85 bimodal pattern
        # observed across 11 eval runs in 2026-05-09).
        if 20 not in applied:
            conn.execute("BEGIN")
            try:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(reflexions)").fetchall()}
                if "is_eval" not in cols:
                    conn.execute("ALTER TABLE reflexions ADD COLUMN is_eval INTEGER DEFAULT 0")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reflexions_is_eval "
                    "ON reflexions (is_eval, quality_score)"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (20,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 21: dedup_decisions instrumentation table ---
        # Captures every Jaccard-based dedup comparison (lesson / curiosity /
        # reflexion) with the score that drove the decision. Lets the operator
        # empirically revisit the hand-tuned thresholds (0.55 / 0.6 / 0.85)
        # that were originally set by symptom rather than measurement.
        # Audit 2026-05-13.
        if 21 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS dedup_decisions ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  entity_type TEXT NOT NULL,"
                    "  jaccard_score REAL NOT NULL,"
                    "  threshold REAL NOT NULL,"
                    "  decision TEXT NOT NULL,"
                    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                    ")"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dedup_decisions_type_time "
                    "ON dedup_decisions (entity_type, created_at)"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (21,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 22: index for time-window monitor_results queries ---
        # get_recent_results filters on created_at alone; the existing
        # idx_monitor_results_monitor(monitor_id, created_at) can't serve it,
        # so every call was a full table scan over rows with multi-KB TEXT
        # columns — one of the contributors to the 2026-06-11 event-loop
        # blocking incident.
        if 22 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_monitor_results_created "
                    "ON monitor_results (created_at)"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (22,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 23: Monitor Intelligence v2 — storylines, forecasts, salience ---
        # Narrative engine (storylines + events), self-scoring forecasts, and the
        # learned owner-interest salience weights. All additive; no existing table touched.
        if 23 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS storylines ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  story_key TEXT UNIQUE NOT NULL,"
                    "  title TEXT NOT NULL,"
                    "  status TEXT DEFAULT 'active',"          # active | dormant | closed
                    "  summary TEXT DEFAULT '',"
                    "  monitors_csv TEXT DEFAULT '',"
                    "  first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                    "  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                    "  update_count INTEGER DEFAULT 0,"
                    "  last_digest_at TIMESTAMP"
                    ")"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_storylines_status ON storylines (status, last_updated)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS storyline_events ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  storyline_id INTEGER REFERENCES storylines(id) ON DELETE CASCADE,"
                    "  summary TEXT NOT NULL,"
                    "  source_monitor TEXT DEFAULT '',"
                    "  item_url TEXT DEFAULT '',"
                    "  published TIMESTAMP,"
                    "  is_new INTEGER DEFAULT 1,"
                    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                    ")"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_storyline_events_story ON storyline_events (storyline_id, created_at)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS forecasts ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  claim TEXT NOT NULL,"
                    "  storyline_key TEXT DEFAULT '',"
                    "  confidence REAL DEFAULT 0.5,"
                    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                    "  resolves_at TIMESTAMP,"
                    "  status TEXT DEFAULT 'open',"            # open | hit | miss | unresolvable
                    "  resolution TEXT DEFAULT '',"
                    "  resolved_at TIMESTAMP,"
                    "  source_monitor TEXT DEFAULT ''"
                    ")"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_forecasts_open ON forecasts (status, resolves_at)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS salience_weights ("
                    "  topic TEXT PRIMARY KEY,"
                    "  weight REAL DEFAULT 0.0,"
                    "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                    ")"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (23,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # --- Migration 24: forecasts.attempts (prevent infinite-retry starvation) ---
        # A permanently-unparseable forecast stayed 'open' and was re-selected by
        # list_due every cycle forever; >=8 such rows starved the queue. Track
        # attempts so they auto-retire to 'unresolvable'.
        if 24 not in applied:
            conn.execute("BEGIN")
            try:
                fc_cols = {row[1] for row in conn.execute("PRAGMA table_info(forecasts)").fetchall()}
                if "attempts" not in fc_cols:
                    conn.execute("ALTER TABLE forecasts ADD COLUMN attempts INTEGER DEFAULT 0")
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (24,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # Migration 25 (2026-08-12): the KNOWING tier — living dossiers.
        # Durable, revisable understanding distilled from digests (which expire
        # on a 30-day retention) + mature storylines. `dossier_revisions` keeps
        # every prior body (valid_from/valid_to) so "what did Nova understand
        # about X on date D" is queryable — same bitemporal philosophy as kg_facts.
        if 25 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS dossiers ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  kind TEXT NOT NULL,"                    # 'domain' | 'storyline'
                    "  dkey TEXT NOT NULL,"                    # stable slug within kind
                    "  title TEXT NOT NULL,"
                    "  body TEXT DEFAULT '',"                  # structured md understanding
                    "  changed_note TEXT DEFAULT '',"          # last CHANGED: line
                    "  update_count INTEGER DEFAULT 0,"
                    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                    "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                    "  UNIQUE(kind, dkey)"
                    ")"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dossiers_kind_updated ON dossiers (kind, updated_at)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS dossier_revisions ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  dossier_id INTEGER REFERENCES dossiers(id) ON DELETE CASCADE,"
                    "  body TEXT NOT NULL,"
                    "  valid_from TIMESTAMP,"                  # prior body's updated_at
                    "  valid_to TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                    ")"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dossier_revisions_d ON dossier_revisions (dossier_id, valid_to)"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (25,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # Migration 26 (2026-08-12): temporal host co-occurrence — the data layer
        # for PARAPHRASE-network detection (the laundering vector text-similarity
        # cannot see). Every digest records which source hosts appeared together;
        # junk-tier pairs that co-occur in nearly every digest either host appears
        # in are a syndication/farm network and collapse to ONE source in the
        # independence clustering. Detection self-arms once the counts exist.
        if 26 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS host_cooccurrence ("
                    "  host_a TEXT NOT NULL,"                # canonical: host_a < host_b
                    "  host_b TEXT NOT NULL,"
                    "  n_cooccur INTEGER DEFAULT 1,"
                    "  first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                    "  last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                    "  PRIMARY KEY (host_a, host_b)"
                    ")"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS host_digest_counts ("
                    "  host TEXT PRIMARY KEY,"
                    "  n_digests INTEGER DEFAULT 1,"
                    "  last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                    ")"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (26,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # Migration 27 (2026-08-12): durable delivery ledger. Monitor alerts that
        # pass dedup/routing used to buffer ONLY in heartbeat memory until the
        # digest flush — any restart in that window silently destroyed completed
        # intelligence (2026-08-12: a finished World Awareness digest sat 29 min
        # in a size-1 buffer, then a restart wiped it; the DB had the result row
        # but the owner never saw it). Rows are written when an alert buffers,
        # deleted on confirmed broadcast, and recovered into the buffer at loop
        # start — delivery becomes at-least-once instead of best-effort.
        if 27 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS pending_deliveries ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  targets TEXT NOT NULL,"          # CSV channel names
                    "  monitor_name TEXT NOT NULL,"
                    "  message TEXT NOT NULL,"
                    "  category TEXT DEFAULT '',"
                    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                    ")"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (27,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # Migration 28 (2026-08-17): fix a duplicate dispatch key that silently
        # neutered scheduled Dream Consolidation. Both "Dream Consolidation" and
        # "Knowledge Consolidation" were seeded with check_type='consolidation',
        # and _CHECK_DISPATCH defined "consolidation" twice — the later (knowing/
        # dossier) handler won, so the Dream monitor ran the dossier cycle and the
        # real dream pipeline (reflexion prune, KG compaction, lesson-contradiction
        # resolution, DPO mining, procedural consolidation) never ran on schedule.
        # The dream handler now dispatches on 'dream_consolidation'; migrate the
        # existing row so it reaches _execute_consolidation.
        if 28 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "UPDATE monitors SET check_type='dream_consolidation' "
                    "WHERE name='Dream Consolidation' AND check_type='consolidation'"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (28,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # Migration 29 (2026-08-18): index host_cooccurrence.n_cooccur. The
        # anti-laundering _network_pairs() query filters `WHERE n_cooccur >= 8`
        # on EVERY digest, but the table only had a (host_a, host_b) PK — so each
        # call full-scanned all rows (32.5k live and growing). A range index turns
        # it into a tiny scan of just the high-cooccurrence tail.
        if 29 not in applied:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_hco_ncooccur "
                    "ON host_cooccurrence(n_cooccur)"
                )
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (29,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        # Migration 30 (2026-08-19): procedure skills — natural-language
        # procedural knowledge matched semantically (no regex trigger, no
        # mechanical steps). The 2026 field converged on NL/markdown skills
        # (Agent Skills spec, SkillPyramid/SkillBrew); Nova's regex+steps
        # formalism was the reason the auto-extractor produced 0 organic
        # skills — the LLM can't reliably author brittle trigger regexes.
        if 30 not in applied:
            conn.execute("BEGIN")
            try:
                for stmt in (
                    "ALTER TABLE skills ADD COLUMN kind TEXT DEFAULT 'exec'",
                    "ALTER TABLE skills ADD COLUMN procedure_text TEXT",
                ):
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError:
                        pass  # column already exists (fresh installs)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (30,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # Statements already reported by _warn_if_event_loop — warn once per
    # statement, capped so a pathological caller can't grow this unbounded.
    _loop_thread_warned: set[str] = set()

    # Startup grace (audit 2026-08-23): app startup runs ~18 one-shot init
    # calls (auth lockout load, monitor seeding, skill revalidation, ...) on
    # the lifespan's loop thread BEFORE any traffic exists — nothing can be
    # blocked, but 18 warnings every boot trained the eye to ignore the
    # tripwire. main.py flips this off when startup completes, so any warning
    # in steady-state logs is a genuine event-loop-blocking offender worth
    # fixing. Processes that never flip it (tests, scripts) simply keep the
    # observability tripwire quiet — it exists for the live app's loop.
    _startup_grace: bool = True

    @classmethod
    def end_startup_grace(cls) -> None:
        cls._startup_grace = False

    def _warn_if_event_loop(self, sql: str) -> None:
        """Flag sync DB calls running on the event-loop thread.

        Such a call blocks every coroutine for the full lock-acquire +
        query duration (incident 2026-06-11: heartbeat blocked the loop
        >60s under lock convoy → container unhealthy). Route async-context
        calls through AsyncSafeDB or asyncio.to_thread instead.
        """
        if SafeDB._startup_grace:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        key = sql[:120]
        if key not in SafeDB._loop_thread_warned and len(SafeDB._loop_thread_warned) < 256:
            SafeDB._loop_thread_warned.add(key)
            # Capture the first frame outside this module — the actual DB caller —
            # so the warning is actionable ("who does sync DB on the loop") instead
            # of just naming the SQL. Walked only when a warning first fires (once
            # per unique statement, capped), so the cost stays off the hot path.
            caller = "?"
            try:
                import os
                import sys as _sys
                fr = _sys._getframe(1)
                for _ in range(25):
                    if fr is None:
                        break
                    if fr.f_code.co_filename != __file__:
                        caller = f"{os.path.basename(fr.f_code.co_filename)}:{fr.f_lineno}"
                        break
                    fr = fr.f_back
            except Exception:
                pass
            logger.warning(
                "Sync DB call on event-loop thread from %s (blocks asyncio; use "
                "AsyncSafeDB or asyncio.to_thread): %s", caller, key
            )

    @contextmanager
    def _acquire_write(self):
        """Writer-lock acquire with a hang ceiling and guaranteed release.

        A plain `with self._write_lock:` waits FOREVER; when the lock leaked
        (2026-07-03) that froze every writer — including the event-loop thread
        — for 54 hours with zero log evidence. A bounded wait converts that
        failure mode into a loud, diagnosable exception.
        """
        if not self._write_lock.acquire(timeout=_WRITE_LOCK_TIMEOUT):
            raise TimeoutError(
                f"SafeDB writer lock not acquired within {_WRITE_LOCK_TIMEOUT:.0f}s — "
                "a writer is wedged or the lock leaked (see incident 2026-07-03)")
        try:
            yield
        finally:
            self._write_lock.release()

    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
        # execute() commits, so it takes the writer lock even for the occasional
        # SELECT-then-fetch caller. Pure reads use fetchone/fetchall (lock-free).
        # The returned cursor and its lastrowid belong to THIS thread's
        # connection, so reading cursor.lastrowid after the lock releases is safe
        # (and no longer races a concurrent insert on a shared connection).
        self._warn_if_event_loop(sql)
        conn = self._get_conn()
        with self._acquire_write():
            try:
                cursor = conn.execute(sql, params)
                conn.commit()
                return cursor
            except BaseException:
                # A failed commit leaves the connection mid-transaction; the next
                # BEGIN on this thread would then raise inside _Transaction.__enter__.
                # Clear it here so no dangling transaction survives this call.
                if conn.in_transaction:
                    conn.rollback()
                raise

    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        self._warn_if_event_loop(sql)
        conn = self._get_conn()
        with self._acquire_write():
            try:
                cursor = conn.executemany(sql, params_list)
                conn.commit()
                return cursor
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise

    class _Transaction:
        """Context manager for atomic multi-statement transactions."""

        def __init__(self, db: "SafeDB") -> None:
            self._db = db
            self._conn: sqlite3.Connection | None = None

        def __enter__(self) -> "_TransactionCursor":
            # A transaction is a write; hold the writer lock for its duration on
            # THIS thread's connection. Readers on other threads run concurrently
            # (WAL). The connection is captured before BEGIN so all statements in
            # the block — including read-backs of just-inserted rows — hit the
            # same connection and see the in-flight changes.
            if not self._db._write_lock.acquire(timeout=_WRITE_LOCK_TIMEOUT):
                raise TimeoutError(
                    f"SafeDB writer lock not acquired within {_WRITE_LOCK_TIMEOUT:.0f}s — "
                    "a writer is wedged or the lock leaked (see incident 2026-07-03)")
            # If anything after acquire() raises (e.g. BEGIN hits 'database is
            # locked'), the `with` body is never entered and __exit__ never runs
            # — so the lock MUST be released here or it leaks forever. That exact
            # leak froze every writer + the event loop for 54h (2026-07-03).
            try:
                self._conn = self._db._get_conn()
                self._conn.execute("BEGIN")
            except BaseException:
                self._db._write_lock.release()
                raise
            return _TransactionCursor(self._conn)

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            try:
                if exc_type is None:
                    self._conn.commit()
                else:
                    self._conn.rollback()
            finally:
                self._db._write_lock.release()

    def transaction(self) -> "_Transaction":
        """Return a context manager for atomic multi-statement transactions.

        Usage:
            with db.transaction() as tx:
                tx.execute("INSERT INTO ...", (...))
                tx.execute("INSERT INTO ...", (...))
            # commits on success, rolls back on exception
        """
        return self._Transaction(self)

    def fetchone(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Row | None:
        # Lock-free: this thread's own connection, WAL gives a consistent
        # snapshot concurrent with any writer. This is the concurrency win.
        self._warn_if_event_loop(sql)
        return self._get_conn().execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        self._warn_if_event_loop(sql)
        return self._get_conn().execute(sql, params).fetchall()

    def close(self) -> None:
        # Close every per-thread connection. Called at shutdown / test teardown
        # when the DB is quiescent. check_same_thread=False lets us close
        # other threads' connections from here.
        with self._all_conns_lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_conns.clear()
        # Reset per-thread storage so the next call re-opens fresh connections.
        self._local = threading.local()


class AsyncSafeDB:
    """Async wrapper around SafeDB — runs blocking DB calls via to_thread."""

    def __init__(self, sync_db: SafeDB) -> None:
        self._sync = sync_db

    def init_schema(self) -> None:
        self._sync.init_schema()

    async def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
        return await asyncio.to_thread(self._sync.execute, sql, params)

    async def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        return await asyncio.to_thread(self._sync.executemany, sql, params_list)

    async def fetchone(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Row | None:
        return await asyncio.to_thread(self._sync.fetchone, sql, params)

    async def fetchall(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        return await asyncio.to_thread(self._sync.fetchall, sql, params)

    async def run_in_transaction(self, fn) -> Any:
        """Run a callable inside a transaction via to_thread.

        Usage:
            result = await adb.run_in_transaction(
                lambda tx: tx.execute("INSERT ...", (...))
            )
        """
        def _run():
            with self._sync.transaction() as tx:
                return fn(tx)
        return await asyncio.to_thread(_run)

    def transaction(self):
        """Passthrough for sync transaction (backward compat)."""
        return self._sync.transaction()

    def close(self) -> None:
        self._sync.close()


def get_db(db_path: str | None = None) -> SafeDB:
    """Get or create a SafeDB singleton for the given path."""
    if db_path is None:
        from app.config import config
        db_path = config.DB_PATH

    with _instance_lock:
        if db_path not in _instances:
            _instances[db_path] = SafeDB(db_path)
        return _instances[db_path]


def get_async_db(db_path: str | None = None) -> AsyncSafeDB:
    """Get an AsyncSafeDB wrapper for the given path."""
    return AsyncSafeDB(get_db(db_path))


def close_all() -> None:
    """Close all SafeDB instances. Call during shutdown."""
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    with _instance_lock:
        for path, db in _instances.items():
            try:
                db.close()
            except Exception as e:
                _logger.error("Failed to close database %s: %s", path, e)
        _instances.clear()


# ---------------------------------------------------------------------------
# Channel conversation persistence
# ---------------------------------------------------------------------------

_CHANNEL_CONV_SCHEMA = """
CREATE TABLE IF NOT EXISTS channel_conversations (
    channel TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (channel, user_id)
);
"""


class ChannelConversationStore:
    """Persist channel user → conversation_id mappings in SQLite."""

    def __init__(self, db: SafeDB):
        self._db = db
        self._db.execute(_CHANNEL_CONV_SCHEMA.strip())

    def get(self, channel: str, user_id: str) -> str | None:
        row = self._db.fetchone(
            "SELECT conversation_id FROM channel_conversations WHERE channel = ? AND user_id = ?",
            (channel, user_id),
        )
        return row["conversation_id"] if row else None

    def set(self, channel: str, user_id: str, conversation_id: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO channel_conversations (channel, user_id, conversation_id, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (channel, user_id, conversation_id),
        )

    def get_all(self, channel: str) -> dict[str, str]:
        rows = self._db.fetchall(
            "SELECT user_id, conversation_id FROM channel_conversations WHERE channel = ?",
            (channel,),
        )
        return {r["user_id"]: r["conversation_id"] for r in rows}
