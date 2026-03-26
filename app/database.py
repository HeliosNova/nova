"""Thread-safe SQLite wrapper with auto-schema creation.

Ported from Nova's battle-tested SafeDB pattern.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any

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
    created_at TEXT DEFAULT (datetime('now'))
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

-- Indexes
CREATE INDEX IF NOT EXISTS idx_reflexions_outcome ON reflexions(outcome);
CREATE INDEX IF NOT EXISTS idx_reflexions_quality ON reflexions(quality_score);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_lessons_topic ON lessons(topic);
CREATE INDEX IF NOT EXISTS idx_skills_trigger ON skills(trigger_pattern);
CREATE INDEX IF NOT EXISTS idx_kg_subject ON kg_facts(subject);
CREATE INDEX IF NOT EXISTS idx_kg_object ON kg_facts(object);
CREATE INDEX IF NOT EXISTS idx_kg_valid_from ON kg_facts(valid_from);
CREATE INDEX IF NOT EXISTS idx_monitors_enabled ON monitors(enabled);
CREATE INDEX IF NOT EXISTS idx_monitor_results_monitor ON monitor_results(monitor_id, created_at);
CREATE INDEX IF NOT EXISTS idx_action_log_type ON action_log(action_type, created_at);
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
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def init_schema(self) -> None:
        """Create all tables. Safe to call multiple times."""
        with self._lock:
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

    def execute(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor

    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.executemany(sql, params_list)
            conn.commit()
            return cursor

    class _Transaction:
        """Context manager for atomic multi-statement transactions."""

        def __init__(self, db: "SafeDB") -> None:
            self._db = db
            self._conn: sqlite3.Connection | None = None

        def __enter__(self) -> "_TransactionCursor":
            self._db._lock.acquire()
            self._conn = self._db._get_conn()
            self._conn.execute("BEGIN")
            return _TransactionCursor(self._conn)

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            try:
                if exc_type is None:
                    self._conn.commit()
                else:
                    self._conn.rollback()
            finally:
                self._db._lock.release()

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
        with self._lock:
            conn = self._get_conn()
            return conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            conn = self._get_conn()
            return conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


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
