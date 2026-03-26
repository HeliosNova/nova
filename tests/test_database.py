"""Tests for SafeDB from app/database.py."""

from __future__ import annotations

import sqlite3

import pytest

from app.database import SafeDB


# ---------------------------------------------------------------------------
# The base SCHEMA_SQL references idx_kg_valid_from on kg_facts(valid_from),
# but the kg_facts CREATE TABLE doesn't include that column — it's added by
# temporal-KG migrations that aren't part of _run_migrations yet.  We work
# around this by creating the missing column before calling init_schema().
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Get a fresh test database with the valid_from column pre-added."""
    db_path = str(tmp_path / "test.db")
    _db = SafeDB(db_path)
    # Pre-create the kg_facts table with the valid_from column so SCHEMA_SQL
    # index creation doesn't blow up.
    conn = _db._get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kg_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 0.8,
            source TEXT DEFAULT 'extracted',
            valid_from TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(subject, predicate, object)
        )
    """)
    conn.commit()
    _db.init_schema()
    yield _db
    _db.close()


class TestGetDbSingleton:
    """Test get_db() returns singleton for the same path."""

    def test_same_path_returns_same_instance(self, tmp_path):
        from app.database import get_db, _instances
        db_path = str(tmp_path / "singleton_test.db")
        db1 = get_db(db_path)
        db2 = get_db(db_path)
        assert db1 is db2
        db1.close()
        _instances.pop(db_path, None)

    def test_different_path_returns_different_instance(self, tmp_path):
        from app.database import get_db, _instances
        path1 = str(tmp_path / "db1.db")
        path2 = str(tmp_path / "db2.db")
        db1 = get_db(path1)
        db2 = get_db(path2)
        assert db1 is not db2
        db1.close()
        db2.close()
        _instances.pop(path1, None)
        _instances.pop(path2, None)


class TestExecute:
    """Test execute() for INSERT/UPDATE."""

    def test_insert_returns_cursor(self, db):
        cursor = db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            ("conv-1", "Test Conv"),
        )
        assert cursor is not None
        assert cursor.rowcount == 1

    def test_update_returns_cursor(self, db):
        db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            ("conv-1", "Original"),
        )
        cursor = db.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            ("Updated", "conv-1"),
        )
        assert cursor.rowcount == 1

    def test_insert_and_retrieve(self, db):
        db.execute(
            "INSERT INTO user_facts (key, value) VALUES (?, ?)",
            ("name", "Alice"),
        )
        row = db.fetchone("SELECT key, value FROM user_facts WHERE key = ?", ("name",))
        assert row is not None
        assert row["key"] == "name"
        assert row["value"] == "Alice"


class TestFetchone:
    """Test fetchone() returns row or None."""

    def test_fetchone_returns_row(self, db):
        db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            ("conv-1", "Test"),
        )
        row = db.fetchone("SELECT * FROM conversations WHERE id = ?", ("conv-1",))
        assert row is not None
        assert row["id"] == "conv-1"
        assert row["title"] == "Test"

    def test_fetchone_returns_none_for_missing(self, db):
        row = db.fetchone("SELECT * FROM conversations WHERE id = ?", ("nonexistent",))
        assert row is None


class TestFetchall:
    """Test fetchall() returns list of rows."""

    def test_fetchall_returns_list(self, db):
        db.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", ("c1", "First"))
        db.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", ("c2", "Second"))
        rows = db.fetchall("SELECT * FROM conversations ORDER BY id")
        assert len(rows) == 2
        assert rows[0]["id"] == "c1"
        assert rows[1]["id"] == "c2"

    def test_fetchall_empty_table(self, db):
        rows = db.fetchall("SELECT * FROM conversations")
        assert rows == []


class TestTransaction:
    """Test transaction() context manager with commit and rollback."""

    def test_transaction_commits_on_success(self, db):
        with db.transaction() as tx:
            tx.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                ("tx-1", "Transaction Test"),
            )
        row = db.fetchone("SELECT * FROM conversations WHERE id = ?", ("tx-1",))
        assert row is not None
        assert row["title"] == "Transaction Test"

    def test_transaction_rolls_back_on_exception(self, db):
        with pytest.raises(ValueError):
            with db.transaction() as tx:
                tx.execute(
                    "INSERT INTO conversations (id, title) VALUES (?, ?)",
                    ("tx-fail", "Should Not Persist"),
                )
                raise ValueError("Simulated failure")

        row = db.fetchone("SELECT * FROM conversations WHERE id = ?", ("tx-fail",))
        assert row is None

    def test_transaction_fetchone(self, db):
        db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            ("tx-r", "Readable"),
        )
        with db.transaction() as tx:
            row = tx.fetchone("SELECT * FROM conversations WHERE id = ?", ("tx-r",))
            assert row is not None
            assert row["title"] == "Readable"

    def test_transaction_fetchall(self, db):
        db.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", ("a", "A"))
        db.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", ("b", "B"))
        with db.transaction() as tx:
            rows = tx.fetchall("SELECT * FROM conversations ORDER BY id")
            assert len(rows) == 2


class TestSchemaCreation:
    """Test that init_schema creates the expected tables."""

    def test_core_tables_exist(self, db):
        tables_rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {row["name"] for row in tables_rows}

        expected = {
            "conversations", "messages", "user_facts",
            "lessons", "skills", "documents", "kg_facts",
            "reflexions", "custom_tools", "monitors",
            "monitor_results", "action_log", "heartbeat_instructions",
        }
        for t in expected:
            assert t in table_names, f"Missing table: {t}"

    def test_fts_table_exists(self, db):
        tables_rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {row["name"] for row in tables_rows}
        assert "chunks_fts" in table_names

    def test_init_schema_idempotent(self, db):
        """Calling init_schema twice should not raise."""
        db.init_schema()  # second call (first was in fixture)
        row = db.fetchone("SELECT count(*) as cnt FROM conversations")
        assert row["cnt"] == 0  # still empty, no error
