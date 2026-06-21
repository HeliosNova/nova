"""Tests for app/core/backup.py — snapshot verification (the restore drill)."""

from __future__ import annotations

import sqlite3

import pytest

from app.core.backup import KEY_TABLES, verify_snapshot


def _make_snapshot(path, rows_in: dict[str, int] | None = None):
    """Create a snapshot file with the key-table schema and optional rows."""
    conn = sqlite3.connect(str(path))
    for t in KEY_TABLES:
        conn.execute(f'CREATE TABLE "{t}" (id INTEGER PRIMARY KEY, content TEXT)')
    for t, n in (rows_in or {}).items():
        for i in range(n):
            conn.execute(f'INSERT INTO "{t}" (content) VALUES (?)', (f"row{i}",))
    conn.commit()
    conn.close()


class TestVerifySnapshot:
    def test_good_snapshot_verifies(self, tmp_path):
        snap = tmp_path / "nova-20260612.db"
        _make_snapshot(snap, {"conversations": 3, "lessons": 2})
        ok, detail = verify_snapshot(snap)
        assert ok, detail
        assert "conversations=3" in detail
        assert "lessons=2" in detail

    def test_missing_file_fails(self, tmp_path):
        ok, detail = verify_snapshot(tmp_path / "nope.db")
        assert not ok
        assert "missing" in detail

    def test_zero_byte_file_fails(self, tmp_path):
        snap = tmp_path / "empty.db"
        snap.write_bytes(b"")
        ok, detail = verify_snapshot(snap)
        assert not ok
        assert "zero bytes" in detail

    def test_garbage_file_fails(self, tmp_path):
        snap = tmp_path / "garbage.db"
        snap.write_bytes(b"this is not a sqlite database, not even close" * 10)
        ok, detail = verify_snapshot(snap)
        assert not ok

    def test_missing_key_table_fails(self, tmp_path):
        snap = tmp_path / "partial.db"
        conn = sqlite3.connect(str(snap))
        conn.execute("CREATE TABLE conversations (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO conversations DEFAULT VALUES")
        conn.commit()
        conn.close()
        ok, detail = verify_snapshot(snap)
        assert not ok
        assert "unreadable" in detail

    def test_all_empty_tables_fail(self, tmp_path):
        """integrity_check passes on a fresh empty schema — but an all-empty
        'backup' of production means the wrong database was snapshotted."""
        snap = tmp_path / "fresh.db"
        _make_snapshot(snap, rows_in=None)
        ok, detail = verify_snapshot(snap)
        assert not ok
        assert "empty" in detail

    def test_corrupted_snapshot_fails(self, tmp_path):
        snap = tmp_path / "corrupt.db"
        _make_snapshot(snap, {"conversations": 200, "messages": 200})
        # Truncate the file mid-way — a torn copy is the classic
        # backup-corruption mode (interrupted transfer, full disk).
        data = snap.read_bytes()
        snap.write_bytes(data[: int(len(data) * 0.6)])
        ok, detail = verify_snapshot(snap)
        assert not ok

    def test_cli_exit_codes(self, tmp_path):
        import subprocess
        import sys
        snap = tmp_path / "cli.db"
        _make_snapshot(snap, {"monitors": 1})
        good = subprocess.run(
            [sys.executable, "-m", "app.core.backup", str(snap)],
            capture_output=True, text=True,
        )
        assert good.returncode == 0
        assert "VERIFIED" in good.stdout
        bad = subprocess.run(
            [sys.executable, "-m", "app.core.backup", str(tmp_path / "absent.db")],
            capture_output=True, text=True,
        )
        assert bad.returncode == 1
        assert "FAILED" in bad.stdout
