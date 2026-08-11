"""SafeDB writer-lock leak regression (incident 2026-07-03 → 2026-07-06).

A `_Transaction.__enter__` that raised AFTER acquiring the writer lock (e.g.
BEGIN hitting 'database is locked' or 'cannot start a transaction within a
transaction') never ran __exit__, leaking the RLock forever. Every subsequent
writer — including the event-loop thread via a sync record_outcome — blocked
permanently: 54 hours of total outage with the container up and "healthy-ish".

py-spy dump evidence: 6 threads waiting at database.py `with self._write_lock:`,
zero threads inside it — the owner was an idle pool thread holding a leaked level.

These tests are deterministic (no LLM, no network) and red on the old code:
  1. a failed BEGIN must not leak the lock,
  2. a failed commit must not leave the connection mid-transaction (the very
     state that makes the NEXT BEGIN raise → the leak trigger),
  3. lock acquisition must time out loudly instead of hanging forever.
"""
from __future__ import annotations

import sqlite3
import threading

import pytest

import app.database as database_mod
from app.database import SafeDB


def _fresh(tmp_path) -> SafeDB:
    db = SafeDB(str(tmp_path / "lockleak.db"))
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
    return db


def _write_from_other_thread(db: SafeDB, timeout: float = 5.0):
    """Attempt a write on a separate thread; return the outcome or None on hang."""
    out: list = []

    def attempt():
        try:
            db.execute("INSERT INTO t (v) VALUES ('probe')")
            out.append("ok")
        except Exception as e:
            out.append(e)

    th = threading.Thread(target=attempt, daemon=True)
    th.start()
    th.join(timeout=timeout)
    return out[0] if out else None


def test_failed_begin_does_not_leak_writer_lock(tmp_path):
    # The incident shape: BEGIN raises after the lock is acquired. Reproduced
    # deterministically via a nested transaction on the same thread — the RLock
    # re-acquires fine, then BEGIN raises 'cannot start a transaction within a
    # transaction'. Old code leaked that recursion level forever.
    db = _fresh(tmp_path)
    with db.transaction() as tx:
        tx.execute("INSERT INTO t (v) VALUES ('outer')")
        with pytest.raises(sqlite3.OperationalError):
            with db.transaction():
                pass  # pragma: no cover — enter must raise

    # The outer transaction exited cleanly; if the failed inner enter leaked its
    # lock level, this write from another thread blocks forever (old behavior).
    assert _write_from_other_thread(db) == "ok", \
        "writer lock leaked by a failed _Transaction.__enter__"
    db.close()


def test_failed_commit_rolls_back_dangling_transaction(tmp_path):
    # A commit() failure inside execute() used to leave the per-thread
    # connection mid-transaction — the state that makes the next BEGIN on that
    # thread raise, which is what triggered the lock leak live.
    db = _fresh(tmp_path)
    real = db._get_conn()

    class _CommitFails:
        def __getattr__(self, name):
            return getattr(real, name)

        def commit(self):
            raise sqlite3.OperationalError("disk I/O error (simulated)")

    db._local.conn = _CommitFails()
    try:
        with pytest.raises(sqlite3.OperationalError):
            db.execute("INSERT INTO t (v) VALUES ('doomed')")
    finally:
        db._local.conn = real

    assert not real.in_transaction, \
        "failed commit left the connection mid-transaction (next BEGIN would raise)"
    # And the follow-up transaction on this thread must work normally.
    with db.transaction() as tx:
        tx.execute("INSERT INTO t (v) VALUES ('after')")
    assert db.fetchone("SELECT COUNT(*) AS c FROM t WHERE v='after'")["c"] == 1
    # The other-thread writer path stays healthy too.
    assert _write_from_other_thread(db) == "ok"
    db.close()


def test_writer_lock_timeout_raises_instead_of_hanging(tmp_path, monkeypatch):
    # If the lock IS ever wedged again, writers must fail loudly with evidence
    # — not freeze the process silently for days.
    db = _fresh(tmp_path)
    monkeypatch.setattr(database_mod, "_WRITE_LOCK_TIMEOUT", 0.2)

    # Wedge the lock from a thread that never releases it. The thread must stay
    # ALIVE while we probe: a dead thread's ident gets recycled, and an RLock
    # would then mistake the probe thread for its owner and let it re-enter.
    acquired = threading.Event()
    done = threading.Event()

    def leak():
        db._write_lock.acquire()
        acquired.set()
        done.wait(timeout=30)

    lt = threading.Thread(target=leak, daemon=True)
    lt.start()
    assert acquired.wait(timeout=5)

    try:
        outcome = _write_from_other_thread(db)
        assert isinstance(outcome, TimeoutError), \
            f"execute under a wedged lock must raise TimeoutError, got {outcome!r}"

        def enter_txn():
            try:
                with db.transaction():
                    pass  # pragma: no cover
                return "ok"
            except Exception as e:
                return e

        out: list = []
        th = threading.Thread(target=lambda: out.append(enter_txn()), daemon=True)
        th.start()
        th.join(timeout=5)
        assert out and isinstance(out[0], TimeoutError), \
            f"transaction enter under a wedged lock must raise TimeoutError, got {out!r}"
    finally:
        done.set()
