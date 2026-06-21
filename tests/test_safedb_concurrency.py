"""SafeDB per-thread-connection concurrency (audit 2026-06-12).

A single shared connection behind one global lock serialized every read and
write, negating WAL and causing the recurring event-loop lock-convoy incidents.
SafeDB now uses one connection per thread + a write-only mutex, so:
  - reads run lock-free and do NOT block behind a held write transaction;
  - each thread gets its own connection (no cross-thread cursor/rowid races).
"""
from __future__ import annotations

import threading
import time

import pytest

from app.database import SafeDB


def _fresh(tmp_path) -> SafeDB:
    db = SafeDB(str(tmp_path / "concurrency.db"))
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
    db.execute("INSERT INTO t (v) VALUES ('seed')")
    return db


def test_read_not_blocked_by_held_write_transaction(tmp_path):
    db = _fresh(tmp_path)
    tx_open = threading.Event()
    release = threading.Event()
    err: list[Exception] = []

    def writer():
        try:
            with db.transaction() as tx:
                tx.execute("INSERT INTO t (v) VALUES ('inflight')")
                tx_open.set()
                release.wait(timeout=5)  # hold the writer lock open
        except Exception as e:  # pragma: no cover
            err.append(e)

    wt = threading.Thread(target=writer)
    wt.start()
    assert tx_open.wait(timeout=5), "writer never opened its transaction"

    # The read happens while the writer holds the write lock. Under the old
    # global-lock design this blocked until the transaction released (it would
    # deadlock here, since release is set only AFTER the read). It must return
    # promptly now.
    start = time.perf_counter()
    rows = db.fetchall("SELECT v FROM t")
    elapsed = time.perf_counter() - start

    release.set()
    wt.join(timeout=5)
    assert not err, f"writer errored: {err}"

    assert elapsed < 2.0, f"read blocked behind the write lock ({elapsed:.2f}s)"
    vals = {r["v"] for r in rows}
    assert "seed" in vals
    # WAL isolation: the other connection must NOT see the uncommitted insert.
    assert "inflight" not in vals


def test_distinct_connection_per_thread(tmp_path):
    db = _fresh(tmp_path)
    conn_ids: dict[str, int] = {}

    def grab(name):
        conn_ids[name] = id(db._get_conn())

    for name in ("a", "b"):
        t = threading.Thread(target=grab, args=(name,))
        t.start()
        t.join()

    main_id = id(db._get_conn())
    assert conn_ids["a"] != conn_ids["b"], "threads must not share a connection"
    assert main_id not in conn_ids.values()


def test_concurrent_writes_serialize_without_corruption(tmp_path):
    db = _fresh(tmp_path)
    errors: list[Exception] = []

    def write(n):
        try:
            for i in range(20):
                db.execute("INSERT INTO t (v) VALUES (?)", (f"{n}-{i}",))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"concurrent writes errored: {errors}"
    count = db.fetchone("SELECT COUNT(*) AS c FROM t")["c"]
    assert count == 1 + 8 * 20  # seed + every write landed, none lost
    db.close()
