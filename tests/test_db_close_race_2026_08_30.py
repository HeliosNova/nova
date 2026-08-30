"""close() must not free a connection while another thread is mid-query.

The full suite crashed with exit 139 (SIGSEGV) — not a test failure — with the
fault at database.py fetchall() on a concurrent.futures worker thread.

Mechanism: SafeDB opens connections with check_same_thread=False and registers
them all, so close() can shut OTHER threads' connections. Its docstring says
that is safe "when the DB is quiescent" — a precondition that was documented
but never enforced. pytest teardown calls close_all() while asyncio.to_thread
workers may still be running queries; closing a connection inside
sqlite3_step() is a C-level use-after-free, which is a segfault rather than a
catchable Python error.

app/main.py already carried three comments about ordering shutdown to dodge
this, i.e. it had bitten production twice and was worked around by sequencing
rather than by making close() safe.
"""

from __future__ import annotations

import threading
import time

import pytest


class TestInFlightAccounting:
    def test_counter_returns_to_zero(self, db):
        db.execute("CREATE TABLE IF NOT EXISTS t_race (k INTEGER)")
        db.fetchall("SELECT * FROM t_race")
        db.fetchone("SELECT 1")
        assert db._inflight == 0, "in-flight counter leaked"

    def test_counter_returns_to_zero_on_error(self, db):
        with pytest.raises(Exception):
            db.fetchall("SELECT * FROM table_that_does_not_exist")
        assert db._inflight == 0, (
            "counter must decrement on the failure path too — a leak here would "
            "make close() block for its full timeout, every time"
        )

    def test_write_path_is_counted(self, db):
        db.execute("CREATE TABLE IF NOT EXISTS t_race2 (k INTEGER)")
        seen = []
        real = db._enter_query

        def spy():
            real()
            seen.append(db._inflight)

        db._enter_query = spy
        try:
            db.execute("INSERT INTO t_race2 (k) VALUES (1)")
        finally:
            db._enter_query = real
        assert seen and seen[0] >= 1, (
            "execute() must be counted: close() takes no write lock, so writes "
            "are as exposed as reads"
        )


class TestCloseWaitsForQuiescence:
    def test_close_waits_for_an_in_flight_query(self, db):
        """close() must not proceed while a query is running."""
        db.execute("CREATE TABLE IF NOT EXISTS t_race3 (k INTEGER)")
        started = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def slow_query():
            db._enter_query()
            started.set()
            order.append("query-start")
            release.wait(timeout=5)
            order.append("query-end")
            db._exit_query()

        t = threading.Thread(target=slow_query, daemon=True)
        t.start()
        started.wait(timeout=5)

        def closer():
            db.close()
            order.append("close-done")

        c = threading.Thread(target=closer, daemon=True)
        c.start()
        time.sleep(0.2)
        assert "close-done" not in order, "close() proceeded while a query was in flight"

        release.set()
        t.join(timeout=5)
        c.join(timeout=5)
        assert order.index("query-end") < order.index("close-done"), (
            "close() must complete only after the in-flight query finishes"
        )

    def test_close_is_bounded_and_does_not_hang(self, db):
        """A stuck query must not block shutdown forever."""
        db._enter_query()          # simulate a query that never returns
        try:
            t0 = time.monotonic()
            db.close()             # must give up after its bounded wait
            elapsed = time.monotonic() - t0
            assert elapsed < 6.0, (
                f"close() blocked {elapsed:.1f}s — the wait must be bounded so a "
                f"hung query cannot wedge shutdown"
            )
        finally:
            db._exit_query()
