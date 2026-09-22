"""The watchdog's staleness rule counts dispatches, not only completions
(2026-09-22).

At 22:20 UTC the watchdog restarted a healthy nova-app: a deploy restart at
21:30 had killed the running batch, the replacement batch of three ~50-minute
digests had produced no completion, and the last completion was 90 minutes
old. The rule read monitors.last_check_at alone. The loop now stamps
system_state.last_dispatch_at when a monitor starts, and the rule takes the
newer of the two. A loop that neither dispatches nor completes is still the
freeze the rule exists for.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import app as _app_pkg
from app.monitors.heartbeat_loop import HeartbeatLoop
from app.monitors.monitor_store import MonitorStore

ROOT = Path(_app_pkg.__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "watchdog.sh"

# The SQL the shell function runs, with the shell's line continuations folded.
STALE_SQL = (
    "SELECT CAST((julianday('now') - julianday(MAX(t)))*1440 AS INTEGER) FROM ( "
    "SELECT MAX(last_check_at) AS t FROM monitors WHERE enabled=1 AND last_check_at IS NOT NULL "
    "UNION ALL "
    "SELECT value AS t FROM system_state WHERE key='last_dispatch_at');"
)


def _ts(minutes_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")


def _con():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE monitors (id INTEGER PRIMARY KEY, enabled INTEGER, last_check_at TEXT)")
    con.execute("CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    return con


@pytest.mark.skipif(not SCRIPT.exists(), reason="scripts/ is not shipped in the image")
def test_the_script_runs_this_query():
    src = SCRIPT.read_text(encoding="utf-8")
    folded = re.sub(r"\\\s*\n\s*", " ", src)
    for fragment in ("MAX(last_check_at) AS t FROM monitors", "UNION ALL",
                     "SELECT value AS t FROM system_state WHERE key='last_dispatch_at'"):
        assert fragment in folded, fragment
    assert "\r" not in src, "CRLF in a shell script kills the alpine watchdog (2026-09-03)"


def test_a_recent_dispatch_keeps_a_batch_in_flight_from_reading_stale():
    con = _con()
    con.execute("INSERT INTO monitors VALUES (1, 1, ?)", (_ts(120),))     # last completion 2 h ago
    assert con.execute(STALE_SQL).fetchone()[0] == 120                    # no dispatch row: the old rule
    con.execute("INSERT INTO system_state VALUES ('last_dispatch_at', ?, ?)", (_ts(5), _ts(5)))
    assert con.execute(STALE_SQL).fetchone()[0] == 5                      # dispatching = alive


def test_a_loop_that_neither_dispatches_nor_completes_still_reads_stale():
    con = _con()
    con.execute("INSERT INTO monitors VALUES (1, 1, ?)", (_ts(200),))
    con.execute("INSERT INTO system_state VALUES ('last_dispatch_at', ?, ?)", (_ts(180), _ts(180)))
    assert con.execute(STALE_SQL).fetchone()[0] == 180


@pytest.mark.asyncio
async def test_the_loop_stamps_every_dispatch(db, monkeypatch):
    import app.core.brain as brain
    from types import SimpleNamespace
    monkeypatch.setattr(brain, "get_services", lambda: SimpleNamespace(kg=None))
    store = MonitorStore(db)
    loop = HeartbeatLoop(store)
    mid = store.create("Dispatch stamp probe", "url", {"url": "https://example.com"},
                       schedule_seconds=3600, cooldown_minutes=0, notify_condition="always")
    monitor = store.get(mid)
    assert db.fetchone("SELECT value FROM system_state WHERE key='last_dispatch_at'") is None
    with patch.object(loop, "_execute_check", new_callable=AsyncMock) as mock_exec, \
         patch.object(loop, "_send_alert", new_callable=AsyncMock) as mock_send:
        mock_exec.return_value = "fresh content"
        mock_send.return_value = True
        await loop._check_monitor(monitor)
    row = db.fetchone("SELECT value FROM system_state WHERE key='last_dispatch_at'")
    assert row is not None
    stamped = datetime.strptime(row["value"], "%Y-%m-%d %H:%M:%S")
    assert abs((datetime.now(timezone.utc).replace(tzinfo=None) - stamped).total_seconds()) < 120
