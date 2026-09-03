"""The guardian must not treat an operator quiet window as a freeze (2026-09-02).

The watchdog restarts nova-app when no monitor has completed in 90 minutes —
"loop alive but monitors dead". A quiet window pauses the LLM lane on purpose,
so that is exactly what it looks like: 90 minutes into a 3.5-hour A/B replay
the watchdog restarted the app and killed the run. The staleness rule is now
suspended while a window is open; the health rule is not.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import app as _app_pkg

ROOT = Path(_app_pkg.__file__).resolve().parents[1]
WATCHDOG = ROOT / "scripts" / "watchdog.sh"

needs_checkout = pytest.mark.skipif(
    not WATCHDOG.exists(),
    reason="reads scripts/watchdog.sh, which the image does not ship")

# The exact SQL the shell function runs, kept in one place so the test breaks
# when the query drifts from the script.
QUIET_SQL = ("SELECT CASE WHEN datetime(value) > datetime('now') THEN 1 ELSE 0 END "
             "FROM system_state WHERE key='quiet_until';")


def _stamp(delta_minutes: int) -> str:
    return (datetime.now(timezone.utc).replace(tzinfo=None)
            + timedelta(minutes=delta_minutes)).strftime("%Y-%m-%d %H:%M:%S")


def test_query_reports_1_only_while_the_window_is_open(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP)")
    con.commit()

    # no row at all — the query returns nothing, and the shell treats empty as
    # "no window", so the normal staleness rules apply
    assert con.execute(QUIET_SQL).fetchall() == []

    con.execute("INSERT INTO system_state (key, value) VALUES ('quiet_until', ?)", (_stamp(120),))
    con.commit()
    assert con.execute(QUIET_SQL).fetchone()[0] == 1        # window open

    con.execute("UPDATE system_state SET value = ? WHERE key = 'quiet_until'", (_stamp(-1),))
    con.commit()
    assert con.execute(QUIET_SQL).fetchone()[0] == 0        # expired, not open

    con.execute("UPDATE system_state SET value = 'not a timestamp' WHERE key = 'quiet_until'")
    con.commit()
    assert con.execute(QUIET_SQL).fetchone()[0] == 0        # garbage never suspends the rule
    con.close()


def test_quiet_module_and_watchdog_agree_on_the_row(db):
    """The writer and the reader must use the same key and format."""
    from app.monitors.quiet import KEY_UNTIL, set_quiet
    set_quiet(db, 2, "A/B")
    assert KEY_UNTIL == "quiet_until"
    row = db.fetchone("SELECT value FROM system_state WHERE key = ?", (KEY_UNTIL,))
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE system_state (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO system_state VALUES ('quiet_until', ?)", (row["value"],))
    assert con.execute(QUIET_SQL).fetchone()[0] == 1
    con.close()


@needs_checkout
def test_watchdog_suspends_only_the_staleness_rule():
    src = WATCHDOG.read_text(encoding="utf-8")
    assert "quiet_window_active()" in src
    assert QUIET_SQL.split(" FROM ")[0][:40] in re.sub(r"\s+", " ", src)

    # the guard sits INSIDE the staleness branch, before the restart call
    stale_branch = src[src.index('if [ "$stale" -ge "$HB_STALE_MIN" ]'):]
    guard = stale_branch.index("quiet_window_active")
    restart = stale_branch.index("restart_target")
    assert guard < restart, "the quiet check must precede the staleness restart"

    # the health rule must NOT consult the quiet window: a wedged container is
    # still restarted while a window is open
    health_branch = src[src.index('if [ "$fails" -ge "$UNHEALTHY_LIMIT" ]'):
                        src.index('if [ "$st" = "healthy" ]')]
    assert "quiet_window_active" not in health_branch
