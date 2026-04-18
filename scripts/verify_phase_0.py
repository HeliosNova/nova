"""Phase-0 bootstrap verification.

Reports pass/fail on each Phase-0 acceptance criterion in one shot.

USAGE
-----
After the stack has been up for ~48h, run:

    docker exec nova-app python -m scripts.verify_phase_0

The script returns exit code 0 if every check passes, otherwise 1.

CHECKS
------
1. daemon_log has >20 entries in the last 48h (real ticks happened)
2. At least one dream-consolidation action recorded in daemon_log
3. The phase_0_bootstrap goal is still present in the goals table
4. ENABLE_SHELL_EXEC is True at runtime (honored by app.config)
5. The 5 Phase-0 ambient monitors are enabled
6. event_queue and daemon_log tables exist (migration 14 applied)

It never mutates state — purely read-only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta


# ──────────────────────────────────────────────────────────────────────────
# Check helpers
# ──────────────────────────────────────────────────────────────────────────


def _cutoff(hours: int) -> str:
    return (datetime.utcnow() - timedelta(hours=hours)).isoformat()


def check_schema(db) -> tuple[bool, str]:
    """Migration 14 applied — daemon_log + event_queue tables exist."""
    names = {
        row[0]
        for row in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = [t for t in ("daemon_log", "event_queue", "goals") if t not in names]
    if missing:
        return False, f"missing tables: {', '.join(missing)}"
    return True, "daemon_log, event_queue, goals present"


def check_daemon_ticks(db, min_ticks: int = 20, hours: int = 48) -> tuple[bool, str]:
    """Daemon_log has >N entries in the last H hours.

    Any category counts — every tick that took a decision writes at least one
    row. Empty-tick ticks (no-op) are not logged, which is by design.
    """
    row = db.fetchone(
        "SELECT COUNT(*) AS c FROM daemon_log WHERE created_at > ?",
        (_cutoff(hours),),
    )
    n = row["c"] if row else 0
    passed = n > min_ticks
    return passed, f"{n} daemon_log entries in last {hours}h (need >{min_ticks})"


def check_dream_action(db, hours: int = 48) -> tuple[bool, str]:
    """At least one dream-consolidation action recorded.

    Written by DreamConsolidator.report() and/or daemon._trigger_dream().
    """
    row = db.fetchone(
        "SELECT COUNT(*) AS c FROM daemon_log "
        "WHERE created_at > ? "
        "  AND (category='dream' OR source='dream_consolidator' "
        "       OR (category='action' AND source='dream'))",
        (_cutoff(hours),),
    )
    n = row["c"] if row else 0
    passed = n > 0
    return passed, f"{n} dream-consolidation entries in last {hours}h (need ≥1)"


def check_bootstrap_goal(db) -> tuple[bool, str]:
    """The Phase-0 bootstrap goal is still present.

    'Still present' is the weakest success criterion — the stronger one is
    that the goal was picked up and pursued (status != 'pending'), which
    requires the will-module to exist. Either state is acceptable for a
    48h-bake pass.
    """
    row = db.fetchone(
        "SELECT id, goal, status, priority FROM goals "
        "WHERE source='phase_0_bootstrap' "
        "ORDER BY id LIMIT 1"
    )
    if not row:
        return False, "bootstrap goal row missing (migration 15 not applied?)"
    return True, (
        f"goal id={row['id']} status={row['status']} "
        f"priority={row['priority']}: {row['goal'][:80]}…"
    )


def check_shell_exec_runtime() -> tuple[bool, str]:
    """ENABLE_SHELL_EXEC is True according to the live config."""
    from app.config import config

    val = bool(config.ENABLE_SHELL_EXEC)
    return val, f"config.ENABLE_SHELL_EXEC = {val}"


def check_ambient_monitors(db) -> tuple[bool, str]:
    """The 5 Phase-0 ambient monitors exist AND are enabled."""
    names = (
        "Lesson Quiz",
        "Skill Validation",
        "Auto-Monitor Detector",
        "Fine-Tune Check",
        "Hacker News Top Stories",
    )
    rows = db.fetchall(
        "SELECT name, enabled FROM monitors WHERE name IN ({})".format(
            ",".join("?" for _ in names)
        ),
        names,
    )
    found = {r["name"]: bool(r["enabled"]) for r in rows}
    missing = [n for n in names if n not in found]
    disabled = [n for n, e in found.items() if not e]
    if missing:
        return False, f"missing monitors (seed did not run?): {', '.join(missing)}"
    if disabled:
        return False, f"present but disabled: {', '.join(disabled)}"
    return True, f"all {len(names)} ambient monitors enabled"


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────


def _fmt(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-0 bootstrap verification")
    parser.add_argument(
        "--min-ticks",
        type=int,
        default=20,
        help="Minimum daemon_log entries required in the window (default 20)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="Lookback window in hours (default 48)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Override DB path (default: config.DB_PATH)",
    )
    args = parser.parse_args()

    # Lazy import so `--help` works without app config being valid.
    from app.database import get_db

    db = get_db(args.db_path)
    # Don't init_schema here — verify is read-only. If the DB is missing,
    # the checks will fail clearly.

    results: list[tuple[str, bool, str]] = []

    name, (ok, detail) = "schema", check_schema(db)
    results.append((name, ok, detail))

    name, (ok, detail) = "daemon_ticks", check_daemon_ticks(
        db, min_ticks=args.min_ticks, hours=args.hours
    )
    results.append((name, ok, detail))

    name, (ok, detail) = "dream_action", check_dream_action(db, hours=args.hours)
    results.append((name, ok, detail))

    name, (ok, detail) = "bootstrap_goal", check_bootstrap_goal(db)
    results.append((name, ok, detail))

    name, (ok, detail) = "shell_exec_runtime", check_shell_exec_runtime()
    results.append((name, ok, detail))

    name, (ok, detail) = "ambient_monitors", check_ambient_monitors(db)
    results.append((name, ok, detail))

    # Report
    print("=" * 72)
    print(f"Phase-0 Verification — {datetime.utcnow().isoformat()}Z")
    print("=" * 72)
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"  [{_fmt(ok)}]  {name:<{width}}  {detail}")
    print("=" * 72)

    all_ok = all(ok for _, ok, _ in results)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
