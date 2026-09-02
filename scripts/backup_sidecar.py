"""Backup sidecar (2026-09-01).

Runs in its own container (network none, nova_data read-only) and owns both
off-volume disaster-recovery legs: it copies the newest VERIFIED snapshot from
/data/backups (written by the maintenance monitor inside nova-app) to
/backups (F:) and /offsite (E:), verifies each copy with an integrity check,
and prunes each leg to the last seven.

Before this, both legs were mounted read-write inside nova-app — the container
that executes LLM-authored `sh -c` — so one injected `rm -rf /offsite/*` could
erase the live data and both mirrors in a single command.

    python scripts/backup_sidecar.py          # loop hourly (container entrypoint)
    python scripts/backup_sidecar.py --once   # one pass (tests / manual)
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

SRC = Path(os.environ.get("BACKUP_SRC_DIR", "/data/backups"))
LEGS = [Path(p) for p in os.environ.get("BACKUP_LEGS", "/backups,/offsite").split(",") if p]
KEEP = int(os.environ.get("BACKUP_KEEP", "7"))
INTERVAL = int(os.environ.get("BACKUP_INTERVAL_SECONDS", "3600"))


def verify(path: Path) -> bool:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            ok = con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            rows = con.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]
        finally:
            con.close()
        return ok and rows > 0
    except Exception:
        return False


def sync_once(src: Path = SRC, legs: list[Path] | None = None, keep: int = KEEP) -> dict:
    """Copy the newest verified snapshot to every mounted leg; prune to `keep`."""
    legs = LEGS if legs is None else legs
    report: dict = {"copied": [], "skipped": [], "failed": []}
    # Newest by mtime, not by name: a lexicographic sort put the stale
    # `nova-premove.db` (Aug 28) after every dated `nova-2026*.db`, and its
    # failed verification blocked the whole sync on the first live run
    # (2026-09-02). Walk newest→oldest and take the first snapshot that verifies.
    snaps = sorted((p for p in src.glob("nova-*.db") if p.is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not snaps:
        report["skipped"].append("no snapshots in source")
        return report
    latest = None
    for cand in snaps:
        if verify(cand):
            latest = cand
            break
        report["failed"].append(f"source snapshot {cand.name} failed verification")
    if latest is None:
        return report
    for leg in legs:
        if not leg.is_dir():
            report["skipped"].append(f"{leg} not mounted")
            continue
        target = leg / latest.name
        if target.exists() and verify(target):
            report["skipped"].append(f"{target} already present")
        else:
            tmp = leg / (latest.name + ".part")
            try:
                shutil.copyfile(latest, tmp)
                if verify(tmp):
                    tmp.replace(target)
                    report["copied"].append(str(target))
                else:
                    tmp.unlink(missing_ok=True)
                    report["failed"].append(f"{target} failed verification after copy")
            except Exception as e:
                tmp.unlink(missing_ok=True)
                report["failed"].append(f"{target}: {e}")
        for old in sorted((p for p in leg.glob("nova-*.db") if p.is_file()),
                          key=lambda p: p.stat().st_mtime)[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass
    return report


def main() -> int:
    once = "--once" in sys.argv
    while True:
        try:
            rep = sync_once()
            print(f"[backup-sidecar] copied={rep['copied']} skipped={rep['skipped']} failed={rep['failed']}", flush=True)
        except Exception as e:  # never die: the next pass may succeed
            print(f"[backup-sidecar] sync failed: {e}", flush=True)
        if once:
            return 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
