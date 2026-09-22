"""SQLite snapshot verification — the "tested" half of tested backups.

A backup that has never been opened is a hope, not a backup. Every
snapshot Nova takes is verified immediately after creation:

  1. PRAGMA integrity_check must return "ok"
  2. every key table must be readable
  3. the snapshot must not be inexplicably empty (a successful VACUUM INTO
     of the wrong/fresh database would pass integrity_check)

The same verifier doubles as the restore drill:

    python -m app.core.backup /path/to/snapshot.db

prints the verdict and row counts, and exits non-zero on failure — run it
against any snapshot before restoring it.

Restore procedure (documented here so it lives next to the verifier):
  1. docker compose stop nova
  2. verify the snapshot:  python -m app.core.backup <snapshot>
  3. inside the volume, replace the live db:
       docker run --rm -v nova_data:/data -v <host-backup-dir>:/backups alpine \
         sh -c "cp /backups/<snapshot> /data/nova.db && rm -f /data/nova.db-wal /data/nova.db-shm"
  4. docker compose up -d nova; wait for healthy
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Tables whose loss would be unrecoverable — all must be readable in a
# verified snapshot. (Row counts are informational except the all-empty
# guard below.)
KEY_TABLES = (
    "conversations",
    "messages",
    "user_facts",
    "lessons",
    "skills",
    "reflexions",
    "kg_facts",
    "monitors",
)


def verify_snapshot(path: str | Path) -> tuple[bool, str]:
    """Open a snapshot read-only and prove it is restorable.

    Returns (ok, detail). Never raises — a verification failure is a
    result, not an exception.
    """
    p = Path(path)
    if not p.exists():
        return False, f"snapshot missing: {p}"
    if p.stat().st_size == 0:
        return False, f"snapshot is zero bytes: {p}"

    try:
        conn = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return False, f"cannot open snapshot: {e}"

    try:
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.Error as e:
            return False, f"integrity_check failed to run: {e}"
        if not row or row[0] != "ok":
            return False, f"integrity_check: {row[0] if row else 'no result'}"

        counts: dict[str, int] = {}
        for table in KEY_TABLES:
            try:
                counts[table] = conn.execute(
                    f'SELECT count(*) FROM "{table}"'
                ).fetchone()[0]
            except sqlite3.Error as e:
                return False, f"key table {table} unreadable: {e}"

        if all(n == 0 for n in counts.values()):
            # integrity_check passes on a freshly-initialized empty schema —
            # but an all-empty "backup" of a production system means we
            # snapshotted the wrong database.
            return False, "all key tables empty — wrong database snapshotted?"

        detail = "ok | " + " ".join(f"{t}={n}" for t, n in counts.items())
        return True, detail
    finally:
        conn.close()


DR_EXTRAS = ("models_manifest.txt", "config_overrides.json")


async def write_dr_extras(dest: str | Path, ollama_url: str,
                          overrides: str | Path = "/data/config_overrides.json") -> list[str]:
    """Write the disaster-recovery extras beside the snapshots in `dest`.

    Thirty gigabytes of model weights are re-pullable, so a MANIFEST of what
    is installed is their backup; the config overrides are tiny and essential
    (a stale LLM_MODEL there once 404'd every generation). The maintenance
    monitor used to write these straight onto the off-volume mount, and when
    the backup sidecar took the legs over on 2026-09-01 they froze at that
    day's copy for three weeks. Written into the volume now, where the sidecar
    picks them up with the snapshot. Never raises; returns what it wrote.
    """
    import asyncio
    import shutil

    dest = Path(dest)
    written: list[str] = []
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("[Backup] extras dir %s unavailable: %s", dest, e)
        return written
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            tags = (await client.get(f"{ollama_url.rstrip('/')}/api/tags")).json()
        names = sorted(m.get("name", "?") for m in tags.get("models", []))
        (dest / "models_manifest.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        written.append("models_manifest.txt")
    except Exception as e:
        logger.warning("[Backup] models manifest failed: %s", e)
    try:
        ov = Path(overrides)
        if ov.is_file():
            await asyncio.to_thread(shutil.copyfile, ov, dest / "config_overrides.json")
            written.append("config_overrides.json")
    except Exception as e:
        logger.warning("[Backup] config override copy failed: %s", e)
    return written


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify a Nova SQLite snapshot is restorable (restore drill)."
    )
    parser.add_argument("snapshot", help="path to the snapshot .db file")
    args = parser.parse_args()

    ok, detail = verify_snapshot(args.snapshot)
    print(("VERIFIED " if ok else "FAILED   ") + detail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
