"""Append-only record of every entailment verdict with its evidence window.

Why (plan 2026-10-09, task P1): the MiniCheck gate decides what ships and
what dies, and its clause/alt rescues re-admit roughly half of what the
primary check rejected (341 anchored per 31 h measured 2026-10-09) — yet no
scored (claim, document, verdict) tuple was ever stored, so the rescue could
not be audited. This writer keeps the tuples; `scripts/planting_probe.py`
reads them back and tests whether rescued verdicts follow the evidence or the
surface overlap.

One JSONL file per UTC day under RECORD_DIR (`/data/entail/gate-YYYY-MM-DD.jsonl`),
files older than KEEP_DAYS pruned on write. The writer never raises into the
gate: any failure is logged and swallowed — a digest must not die because its
audit trail could not be written.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

RECORD_DIR = Path(os.environ.get("ENTAIL_RECORD_DIR", "/data/entail"))
KEEP_DAYS = 14
FIELDS = ("ts", "label", "site", "claim", "hosts", "doc", "prob", "supported", "narrow")


def _path_for(day: datetime) -> Path:
    return RECORD_DIR / f"gate-{day.strftime('%Y-%m-%d')}.jsonl"


def record(label: str, site: str, rows: Iterable[dict]) -> int:
    """Append one line per scored pair. Returns the number written (0 on any failure)."""
    now = datetime.now(timezone.utc)
    written = 0
    try:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        _prune(now)
        with _path_for(now).open("a", encoding="utf-8") as fh:
            for row in rows:
                line = {
                    "ts": now.isoformat(timespec="seconds"),
                    "label": label,
                    "site": site,
                    "claim": str(row.get("claim", "")),
                    "hosts": list(row.get("hosts") or []),
                    "doc": str(row.get("doc", "")),
                    "prob": row.get("prob"),
                    "supported": bool(row.get("supported")),
                    "narrow": bool(row.get("narrow")),
                }
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
    except Exception as e:  # never into the gate
        logger.warning("[gate-record] write failed (%s/%s): %s", label, site, e)
        return 0
    return written


def _prune(now: datetime) -> None:
    try:
        for p in RECORD_DIR.glob("gate-*.jsonl"):
            try:
                day = datetime.strptime(p.stem[5:], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if (now - day).days > KEEP_DAYS:
                p.unlink(missing_ok=True)
    except Exception as e:
        logger.debug("[gate-record] prune skipped: %s", e)


def read(days: int = KEEP_DAYS) -> Iterator[dict]:
    """Yield records from the newest `days` files, oldest first."""
    try:
        files = sorted(RECORD_DIR.glob("gate-*.jsonl"))[-days:]
    except Exception:
        return
    for p in files:
        try:
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
