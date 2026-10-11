"""Rule C2 (plan 2026-10-09): do the answerability probes predict the outcome?

AUROC of the probe score (answerable + evidence_ok, 0..2) against the terminal
outcome (resolved = 1, failed = 0) over probed curiosity items. Rule C2, read
after 40 probed attempts: AUROC > 0.85 keeps the pick order; otherwise the
probes stay as instruments and the pick order reverts.

    python scripts/curiosity_triage_auroc.py [--db /data/nova.db]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def auroc(pairs: list[tuple[float, int]]) -> float | None:
    """Mann-Whitney AUROC over (score, label) pairs; None with one class only."""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def rows_from_db(path: str) -> list[tuple[float, int]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    out = []
    for r in conn.execute(
        "SELECT answerable, evidence_ok, status FROM curiosity_queue "
        "WHERE probe_at IS NOT NULL AND status IN ('resolved', 'failed')"):
        score = float((r["answerable"] or 0) + (r["evidence_ok"] or 0))
        out.append((score, 1 if r["status"] == "resolved" else 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default="/data/nova.db")
    args = ap.parse_args(argv)
    pairs = rows_from_db(args.db)
    a = auroc(pairs)
    print(json.dumps({"n": len(pairs), "resolved": sum(y for _, y in pairs),
                      "auroc": None if a is None else round(a, 3),
                      "rule_c2_keep_pick_order": bool(a is not None and len(pairs) >= 40 and a > 0.85)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
