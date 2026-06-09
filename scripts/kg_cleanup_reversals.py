"""One-shot cleanup of reversed-direction triples in kg_facts.

Identifies obvious reversals (e.g., "Russia capital_of Moscow") using the
same heuristics as is_garbage_triple, then either:
  - FLIPS them when there's no existing forward triple
  - SUPERSEDES them when a forward triple already exists

Run once after deploying the prompt fix:
    docker exec nova-app python /app/scripts/kg_cleanup_reversals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.kg import _is_country, _is_org


def main() -> int:
    import sqlite3
    db = sqlite3.connect("/data/nova.db")
    db.row_factory = sqlite3.Row

    flipped = 0
    superseded = 0
    skipped = 0

    rows = db.execute(
        "SELECT id, subject, predicate, object FROM kg_facts WHERE valid_to IS NULL"
    ).fetchall()
    print(f"Scanning {len(rows)} current facts ...")

    for r in rows:
        fid, s, p, o = r["id"], r["subject"], r["predicate"], r["object"]
        p_low = (p or "").lower()
        s_low = (s or "").strip().lower()
        o_low = (o or "").strip().lower()

        is_reversal = False
        if p_low == "capital_of" and _is_country(s):
            is_reversal = True
        elif p_low in ("works_at", "leads") and _is_org(s) and not _is_org(o):
            is_reversal = True

        if not is_reversal:
            continue

        # Does a correctly-oriented triple already exist?
        existing = db.execute(
            "SELECT id FROM kg_facts WHERE lower(subject)=? AND predicate=? AND lower(object)=? AND valid_to IS NULL",
            (o_low, p, s_low),
        ).fetchone()

        if existing:
            # Forward already there — supersede the reversed one
            db.execute(
                "UPDATE kg_facts SET valid_to=datetime('now'), superseded_by=? WHERE id=?",
                (existing["id"], fid),
            )
            superseded += 1
            print(f"  superseded #{fid}: {s} {p} {o}  (forward #{existing['id']})")
        else:
            # Flip
            db.execute(
                "UPDATE kg_facts SET subject=?, object=? WHERE id=?",
                (o, s, fid),
            )
            flipped += 1
            print(f"  flipped    #{fid}: {s} {p} {o}  →  {o} {p} {s}")

        if (flipped + superseded) % 25 == 0:
            db.commit()

    db.commit()
    print(f"\nDone. flipped={flipped}, superseded={superseded}, scanned={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
