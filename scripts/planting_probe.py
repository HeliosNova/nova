"""Planting probe for the entailment gate (plan 2026-10-09, task P1).

Question: do the clause/alt rescues admit claims because the evidence entails
them, or because the words overlap? Method: take N claims the rescue accepted
(site clause/alt, supported) and N the primary gate accepted first pass (site
gate, supported); PLANT each claim into a passage from a NON-cited article
(the doc of another record whose hosts do not overlap) and score the planted
pair with the same sidecar. A verdict that MIGRATES (planted pair supported)
means the model keyed on the claim's own words, not the evidence.

Pre-registered rule P1: rescued minus first-pass migration > 5 pp disables the
clause/alt rescue (`_CLAUSE_TOTAL = 0`, `_ALT_TOTAL = 0`) pending redesign.

Usage (inside a quiet window — the sidecar is CPU-only):
    python scripts/planting_probe.py --n 200 [--url http://minicheck:8001]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESCUE_SITES = ("clause", "alt")


def sample(records, n: int, *, seed: int = 7) -> tuple[list[dict], list[dict]]:
    """N rescued (clause/alt, supported) and N first-pass (gate, supported)
    records with a usable claim and document. Deterministic for a seed."""
    rescued = [r for r in records if r.get("site") in RESCUE_SITES and r.get("supported")
               and r.get("claim") and r.get("doc")]
    first = [r for r in records if r.get("site") == "gate" and r.get("supported")
             and r.get("claim") and r.get("doc")]
    rng = random.Random(seed)
    rng.shuffle(rescued)
    rng.shuffle(first)
    return rescued[:n], first[:n]


def plant(claim: str, uncited_doc: str) -> str:
    """Insert the claim sentence into the middle of an unrelated passage."""
    claim = claim.strip()
    if not claim.endswith((".", "!", "?")):
        claim += "."
    doc = (uncited_doc or "").strip()
    if not doc:
        return claim
    cut = doc.find(". ", len(doc) // 2)
    if cut == -1:
        return f"{doc} {claim}"
    return f"{doc[:cut + 1]} {claim}{doc[cut + 1:]}"


def uncited_doc_for(rec: dict, pool: list[dict], *, seed: int = 7) -> str | None:
    """A document from another record whose hosts do not overlap this one's."""
    hosts = set(rec.get("hosts") or [])
    rng = random.Random(f"{seed}:{rec.get('claim', '')[:40]}")
    candidates = [p for p in pool if p.get("doc") and not (set(p.get("hosts") or []) & hosts)
                  and p.get("claim") != rec.get("claim")]
    if not candidates:
        return None
    return rng.choice(candidates)["doc"]


def migration_rate(pairs: list[tuple[str, str]], scorer) -> float | None:
    """Share of planted pairs the scorer calls supported. `scorer(pairs) -> list[dict]`."""
    if not pairs:
        return None
    verdicts = scorer([{"doc": d, "claim": c} for c, d in pairs])
    return sum(1 for v in verdicts if v.get("supported")) / len(pairs)


def run(records, n: int, scorer, *, seed: int = 7) -> dict:
    rescued, first = sample(records, n, seed=seed)
    pool = [r for r in records if r.get("doc")]
    out = {"n_rescued": len(rescued), "n_first_pass": len(first)}
    for key, group in (("rescued", rescued), ("first_pass", first)):
        pairs = []
        for rec in group:
            doc = uncited_doc_for(rec, pool, seed=seed)
            if doc is None:
                continue
            pairs.append((rec["claim"], plant(rec["claim"], doc)))
        out[f"{key}_planted"] = len(pairs)
        out[f"{key}_migration"] = migration_rate(pairs, scorer)
    r, f = out.get("rescued_migration"), out.get("first_pass_migration")
    out["delta_pp"] = None if r is None or f is None else round((r - f) * 100, 1)
    out["rule_p1_disable_rescue"] = bool(out["delta_pp"] is not None and out["delta_pp"] > 5)
    return out


def _sidecar_scorer(url: str):
    import httpx

    def _score(pairs: list[dict]) -> list[dict]:
        results: list[dict] = []
        with httpx.Client(timeout=600) as client:
            for i in range(0, len(pairs), 8):
                r = client.post(f"{url.rstrip('/')}/check_batch", json={"pairs": pairs[i:i + 8]})
                r.raise_for_status()
                results.extend(r.json()["results"])
        return results
    return _score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--url", default="http://minicheck:8001")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args(argv)
    from app.monitors.gate_record import read
    records = list(read(days=args.days))
    report = run(records, args.n, _sidecar_scorer(args.url))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
