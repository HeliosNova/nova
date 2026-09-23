r"""Does evidence at mint time, or an explicit outside view, move the forecast?

Written 2026-09-23, after the estimator bake-off on the 144 resolved claims of
the dumped record showed the lever is context, not the model: the number
STATED at mint (written with the digest in view) scored Brier 0.2287, level
with the base rate 0.2267, while every blind re-estimate of the claim text
alone scored 0.31-0.33. This measures two ways of giving the minter more of
that context, leak-free, on the same frozen claims and outcomes:

  claim               the claim text alone (the bake-off arm, re-run for pairing)
  evidence            + web evidence published ON OR BEFORE the claim's own
                        created_at, undated results dropped (HINDCAST rule)
  baserate            + one reference-class call (class + base rate) prepended
  evidence+baserate   both

All arms run on the same model with k samples each. Pre-registered rules:
  * adopt evidence at mint if, on the claims where any prior-dated evidence
    was found, the evidence arm is closer than claim-only on >=60% and its
    Brier is lower by >=0.02;
  * adopt the reference-class step if it is closer than claim-only on >=60%
    of all claims and its Brier is lower by >=0.01;
  * the stated-as-minted Brier is the ceiling to compare against.

    MSYS_NO_PATHCONV=1 docker run --rm -v "F:\Helios Project\nova_:/app" \
        -v nova__nova_data:/data --network nova__default -w /app \
        -e PYTHONPATH=/app -e LLM_MODEL=nova-ft nova-app:latest \
        python scripts/forecast_context_backtest.py \
            --dump /data/backups/forecasts_reset_2026-09-22.json [--limit N]

Same model as the digest lane, so no residency swap; it shares the card and
SearXNG with the running digests. Rows append to <out>.rows.jsonl as they
finish. Caveat shared by every arm: the outcomes are Nova's own self-grading
under the pre-reset regimes; search engines surface what is findable TODAY,
so the prior-evidence arm is bounded by how much pre-claim reporting survives
in the index — the row records how many items it found.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from scripts.forecast_bakeoff import _bands, _brier, _load

ARMS = ["claim", "evidence", "baserate", "evidence+baserate"]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None)
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="/data/ceiling/forecast_context_2026-09-23")
    args = ap.parse_args()
    from app.core.forecasts import (_ensemble_confidence, gather_prior_evidence,
                                    outside_view_block, reference_class)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows = _load(args.dump, args.limit)
    if not rows:
        print("no resolved forecasts to replay")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out.with_suffix(".rows.jsonl")
    print(f"{len(rows)} resolved forecasts; model {args.model}; arms {arms}; rows -> {rows_path}", flush=True)

    est: dict[str, list[tuple[float, int, bool]]] = {a: [] for a in arms}   # (p, outcome, had_evidence)
    stated: list[tuple[float, int, bool]] = []
    found = 0
    t0 = time.monotonic()
    for i, r in enumerate(rows, 1):
        outcome = 1 if r["status"] == "hit" else 0
        claim = r["claim"]
        evidence = ""
        if any(a.startswith("evidence") for a in arms):
            evidence = await gather_prior_evidence(claim, as_of=r.get("created_at"), strict_dates=True)
        n_ev = evidence.count("\n- ") + (1 if evidence.startswith("- ") else 0)
        had = n_ev > 0
        found += had
        ref = None
        if any(a.endswith("baserate") for a in arms):
            ref = await reference_class(claim, r.get("resolves_at"), model=args.model)
        block = outside_view_block(ref)
        stated.append((float(r.get("confidence") or 0.5), outcome, had))
        for a in arms:
            ctx = {
                "claim": claim,
                "evidence": (f"EVIDENCE (published before the claim):\n{evidence}" if had else claim),
                "baserate": block + claim,
                "evidence+baserate": block + (f"EVIDENCE (published before the claim):\n{evidence}" if had else claim),
            }[a]
            mean, spread = await _ensemble_confidence(claim, ctx, k=args.k, model=args.model)
            if mean is None:
                print(f"  [{a}] #{r.get('id')} no estimate — skipped", file=sys.stderr, flush=True)
                continue
            est[a].append((mean, outcome, had))
            with rows_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"arm": a, "id": r.get("id"), "claim": claim[:200], "outcome": outcome,
                                     "stated": r.get("confidence"), "estimate": round(mean, 4),
                                     "spread": spread, "n_evidence": n_ev,
                                     "reference_class": ref[0] if ref else None,
                                     "base_rate": ref[1] if ref else None}) + "\n")
        if i % 10 == 0:
            print(f"  {i}/{len(rows)} ({time.monotonic() - t0:.0f}s, prior evidence found for {found})", flush=True)

    print("\n=== results (Brier, lower is better) ===")
    hit_rate = statistics.mean(o for _, o, _ in stated)
    base = hit_rate * (1 - hit_rate)

    def _skill(b: float) -> str:
        return f"skill {1 - b / base:+.3f}" if base > 0 else "skill n/a"

    def _pairs(xs, subset=None):
        return [(p, o) for p, o, h in xs if subset is None or h == subset]

    print(f"n={len(stated)} hit rate {hit_rate:.3f} base-rate Brier {base:.4f}; prior evidence found for {found}/{len(stated)}")
    print(f"{'stated (as minted)':<22} Brier {_brier(_pairs(stated)):.4f}  {_skill(_brier(_pairs(stated)))}")
    for a in arms:
        pairs = _pairs(est[a])
        if not pairs:
            print(f"{a:<22} no estimates")
            continue
        b = _brier(pairs)
        print(f"{a:<22} Brier {b:.4f}  {_skill(b)}  n={len(pairs)}  mean conf {statistics.mean(p for p, _ in pairs):.3f}")
        for lo, hi, n, mp, mo in _bands(pairs):
            print(f"    [{lo:.2f},{hi:.2f}) n={n:<3} said {mp:.2f} happened {mo:.2f}")

    if "claim" in arms:
        ref_arm = est["claim"]
        for a in arms:
            if a == "claim" or not est[a]:
                continue
            by_id = min(len(ref_arm), len(est[a]))
            wins = sum(1 for (pa, o, _), (pb, _, _) in zip(ref_arm[:by_id], est[a][:by_id]) if (pb - o) ** 2 < (pa - o) ** 2)
            print(f"paired vs claim: {a} closer on {wins}/{by_id}")
        if found:
            print("\n--- on the claims WITH prior-dated evidence ---")
            print(f"{'stated (as minted)':<22} Brier {_brier(_pairs(stated, True)):.4f}  n={len(_pairs(stated, True))}")
            for a in arms:
                sub = _pairs(est[a], True)
                if sub:
                    print(f"{a:<22} Brier {_brier(sub):.4f}  n={len(sub)}")
            ca = [(p, o) for p, o, h in est["claim"] if h]
            for a in arms:
                if a == "claim":
                    continue
                ea = [(p, o) for p, o, h in est[a] if h]
                n = min(len(ca), len(ea))
                wins = sum(1 for (pa, o), (pb, _) in zip(ca[:n], ea[:n]) if (pb - o) ** 2 < (pa - o) ** 2)
                print(f"paired vs claim (evidence subset): {a} closer on {wins}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
