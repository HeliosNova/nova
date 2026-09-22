r"""Which estimator should mint Nova's forecasts? Ask the record that resolved.

Written 2026-09-22, the day the forecast record was reset to zero. The old
record (969 rows, 144 resolved hit/miss) was dumped to
/data/backups/forecasts_reset_2026-09-22.json before the reset; its claims and
outcomes are fixed, so an estimator is the only free variable — a paired
comparison on frozen data, the same shape as scripts/forecast_backtest.py.

Arms are Ollama model tags. The default pair is the resident 27B against
OpenForecaster-8B (Qwen3-8B trained with GRPO on accuracy + Brier over ~52k
news-derived questions; matched 100B+ models on Brier in its paper):

    MSYS_NO_PATHCONV=1 docker run --rm -v "F:\Helios Project\nova_:/app" \
        -v nova__nova_data:/data --network nova__default -w /app \
        -e PYTHONPATH=/app -e LLM_MODEL=nova-ft nova-app:latest \
        python scripts/forecast_bakeoff.py --dump /data/backups/forecasts_reset_2026-09-22.json \
            --models qwen3.8:27b,hf.co/mradermacher/OpenForecaster-8B-GGUF:Q8_0 [--limit N]

Honest caveats, all of which handicap every arm equally:
  * the re-estimate sees the claim text alone — the digest that produced the
    stated number expired with its monitor result;
  * the outcomes are Nova's own self-grading under the pre-reset regimes;
  * OpenForecaster reasons before it answers; the JSON-mode call here asks
    for the number directly, so it runs below its paper setting.
Each arm is a model load, so RUN THIS IN A QUIET WINDOW (POST
/api/monitors/quiet {"hours": 2}); it competes with the digest chain for the
card. Rows are appended to <out>.rows.jsonl as they finish, so a partial run
is still analysable. Brier: lower is better; the base-rate Brier is
hit_rate * (1 - hit_rate); skill = 1 - brier / base_brier.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path


def _brier(pairs: list[tuple[float, int]]) -> float:
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def _bands(pairs: list[tuple[float, int]], bins=(0.0, 0.5, 0.65, 0.8, 1.01)):
    out = []
    for lo, hi in zip(bins, bins[1:]):
        band = [(p, o) for p, o in pairs if lo <= p < hi]
        if band:
            out.append((lo, hi, len(band), statistics.mean(p for p, _ in band),
                        statistics.mean(o for _, o in band)))
    return out


def _load(dump: str | None, limit: int | None) -> list[dict]:
    if dump:
        data = json.load(open(dump, encoding="utf-8"))
        rows = data["rows"] if isinstance(data, dict) else data
    else:
        import sqlite3
        conn = sqlite3.connect("file:/data/nova.db?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM forecasts")]
    rows = [r for r in rows if r.get("status") in ("hit", "miss") and r.get("claim")]
    rows.sort(key=lambda r: r.get("id") or 0)
    return rows[:limit] if limit else rows


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None)
    ap.add_argument("--models", default="qwen3.8:27b,hf.co/mradermacher/OpenForecaster-8B-GGUF:Q8_0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="/data/ceiling/forecast_bakeoff_2026-09-22")
    args = ap.parse_args()
    from app.core.forecasts import _ensemble_confidence

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = _load(args.dump, args.limit)
    if not rows:
        print("no resolved forecasts to replay")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out.with_suffix(".rows.jsonl")
    print(f"{len(rows)} resolved forecasts; arms: {models}; rows -> {rows_path}")

    est: dict[str, list[tuple[float, int]]] = {m: [] for m in models}
    stated: list[tuple[float, int]] = []
    t0 = time.monotonic()
    # One model at a time across ALL claims, so the card swaps once per arm
    # instead of once per claim.
    for m in models:
        for i, r in enumerate(rows, 1):
            outcome = 1 if r["status"] == "hit" else 0
            mean, spread = await _ensemble_confidence(r["claim"], r["claim"], k=args.k, model=m)
            if mean is None:
                print(f"  [{m}] #{r.get('id')} no estimate — skipped", file=sys.stderr)
                continue
            est[m].append((mean, outcome))
            if m == models[0]:
                stated.append((float(r.get("confidence") or 0.5), outcome))
            with rows_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"model": m, "id": r.get("id"), "claim": r["claim"][:200],
                                     "outcome": outcome, "stated": r.get("confidence"),
                                     "estimate": round(mean, 4), "spread": spread,
                                     "regime": r.get("regime")}) + "\n")
            if i % 10 == 0:
                print(f"  [{m}] {i}/{len(rows)} ({time.monotonic() - t0:.0f}s)")

    print("\n=== results (Brier, lower is better) ===")
    if stated:
        hit_rate = statistics.mean(o for _, o in stated)
        base = hit_rate * (1 - hit_rate)

        def _skill(b: float) -> str:
            # all outcomes equal → the base rate is a perfect predictor; skill undefined
            return f"skill {1 - b / base:+.3f}" if base > 0 else "skill n/a (one outcome)"

        print(f"n={len(stated)} hit rate {hit_rate:.3f} base-rate Brier {base:.4f}")
        print(f"stated (as minted)         Brier {_brier(stated):.4f}  {_skill(_brier(stated))}")
        for m in models:
            pairs = est[m]
            if not pairs:
                print(f"{m:<28} no estimates")
                continue
            b = _brier(pairs)
            print(f"{m:<28} Brier {b:.4f}  {_skill(b)}  n={len(pairs)}  "
                  f"mean conf {statistics.mean(p for p, _ in pairs):.3f}")
            for lo, hi, n, mp, mo in _bands(pairs):
                print(f"    [{lo:.2f},{hi:.2f}) n={n:<3} said {mp:.2f} happened {mo:.2f}")
        if len(models) == 2 and est[models[0]] and est[models[1]]:
            a, b_ = est[models[0]], est[models[1]]
            n = min(len(a), len(b_))
            wins = sum(1 for (pa, o), (pb, _) in zip(a[:n], b_[:n]) if (pb - o) ** 2 < (pa - o) ** 2)
            print(f"paired: {models[1]} closer on {wins}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
