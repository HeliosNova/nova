r"""Fit the one calibration parameter 105 outcomes can actually support.

A learned Beta-Bernoulli calibrator wants thousands of resolved forecasts; Nova
has 105 and mints ~25 a day against horizons of weeks, so that is years away.
That is a reason not to fit a CALIBRATOR, not a reason to stay uncalibrated.

The legacy record has one dominant, simple defect: overconfidence. Stated
confidence averages ~0.75 and the claims come true ~60% of the time. One
parameter fixes that shape -- shrink every stated probability toward the base
rate:

    p_adjusted = base + k * (p_stated - base)

k=1 leaves it alone, k=0 collapses everything to the base rate. One parameter
against 105 outcomes is a fit those outcomes can support, where 20 would not be.

Scored leave-one-out, because fitting and evaluating on the same points is how
you talk yourself into a number that does not survive contact: each forecast is
scored with a k fitted on the OTHER 104.

Deterministic and offline -- no model calls, no GPU, runs in a second:

    MSYS_NO_PATHCONV=1 docker run --rm -v "F:\Helios Project\nova_:/app" \
        -v nova__nova_data:/data -w /app -e PYTHONPATH=/app \
        nova-app:latest python scripts/forecast_shrinkage.py
"""
from __future__ import annotations

import sqlite3
import statistics
import sys

GRID = [i / 100 for i in range(0, 141)]          # k from 0.00 to 1.40


def brier(pairs: list[tuple[float, int]]) -> float:
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def shrink(p: float, base: float, k: float) -> float:
    return min(1.0, max(0.0, base + k * (p - base)))


def best_k(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    base = statistics.mean(o for _p, o in pairs)
    scored = [(brier([(shrink(p, base, k), o) for p, o in pairs]), k) for k in GRID]
    b, k = min(scored)
    return k, base


def main(db_path: str = "file:/data/nova.db?mode=ro") -> None:
    conn = sqlite3.connect(db_path, uri=True)
    rows = conn.execute(
        "SELECT confidence, status, source_monitor FROM forecasts "
        "WHERE status IN ('hit','miss') AND confidence IS NOT NULL").fetchall()
    pairs = [(float(c), 1 if s == "hit" else 0) for c, s, _m in rows]
    if len(pairs) < 20:
        print(f"only {len(pairs)} resolved forecasts — not enough to fit anything")
        return

    base = statistics.mean(o for _p, o in pairs)
    mean_p = statistics.mean(p for p, _o in pairs)
    print(f"resolved forecasts: {len(pairs)}")
    print(f"mean stated confidence {mean_p:.3f}   actual hit rate {base:.3f}   "
          f"overconfident by {mean_p - base:+.3f}")
    print(f"\nBrier as-is              {brier(pairs):.4f}")
    print(f"Brier if always {base:.2f}     "
          f"{brier([(base, o) for _p, o in pairs]):.4f}   (the do-nothing baseline)")

    k_all, _ = best_k(pairs)
    fitted = [(shrink(p, base, k_all), o) for p, o in pairs]
    print(f"Brier shrunk k={k_all:.2f}      {brier(fitted):.4f}   (fitted AND scored on "
          f"all {len(pairs)} — optimistic)")

    # Leave-one-out: the honest number.
    loo: list[tuple[float, int]] = []
    ks: list[float] = []
    for i in range(len(pairs)):
        rest = pairs[:i] + pairs[i + 1:]
        k, b = best_k(rest)
        ks.append(k)
        loo.append((shrink(pairs[i][0], b, k), pairs[i][1]))
    print(f"Brier shrunk, leave-one-out  {brier(loo):.4f}   "
          f"(k ranged {min(ks):.2f}-{max(ks):.2f}, median {statistics.median(ks):.2f})")

    delta = brier(pairs) - brier(loo)
    print(f"\nhonest improvement {delta:+.4f} ({delta / brier(pairs) * 100:+.1f}%)")
    if brier(loo) >= brier([(base, o) for _p, o in pairs]):
        print("NOTE: shrinking does not beat simply always predicting the base rate. "
              "That means the stated confidences carry no usable signal yet, and the "
              "honest move is to say so rather than dress them up.")

    print("\ncalibration bands, before -> after (leave-one-out)")
    print(f"{'band':<12}{'n':>5}{'stated':>9}{'shrunk':>9}{'actual':>9}")
    for lo, hi in ((0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        idx = [i for i, (p, _o) in enumerate(pairs) if lo <= p < hi]
        if not idx:
            continue
        print(f"{lo:.2f}-{hi:.2f}  {len(idx):>5}"
              f"{statistics.mean(pairs[i][0] for i in idx):>9.2f}"
              f"{statistics.mean(loo[i][0] for i in idx):>9.2f}"
              f"{statistics.mean(pairs[i][1] for i in idx):>9.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "file:/data/nova.db?mode=ro")
