r"""Is the new confidence estimator better? Ask the forecasts that already resolved.

Written 2026-09-04. Nova mints ~20-35 forecasts a day against horizons of weeks
to months, so "wait for the new regime to resolve" means waiting a month to
learn anything - and a learned recalibrator wants thousands of outcomes, which
at this rate is years away. Neither is a reason not to know today.

105 forecasts have already resolved hit/miss under the OLD estimator: a single
verbalized probability, taken from one generation. The new one (2026-09-04) is
the mean of that stated number and k independently sampled estimates. The claims
and the outcomes are fixed, so the estimator is the only thing that changes -
which makes this a paired comparison on frozen evidence, the same shape as the
priming A/B.

Two honest caveats, both of which handicap the NEW arm:

  * The original stated confidence was written with the full digest and its
    sources in view. That context expired with the digest, so the re-estimate
    sees the claim text alone. If it wins anyway, it wins from behind.
  * The outcomes are Nova's own self-grading, and the grader changed on
    2026-09-02. That error is real, but it falls on both arms equally, because
    both are scored against the same labels.

Brier score, lower is better; 0.25 is what you get by always saying 0.5.

    MSYS_NO_PATHCONV=1 docker run --rm -v "F:\Helios Project\nova_:/app" \
        -v nova__nova_data:/data --network nova__default -w /app \
        nova-app:latest python scripts/forecast_backtest.py [limit]
"""
from __future__ import annotations

import asyncio
import sqlite3
import statistics
import sys


def _brier(pairs: list[tuple[float, int]]) -> float:
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def _calibration(pairs: list[tuple[float, int]], bins=(0.0, 0.5, 0.65, 0.8, 1.01)) -> list[tuple]:
    """Stated probability vs how often it actually happened, per band."""
    out = []
    for lo, hi in zip(bins, bins[1:]):
        band = [(p, o) for p, o in pairs if lo <= p < hi]
        if band:
            out.append((lo, hi, len(band),
                        statistics.mean(p for p, _ in band),
                        statistics.mean(o for _, o in band)))
    return out


async def main(limit: int | None = None) -> None:
    from app.core.forecasts import _CONF_SAMPLES, _ensemble_confidence

    conn = sqlite3.connect("file:/data/nova.db?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT id, claim, confidence, status FROM forecasts "
        "WHERE status IN ('hit', 'miss') AND confidence IS NOT NULL "
        "ORDER BY resolved_at DESC").fetchall()
    if limit:
        rows = rows[:limit]
    print(f"resolved forecasts scored: {len(rows)}", flush=True)

    old: list[tuple[float, int]] = []
    new: list[tuple[float, int]] = []
    spreads: list[float] = []
    failed = 0

    for i, (fid, claim, stated, status) in enumerate(rows, 1):
        outcome = 1 if status == "hit" else 0
        stated = float(stated)
        mean, spread = await _ensemble_confidence(claim, claim, k=_CONF_SAMPLES)
        if mean is None:
            failed += 1
            continue
        conf = (stated + mean * _CONF_SAMPLES) / (1 + _CONF_SAMPLES)
        old.append((stated, outcome))
        new.append((conf, outcome))
        spreads.append(spread or 0.0)
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}  old {_brier(old):.4f}  new {_brier(new):.4f}",
                  flush=True)

    if not new:
        print("no forecast could be re-estimated — is Ollama reachable?")
        return

    print(f"\n=== {len(new)} paired forecasts, {failed} unscorable ===")
    print(f"{'arm':<26}{'Brier':>9}{'mean conf':>11}{'hit rate':>10}{'gap':>8}")
    for name, pairs in (("old  stated only", old), ("new  ensembled", new)):
        mc = statistics.mean(p for p, _ in pairs)
        hr = statistics.mean(o for _, o in pairs)
        print(f"{name:<26}{_brier(pairs):>9.4f}{mc:>11.3f}{hr:>10.3f}{mc - hr:>+8.3f}")
    print("\ngap = mean confidence minus how often it happened; positive is overconfident.")

    delta = _brier(old) - _brier(new)
    wins = sum(1 for (po, o), (pn, _) in zip(old, new)
               if (pn - o) ** 2 < (po - o) ** 2)
    print(f"\nBrier improvement {delta:+.4f}  ({delta / _brier(old) * 100:+.1f}%), "
          f"new closer on {wins}/{len(new)}")
    if spreads:
        print(f"sample spread: mean {statistics.mean(spreads):.3f}, "
              f"max {max(spreads):.3f}")
        wide = [(abs(p - o), s) for (p, o), s in zip(new, spreads) if s > 0.2]
        if len(wide) >= 5:
            narrow = [(abs(p - o), s) for (p, o), s in zip(new, spreads) if s <= 0.2]
            print(f"  disagreement predicts error? wide-spread mean |p-o| "
                  f"{statistics.mean(e for e, _ in wide):.3f} "
                  f"vs narrow {statistics.mean(e for e, _ in narrow):.3f} "
                  f"({len(wide)} wide / {len(narrow)} narrow)")

    print("\ncalibration (old -> new), band: n, mean stated, actual hit rate")
    for arm, pairs in (("old", old), ("new", new)):
        for lo, hi, n, mp, mo in _calibration(pairs):
            print(f"  {arm}  {lo:.2f}-{hi:.2f}  n={n:<4} stated {mp:.2f}  actual {mo:.2f}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
