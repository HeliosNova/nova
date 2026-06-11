"""Conformal Abstention — calibrated confidence thresholds from reflexion history.

Instead of hard-coded thresholds (0.40/0.50/0.60) for the confidence footer
tiers in brain.py, this module computes adaptive percentile cutoffs from the
last N reflexion scores. As reflexion grading drifts (model gets better or
the critique prompt is retuned), the footer thresholds shift with it — so we
flag the same fraction of "low confidence" answers regardless of absolute
score scale.

This is the practical version of arxiv 2405.01563 (Conformal Abstention) —
we don't have ground truth labels for "was the answer correct?", but the
reflexion score is the best signal we have, and a percentile-based threshold
gives us a statistical guarantee on our own abstention rate.

Cached: thresholds recompute every CONFORMAL_RECOMPUTE_INTERVAL queries
(default 100) so the math doesn't run on every chat.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from app.config import config
from app.database import get_db

logger = logging.getLogger(__name__)


# Tunables (kept here until proven worth surfacing)
_CALIB_RECOMPUTE_INTERVAL_SECONDS = 600.0   # refresh every 10 min
_CALIB_SAMPLE_SIZE = 500                    # last N reflexions for percentile calc
_CALIB_MIN_REFLEXIONS = 30                  # below this, fall back to defaults

# Defaults if no calibration data is available — match the historical hard-coded
# values in brain.py so behavior is identical when the table is empty.
_DEFAULT_TIER1_LOW = 0.40
_DEFAULT_TIER1_HIGH = 0.60
_DEFAULT_TIER2 = 0.50
_DEFAULT_TIER3 = 0.60


@dataclass
class FooterThresholds:
    """Calibrated cutoffs for the three confidence-footer tiers.

    tier1: ungrounded answer — fire footer when low <= score < high
    tier2: tools-used-but-thin — fire footer when score < tier2
    tier3: hedging-language — fire footer when score < tier3
    n_samples: how many reflexions the calibration was based on
    is_default: True if defaults were used (insufficient data)
    """
    tier1_low: float
    tier1_high: float
    tier2: float
    tier3: float
    n_samples: int = 0
    is_default: bool = False


_LOCK = threading.Lock()
_CACHE: FooterThresholds | None = None
_CACHE_BUILT_AT: float = 0.0


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile. p in [0, 1]."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = p * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def _compute_thresholds() -> FooterThresholds:
    """Read recent reflexion scores and compute percentile cutoffs.

    Tier 1 (ungrounded): hits in the [P25, P50] band — moderately low quality
        relative to recent history. Below P25 means very poor; above P50 is
        near average — neither is the "uncertain but trying" zone the footer
        is designed for.
    Tier 2 (tools-used-but-thin): below P25 — the answer used tools but the
        outcome scored worse than three-quarters of recent answers.
    Tier 3 (hedging-language): below P50 — any below-median quality where
        the model itself signaled uncertainty in prose.

    These percentile choices roughly preserve the original hard-coded
    semantics (40-60 = lower-half band, 50 = lower quartile of well-grounded,
    60 = below median) while adapting to score drift.
    """
    try:
        db = get_db()
        rows = db.fetchall(
            "SELECT quality_score FROM reflexions "
            "WHERE quality_score IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (_CALIB_SAMPLE_SIZE,),
        )
    except Exception as e:
        logger.warning("[calibration] reflexion read failed: %s — using defaults", e)
        return FooterThresholds(
            _DEFAULT_TIER1_LOW, _DEFAULT_TIER1_HIGH,
            _DEFAULT_TIER2, _DEFAULT_TIER3,
            n_samples=0, is_default=True,
        )

    scores = [row["quality_score"] for row in rows if row["quality_score"] is not None]
    if len(scores) < _CALIB_MIN_REFLEXIONS:
        return FooterThresholds(
            _DEFAULT_TIER1_LOW, _DEFAULT_TIER1_HIGH,
            _DEFAULT_TIER2, _DEFAULT_TIER3,
            n_samples=len(scores), is_default=True,
        )

    scores.sort()
    # Variance check: if the distribution is bunched up (e.g. table has only
    # success-pattern reflexions because failures got pruned), percentile
    # cutoffs collapse onto the same value and the footer would either fire
    # for everything or nothing. Fall back to defaults when std is degenerate.
    mean_s = sum(scores) / len(scores)
    var_s = sum((s - mean_s) ** 2 for s in scores) / len(scores)
    std_s = var_s ** 0.5
    if std_s < 0.05:
        logger.info(
            "[calibration] reflexion distribution too tight (std=%.3f, n=%d) — "
            "using historical defaults rather than degenerate percentiles",
            std_s, len(scores),
        )
        return FooterThresholds(
            _DEFAULT_TIER1_LOW, _DEFAULT_TIER1_HIGH,
            _DEFAULT_TIER2, _DEFAULT_TIER3,
            n_samples=len(scores), is_default=True,
        )

    p25 = _percentile(scores, 0.25)
    p50 = _percentile(scores, 0.50)

    # Construct tier thresholds. Clamp to [0, 1] and enforce ordering — pathological
    # data (all identical scores) shouldn't produce inverted ranges.
    tier1_low = max(0.0, min(p25, 1.0))
    tier1_high = max(tier1_low + 0.01, min(p50, 1.0))
    tier2 = max(0.0, min(p25, 1.0))
    tier3 = max(0.0, min(p50, 1.0))

    logger.info(
        "[calibration] thresholds: tier1=[%.2f, %.2f] tier2=%.2f tier3=%.2f (from %d reflexions)",
        tier1_low, tier1_high, tier2, tier3, len(scores),
    )
    return FooterThresholds(
        tier1_low=tier1_low,
        tier1_high=tier1_high,
        tier2=tier2,
        tier3=tier3,
        n_samples=len(scores),
        is_default=False,
    )


def get_thresholds() -> FooterThresholds:
    """Return cached thresholds, recomputing if expired or absent.

    Falls back to defaults if reflexion data is insufficient or unavailable.
    Safe to call from a hot path — work is amortized over the cache TTL.
    """
    global _CACHE, _CACHE_BUILT_AT

    if not getattr(config, "ENABLE_CONFORMAL_ABSTENTION", False):
        return FooterThresholds(
            _DEFAULT_TIER1_LOW, _DEFAULT_TIER1_HIGH,
            _DEFAULT_TIER2, _DEFAULT_TIER3,
            n_samples=0, is_default=True,
        )

    with _LOCK:
        now = time.monotonic()
        if _CACHE is None or now - _CACHE_BUILT_AT > _CALIB_RECOMPUTE_INTERVAL_SECONDS:
            _CACHE = _compute_thresholds()
            _CACHE_BUILT_AT = now
        return _CACHE


def invalidate() -> None:
    """Force recompute on next call. Use after a fine-tune or critique-prompt change."""
    global _CACHE, _CACHE_BUILT_AT
    with _LOCK:
        _CACHE = None
        _CACHE_BUILT_AT = 0.0
