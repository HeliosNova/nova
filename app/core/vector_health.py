"""Vector-index rot detection + failure telemetry (2026-08-25).

hnswlib never compacts delete-tombstones. Chroma's ``delete()`` only
marks elements dead in the HNSW graph, so a churny collection
accumulates tombstones until queries with ``n_results`` at or above the
surviving neighborhood fail with "Cannot return the results in a
contigious 2D array" (hnswlib's own spelling). The lessons collection
hit this on 2026-08-22 — ~335 tombstones against 42 live rows — and the
vector arm of lesson retrieval silently degraded to keyword-only for
three days: 133 warnings, zero alerts, and no code path that could heal
it (the guarded reindex early-returns on count>0; upserts and ghost
prunes cannot clear tombstones; only a drop+rebuild can).

Three defenses live here:

- ``record_failure()`` / ``failures_in_window()``: in-memory telemetry
  that the System Health monitor turns into a loud alert. The silent
  degradation was the real failure, not the rot itself.
- ``assess()`` / ``sweep()``: rot detection for daily maintenance —
  a canary probe (catches the terminal state) plus a churn estimate
  (catches it BEFORE queries start failing). Churn is estimated from
  SQL alone: AUTOINCREMENT max(id) is "rows ever created" and the live
  count is what survives, so ever-live approximates deletes-ever ≈
  tombstones. A per-store watermark recorded at rebuild time stops the
  historical churn from re-triggering a rebuild every day forever.
- watermark persistence (``get_watermark``/``set_watermark``) in a tiny
  ``vector_index_meta`` table.

The stores themselves own their rebuild methods (drop + backfill from
the SQLite source of truth): ``LearningEngine.rebuild_lessons_vectors``
and ``KnowledgeGraph.rebuild_vectors``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Tombstones tolerated before a rebuild, whichever bound is larger:
# a multiple of the live set (dilution ratio) or an absolute floor so a
# tiny collection isn't rebuilt over a handful of deletes.
DEFAULT_RATIO = 2.0
DEFAULT_MIN_EXCESS = 100

# Retry k for the in-request degrade path: the live failure floor was
# k>=8, and probes at k<=5 kept working on the saturated index.
DEGRADE_K = 5

_LOCK = threading.Lock()
_FAILURES: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=512))

def is_tombstone_error(exc: BaseException) -> bool:
    """True when `exc` is hnswlib's tombstone-saturation query failure."""
    msg = str(exc).lower()
    return "2d array" in msg and ("contigious" in msg or "contiguous" in msg)


def record_failure(store: str, when: float | None = None) -> None:
    """Note a vector-arm query failure for `store` (thread-safe)."""
    with _LOCK:
        _FAILURES[store].append(when if when is not None else time.time())


def failures_in_window(hours: float = 24.0) -> dict[str, int]:
    """Per-store count of failures recorded in the last `hours`."""
    cutoff = time.time() - hours * 3600
    with _LOCK:
        return {
            store: sum(1 for t in stamps if t >= cutoff)
            for store, stamps in _FAILURES.items()
        }


def reset() -> None:
    """Clear telemetry (tests)."""
    with _LOCK:
        _FAILURES.clear()


def _ensure_meta_table(db) -> None:
    # Unconditional IF NOT EXISTS: watermark ops run ~once per day from
    # maintenance threads, and a cached "done" flag breaks the moment two
    # SafeDB instances (tests, migrations) pass through this module.
    db.execute(
        "CREATE TABLE IF NOT EXISTS vector_index_meta ("
        "  store TEXT PRIMARY KEY,"
        "  rebuilt_at TEXT NOT NULL,"
        "  ever_at_rebuild INTEGER NOT NULL,"
        "  live_at_rebuild INTEGER NOT NULL"
        ")"
    )


def get_watermark(db, store: str) -> tuple[int, int] | None:
    """(ever, live) recorded at the store's last rebuild, or None."""
    _ensure_meta_table(db)
    row = db.fetchone(
        "SELECT ever_at_rebuild, live_at_rebuild FROM vector_index_meta "
        "WHERE store = ?",
        (store,),
    )
    if row is None:
        return None
    return (row["ever_at_rebuild"], row["live_at_rebuild"])


def set_watermark(db, store: str, *, ever: int, live: int) -> None:
    _ensure_meta_table(db)
    db.execute(
        "INSERT INTO vector_index_meta (store, rebuilt_at, ever_at_rebuild, live_at_rebuild) "
        "VALUES (?, datetime('now'), ?, ?) "
        "ON CONFLICT(store) DO UPDATE SET "
        "  rebuilt_at = excluded.rebuilt_at,"
        "  ever_at_rebuild = excluded.ever_at_rebuild,"
        "  live_at_rebuild = excluded.live_at_rebuild",
        (store, ever, live),
    )


def assess(
    *,
    live: int,
    ever: int,
    canary,
    watermark: tuple[int, int] | None = None,
    ratio: float = DEFAULT_RATIO,
    min_excess: int = DEFAULT_MIN_EXCESS,
) -> dict:
    """Decide whether a collection needs a drop+rebuild.

    `canary` is a zero-arg callable performing a representative query
    (k=10 — the k that dies first); only a tombstone-shaped failure
    counts as rot, any other exception is an availability problem the
    rebuild wouldn't fix.
    """
    try:
        canary()
    except Exception as e:  # noqa: BLE001 — classified below
        if is_tombstone_error(e):
            return {"needs_rebuild": True, "reason": "canary"}
        logger.warning("vector canary errored (not rot): %s", e)

    if watermark is not None:
        wm_ever, wm_live = watermark
        tombstones = max(0, (wm_live + max(0, ever - wm_ever)) - live)
    else:
        tombstones = max(0, ever - live)
    if tombstones > max(ratio * live, min_excess):
        return {"needs_rebuild": True, "reason": "churn", "tombstones": tombstones}
    return {"needs_rebuild": False, "reason": "healthy", "tombstones": tombstones}


def sweep(targets: list[dict]) -> list[str]:
    """Run rot assessment over `targets`, rebuilding where needed.

    Each target: {name, live, ever, canary, watermark, rebuild,
    record_watermark}. Returns human-readable summary lines for the
    maintenance report. Never raises — a failed rebuild is reported and
    the sweep moves on.
    """
    lines: list[str] = []
    for t in targets:
        name = t["name"]
        try:
            verdict = assess(
                live=t["live"], ever=t["ever"], canary=t["canary"],
                watermark=t.get("watermark"),
            )
            if not verdict["needs_rebuild"]:
                lines.append(
                    f"{name}: healthy (~{verdict.get('tombstones', 0)} tombstones)"
                )
                continue
            logger.warning(
                "Vector index %r is rotten (%s) — rebuilding", name, verdict["reason"]
            )
            n = t["rebuild"]()
            t["record_watermark"]()
            lines.append(f"{name}: REBUILT ({verdict['reason']}) — {n} re-embedded")
        except Exception as e:  # noqa: BLE001 — sweep must complete
            logger.error("Vector index sweep FAILED for %r: %s", name, e)
            lines.append(f"{name}: rebuild FAILED — {e}")
    return lines
