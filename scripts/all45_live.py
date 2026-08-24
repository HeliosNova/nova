"""Run ALL domain/news monitors end-to-end with the REAL pipeline, everything live.

Real dependency injection — NOT a monkeypatch: builds the genuine `Services`
dataclass (the same class main.py assembles) with a real `KnowledgeGraph(get_db())`
and registers it via the official `set_services()`. `run_domain_study` reads
`services.kg` exactly as in production, so every monitor takes the real path —
deep-research overview / native list, best-of-N synthesis, numeric grounding +
corroboration, primary-source preference, and live KG fact-banking. It returns
the digest string and never touches `_send_alert`, so no channel broadcast (this
mirrors the prior all_digests.md capture runs).

Selects the 44 enabled `query`-type domain/intelligence monitors (operational /
self-improvement monitors excluded by name). Paced at concurrency 3 so the search
channels aren't CAPTCHA-throttled; the curated feeds are search-independent and
carry every domain regardless.

Usage (inside nova-app):  python /data/all45_live.py [concurrency]
"""
import asyncio
import logging
import re
import time
from datetime import datetime, timezone

logging.disable(logging.CRITICAL)

from app.core.brain import Services, set_services
from app.core.kg import KnowledgeGraph
from app.database import get_db
from app.monitors.domain_study_runner import run_domain_study

OUT = "/data/all_digests_live.md"
_SKIP = {
    "morning check-in", "system health", "system maintenance", "fine-tune check",
    "auto-monitor detector", "lesson quiz", "skill validation", "curiosity research",
    "quality eval harness", "prompt optimizer", "storyline tracker",
    "forecast resolution", "cross-monitor synthesis", "self-reflection",
}

_SRC = re.compile(r"read\s+(\d+)\s+sources", re.I)
# Native-list footer is "**<Label>**  N items from host, host." — the count is
# OUTSIDE the bold markers, so match "N items" plainly (not **N items**).
_ITEMS = re.compile(r"(\d+)\s+items\s+from", re.I)
_FB = re.compile(r"no readable credible sources|synthesis unavailable|background[- ]context", re.I)


def _kg_count(db):
    try:
        r = db.fetchone("SELECT COUNT(*) AS c FROM kg_facts WHERE superseded_at IS NULL")
        return r["c"] if r else -1
    except Exception:
        return -1


def _metrics(value):
    if not value:
        return 0, True
    m = _SRC.search(value) or _ITEMS.search(value)
    n = int(m.group(1)) if m else 0
    return n, bool(_FB.search(value)) or (n <= 1 and not _ITEMS.search(value))


def _targets(db):
    rows = db.fetchall(
        "SELECT id, name FROM monitors WHERE enabled=1 AND check_type='query' ORDER BY id")
    out = []
    for r in rows:
        if (r["name"] or "").strip().lower() in _SKIP:
            continue
        out.append(r["name"])
    return out


async def _run_one(name, sem, results, total):
    async with sem:
        t0 = time.monotonic()
        try:
            digest = await asyncio.wait_for(run_domain_study(name), 1500)
        except Exception as e:
            digest = f"(ERROR: {type(e).__name__}: {e})"
        dt = time.monotonic() - t0
        is_err = digest.startswith("(ERROR")
        n, fb = _metrics("" if is_err else digest)
        results[name] = {"digest": digest, "n": n, "fb": fb or is_err, "secs": dt, "err": is_err}
        flag = "ERROR" if is_err else ("FALLBACK" if fb else "ok")
        print(f"  [{len(results):>2}/{total}] {name[:40]:<40} {n:>3} src  {dt:>5.0f}s  {flag}",
              flush=True)


async def main():
    import sys
    conc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    db = get_db()
    set_services(Services(kg=KnowledgeGraph(db)))

    targets = _targets(db)
    kg_before = _kg_count(db)
    print(f"=== ALL-{len(targets)} LIVE RUN (real Services.kg, conc={conc}) "
          f"{datetime.now(timezone.utc).isoformat()} ===", flush=True)
    print(f"KG facts before: {kg_before}", flush=True)

    sem = asyncio.Semaphore(conc)
    results = {}
    t_start = time.monotonic()
    await asyncio.gather(*[_run_one(n, sem, results, len(targets)) for n in targets])

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"# All-{len(targets)} live digests — {today}\n\n")
        for name in targets:
            r = results.get(name, {})
            f.write(f"\n\n{'='*92}\n## {name}  —  {r.get('n',0)} sources · "
                    f"{r.get('secs',0):.0f}s{' · FALLBACK' if r.get('fb') else ''}"
                    f"{' · ERROR' if r.get('err') else ''}\n{'='*92}\n\n")
            f.write(r.get("digest") or "(no output)")

    kg_after = _kg_count(db)
    total_src = sum(r["n"] for r in results.values())
    fb = [n for n, r in results.items() if r["fb"]]
    rich = sum(1 for r in results.values() if r["n"] >= 5 and not r["err"])
    print("\n=== SUMMARY ===", flush=True)
    print(f"ran                : {len(results)}/{len(targets)}", flush=True)
    print(f">=5 sources        : {rich}", flush=True)
    print(f"fallbacks/errors   : {len(fb)}  {fb}", flush=True)
    print(f"total sources read : {total_src}  (avg {total_src/max(len(results),1):.1f}/domain)", flush=True)
    print(f"KG facts: {kg_before} -> {kg_after}  (+{kg_after-kg_before} banked)", flush=True)
    print(f"wall time          : {(time.monotonic()-t_start)/60:.1f} min", flush=True)
    print(f"digests written    : {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
