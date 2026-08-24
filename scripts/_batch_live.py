"""Run a specific LIST of monitors live and capture digests — targeted verification
that cumulative fixes lifted the previously-worst monitors. Usage:
  python /data/_batch_live.py "Finance" "Europe and EU" ... [conc]
"""
import asyncio
import logging
import sys
import time

logging.disable(logging.CRITICAL)

from app.core.brain import Services, set_services
from app.core.kg import KnowledgeGraph
from app.database import get_db
from app.monitors.domain_study_runner import run_domain_study

OUT = "/data/_batch_digests.md"


async def _run(name, sem, results):
    async with sem:
        t = time.monotonic()
        try:
            d = await asyncio.wait_for(run_domain_study(name), 1600)
        except Exception as e:
            d = f"(ERROR {type(e).__name__}: {e})"
        secs = round(time.monotonic() - t)
        results[name] = (d, secs)
        print(f"  done: {name[:32]:<32} {secs:>4}s  {len(d):>5} chars", flush=True)


async def main():
    args = sys.argv[1:]
    conc = int(args[-1]) if args and args[-1].isdigit() else 1
    names = args[:-1] if args and args[-1].isdigit() else args
    set_services(Services(kg=KnowledgeGraph(get_db())))
    print(f"=== batch live: {len(names)} monitors, conc={conc} ===", flush=True)
    sem = asyncio.Semaphore(conc)
    results = {}
    await asyncio.gather(*[_run(n, sem, results) for n in names])
    with open(OUT, "w", encoding="utf-8") as f:
        for n in names:
            d, s = results.get(n, ("(none)", 0))
            f.write(f"\n\n{'=' * 92}\n## {n}  ({s}s)\n{'=' * 92}\n\n{d}")
    print(f"\nwritten: {OUT}", flush=True)


asyncio.run(main())
