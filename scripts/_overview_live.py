"""Live test of the wired domain_overview: breadth (top-N stories) over a deep
base (full-body reads) + recency gating + one grounded synthesis. Confirms the
stale/single-source failures are gone and the overview is rich + cited.

Run inside nova-app:  PYTHONPATH=/app python -u /data/_overview_live.py
"""
import asyncio
import time

from app.monitors.deep_research import domain_overview


async def run(label):
    t0 = time.time()
    try:
        out = await domain_overview(label, kg=None, n_stories=3)
    except Exception as e:
        import traceback; traceback.print_exc()
        out = f"!! failed: {e}"
    print(f"\n{'='*80}\n# {label}   [{time.time()-t0:.0f}s]\n{'='*80}\n{out}", flush=True)


import sys


async def main():
    labels = sys.argv[1:] or ["Finance", "Geopolitics", "Cybersecurity"]
    for label in labels:
        await run(label)


if __name__ == "__main__":
    asyncio.run(main())
