"""Run ONE domain monitor live through the REAL pipeline (27B synthesis) and report.

Verifies fix #1 end-to-end: the per-story deep-analysis now reads FULL article
bodies (not 240-token stubs). Surfaces the [DeepResearch] INFO lines so the
deep-analysis story/call counts are visible. Usage: python /data/_one_live.py "AI & ML"
"""
import asyncio
import logging
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("app.monitors.deep_research").setLevel(logging.INFO)

from app.core.brain import Services, set_services
from app.core.kg import KnowledgeGraph
from app.database import get_db
from app.monitors.domain_study_runner import run_domain_study


def _kgcount(db):
    try:
        r = db.fetchone("SELECT COUNT(*) AS c FROM kg_facts WHERE superseded_at IS NULL")
        return r["c"] if r else -1
    except Exception:
        return -1


async def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "AI & ML"
    db = get_db()
    set_services(Services(kg=KnowledgeGraph(db)))
    before = _kgcount(db)
    t = time.monotonic()
    digest = await run_domain_study(name)
    secs = round(time.monotonic() - t)
    after = _kgcount(db)
    print(f"\n===== {name} | {secs}s | KG {before}->{after} (+{after - before}) =====\n", flush=True)
    print(digest, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
