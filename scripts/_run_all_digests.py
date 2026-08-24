"""Run the full digest for EVERY domain monitor — matching PRODUCTION routing:
specialized monitors -> native list renderer; all others -> deep-research overview.
Live KG so facts bank too. Writes each full digest to /data/all_digests.md and
prints a one-line progress summary per monitor."""
import asyncio
import time

from app.monitors.domain_study_runner import (
    _DOMAIN_PROFILES, _SPECIALIZED, _render_native_list,
)
from app.monitors.deep_research import domain_overview

KG = None
try:
    from app.database import get_db
    from app.core.kg import KnowledgeGraph
    KG = KnowledgeGraph(get_db())
except Exception as e:
    print(f"[no KG: {type(e).__name__}]", flush=True)

OUT = "/data/all_digests.md"

# News domains first, specialized list-monitors last.
keys = [k for k in _DOMAIN_PROFILES if k not in _SPECIALIZED] + \
       [k for k in _DOMAIN_PROFILES if k in _SPECIALIZED]


async def main():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Nova — all monitor digests (v2, post-polish)\n\n")
    print(f"=== RUNNING {len(keys)} DOMAINS ===", flush=True)
    for i, key in enumerate(keys, 1):
        emoji, label, _kw = _DOMAIN_PROFILES[key]
        t = time.time()
        try:
            if key in _SPECIALIZED:
                out = await _render_native_list(key, label, emoji)
                kind = "NATIVE"
            else:
                out = await domain_overview(label, kg=KG, n_stories=5, feed_key=key)
                kind = "OVERVIEW" if "domain overview" in out else (
                    "BRIEFING" if "researched briefing" in out else "THIN")
        except Exception as e:
            import traceback; traceback.print_exc()
            out = f"## {label} — FAILED: {e}"
            kind = "FAIL"
        dt = time.time() - t
        hdr = next((l for l in out.splitlines() if "sources:" in l or "items from" in l), "").strip().strip("_")
        print(f"[{i:2}/{len(keys)}] {key:30} {kind:9} [{dt:3.0f}s]  {hdr[:78]}", flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(f"\n\n{'='*90}\n{out}\n")
    print("=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
