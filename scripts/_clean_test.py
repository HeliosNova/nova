"""Clean (post-cooldown) measurement of the deep-research engine. Answers:
  1. Are the aux channels (GDELT/Bing) healthy when not throttled?
  2. For Geopolitics: feed vs aux candidate counts, and is breadth consistent
     across repeated runs?
  3. Baselines: Cybersecurity (strong) and AI/ML.
"""
import asyncio
import time
from app.monitors import deep_research as dr


async def aux_probe():
    g = await dr._gdelt("Iran Israel Hormuz")
    await asyncio.sleep(6)
    b = await dr._bing_news("Iran Israel Hormuz")
    print(f"[AUX PROBE] gdelt={len(g)} bing={len(b)} "
          f"gdelt_hosts={sorted({dr._host(c.url) for c in g})[:6]}", flush=True)


async def candidates(label):
    subs = await dr._focus_subjects(label, n=5)
    fc = await dr._feed_candidates(label)
    await asyncio.sleep(6)
    ac = await dr._aux_news(subs[:3])
    print(f"[{label} CANDIDATES] feed={len(fc)} aux={len(ac)} "
          f"aux_hosts={sorted({dr._host(c.url) for c in ac})[:8]}", flush=True)


async def overview(label, n):
    t = time.time()
    out = await dr.domain_overview(label, kg=None, n_stories=5)
    fmt = "OVERVIEW" if "domain overview" in out else ("FALLBACK" if "researched briefing" in out else "?")
    hdr = next((l for l in out.splitlines() if "sources:" in l), "").strip().strip("_")
    print(f"[{label} #{n}] {fmt} [{time.time()-t:.0f}s]  {hdr}", flush=True)


async def main():
    print("=== CLEAN TEST (post-cooldown) ===", flush=True)
    await aux_probe()
    await candidates("Geopolitics")
    for i in range(1, 4):
        await overview("Geopolitics", i)
    await overview("Cybersecurity", 1)
    await overview("AI and ML", 1)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
