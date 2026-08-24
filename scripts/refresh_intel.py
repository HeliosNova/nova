"""One-off: re-run the 7 non-contracts Monitor-Intelligence query monitors so any
stale link-only digests are replaced by the fixed code, then report each digest's
shape (item count, bare-link rows, self-flag insight). Run in-container:
  docker exec nova-app python /data/refresh_intel.py
"""
import asyncio, os, re, sqlite3
import httpx

KEY = os.environ.get("NOVA_API_KEY", "")
NAMES = [
    "SEC Insider Trading", "FDA Drug Approvals", "FOMC and Fed Watch",
    "GitHub Security Advisories", "Hacker News Top Stories",
    "Product Hunt Trending", "World Awareness",
]
# feeds where the TITLE is the item (a bare title+link is fine, not "just links")
TITLE_IS_ITEM = {"SEC Insider Trading", "Hacker News Top Stories", "Product Hunt Trending"}


def _id(name):
    db = sqlite3.connect("file:/data/nova.db?mode=ro", uri=True)
    r = db.execute("SELECT id FROM monitors WHERE name=?", (name,)).fetchone()
    return r[0] if r else None


async def main():
    async with httpx.AsyncClient(timeout=290) as c:
        for n in NAMES:
            mid = _id(n)
            try:
                r = await c.post(f"http://localhost:8000/api/monitors/{mid}/trigger",
                                 headers={"Authorization": f"Bearer {KEY}"})
                print(f"triggered {n}: HTTP {r.status_code}", flush=True)
            except Exception as e:
                print(f"trigger {n} FAILED: {e}", flush=True)

    print("\n=== ANALYSIS (incl. Government Contract Awards) ===", flush=True)
    db = sqlite3.connect("file:/data/nova.db?mode=ro", uri=True)
    for n in NAMES + ["Government Contract Awards"]:
        mid = _id(n)
        row = db.execute("SELECT created_at,value FROM monitor_results WHERE monitor_id=? "
                         "ORDER BY created_at DESC LIMIT 1", (mid,)).fetchone()
        if not row:
            print(f"  {n:28} (no result)", flush=True); continue
        ts, v = row
        blocks = [b for b in re.split(r"\n(?=\*\*`\d+\.`)", v) if b.startswith("**`")]
        bare = sum(1 for b in blocks if len([l for l in b.split("\n") if l.strip()]) <= 2)
        flag = "SELF-FLAG!" if re.search(r"lacks|only date headers|no specific|no detail", v, re.I) else "ok"
        note = "(title=item, bare ok)" if n in TITLE_IS_ITEM else ""
        print(f"  {n[:28]:28} items={len(blocks):<2} bare={bare:<2} insight={flag:<10} "
              f"len={len(v):<5} {ts[:16]} {note}", flush=True)

asyncio.run(main())
