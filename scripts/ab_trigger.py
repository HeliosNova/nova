"""A/B helper: trigger a set of monitors, wait until each stores a FRESH result.
Tolerates the client-side timeout (the server keeps running the digest after the
HTTP request disconnects). Prints the stored ids when all are fresh.
"""
import json, os, sqlite3, sys, time, urllib.request

DB = "/data/nova.db"
MIDS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "25,30,7").split(",")]
KEY = os.environ.get("NOVA_API_KEY", "")


def latest(mid):
    c = sqlite3.connect(DB)
    r = c.execute("SELECT id, created_at FROM monitor_results WHERE monitor_id=? ORDER BY id DESC LIMIT 1", (mid,)).fetchone()
    c.close()
    return r


start = {mid: (latest(mid)[0] if latest(mid) else 0) for mid in MIDS}
print("baseline latest ids:", start, flush=True)

for mid in MIDS:
    try:
        req = urllib.request.Request(f"http://localhost:8000/api/monitors/{mid}/trigger",
                                     data=b"", method="POST", headers={"Authorization": f"Bearer {KEY}"})
        urllib.request.urlopen(req, timeout=780)
        print(f"monitor {mid}: trigger returned", flush=True)
    except Exception as e:
        print(f"monitor {mid}: client timeout/err ({type(e).__name__}) — server continues", flush=True)

# Wait for all to produce a new row (id greater than the pre-trigger latest).
deadline = time.time() + 2400  # 40 min hard cap
done = set()
while time.time() < deadline and len(done) < len(MIDS):
    for mid in MIDS:
        if mid in done:
            continue
        r = latest(mid)
        if r and r[0] > start[mid]:
            done.add(mid)
            print(f"monitor {mid}: FRESH result id={r[0]} at {r[1]}", flush=True)
    if len(done) < len(MIDS):
        time.sleep(20)

print("DONE. fresh:", sorted(done), "| missing:", sorted(set(MIDS) - done), flush=True)
