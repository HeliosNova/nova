"""Regenerate every monitor digest that is defective (think-leak or truncated
mid-sentence), self-verifying: re-scan, re-trigger, re-check until zero remain
or the pass budget is exhausted. Run detached:

    docker exec -d nova-app python scripts/regen_defective_digests.py > /data/regen.log 2>&1

Written as a file (not a shell heredoc) so f-string quoting can't get mangled.
"""

import os
import sqlite3
import time
import urllib.request

KEY = os.environ.get("NOVA_API_KEY", "")
CLEAN_END = set(".!?)”\":_")
DB = "/data/nova.db"
API = "http://localhost:8000"


def scan():
    c = sqlite3.connect(DB)
    rows = c.execute(
        "SELECT m.id, m.name, r.value FROM monitor_results r "
        "JOIN monitors m ON m.id = r.monitor_id "
        "WHERE r.id IN (SELECT MAX(id) FROM monitor_results GROUP BY monitor_id) "
        "AND m.check_type = 'query' AND length(r.value) > 200"
    ).fetchall()
    c.close()
    bad = []
    for mid, name, v in rows:
        tail = v.rstrip()
        think = "<think>" in v.lower()
        trunc = len(tail) > 3000 and tail[-1] not in CLEAN_END
        if think or trunc:
            bad.append((mid, name, "THINK" if think else "TRUNC"))
    return bad


def trigger(mid):
    req = urllib.request.Request(
        API + "/api/monitors/%d/trigger" % mid,
        method="POST", headers={"Authorization": "Bearer " + KEY})
    urllib.request.urlopen(req, timeout=2400).read()


def status(mid):
    c = sqlite3.connect(DB)
    v = c.execute(
        "SELECT value FROM monitor_results WHERE monitor_id = ? "
        "ORDER BY id DESC LIMIT 1", (mid,)).fetchone()[0]
    c.close()
    tail = v.rstrip()
    return len(v), ("<think>" in v.lower()), (tail[-1] in CLEAN_END)


def main():
    done = set()
    for it in range(3):
        bad = [b for b in scan() if b[0] not in done]
        print("--- pass %d: %d defective ---" % (it, len(bad)), flush=True)
        if not bad:
            break
        for mid, name, kind in bad:
            t0 = time.time()
            try:
                trigger(mid)
                ln, think, clean = status(mid)
                ok = (not think) and clean
                if ok:
                    done.add(mid)
                verdict = "OK" if ok else "STILL-BAD"
                print("[%s] %-34s len=%d think=%s clean=%s %s (%ds)" % (
                    kind, name[:34], ln, think, clean, verdict, round(time.time() - t0)), flush=True)
            except Exception as e:
                print("[%s] %s: FAILED %r" % (kind, name[:34], e), flush=True)
    final = scan()
    print("=== FINAL: %d still defective ===" % len(final), flush=True)
    for mid, name, kind in final:
        print("  STILL %s: %s" % (kind, name), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
