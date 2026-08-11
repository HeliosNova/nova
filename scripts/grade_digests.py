"""Grade recently-stored monitor digests with the RACE/FACT grader, using a PINNED
judge model so a 9B-vs-27B synthesis A/B is fair (same judge both sides).

Thesis test (owner 2026-07-10, "test limits, don't import them"): does the 9B +
the full aggregation/grounding/fresh-check/corroboration loop match the stateless
27B on synthesis quality? If yes, MONITOR_SYNTHESIS_MODEL→9B kills the GPU thrash.

Usage (inside nova-app):
  python -m scripts.grade_digests --monitors 25,30,7 --judge gemma4:e4b --tag 27b
Run once before the config flip (tag 27b) and once after (tag 9b); compare means.
"""
import argparse, asyncio, json, re, sqlite3, statistics

DB = "/data/nova.db"
_HDR_HOSTS = re.compile(r"read \d+ sources?:?\s*([^\n_]+)", re.IGNORECASE)


def _read_hosts(text: str) -> list[str]:
    m = _HDR_HOSTS.search(text or "")
    if not m:
        return []
    return [h.strip() for h in re.split(r"[,·]", m.group(1)) if "." in h and len(h.strip()) < 40]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitors", default="", help="comma monitor ids; blank = most recent overall")
    ap.add_argument("--judge", default="gemma4:e4b")
    ap.add_argument("--per", type=int, default=1, help="latest N digests per monitor")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from app.monitors.report_grader import grade_report

    c = sqlite3.connect(DB)
    rows = []
    if args.monitors:
        for mid in [x for x in args.monitors.split(",") if x.strip()]:
            for (v, ts, name) in c.execute(
                "SELECT r.value, r.created_at, m.name FROM monitor_results r JOIN monitors m ON m.id=r.monitor_id "
                "WHERE r.monitor_id=? AND length(r.value)>1500 ORDER BY r.id DESC LIMIT ?", (int(mid), args.per)):
                rows.append((name, ts, v))
    else:
        for (v, ts, name) in c.execute(
            "SELECT r.value, r.created_at, m.name FROM monitor_results r JOIN monitors m ON m.id=r.monitor_id "
            "WHERE length(r.value)>1500 AND r.value LIKE '%ead development%' ORDER BY r.id DESC LIMIT 6"):
            rows.append((name, ts, v))

    graded = []
    for name, ts, v in rows:
        g = await grade_report(v, name, model=args.judge)
        g["_name"], g["_ts"] = name, ts
        graded.append(g)
        print(f"[{args.tag}] {name[:34]:34} ts={ts} overall={g['overall']:.3f} "
              f"race={g['race_avg']:.2f} support={g['fact']['support']:.2f} fab={g['fact']['fabricated_rate']:.2f}")

    if graded:
        print(f"\n[{args.tag}] MEANS over {len(graded)}: "
              f"overall={statistics.mean(x['overall'] for x in graded):.3f} "
              f"race={statistics.mean(x['race_avg'] for x in graded):.3f} "
              f"support={statistics.mean(x['fact']['support'] for x in graded):.3f} "
              f"fabricated={statistics.mean(x['fact']['fabricated_rate'] for x in graded):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
