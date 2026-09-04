"""How much work is Nova actually delivering, and what is it waiting on?

Written 2026-09-04, after a throughput regression sat undetected for a week.
Delivered monitor runs per active hour fell from 6.4 to 3.8 on 2026-08-28 and
stayed there. Nothing alerted, because nothing was broken in the way monitoring
looks for: the app was up ~24 hours a day, digests were the same length, cited
the same number of sources, and scored the same. Each monitor merely ran a
little less often, and no single one looked wrong.

The cause was two correct fixes whose cost nobody measured. On 2026-08-26 the
MiniCheck sidecar serialized scoring behind one lock (it had to: concurrent
batches were thrashing the CPU and failing chat open 27 times in 48 hours), and
on 2026-08-29 the enrichment entailment gate stopped timing out into unverified
publishes (it had to: a timeout was wearing the costume of a clean run). Between
them, every entailment call in the system became a queue, with more work in it.

So this panel reports the things that would have made that visible on day one:

  runs            monitor results written that day
  active hrs      hours with at least one result - separates "down" from "idle"
  runs/hr         the headline. A step here is a cost change, not weather.
  idle hrs        time inside gaps longer than THRESH_MIN, i.e. the loop waiting
  min/run         active minutes divided by runs - the inverse of the
                  headline, and the number to quote when comparing deploys
  med gap         median time between consecutive runs, UNCAPPED

Uncapped is the point. An earlier version of this analysis capped gaps at 45
minutes to exclude idle stretches and concluded digests had not slowed down -
the cap had excluded exactly the expensive runs. A single digest measured 52
minutes the morning this was written.

Read-only. Run it before and after any deploy that touches the digest chain:

    docker exec nova-app python /app/scripts/throughput_panel.py [days]
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime

DB = "file:/data/nova.db?mode=ro"
THRESH_MIN = 45.0          # a gap longer than this is the loop waiting, not working


def main(days: int = 21) -> None:
    conn = sqlite3.connect(DB, uri=True)
    rows = conn.execute(
        "SELECT mr.created_at, m.name, m.category "
        "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
        "WHERE mr.created_at >= date('now', ?) ORDER BY mr.created_at",
        (f"-{int(days)} days",)).fetchall()
    if not rows:
        print("no monitor results in the window")
        return

    per_day: dict[str, dict] = defaultdict(
        lambda: {"runs": 0, "digests": 0, "hours": set(), "idle": 0.0,
                 "gaps": [], "stalls": 0})
    blockers: dict[str, int] = defaultdict(int)

    prev_t = prev_name = None
    for ts, name, category in rows:
        day = ts[:10]
        e = per_day[day]
        e["runs"] += 1
        e["digests"] += (category == "content")
        e["hours"].add(ts[11:13])
        t = datetime.fromisoformat(ts)
        if prev_t is not None:
            gap = (t - prev_t).total_seconds() / 60.0
            if gap >= 0:
                e["gaps"].append(gap)
                if gap > THRESH_MIN:
                    e["idle"] += gap
                    e["stalls"] += 1
                    blockers[prev_name] += 1
        prev_t, prev_name = t, name

    print(f"{'day':<12}{'runs':>6}{'digests':>9}{'active hrs':>12}"
          f"{'runs/hr':>9}{'idle hrs':>10}{'min/run':>9}{'med gap':>9}")
    series: list[tuple[str, float]] = []
    for day in sorted(per_day):
        e = per_day[day]
        hrs = len(e["hours"])
        rate = e["runs"] / hrs if hrs else 0.0
        med = statistics.median(e["gaps"]) if e["gaps"] else 0.0
        per_run = (hrs * 60 / e["runs"]) if e["runs"] else 0.0
        series.append((day, rate))
        print(f"{day:<12}{e['runs']:>6}{e['digests']:>9}{hrs:>12}"
              f"{rate:>9.1f}{e['idle'] / 60:>10.1f}{per_run:>9.1f}{med:>9.1f}")

    if len(series) >= 8:
        # A step is what matters, so compare the oldest and newest thirds rather
        # than fitting a trend through weather.
        k = max(3, len(series) // 3)
        old = statistics.mean(r for _d, r in series[:k])
        new = statistics.mean(r for _d, r in series[-k:])
        delta = (new - old) / old * 100 if old else 0.0
        print(f"\nfirst {k} days {old:.1f} runs/hr -> last {k} days {new:.1f} "
              f"({delta:+.0f}%)")
        if delta <= -20:
            print("STEP DOWN: something is costing more per run than it used to. "
                  "Check what shipped at the break, not what looks broken now.")

    print("\nmonitor that ran last before a stall "
          f"(> {THRESH_MIN:.0f} min), whole window:")
    for name, n in sorted(blockers.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:>4}  {name}")
    print("\nA spread of names here means the loop is waiting on shared cost "
          "(the entailment sidecar, model residency).\nOne name repeating means "
          "that monitor is the blocker.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 21)
