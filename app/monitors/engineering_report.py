"""One daily readout of whether Nova is delivering, and what to look at.

The 2026-08-28 throughput regression ran for a WEEK. Nothing was broken in the
way monitoring looks for: the app was up ~24 hours a day, digests kept their
length, their sources and their judge scores, and every monitor merely ran a
little less often. The signal existed — it was spread across twenty system
monitors and five operator scripts, each reporting its own slice against its own
threshold, and no single one owned the question "is Nova delivering less than it
was".

This owns that question. It measures nothing new: every number here comes from a
function that already existed by 2026-09-04, assembled in one place and
DELIVERED whether or not anything is wrong. That last part is the point — a
report that only speaks up on a threshold is another thing that stayed quiet for
a week.

The closing section is the deliverable: not numbers, but the ones that crossed a
bar, in the order worth looking at.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# The cascade's own summary line, which nothing parsed until now:
#   [entail-cascade] <label>/<site>: N pair(s), M scored narrow (support P%), K read at full width
_CASCADE_RE = re.compile(
    r"\[entail-cascade\] .*?/(\w+): (\d+) pair\(s\), (\d+) scored narrow "
    r"\(support (\d+)%\), (\d+) read at full width")


def cascade_support(days: int = 1, log_glob: str = "/data/logs/nova-app.log*",
                    today: str | None = None) -> dict | None:
    """How often the NARROW document alone entailed a claim, per call site.

    The pre-registered check for the 2026-09-04 chrome rules: navigation menus
    were winning evidence windows the article should have won, so removing them
    should RAISE narrow support. It sat at 12-17% when the cascade shipped.
    Nothing read this line, so the answer would have gone unmeasured.
    """
    import glob as _glob
    import os
    import time
    from datetime import date, timedelta

    base = date.fromisoformat(today) if today else date.today()
    cutoff = (base - timedelta(days=max(1, days))).isoformat()
    sites: dict[str, list[int]] = {}
    pairs = 0
    try:
        paths = _glob.glob(log_glob)
    except OSError:
        return None
    for lp in paths:
        try:
            if os.path.getmtime(lp) < (time.time() - (days + 1) * 86400):
                continue
            with open(lp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = _CASCADE_RE.search(line)
                    if not m or line[:10] < cutoff:
                        continue
                    sites.setdefault(m.group(1), []).append(int(m.group(4)))
                    pairs += int(m.group(2))
        except OSError:
            continue
    if not sites:
        return None
    return {"pairs": pairs,
            "by_site": {k: round(sum(v) / len(v)) for k, v in sorted(sites.items())},
            "runs": {k: len(v) for k, v in sorted(sites.items())}}


def _short(item: str) -> str:
    """The headline of one attention line — the summary is capped at 80 chars,
    so three of them have to fit in the part a reader always sees."""
    head = item.split(" — ")[0].split(" (")[0]
    return head[:34].rstrip()


def _curiosity(db) -> dict:
    """Queue depth and the latency that made it a C grade."""
    out: dict = {}
    try:
        out["pending"] = db.fetchone(
            "SELECT COUNT(*) AS c FROM curiosity_queue WHERE status = 'pending'")["c"]
        out["resolved_24h"] = db.fetchone(
            "SELECT COUNT(*) AS c FROM curiosity_queue WHERE status = 'resolved' "
            "AND resolved_at >= datetime('now', '-1 day')")["c"]
        row = db.fetchone(
            "SELECT MIN(created_at) AS oldest FROM curiosity_queue WHERE status = 'pending'")
        out["oldest_pending"] = (row or {})["oldest"]
        row = db.fetchone(
            "SELECT AVG(julianday(resolved_at) - julianday(created_at)) AS d "
            "FROM curiosity_queue WHERE status = 'resolved' "
            "AND resolved_at >= datetime('now', '-14 days')")
        out["latency_days"] = round(row["d"], 1) if row and row["d"] is not None else None
    except Exception as e:
        logger.debug("[EngReport] curiosity block failed: %r", e)
    return out


def _knowing(db) -> dict:
    out: dict = {}
    for key, sql in (
        ("kg_facts_24h", "SELECT COUNT(*) AS c FROM kg_facts "
                         "WHERE created_at >= datetime('now', '-1 day')"),
        ("dossiers", "SELECT COUNT(*) AS c FROM dossiers"),
        ("open_questions", "SELECT COUNT(*) AS c FROM dossier_questions "
                           "WHERE status = 'open'"),
        ("forecasts_open", "SELECT COUNT(*) AS c FROM forecasts WHERE status = 'open'"),
    ):
        try:
            out[key] = db.fetchone(sql)["c"]
        except Exception:
            out[key] = None
    return out


def build_report(db) -> tuple[str, str, dict]:
    """(status, summary, fields) — always delivers, never only on a threshold."""
    from app.core.forecasts import calibration
    from app.monitors.health_checks import entail_gate_totals
    from app.monitors.pathways import (
        constant_monitors,
        schedule_pressure,
        snapshot,
        throughput_step,
    )

    fields: dict[str, str | int | float] = {}
    attention: list[str] = []

    step = throughput_step(db)
    if step:
        fields["delivery"] = (f"{step['after']:.1f} runs/active hour "
                              f"({step['change']:+.0%} over {step['days']}d)")
        if step["stepped_down"]:
            attention.append(
                f"delivery is DOWN {abs(step['change']):.0%} ({step['before']:.1f} -> "
                f"{step['after']:.1f} runs/active hour) — something costs more per run")

    press = schedule_pressure(db)
    if press.get("ratio") is not None:
        fields["schedule"] = (f"{press['ratio']:.0%} of demanded runs delivered "
                              f"({press['delivered']}/{press['demanded']}, 7d)")
        if press["starved"]:
            worst = press["starved"][0]
            attention.append(f"{worst['name']} is running at {worst['ratio']:.0%} "
                             f"of its declared cadence")

    week = entail_gate_totals(7)
    day = entail_gate_totals(1)
    if week[0]:
        w_rate = week[1] / week[0]
        fields["entail_drop"] = f"{w_rate:.0%} over 7d"
        if day[0] >= 100:
            d_rate = day[1] / day[0]
            fields["entail_drop"] += f", {d_rate:.0%} today"
            if d_rate - w_rate >= 0.12:
                attention.append(f"entail drop-rate jumped to {d_rate:.0%} today "
                                 f"against {w_rate:.0%} for the week")

    casc = cascade_support(1)
    if casc:
        fields["narrow_support"] = ", ".join(
            f"{k} {v}%" for k, v in casc["by_site"].items())

    cur = _curiosity(db)
    if cur:
        bits = [f"{cur.get('pending', '?')} pending",
                f"{cur.get('resolved_24h', 0)} resolved in 24h"]
        if cur.get("latency_days") is not None:
            bits.append(f"{cur['latency_days']}d to answer")
            if cur["latency_days"] > 5:
                attention.append(
                    f"curiosity takes {cur['latency_days']} days to answer a question")
        fields["curiosity"] = ", ".join(bits)

    kn = _knowing(db)
    fields["knowing"] = (f"+{kn.get('kg_facts_24h', 0)} facts/24h, "
                         f"{kn.get('dossiers', 0)} dossiers, "
                         f"{kn.get('open_questions', 0)} open questions, "
                         f"{kn.get('forecasts_open', 0)} forecasts open")

    cal = calibration(db, min_n=20) or calibration(db, min_n=20, regime=None)
    if cal:
        fields["forecast_skill"] = (
            f"{cal['skill']:+.2f} (n={cal['n']}, Brier {cal['brier']:.3f} vs "
            f"{cal['base_brier']:.3f} base)" if cal.get("skill") is not None
            else f"n={cal['n']}, Brier {cal['brier']:.3f}")
        if cal.get("skill") is not None and cal["skill"] <= 0:
            attention.append("forecast confidence still has no edge over the base rate")

    rows = snapshot(db)
    dead = [r["name"] for r in rows if r["verdict"] in ("dead", "unknown")]
    fields["pathways"] = (f"{sum(1 for r in rows if r['verdict'] == 'alive')} alive, "
                          f"{sum(1 for r in rows if r['verdict'] == 'idle')} idle, "
                          f"{sum(1 for r in rows if r['verdict'] == 'off')} off")
    if dead:
        attention.insert(0, f"pathway(s) DEAD: {', '.join(dead)}")

    quiet = constant_monitors(db)
    if quiet:
        fields["saying_nothing"] = ", ".join(
            f"{c['name']} ({c['runs']}x identical)" for c in quiet[:3])

    if attention:
        # The rendered line is capped at 400 characters and fields are dropped
        # from the END, so the actionable ones go FIRST — the numbers below are
        # context and can afford to fall off. Learned on 2026-09-03, when dead
        # pathway names were being pushed off by schedule stats, and re-learned
        # here on the first live run: `look_at` was last and vanished entirely.
        fields = {**{f"look_at_{i + 1}": a for i, a in enumerate(attention[:3])},
                  **fields}
        status = "error" if dead else "warning"
        summary = f"{len(attention)} to look at: " + "; ".join(
            _short(a) for a in attention[:3])
    else:
        status = "info"
        summary = "nothing crossed a bar today"
    return status, summary, fields
