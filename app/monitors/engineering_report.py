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


_STAGE1_RE = re.compile(r"\[Curiosity\] stage-1 reject \(([^)]+)\)")
# The caller, then the ceiling. Older lines name the helper ("invoke_nothink")
# because until 2026-09-06 the tripwire could not see past it; those still
# parse, they just cannot be attributed.
_TRUNC_RE = re.compile(r"\[truncation\] ([\w.]+) hit max_tokens \((\d+)\)")
_PLAN_FAIL_RE = re.compile(r"Planning failed: (\w+)")
# The denominator. Counting failures alone was this field's own version of
# the mistake schedule_pressure made: "planner failed: TimeoutError x3" reads
# identically at 3-of-3 (catastrophic) and 3-of-38 (a bad afternoon). The
# measured spread across one week was 40% down to 0%.
# The DENOMINATOR comes from brain's pre-existing success line, which has
# history; the duration comes from the line added 2026-09-06, which does not.
# Reading the count off the new line made the field report "0 ok / 3 timeout
# (100%)" on the day it shipped — the three failures were real and from a day
# with 35 uninstrumented successes beside them. Absent evidence of success is
# not evidence of total failure, the same trap as reading a missing cascade
# line as 0% support.
_PLAN_OK_RE = re.compile(r"Query planned: \d+ steps")
_PLAN_SECS_RE = re.compile(r"\[planning\] plan ready in ([0-9.]+)s")


def _scan(days: int, log_glob: str, today: str | None, handler) -> None:
    """Walk the persisted logs inside a window, feeding lines to `handler`.

    The container's own log is lost on every restart — and this session
    restarted nova-app nine times in a day — so everything here reads
    /data/logs, which lives on the volume and survives.
    """
    import glob as _glob
    import os
    import time as _time
    from datetime import date, timedelta

    base = date.fromisoformat(today) if today else date.today()
    cutoff = (base - timedelta(days=max(1, days))).isoformat()
    try:
        paths = _glob.glob(log_glob)
    except OSError:
        return
    for lp in paths:
        try:
            if os.path.getmtime(lp) < (_time.time() - (days + 1) * 86400):
                continue
            with open(lp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line[:10] >= cutoff:
                        handler(line)
        except OSError:
            continue


def curiosity_stage1(days: int = 1, log_glob: str = "/data/logs/nova-app.log*",
                     today: str | None = None) -> dict | None:
    """Why curiosity's cheap pre-filter rejected an answer, by reason.

    Half of all closure failures came from stage 1 and logged nothing until
    2026-09-06 (171 of 340 since 08-20). The open question this answers: are
    the deflection markers killing answers that actually settled the question
    but hedged one sub-part? A marker dominating this list is the suspect.
    """
    from collections import Counter
    reasons: Counter = Counter()

    def _h(line: str) -> None:
        m = _STAGE1_RE.search(line)
        if m:
            r = m.group(1)
            reasons["too short" if r.startswith("too short") else
                    r.split(" at ")[0].replace("deflection ", "")] += 1

    _scan(days, log_glob, today, _h)
    return dict(reasons.most_common(5)) if reasons else None


def planner_health(days: int = 1, log_glob: str = "/data/logs/nova-app.log*",
                   today: str | None = None) -> dict | None:
    """Whether the planner is still being cut off, and how often it times out.

    The ceiling went 512 -> 900 on 2026-09-05 with a caveat: the A/B that
    justified it never reproduced the long queries that truncate, so "900 is
    enough" was untested. A `900` here says it is not. Timeouts are the larger,
    untouched problem — 60s against a GPU the digest chain owns.
    """
    from collections import Counter
    caps: Counter = Counter()
    fails: Counter = Counter()
    secs: list[float] = []
    oks = [0]

    def _h(line: str) -> None:
        m = _TRUNC_RE.search(line)
        if m:
            who, cap = m.group(1), m.group(2)
            # Legacy lines name only the helper and carry no attribution, so
            # they stay keyed by ceiling alone rather than pretending otherwise.
            caps[cap if who == "invoke_nothink" else f"{who}@{cap}"] += 1
        f = _PLAN_FAIL_RE.search(line)
        if f:
            fails[f.group(1)] += 1
        if _PLAN_OK_RE.search(line):
            oks[0] += 1
        d = _PLAN_SECS_RE.search(line)
        if d:
            secs.append(float(d.group(1)))

    _scan(days, log_glob, today, _h)
    if not caps and not fails and not oks[0]:
        return None
    out: dict = {"truncations": dict(caps.most_common(4)),
                 "plan_failures": dict(fails.most_common(3))}
    nfail = sum(fails.values())
    if oks[0] or nfail:
        out["planned"] = oks[0]
        out["failed"] = nfail
        total = oks[0] + nfail
        out["fail_rate"] = (nfail / total) if total else None
        if secs:
            # The slowest plan that SUCCEEDED, against a 60s ceiling. Well under
            # it means a timeout is queueing or a model swap, and raising the
            # ceiling is the wrong fix; close to it means 60s is simply tight.
            out["slowest_ok"] = max(secs)
    return out


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


# ---------------------------------------------------------------------------
# What the digests LOOK like. Deterministic, no model, no network.
#
# The report above answers "is Nova delivering less than it was". It cannot see
# the other regression: delivering the same amount, worse. On 2026-09-04 digests
# citing "(deep analysis)" climbed 10% -> 34% -> 45% -> 55% over three days with
# the suite green and every throughput number flat, and the OWNER found it.
#
# These are the same measures scripts/quality_panel.py uses, defined HERE and
# imported there, because a second copy of a regex is how `strip_markup` came to
# be defined twice with the second silently winning.
# ---------------------------------------------------------------------------
_PAREN = re.compile(r"\(([^)]{0,80})\)")
_ANALYSIS = re.compile(r"(?i)\banalys[ei]s\b")
_DOMAINISH = re.compile(r"[a-z0-9-]+\.[a-z]{2,}")
_CITE = re.compile(r"\(([a-z0-9-]+\.[a-z]{2,})\)")
_LEAKS = (
    re.compile(r"(?i)\bas an ai\b"),
    re.compile(r"(?i)\b(?:step|stage) \d+/\d+\b"),
    re.compile(r"(?i)\bnot specified here\b"),
    re.compile(r"(?i)\bsearch results?\b"),
    re.compile(r"</?tool_call>"),
    re.compile(r"(?i)\bI (?:cannot|can't) (?:access|browse)\b"),
)


# Post-fix (2026-09-04) this measured 0.00 across 125 digests while the four
# days before it ran 1.41-2.36, so anything above a rounding error is a
# regression and not a topic-mix wobble.
_PSEUDO_FLOOR = 0.25


def digest_shape(value: str) -> dict:
    """The deterministic fingerprint of one digest.

    `pseudo` is the load-bearing one: a short parenthetical naming "analysis"
    with no domain token in it is the briefing citing its own reasoning instead
    of a source. `linkonly` is the owner's recurring complaint in numeric form.
    """
    return {
        "chars": len(value),
        "cites": len(_CITE.findall(value)),
        "pseudo": sum(1 for m in _PAREN.finditer(value)
                      if _ANALYSIS.search(m.group(1))
                      and not _DOMAINISH.search(m.group(1))),
        "leaks": sum(1 for rx in _LEAKS if rx.search(value)),
        "linkonly": 1 if (len(value) < 600 and "http" in value) else 0,
        "thin": 1 if len(value) < 2500 else 0,
    }


def product_quality(db, days: int = 1, baseline_days: int = 7) -> dict | None:
    """Yesterday's digest shape against the week before it.

    Absolute values move with topic mix; a STEP against the trailing week is
    what a prompt or gate change looks like, which is the only comparison worth
    waking someone for.

    `check_type='query'` and NOT `category='content'`, which was the first
    version and was wrong. The content category also carries curiosity answers
    (470 chars on average), storyline summaries (3,786) and forecast resolution
    notes (770) — none of which are briefings. Measured that way, raising
    `_CURIOSITY_BATCH` to 3 on 2026-09-04 read as digests getting shorter and
    the thin count tripling: a change I made deliberately, arriving as a quality
    alarm. Grading an artifact against a shape it never had is this codebase's
    most repeated mistake; do not widen this back to the category.
    """
    def _window(where: str, args: tuple) -> dict | None:
        try:
            rows = db.fetchall(
                "SELECT mr.value FROM monitor_results mr "
                "JOIN monitors m ON m.id = mr.monitor_id "
                "WHERE m.check_type = 'query' AND mr.value IS NOT NULL "
                "AND LENGTH(mr.value) > 400 AND " + where, args)
        except Exception:
            return None
        if not rows:
            return None
        tot = {"n": len(rows), "chars": 0, "cites": 0, "pseudo": 0,
               "leaks": 0, "linkonly": 0, "thin": 0}
        for r in rows:
            for k, v in digest_shape(r["value"]).items():
                tot[k] += v
        return tot

    recent = _window("mr.created_at > datetime('now', ?)", (f"-{days} days",))
    if not recent:
        return None
    base = _window(
        "mr.created_at <= datetime('now', ?) AND mr.created_at > datetime('now', ?)",
        (f"-{days} days", f"-{days + baseline_days} days"))
    out = {"n": recent["n"], "chars": recent["chars"] // recent["n"],
           "cites": recent["cites"] / recent["n"],
           "pseudo": recent["pseudo"] / recent["n"],
           "leaks": recent["leaks"], "linkonly": recent["linkonly"],
           "thin": recent["thin"]}
    if base and base["n"] >= 5:
        out["base_chars"] = base["chars"] // base["n"]
        out["base_cites"] = base["cites"] / base["n"]
        out["base_pseudo"] = base["pseudo"] / base["n"]
        out["base_n"] = base["n"]
    return out


SNAPSHOT_PATH = "/data/eng_report.jsonl"


def append_snapshot(status: str, summary: str, fields: dict,
                    path: str = SNAPSHOT_PATH) -> bool:
    """Keep the FULL field set, because the delivered line cannot.

    format_monitor_result caps a result at 400 characters and drops fields from
    the end, so the row stored in monitor_results holds the findings and the
    first few numbers and loses the rest. That is right for a message and wrong
    for a record: comparing this morning with last week is the entire reason
    the report exists, and half its fields would not survive to be compared.

    One JSON line per run, on the data volume so it outlives a container. Never
    raises — it runs inside a monitor.
    """
    import json
    from datetime import datetime, timezone
    try:
        row = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
               "status": status, "summary": summary, **fields}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return True
    except Exception as e:
        logger.warning("[EngReport] snapshot not written: %r", e)
        return False


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
        # Carry the window's BEST alongside the trend: the rolling comparison
        # says whether things are getting worse, and cannot say whether they are
        # back to normal — as a regression ages it contaminates its own
        # baseline. "4.7 now, best 6.7" tells a reader both.
        best = step.get("best")
        fields["delivery"] = (f"{step['after']:.1f} runs/active hour "
                              f"({step['change']:+.0%} vs prior week"
                              + (f", best {best:.1f}" if best else "") + ")")
        if step["stepped_down"]:
            attention.append(
                f"delivery is DOWN {abs(step['change']):.0%} ({step['before']:.1f} -> "
                f"{step['after']:.1f} runs/active hour vs the prior week) — "
                f"something costs more per run")

    prod = product_quality(db)
    if prod:
        bits = [f"{prod['n']} digests", f"{prod['chars']} chars",
                f"{prod['cites']:.1f} cites"]
        if prod["pseudo"]:
            bits.append(f"{prod['pseudo']:.2f} self-cites")
        if prod["thin"]:
            bits.append(f"{prod['thin']} thin")
        fields["product"] = ", ".join(bits)
        # The owner's recurring complaint, in numbers. A digest under 600 chars
        # carrying a URL is the link-only failure, and is never acceptable.
        if prod["linkonly"]:
            attention.append(
                f"{prod['linkonly']} digest(s) came back as bare links in 24h")
        if prod["leaks"]:
            attention.append(
                f"{prod['leaks']} digest(s) leaked scaffolding text in 24h")
        # Self-citation is a prompt/guard mismatch — the shape that ran 10% ->
        # 55% over three days in 2026-09 with every other number flat.
        #
        # An ABSOLUTE floor, not only a step, because a rolling baseline cannot
        # see a return to a level it still contains. The 09-04 fix took this to
        # 0.00 across 125 digests while the trailing week still averages 1.67,
        # so for the next week a regression all the way back to 1.0 would
        # compute as an IMPROVEMENT and say nothing. Same trap the fixed-thirds
        # throughput window had, one metric over.
        bp = prod.get("base_pseudo")
        stepped = bp is not None and prod["pseudo"] >= 0.20 and prod["pseudo"] > bp + 0.15
        if prod["pseudo"] >= _PSEUDO_FLOOR or stepped:
            was = f" vs {bp:.2f} the week before" if bp is not None else ""
            attention.append(
                f"digests are citing their own analysis again "
                f"({prod['pseudo']:.2f}/digest{was})")
        bc = prod.get("base_chars")
        if bc and prod["chars"] < bc * 0.75:
            attention.append(
                f"digests got shorter: {prod['chars']} chars vs {bc} the week "
                f"before — check the synthesis ceiling, not the schedule")

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

    st1 = curiosity_stage1(1)
    if st1:
        fields["curiosity_rejects"] = ", ".join(f"{k} x{v}" for k, v in st1.items())
        top, n = next(iter(st1.items()))
        if top != "too short" and n >= 3:
            attention.append(
                f"curiosity's stage-1 filter killed {n} answer(s) on {top!r} "
                f"before the judge saw them")

    ph = planner_health(1)
    if ph:
        # NOT called "planner": the tripwire logs the LLM helper's name, never
        # the caller, so these ceilings belong to whoever asked for them —
        # critique and storylines at 700, brain synthesis and tool_triggers at
        # 800. Reporting them under a planner heading was a first-day mistake.
        bits = []
        if ph["truncations"]:
            bits.append("cut at " + ", ".join(f"{k}x{v}" for k, v in ph["truncations"].items()))
        if ph.get("fail_rate") is not None:
            bits.append(f"planning {ph['planned']} ok / {ph['failed']} timeout "
                        f"({ph['fail_rate']:.0%})")
        if ph.get("slowest_ok") is not None:
            bits.append(f"slowest plan {ph['slowest_ok']:.0f}s")
        if ph["plan_failures"] and not ph.get("failed"):
            bits.append("planner failed: "
                        + ", ".join(f"{k} x{v}" for k, v in ph["plan_failures"].items()))
        fields["truncation"] = "; ".join(bits)
        # 900 is NOT unique to the planner — heartbeat_loop's curiosity answer and
        # the search agent ask for it too, which is why the tripwire had to learn
        # to name its caller before this rule could mean anything.
        # A RATE, not a count. 3 timeouts out of 38 is a bad afternoon; 3 out of
        # 3 is an outage, and the old field rendered them identically. The bar is
        # 15% because the measured week ran 40% at its worst and 0% at its best.
        if (ph.get("fail_rate") or 0) >= 0.15 and (ph["planned"] + ph["failed"]) >= 10:
            slow = ph.get("slowest_ok")
            hint = (f"slowest plan that DID finish took {slow:.0f}s of 60 — "
                    + ("the ceiling is tight" if slow and slow > 30
                       else "so this is queueing behind a model load, not slow generation")
                    ) if slow else "no plan finished, so the ceiling tells us nothing"
            attention.append(
                f"{ph['fail_rate']:.0%} of plans timed out "
                f"({ph['failed']}/{ph['planned'] + ph['failed']}) — {hint}")

        n900 = sum(v for k, v in ph["truncations"].items()
                   if k.endswith("@900") and k.startswith("planning"))
        if n900:
            attention.append(
                f"the planner is STILL truncating at its new 900 ceiling "
                f"({n900}x) — it needs more")

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
