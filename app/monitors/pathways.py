"""Pathway liveness registry (2026-09-02).

Every optional pathway in Nova — storylines, dossiers, forecasts, curiosity,
KG banking, output eval, the delivery ledger — is a background WRITER whose
failure mode is silence, not an error: the code path stops being reached and
its table simply stops growing. Storylines were dead for five weeks
(2026-08-11) and KG extraction for three days (2026-08-18) before log
archaeology noticed; nothing in the system could say "this table used to grow
and no longer does".

This module is the ONE place that lists those writers, the table (or file)
each one writes, the config flag and heartbeat monitor that gate it, and how
long it may go quiet before that silence is a fault. Two consumers:

  * the "Pathway Liveness" monitor (fast lane: no LLM, no network) turns the
    list into a deterministic verdict every 6h and alerts on the first dead
    pathway — and again on recovery;
  * /api/system/status exposes the same snapshot so the frontend can show
    "last written" per pathway.

Verdicts: alive (wrote inside its window), dead (silent past the window),
idle (usage-gated — only writes when the owner talks to Nova — so silence is
not a fault), off (flag false or driving monitor disabled), warming (the
install is younger than the window), unknown (probe failed — missing table,
treated as dead).
"""
from __future__ import annotations

import hashlib

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_TS = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class Pathway:
    """One background writer and the silence it is allowed."""

    name: str
    cadence_hours: float            # silence longer than this is a fault
    table: str = ""                 # writer table (or "" for a file probe)
    time_col: str = ""              # timestamp column proving a write
    where: str = ""                 # extra SQL filter (e.g. source = 'storyline')
    flag: str | None = None         # config.ENABLE_* that gates the pathway
    monitor: str | None = None      # heartbeat monitor that drives the writer
    usage_gated: bool = False       # only writes when the owner uses Nova
    min_rows: int = 1               # rows expected inside the window
    path: str | None = None         # file probe by mtime instead of a table
    describe: str = ""


# The effective window is max(cadence, 2 × driving monitor schedule + 1h), so
# a monitor that legitimately runs every 12h is never flagged for a 25h gap.
PATHWAYS: tuple[Pathway, ...] = (
    Pathway("heartbeat_results", 3, "monitor_results", "created_at",
            describe="any monitor stored a result"),
    Pathway("alert_delivery", 12, "monitors", "last_alert_at",
            describe="an alert reached a channel"),
    Pathway("kg_digest_facts", 12, "kg_facts", "created_at", where="source = 'researched'",
            describe="deep research banked facts from a digest"),
    Pathway("kg_extracted_facts", 12, "kg_facts", "created_at", where="source = 'extracted'",
            describe="post-digest / chat triple extraction"),
    Pathway("digest_independence", 24, "host_cooccurrence", "last_seen",
            describe="source-network (laundering) layer updated"),
    Pathway("storylines", 36, "storyline_events", "created_at",
            flag="ENABLE_STORYLINES", monitor="Storyline Tracker",
            describe="story threads moved"),
    Pathway("dossiers", 36, "dossiers", "updated_at",
            flag="ENABLE_DOSSIERS", monitor="Knowledge Consolidation",
            describe="a dossier was (re)consolidated"),
    # valid_from on a revision row is the SUPERSEDED body's updated_at (days
    # old by construction), so the window is a week: quiet domains revise
    # rarely, but across 90 dossiers some prior version from the last week is
    # always being retired while the trail is alive.
    Pathway("dossier_history", 168, "dossier_revisions", "valid_from",
            flag="ENABLE_DOSSIERS", monitor="Knowledge Consolidation",
            describe="bitemporal revision trail appended"),
    Pathway("question_ledger", 48, "dossier_questions", "last_seen_at",
            flag="ENABLE_DOSSIERS", monitor="Knowledge Consolidation",
            describe="the open-questions frontier was reconciled"),
    Pathway("forecast_minting", 48, "forecasts", "created_at",
            flag="ENABLE_FORECASTS", describe="a falsifiable forecast was minted"),
    Pathway("forecast_resolution", 48, "forecasts", "resolved_at",
            flag="ENABLE_FORECASTS", monitor="Forecast Resolution",
            describe="a due forecast was graded"),
    Pathway("curiosity_intake", 48, "curiosity_queue", "created_at",
            flag="ENABLE_CURIOSITY", describe="an open question entered the queue"),
    Pathway("curiosity_research", 48, "curiosity_queue", "resolved_at",
            flag="ENABLE_CURIOSITY", monitor="Curiosity Research",
            describe="a queued question was researched"),
    Pathway("output_eval", 48, "output_quality_log", "created_at",
            monitor="Output Quality Eval", describe="a digest was graded"),
    Pathway("cross_synthesis", 168, "kg_facts", "created_at", where="source = 'cross_synthesis'",
            monitor="Cross-Monitor Synthesis", describe="a cross-monitor theme was written"),
    Pathway("kg_communities", 48, "kg_communities", "created_at",
            monitor="Dream Consolidation", describe="KG community summaries rebuilt"),
    Pathway("dream_consolidation", 48, "daemon_log", "created_at",
            monitor="Dream Consolidation", describe="the dream/daemon cycle logged"),
    Pathway("maintenance_ran", 48, "monitor_results", "created_at",
            where="status IN ('ok','changed','alert') AND monitor_id IN "
                  "(SELECT id FROM monitors WHERE name = 'System Maintenance')",
            monitor="System Maintenance", describe="daily maintenance completed"),
    # No trust_ledger pathway. Trust was retired from the database on
    # 2026-09-01 (one UPDATE per tool call plus an audit row, gating nothing:
    # can_use() is always True) and is kept in memory for self-awareness. The
    # pathway outlived its writer and read DEAD from that day on, which breaks
    # the "all pathways alive" marker permanently — so the canary delivers one
    # dead verdict, never recovers, and a REAL death after it looks like the
    # same standing complaint. Reviving this entry means reviving the writer.
    Pathway("dedup_metrics", 48, "dedup_decisions", "created_at",
            describe="a dedup decision was recorded"),
    Pathway("kg_retrieval", 48, "kg_facts", "last_retrieved_at",
            describe="KG facts were read into a prompt"),
    Pathway("eval_harness", 48, path="{EVAL_REPORT_PATH}/eval_history.jsonl",
            flag="ENABLE_EVAL_HARNESS", monitor="Quality Eval Harness",
            describe="the nightly eval appended its history"),
    # Usage-gated: these only write when the owner talks to Nova. Silence is
    # reported as idle, never dead.
    Pathway("chat_messages", 168, "messages", "created_at", usage_gated=True,
            describe="a chat turn was stored"),
    Pathway("lessons", 168, "lessons", "created_at", usage_gated=True,
            describe="a correction became a lesson"),
    Pathway("reflexions", 168, "reflexions", "created_at", usage_gated=True,
            describe="a response was critiqued"),
    Pathway("skills", 168, "skills", "created_at", usage_gated=True,
            describe="a skill was induced"),
    Pathway("tool_actions", 168, "action_log", "created_at", usage_gated=True,
            describe="a tool call was logged"),
)

_BY_NAME: dict[str, Pathway] = {p.name: p for p in PATHWAYS}

HEALTHY_MARKER = "all pathways alive"


def get_pathway(name: str) -> Pathway | None:
    return _BY_NAME.get(name)


def _parse_ts(value) -> datetime | None:
    """Tolerant parser for the three timestamp spellings in the DB
    (SQLite datetime('now'), ISO with 'T', ISO with offset/microseconds)."""
    if not value:
        return None
    s = str(value).strip().replace("T", " ")
    for sep in ("+", "Z"):
        s = s.split(sep)[0]
    if "." in s:
        s = s.split(".")[0]
    try:
        return datetime.strptime(s[:19], _TS)
    except ValueError:
        return None


def _monitor_index(db) -> dict[str, dict]:
    try:
        rows = db.fetchall("SELECT name, enabled, schedule_seconds FROM monitors")
    except Exception:
        return {}
    return {
        r["name"]: {"enabled": bool(r["enabled"]),
                    "schedule_seconds": int(r["schedule_seconds"] or 0)}
        for r in rows
    }


def _install_age_hours(db, now: datetime) -> float | None:
    try:
        row = db.fetchone("SELECT MIN(applied_at) AS t FROM schema_version")
    except Exception:
        return None
    t = _parse_ts(row["t"] if row else None)
    return (now - t).total_seconds() / 3600 if t else None


def _probe_table(db, p: Pathway, cutoff: str) -> tuple[str | None, int]:
    sql = (f"SELECT MAX({p.time_col}) AS last_at, "
           f"SUM(CASE WHEN {p.time_col} > ? THEN 1 ELSE 0 END) AS recent "
           f"FROM {p.table} WHERE {p.time_col} IS NOT NULL")
    if p.where:
        sql += f" AND ({p.where})"
    try:
        row = db.fetchone(sql, (cutoff,))
    except Exception as e:
        # Several writers create their table lazily on first use (curiosity,
        # output eval, trust, communities): no table = the writer never ran,
        # which is warm-up on a fresh install and DEAD on an old one — the
        # ordinary verdict rule, not a probe error.
        if "no such table" in str(e).lower():
            return None, 0
        raise
    if not row:
        return None, 0
    return row["last_at"], int(row["recent"] or 0)


def _probe_file(p: Pathway, cfg, cutoff: str) -> tuple[str | None, int]:
    path = (p.path or "").format(
        EVAL_REPORT_PATH=getattr(cfg, "EVAL_REPORT_PATH", "/data/eval_reports"))
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None, 0
    last = datetime.fromtimestamp(mtime, tz=timezone.utc).replace(tzinfo=None).strftime(_TS)
    return last, (1 if last > cutoff else 0)


def effective_window_hours(p: Pathway, monitors: dict[str, dict]) -> float:
    window = float(p.cadence_hours)
    drv = monitors.get(p.monitor) if p.monitor else None
    if drv and drv["schedule_seconds"]:
        window = max(window, 2 * drv["schedule_seconds"] / 3600 + 1)
    return window


def snapshot(db, *, cfg=None, now: datetime | None = None) -> list[dict]:
    """One dict per pathway: name, verdict, last_at, age_hours, window_hours,
    recent_rows, flag, monitor, describe. Pure DB reads — safe on the API path."""
    if cfg is None:
        from app.config import config as cfg  # noqa: N813 — late import keeps this module import-light
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    monitors = _monitor_index(db)
    install_age = _install_age_hours(db, now)
    out: list[dict] = []
    for p in PATHWAYS:
        window = effective_window_hours(p, monitors)
        entry: dict = {
            "name": p.name, "describe": p.describe, "flag": p.flag, "monitor": p.monitor,
            "window_hours": round(window, 1), "last_at": None, "age_hours": None,
            "recent_rows": 0, "verdict": "alive",
        }
        if p.flag and not getattr(cfg, p.flag, True):
            entry["verdict"] = "off"
            out.append(entry)
            continue
        if p.monitor:
            drv = monitors.get(p.monitor)
            if drv is None or not drv["enabled"]:
                entry["verdict"] = "off"
                out.append(entry)
                continue
        cutoff = (now - timedelta(hours=window)).strftime(_TS)
        try:
            if p.path:
                last_at, recent = _probe_file(p, cfg, cutoff)
            else:
                last_at, recent = _probe_table(db, p, cutoff)
        except Exception as e:
            logger.warning("[Pathways] probe failed for %s: %s", p.name, e)
            entry["verdict"] = "unknown"
            entry["error"] = str(e)[:120]
            out.append(entry)
            continue
        last_dt = _parse_ts(last_at)
        entry["last_at"] = last_dt.strftime(_TS) if last_dt else None
        entry["age_hours"] = (round((now - last_dt).total_seconds() / 3600, 1)
                              if last_dt else None)
        entry["recent_rows"] = recent
        if recent >= p.min_rows:
            verdict = "alive"
        elif p.usage_gated:
            verdict = "idle"
        elif install_age is not None and install_age < window:
            verdict = "warming"
        else:
            verdict = "dead"
        entry["verdict"] = verdict
        out.append(entry)
    return out


# Below this share of demanded runs the schedule is not a schedule any more —
# it is a wish list, and the "overdue" signal every priority rule depends on
# degenerates because everything is overdue at once. Measured 2026-09-03:
# 1,646 runs demanded per week, 646 delivered (39%), Curiosity Research at 20%
# of its declared hourly cadence. Reported always; escalated only when severe,
# because an operator may oversubscribe deliberately.
SCHEDULE_PRESSURE_FLOOR = 0.25
# A ratio computed from a handful of runs says nothing. Below this many
# delivered runs the pressure is reported but never escalated (a two-pathway
# probe with one stored result escalated to warning, 2026-09-03).
SCHEDULE_MIN_RUNS = 20


def schedule_pressure(db, *, days: int = 7) -> dict:
    """Delivered vs demanded monitor runs over the window.

    `demanded` is what the enabled cadences add up to; `delivered` is how many
    results were actually stored. A ratio well under 1 means the card cannot
    keep up with the schedule, which is invisible from any single monitor: each
    one merely looks a bit late. Anchored dailies are counted once per day
    rather than by their raw interval.

    The window shrinks to the install's own age, and an install younger than a
    day reports no ratio at all: a two-hour-old system has delivered none of a
    week's runs, which is warm-up rather than saturation (a fresh-install probe
    caught this reporting 0% and escalating, 2026-09-03).
    """
    try:
        monitors = db.fetchall(
            "SELECT id, name, schedule_seconds, check_config FROM monitors WHERE enabled = 1")
        counts = {r["monitor_id"]: int(r["n"]) for r in db.fetchall(
            "SELECT monitor_id, COUNT(*) AS n FROM monitor_results "
            "WHERE created_at > datetime('now', ?) GROUP BY monitor_id", (f"-{days} days",))}
        # counts use the full requested window; `window` below bounds what the
        # cadences may DEMAND, so a young install cannot look starved
    except Exception as e:
        logger.warning("[Pathways] schedule pressure unreadable: %s", e)
        return {"demanded": 0, "delivered": 0, "ratio": None, "starved": []}

    age_h = _install_age_hours(db, datetime.now(timezone.utc).replace(tzinfo=None))
    window = float(days) if age_h is None else max(0.0, min(float(days), age_h / 24.0))
    if window < 1.0:
        return {"demanded": 0, "delivered": 0, "ratio": None, "starved": [],
                "note": "install younger than a day — no schedule history yet"}

    demanded = delivered = 0.0
    per: list[tuple[float, str, int, float]] = []
    for m in monitors:
        sched = max(int(m["schedule_seconds"] or 0), 1)
        want = window * 86400.0 / sched
        if "anchor_hour" in (m["check_config"] or ""):
            want = min(want, window)          # anchored dailies run once a day
        got = counts.get(m["id"], 0)
        demanded += want
        delivered += min(got, want)           # a monitor cannot bank credit
        if want >= 3:
            per.append((got / want, m["name"], got, want))
    per.sort()
    return {
        "demanded": round(demanded),
        "delivered": round(delivered),
        "ratio": round(delivered / demanded, 3) if demanded else None,
        "starved": [{"name": n, "ratio": round(r, 3), "delivered": g, "demanded": round(w)}
                    for r, n, g, w in per[:5]],
    }


CONSTANT_MIN_RUNS = 8
CONSTANT_WINDOW_DAYS = 14


def constant_monitors(db, *, days: int = CONSTANT_WINDOW_DAYS,
                      min_runs: int = CONSTANT_MIN_RUNS) -> list[dict]:
    """Monitors whose output has been byte-identical every run in the window.

    A pathway check asks whether a writer still writes. This asks the next
    question: is what it writes worth the slot? Found on 2026-09-04 by one ad-hoc
    query — Training Job Watch had returned "no training history yet" 101 times
    in 14 days, hourly, for a trainer archived in June, while the schedule was
    delivering 37% of what it demanded. Goal Derivation and Auto-Tool Synthesis
    were the same shape.

    A constant monitor is not necessarily broken: it can be legitimately idle
    for want of input, which is why this is a report and not an alarm. What it
    is, always, is a candidate for a longer cadence.
    """
    try:
        rows = db.fetchall(
            "SELECT m.name AS name, mr.value AS value "
            "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
            "WHERE m.enabled = 1 AND mr.value IS NOT NULL "
            f"AND mr.created_at >= date('now', '-{int(days)} days')",
        )
    except Exception:
        return []
    seen: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for r in rows:
        name = r["name"]
        seen.setdefault(name, set()).add(hashlib.md5(
            (r["value"] or "").encode("utf-8", "replace")).hexdigest())
        counts[name] = counts.get(name, 0) + 1
    out = [{"name": n, "runs": counts[n], "distinct": len(h)}
           for n, h in seen.items()
           if len(h) == 1 and counts[n] >= min_runs]
    out.sort(key=lambda d: -d["runs"])
    return out


THROUGHPUT_STEP_DROP = 0.20      # a fall this deep is a cost change, not weather


def throughput_step(db, *, days: int = 18) -> dict | None:
    """Delivered runs per ACTIVE hour, oldest third against newest third.

    Schedule pressure says the cadences are not being met; it cannot say when
    that started or that anything changed, because it has no before. This does.
    Measured 2026-09-04: runs per active hour fell 6.5 -> 4.1 on 2026-08-28 and
    stayed there for a week with nothing noticing — the app was up ~24 hours a
    day, digests kept their length, their sources and their scores, and every
    monitor merely ran a little less often.

    Per ACTIVE hour, not per day, so a genuine outage reads as downtime rather
    than as a cost regression.
    """
    try:
        rows = db.fetchall(
            "SELECT substr(created_at, 1, 10) AS day, "
            "COUNT(*) AS runs, COUNT(DISTINCT substr(created_at, 12, 2)) AS hrs "
            "FROM monitor_results "
            f"WHERE created_at >= date('now', '-{int(days)} days') "
            "GROUP BY day ORDER BY day")
    except Exception:
        return None
    # Drop today: a partial day always reads as a fall.
    series = [(r["day"], r["runs"] / r["hrs"]) for r in rows if r["hrs"]][:-1]
    if len(series) < 8:
        return None
    k = max(3, len(series) // 3)
    old = sum(v for _d, v in series[:k]) / k
    new = sum(v for _d, v in series[-k:]) / k
    if not old:
        return None
    return {"before": round(old, 2), "after": round(new, 2),
            "change": round((new - old) / old, 3), "days": len(series),
            "stepped_down": (new - old) / old <= -THROUGHPUT_STEP_DROP}


def liveness_report(db, *, cfg=None, now: datetime | None = None
                    ) -> tuple[str, str, dict[str, str | int | float]]:
    """(status, summary, fields) for the Pathway Liveness monitor.

    The healthy summary always carries HEALTHY_MARKER and no drifting numbers,
    so the canary gate can suppress healthy→healthy repeats and deliver only
    the first dead verdict and the recovery edge.
    """
    rows = snapshot(db, cfg=cfg, now=now)
    dead = [r for r in rows if r["verdict"] in ("dead", "unknown")]
    alive = [r for r in rows if r["verdict"] == "alive"]
    idle = [r for r in rows if r["verdict"] == "idle"]
    off = [r for r in rows if r["verdict"] == "off"]
    warming = [r for r in rows if r["verdict"] == "warming"]
    fields: dict[str, str | int | float] = {"alive": len(alive), "idle": len(idle), "off": len(off)}
    press = schedule_pressure(db)

    def _with_pressure(f: dict) -> dict:
        """Schedule fields go LAST: the rendered line is length-capped and the
        dead pathway names are the actionable part (they were being pushed off
        the end, 2026-09-03)."""
        if press.get("ratio") is not None:
            f["schedule"] = (f"{press['ratio']:.0%} of demanded runs delivered "
                             f"({press['delivered']}/{press['demanded']}, 7d)")
            if press["starved"]:
                worst = press["starved"][0]
                f["most_starved"] = f"{worst['name']} at {worst['ratio']:.0%}"
        const = constant_monitors(db)
        if const:
            f["saying_nothing"] = ", ".join(
                f"{c['name']} ({c['runs']}x identical)" for c in const[:3])
        step = throughput_step(db)
        if step and step["stepped_down"]:
            f["throughput_step"] = (
                f"{step['before']:.1f} -> {step['after']:.1f} runs/active hour "
                f"({step['change']:+.0%} over {step['days']}d) — something costs "
                f"more per run than it used to")
        return f
    if warming:
        fields["warming"] = len(warming)
    if dead:
        for r in dead:
            if r["verdict"] == "unknown":
                fields[r["name"]] = f"probe error: {r.get('error', '')}"
            elif r["age_hours"] is None:
                fields[r["name"]] = f"never wrote (window {r['window_hours']:.0f}h)"
            else:
                fields[r["name"]] = f"{r['age_hours']:.0f}h silent (window {r['window_hours']:.0f}h)"
        names = ", ".join(r["name"] for r in dead)
        return "error", f"{len(dead)} pathway(s) DEAD: {names}", _with_pressure(fields)
    tail = f"{len(alive)} writing, {len(idle)} idle, {len(off)} off"
    if warming:
        tail += f", {len(warming)} warming up"
    # Every writer alive but the schedule badly unmet is its own failure: the
    # pathways are working, there is simply not enough capacity to run them as
    # often as declared. Severe cases break the healthy marker so they deliver.
    if (press.get("ratio") is not None
            and press["ratio"] < SCHEDULE_PRESSURE_FLOOR
            and press["delivered"] >= SCHEDULE_MIN_RUNS):
        return ("warning",
                f"pathways all writing but the schedule is not being met: "
                f"{press['ratio']:.0%} of demanded runs delivered", _with_pressure(fields))
    return "info", f"{HEALTHY_MARKER} ({tail})", _with_pressure(fields)
