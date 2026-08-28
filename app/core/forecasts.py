"""Self-scoring forecasts (Monitor Intelligence v2, Phase C).

Beyond reporting the past: Nova makes explicit, falsifiable calls ("watch for X
within N days") and GRADES ITSELF when they come due. Over time this yields a
real track record — and, later, source-credibility weighting.

Storage: `forecasts` table (status open|hit|miss|unresolvable). Generation is
opportunistic (the storyline tracker may emit one); resolution is a scheduled
monitor that grades due forecasts with one grounded LLM call.
"""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlparse

from app.core import llm

logger = logging.getLogger(__name__)

# Clamp a forecast horizon to a sane window (days).
_MIN_DAYS, _MAX_DAYS = 1, 30


def create_forecast(db, claim: str, *, days: int, confidence: float,
                    storyline_key: str = "", source_monitor: str = "") -> int | None:
    """Record a falsifiable forecast resolving in `days`. Returns id or None."""
    claim = (claim or "").strip()
    if len(claim) < 12:
        return None
    days = max(_MIN_DAYS, min(_MAX_DAYS, int(days)))
    confidence = max(0.3, min(0.95, float(confidence)))
    try:
        cur = db.execute(
            "INSERT INTO forecasts (claim, storyline_key, confidence, resolves_at, status, source_monitor) "
            "VALUES (?, ?, ?, datetime('now', ?), 'open', ?)",
            (claim[:500], storyline_key[:80], confidence, f"+{days} days", source_monitor[:80]),
        )
        return cur.lastrowid
    except Exception as e:
        logger.warning("[Forecast] create failed: %s", e)
        return None


# Parses a "FORECAST: <claim> | <N> days | <0.x confidence>" line the LLM emits.
# The trailing 'confidence' word is tolerated: the dossier prompt's own template
# reads '<0.x confidence>', and the 27B dutifully writes the word — the v1 regex
# demanded end-of-line right after the number, so the exact format the prompt
# requested could never parse (found via live probe 2026-08-13).
_FORECAST_LINE = re.compile(
    r"(?im)^\s*FORECAST:\s*(?P<claim>.+?)\s*\|\s*(?P<days>\d+)\s*(?:days?)?\s*"
    r"(?:\|\s*(?P<conf>0?\.\d+|1(?:\.0)?)\s*(?:confidence)?)?\s*$"
)


def parse_and_store_forecast(db, text: str, *, storyline_key: str = "",
                             source_monitor: str = "") -> int | None:
    """Extract a 'FORECAST: ... | N days | 0.x' line from LLM output and store it."""
    m = _FORECAST_LINE.search(text or "")
    if not m:
        return None
    claim = m.group("claim").strip()
    days = int(m.group("days"))
    conf = float(m.group("conf")) if m.group("conf") else 0.55
    return create_forecast(db, claim, days=days, confidence=conf,
                           storyline_key=storyline_key, source_monitor=source_monitor)


def list_due(db, limit: int = 12) -> list[dict]:
    # limit 8→12 + the monitor going 6-hourly (2026-08-27) = 48 gradings/day
    # capacity. Minting runs ~40/day (measured); at the old daily×8 the
    # self-scoring loop would drown as soon as the mint wave matured, grading
    # a 20% oldest-first sample and growing the overdue pile forever.
    """Open forecasts whose resolution date has passed."""
    try:
        return [dict(r) for r in db.fetchall(
            "SELECT * FROM forecasts WHERE status = 'open' AND resolves_at <= datetime('now') "
            "ORDER BY resolves_at ASC LIMIT ?", (limit,),
        )]
    except Exception:
        return []


_RESOLVE_PROMPT = (
    "You made this forecast on {created}, now due:\n\n"
    "  \"{claim}\"\n\n"
    "RECENT EVIDENCE (live web search run today):\n{evidence}\n\n"
    "Judge STRICTLY from the evidence above — do NOT use prior knowledge or "
    "assumptions about what probably happened. A 9B model's memory of recent "
    "events is unreliable; the evidence is the only ground truth here.\n"
    "Reply JSON only: {{\"verdict\": \"hit\"|\"miss\"|\"unresolvable\", "
    "\"reason\": \"one sentence pointing to the evidence\"}}. Use 'unresolvable' "
    "if the evidence does not clearly settle whether the forecast came true."
)


async def _gather_evidence(claim: str, *, max_results: int = 8) -> str:
    """Live web evidence for grading a forecast — recent news, falling back to
    general results. Returns a compact titled-snippet block, or '' if nothing
    usable (in which case the caller must NOT let the model guess). This is what
    makes the self-scoring honest: the verdict traces to today's web, not to the
    frozen-cutoff model's recollection of a near-future event."""
    try:
        from app.tools import native_search
        results = await native_search.search(claim, max_results=max_results, mode="news")
        if len(results) < 3:
            results = list(results) + await native_search.search(
                claim, max_results=max_results, mode="general")
    except Exception as e:
        logger.debug("[Forecast] evidence search failed: %s", e)
        return ""
    seen: set[str] = set()
    lines: list[str] = []
    for r in results:
        url = (getattr(r, "url", "") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = (getattr(r, "title", "") or "").strip()
        snippet = (getattr(r, "snippet", "") or "").strip()
        date = (getattr(r, "published_date", "") or "").strip()
        host = (urlparse(url).netloc or "").replace("www.", "")
        entry = f"- {title}" + (f" ({date})" if date else "") + (f" [{host}]" if host else "")
        if snippet:
            entry += f": {snippet[:300]}"
        lines.append(entry)
        if len(lines) >= 8:
            break
    return "\n".join(lines)


_MAX_RESOLVE_ATTEMPTS = 3


def _bump_attempts(db, fc: dict) -> str:
    """Non-terminal outcome: count the attempt; auto-retire after the cap so a
    permanently-unparseable forecast can't re-process every cycle forever."""
    attempts = int(fc.get("attempts", 0) or 0) + 1
    if attempts >= _MAX_RESOLVE_ATTEMPTS:
        try:
            db.execute(
                "UPDATE forecasts SET status = 'unresolvable', attempts = ?, "
                "resolution = 'auto-retired: could not be graded', resolved_at = datetime('now') WHERE id = ?",
                (attempts, fc["id"]),
            )
        except Exception:
            pass
        return "unresolvable"
    try:
        db.execute("UPDATE forecasts SET attempts = ? WHERE id = ?", (attempts, fc["id"]))
    except Exception:
        pass
    return "open"


async def resolve_one(db, fc: dict) -> str:
    """Grade one due forecast against LIVE web evidence. Returns the verdict.

    The grade is grounded: we web-search the claim first and force the model to
    judge only from those results. If no evidence is found, we DEFER (count the
    attempt) rather than let a frozen-cutoff 9B invent a verdict — an honest
    'couldn't verify yet' beats a fabricated track record."""
    evidence = await _gather_evidence(fc["claim"])
    if not evidence:
        logger.info("[Forecast] no web evidence to grade %r — deferring", fc["claim"][:80])
        return _bump_attempts(db, fc)
    prompt = _RESOLVE_PROMPT.format(
        created=fc.get("created_at", "?"), claim=fc["claim"], evidence=evidence)
    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True, json_prefix="{", max_tokens=160, temperature=0.1,
        )
        data = llm.extract_json_object(raw) if raw else {}
    except Exception as e:
        logger.warning("[Forecast] resolve LLM failed: %s", e)
        return _bump_attempts(db, fc)
    verdict = str(data.get("verdict", "")).lower().strip()
    if verdict not in ("hit", "miss", "unresolvable"):
        return _bump_attempts(db, fc)  # count it; retire if chronically ungradeable
    reason = str(data.get("reason", ""))[:300]
    try:
        db.execute(
            "UPDATE forecasts SET status = ?, resolution = ?, resolved_at = datetime('now') WHERE id = ?",
            (verdict, reason, fc["id"]),
        )
    except Exception as e:
        logger.warning("[Forecast] resolve update failed: %s", e)
    return verdict


def accuracy(db) -> dict:
    """Rolling track record over resolved (hit/miss) forecasts."""
    try:
        rows = db.fetchall("SELECT status FROM forecasts WHERE status IN ('hit','miss')")
    except Exception:
        return {"resolved": 0, "hits": 0, "rate": None}
    hits = sum(1 for r in rows if r["status"] == "hit")
    n = len(rows)
    return {"resolved": n, "hits": hits, "rate": round(hits / n, 2) if n else None}


def calibration(db, *, key_prefix: str | None = None, min_n: int = 1) -> dict | None:
    """Calibration over resolved forecasts: is stated confidence WORTH its
    number? (judgment rung, 2026-08-14). gap > 0 = overconfident (claimed more
    than delivered); Brier score penalizes both miscalibration and hedging.
    `key_prefix` scopes to one storyline/dossier family ('dossier:finance');
    returns None below `min_n` resolved — no lessons from tiny samples."""
    where = "status IN ('hit','miss')"
    args: tuple = ()
    if key_prefix:
        where += " AND storyline_key LIKE ?"
        args = (key_prefix + "%",)
    try:
        rows = db.fetchall(
            f"SELECT confidence, status FROM forecasts WHERE {where}", args)
    except Exception:
        return None
    if len(rows) < max(1, min_n):
        return None
    outcomes = [(float(r["confidence"] or 0.55), 1.0 if r["status"] == "hit" else 0.0)
                for r in rows]
    n = len(outcomes)
    hit_rate = sum(o for _, o in outcomes) / n
    mean_conf = sum(c for c, _ in outcomes) / n
    brier = sum((c - o) ** 2 for c, o in outcomes) / n
    return {"n": n, "hit_rate": round(hit_rate, 3), "mean_conf": round(mean_conf, 3),
            "gap": round(mean_conf - hit_rate, 3), "brier": round(brier, 3)}


async def resolve_due(db) -> str:
    """Resolve all due forecasts; return a digest-ready summary."""
    # These read/aggregate the DB synchronously; off-load so resolve_due (awaited
    # directly by the heartbeat) never blocks the event loop (2026-08-14 audit).
    due = await asyncio.to_thread(list_due, db)
    if not due:
        acc = await asyncio.to_thread(accuracy, db)
        if acc["resolved"]:
            return f"FORECASTS | none due. Track record: {acc['hits']}/{acc['resolved']} hits ({acc['rate']})"
        return "FORECASTS | none due"
    lines = []
    for fc in due:
        v = await resolve_one(db, fc)
        icon = {"hit": "✅", "miss": "❌", "unresolvable": "❓"}.get(v, "•")
        if v in ("hit", "miss", "unresolvable"):
            lines.append(f"{icon} {fc['claim'][:120]}")
    acc = await asyncio.to_thread(accuracy, db)
    if not lines:
        return "FORECASTS | due forecasts not yet judgeable"
    header = f"## 🔮 FORECAST RESOLUTIONS ({acc['hits']}/{acc['resolved']} hits, {acc['rate']})"
    cal = await asyncio.to_thread(calibration, db, min_n=5)
    if cal:
        direction = "overconfident" if cal["gap"] > 0.05 else (
            "underconfident" if cal["gap"] < -0.05 else "well calibrated")
        header += (f"\ncalibration: stated {cal['mean_conf']:.0%} vs delivered "
                   f"{cal['hit_rate']:.0%} — {direction} (Brier {cal['brier']:.2f}, n={cal['n']})")
    return header + "\n" + "\n".join(lines)
