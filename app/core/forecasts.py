"""Self-scoring forecasts (Monitor Intelligence v2, Phase C).

Beyond reporting the past: Nova makes explicit, falsifiable calls ("watch for X
within N days") and GRADES ITSELF when they come due. Over time this yields a
real track record — and, later, source-credibility weighting.

Storage: `forecasts` table (status open|hit|miss|unresolvable). Generation is
opportunistic (the storyline tracker may emit one); resolution is a scheduled
monitor that grades due forecasts with one grounded LLM call.
"""

from __future__ import annotations

import logging
import re

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


# Parses an optional "FORECAST: <claim> | <N> days | <0.x>" line the LLM may emit.
_FORECAST_LINE = re.compile(
    r"(?im)^\s*FORECAST:\s*(?P<claim>.+?)\s*\|\s*(?P<days>\d+)\s*(?:days?)?\s*"
    r"(?:\|\s*(?P<conf>0?\.\d+|1(?:\.0)?))?\s*$"
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


def list_due(db, limit: int = 8) -> list[dict]:
    """Open forecasts whose resolution date has passed."""
    try:
        return [dict(r) for r in db.fetchall(
            "SELECT * FROM forecasts WHERE status = 'open' AND resolves_at <= datetime('now') "
            "ORDER BY resolves_at ASC LIMIT ?", (limit,),
        )]
    except Exception:
        return []


_RESOLVE_PROMPT = (
    "You made this forecast on {created}, due now:\n\n"
    "  \"{claim}\"\n\n"
    "Given what is known as of today, did it come true?\n"
    "Reply JSON only: {{\"verdict\": \"hit\"|\"miss\"|\"unresolvable\", "
    "\"reason\": \"one sentence\"}}. Use 'unresolvable' only if it genuinely "
    "cannot be judged yet."
)


async def resolve_one(db, fc: dict) -> str:
    """Grade one due forecast via an LLM judgment. Returns the verdict."""
    prompt = _RESOLVE_PROMPT.format(created=fc.get("created_at", "?"), claim=fc["claim"])
    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True, json_prefix='{"', max_tokens=160, temperature=0.1,
        )
        data = llm.extract_json_object(raw) if raw else {}
    except Exception as e:
        logger.warning("[Forecast] resolve LLM failed: %s", e)
        return "open"
    verdict = str(data.get("verdict", "")).lower().strip()
    if verdict not in ("hit", "miss", "unresolvable"):
        return "open"  # leave open, try again next cycle
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


async def resolve_due(db) -> str:
    """Resolve all due forecasts; return a digest-ready summary."""
    due = list_due(db)
    if not due:
        acc = accuracy(db)
        if acc["resolved"]:
            return f"FORECASTS | none due. Track record: {acc['hits']}/{acc['resolved']} hits ({acc['rate']})"
        return "FORECASTS | none due"
    lines = []
    for fc in due:
        v = await resolve_one(db, fc)
        icon = {"hit": "✅", "miss": "❌", "unresolvable": "❓"}.get(v, "•")
        if v in ("hit", "miss", "unresolvable"):
            lines.append(f"{icon} {fc['claim'][:120]}")
    acc = accuracy(db)
    if not lines:
        return "FORECASTS | due forecasts not yet judgeable"
    header = f"## 🔮 FORECAST RESOLUTIONS ({acc['hits']}/{acc['resolved']} hits, {acc['rate']})"
    return header + "\n" + "\n".join(lines)
