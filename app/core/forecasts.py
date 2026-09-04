"""Self-scoring forecasts (Monitor Intelligence v2, Phase C).

Beyond reporting the past: Nova makes explicit, falsifiable calls ("watch for X
by <date>") and GRADES ITSELF when they come due. Over time this yields a real
track record and a calibration signal that feeds back into the next mint.

Storage: `forecasts` table (status open|hit|miss|unresolvable|restated).
Generation is opportunistic (dossier consolidation and the storyline tracker
emit a FORECAST line); resolution is a scheduled monitor that grades due
forecasts with one grounded LLM call.

Forecasting discipline (2026-09-01) — measured before the change: Brier 0.253
against a 0.250 coin-flip baseline, stated 0.8 delivering 61 %, 506/723
forecasts clamped to exactly 30 days, 8/99 resolutions citing evidence that
predated the forecast, "guided for $50B" graded as a hit:
  * a forecast carries an explicit resolution DATE (`| resolves YYYY-MM-DD |`,
    horizons up to 365 days) and a deadline stated inside the claim ("by Q4
    2026", "through November 2026") extends the resolution date;
  * evidence published before the forecast was made is dropped before the
    judge sees it, and low-credibility hosts are dropped too;
  * the judge is told the window and that guidance / targets / projections are
    not outcomes; a hit must point at dated evidence inside the window when
    dated evidence exists;
  * near-duplicate open claims in the same family are recorded as `restated`
    (a confidence update), not a second bet;
  * a global calibration record (per stated-confidence bucket) is available to
    every mint prompt (`global_calibration_note`).
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from app.core import llm

logger = logging.getLogger(__name__)

# Clamp a forecast horizon to a sane window (days). 30 -> 365 (2026-09-01):
# the 30-day cap silently rewrote "by Q4 2026" into a month and graded the
# forecast before its own deadline.
_MIN_DAYS, _MAX_DAYS = 1, 365
# Scoring REGIME. A track record is only evidence about the process that
# produced it, and this one changed on 2026-09-02: before that date every
# forecast was clamped to a 30-day horizon, the resolver searched the raw claim
# with no date filter, and the judge was never told its window. Pooling those
# outcomes with new ones answers no question you would want to ask — on
# 2026-09-04 all 103 resolved forecasts came from the old regime, so the 0.60
# hit rate against 0.75 stated confidence was a verdict on code that no longer
# runs. Calibration therefore scores WITHIN a regime, and any future change to
# how forecasts are minted or graded bumps this string so the effect of the
# change is measurable instead of averaged away.
REGIME = "2026-09-02-dated"
REGIME_LEGACY = "pre-2026-09-02"
REGIME_CUTOVER = "2026-09-02 03:00:00"

_DEFAULT_DAYS = 30

# Open claims at or above this token-Jaccard within the same family are the
# same bet restated, not a new forecast (15 clusters of near-duplicates live).
_DUP_JACCARD = 0.6


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _parse_ts(value) -> datetime | None:
    """Parse the SQLite/ISO timestamps this table stores; None when absent."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19] if "T" in s or " " in s else s[:10], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Deadlines stated inside the claim
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_Q_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
    r"\s*(\d{1,2})?(?:st|nd|rd|th)?,?\s*(20\d{2})?\b"
)
_QUARTER_RE = re.compile(r"\bq([1-4])\s*(?:of\s*)?(?:fy\s*)?(20\d{2})\b|\b(20\d{2})\s*q([1-4])\b")
_HALF_RE = re.compile(r"\bh([12])\s*(?:of\s*)?(20\d{2})\b")
_YEAR_RE = re.compile(
    r"\b(by|before|in|through|throughout|during|until|end of|end-|mid-|mid|late|early)"
    r"\s*(?:the\s+)?(?:end\s+of\s+)?(?:year\s+)?(20\d{2})\b"
)
_WITHIN_RE = re.compile(r"\bwithin\s+(?:the\s+next\s+)?(\d{1,3})\s*(day|week|month)s?\b")
_END_OF_YEAR_RE = re.compile(r"\bby\s+(?:the\s+)?end\s+of\s+(?:this\s+|the\s+)?year\b")


def claim_deadline(claim: str, *, now: datetime | None = None) -> datetime | None:
    """The latest explicit deadline stated INSIDE a forecast claim, or None.

    "…by Q4 2026" -> 2026-12-31; "…in 2027" -> 2027-12-31; "…through November
    2026" -> 2026-11-30; "…by March 15, 2027" -> 2027-03-15; "…within 14 days"
    -> now + 14 d; "…by mid-2027" -> 2027-06-30. Dates already in the past are
    ignored (they describe context, not the deadline). When several future
    dates appear the latest one wins — the outcome is knowable only after it.
    """
    text = (claim or "").lower()
    if not text:
        return None
    now = now or _now()
    cands: list[datetime] = []
    for m in _QUARTER_RE.finditer(text):
        if m.group(1):
            q, y = int(m.group(1)), int(m.group(2))
        else:
            y, q = int(m.group(3)), int(m.group(4))
        mo, d = _Q_END[q]
        cands.append(datetime(y, mo, d))
    for m in _HALF_RE.finditer(text):
        h, y = int(m.group(1)), int(m.group(2))
        cands.append(datetime(y, 6, 30) if h == 1 else datetime(y, 12, 31))
    for m in _MONTH_RE.finditer(text):
        name = m.group(1)
        day = int(m.group(2)) if m.group(2) else None
        year = int(m.group(3)) if m.group(3) else None
        # A bare modal 'may' or a bare abbreviation ('sep', 'mar') with neither
        # a day nor a year is prose, not a deadline.
        if day is None and year is None and (name == "may" or len(name) <= 4):
            continue
        mo = _MONTHS[name[:3]]
        if year is None:
            year = now.year
            probe = datetime(year, mo, min(day or calendar.monthrange(year, mo)[1],
                                           calendar.monthrange(year, mo)[1]))
            if probe < now:
                year += 1
        last = calendar.monthrange(year, mo)[1]
        d = min(day or last, last)
        cands.append(datetime(year, mo, d))
    for m in _YEAR_RE.finditer(text):
        kw, y = m.group(1), int(m.group(2))
        if kw.startswith("mid"):
            cands.append(datetime(y, 6, 30))
        elif kw == "early":
            cands.append(datetime(y, 3, 31))
        else:
            cands.append(datetime(y, 12, 31))
    for m in _WITHIN_RE.finditer(text):
        n, unit = int(m.group(1)), m.group(2)
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
        cands.append(now + timedelta(days=days))
    if _END_OF_YEAR_RE.search(text):
        cands.append(datetime(now.year, 12, 31))
    future = [c for c in cands if c > now]
    if not future:
        return None
    return max(future)


# ---------------------------------------------------------------------------
# Minting
# ---------------------------------------------------------------------------

def _find_open_duplicate(db, claim: str, storyline_key: str):
    """An open forecast in the same family whose claim is the same bet."""
    try:
        from app.core.text_utils import normalize_words
    except Exception:
        return None
    words = normalize_words(claim, min_length=2)
    if len(words) < 4:
        return None
    try:
        if storyline_key:
            rows = db.fetchall(
                "SELECT id, claim, resolves_at FROM forecasts WHERE status = 'open' "
                "AND storyline_key = ? ORDER BY id DESC LIMIT 200", (storyline_key,))
        else:
            rows = db.fetchall(
                "SELECT id, claim, resolves_at FROM forecasts WHERE status = 'open' "
                "AND created_at > datetime('now', '-60 days') ORDER BY id DESC LIMIT 400")
    except Exception:
        return None
    best, best_j = None, 0.0
    for r in rows:
        other = normalize_words(r["claim"] or "", min_length=2)
        if not other:
            continue
        j = len(words & other) / len(words | other)
        if j > best_j:
            best, best_j = r, j
    return best if best_j >= _DUP_JACCARD else None


def create_forecast(db, claim: str, *, days: int | None = None, confidence: float,
                    storyline_key: str = "", source_monitor: str = "",
                    resolves_on: str | None = None) -> int | None:
    """Record a falsifiable forecast. Returns the row id or None.

    Resolution date = the explicit `resolves_on` date (or `days` from now,
    default 30), extended to any later deadline stated inside the claim, and
    clamped to [1, 365] days from now. A near-duplicate of an open forecast in
    the same family is stored as `restated` (a confidence update, not a bet).
    """
    from app.core.text_utils import strip_markup
    # The claim is web-searched verbatim when the forecast is graded, so
    # emphasis marks go in as query text (2026-09-04).
    claim = strip_markup(claim)
    if len(claim) < 12:
        return None
    now = _now()
    target: datetime | None = None
    if resolves_on:
        target = _parse_ts(resolves_on)
    if target is None:
        d = _DEFAULT_DAYS if days is None else int(days)
        d = max(_MIN_DAYS, min(_MAX_DAYS, d))
        target = now + timedelta(days=d)
    deadline = claim_deadline(claim, now=now)
    if deadline is not None and deadline > target:
        target = deadline
    lo, hi = now + timedelta(days=_MIN_DAYS), now + timedelta(days=_MAX_DAYS)
    target = max(lo, min(hi, target))
    confidence = max(0.3, min(0.95, float(confidence)))
    dup = _find_open_duplicate(db, claim, storyline_key)
    try:
        if dup is not None:
            cur = db.execute(
                "INSERT INTO forecasts (claim, storyline_key, confidence, resolves_at, status, "
                "resolution, source_monitor, regime) VALUES (?, ?, ?, ?, 'restated', ?, ?, ?)",
                (claim[:500], storyline_key[:80], confidence, dup["resolves_at"],
                 f"restates #{dup['id']}", source_monitor[:80], REGIME),
            )
            logger.info("[Forecast] restated #%d (%.2f): %s", dup["id"], confidence, claim[:80])
            return cur.lastrowid
        cur = db.execute(
            "INSERT INTO forecasts (claim, storyline_key, confidence, resolves_at, status, "
            "source_monitor, regime) VALUES (?, ?, ?, ?, 'open', ?, ?)",
            (claim[:500], storyline_key[:80], confidence,
             target.strftime("%Y-%m-%d %H:%M:%S"), source_monitor[:80], REGIME),
        )
        logger.info("[Forecast] minted #%s resolves %s (%.2f): %s",
                    cur.lastrowid, target.date().isoformat(), confidence, claim[:80])
        return cur.lastrowid
    except Exception as e:
        logger.warning("[Forecast] create failed: %s", e)
        return None


# Parses a "FORECAST: <claim> | resolves YYYY-MM-DD | <0.x confidence>" line
# (2026-09-01) or the legacy "<N> days" horizon. The trailing 'confidence'
# word is tolerated: the dossier prompt's own template reads '<0.x
# confidence>', and the 27B dutifully writes the word.
_FORECAST_LINE = re.compile(
    r"(?im)^\s*FORECAST:\s*(?P<claim>.+?)\s*\|\s*"
    r"(?:(?:resolves|resolve|resolution|by|due|on)\s*:?\s*)?"
    r"(?:(?P<date>20\d{2}-\d{2}-\d{2})|(?P<days>\d+)\s*(?:days?|d)?)\s*"
    r"(?:\|\s*(?P<conf>0?\.\d+|1(?:\.0)?)\s*(?:confidence)?)?\s*$"
)


def parse_and_store_forecast(db, text: str, *, storyline_key: str = "",
                             source_monitor: str = "") -> int | None:
    """Extract a FORECAST line from LLM output and store it."""
    m = _FORECAST_LINE.search(text or "")
    if not m:
        return None
    claim = m.group("claim").strip()
    conf = float(m.group("conf")) if m.group("conf") else 0.55
    date = m.group("date")
    days = int(m.group("days")) if m.group("days") else None
    return create_forecast(db, claim, days=days, resolves_on=date, confidence=conf,
                           storyline_key=storyline_key, source_monitor=source_monitor)


def list_due(db, limit: int = 12) -> list[dict]:
    # limit 8→12 + the monitor going 6-hourly (2026-08-27) = 48 gradings/day
    # capacity. Minting runs ~35/day (measured).
    """Open forecasts whose resolution date has passed."""
    try:
        return [dict(r) for r in db.fetchall(
            "SELECT * FROM forecasts WHERE status = 'open' AND resolves_at <= datetime('now') "
            "ORDER BY resolves_at ASC LIMIT ?", (limit,),
        )]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

_RESOLVE_PROMPT = (
    "You made this forecast on {created}; it resolves on {due}; today is {today}:\n\n"
    "  \"{claim}\"\n\n"
    "RECENT EVIDENCE (live web search run today; each line carries the article's "
    "date where known):\n{evidence}\n\n"
    "Judge STRICTLY from the evidence above — do NOT use prior knowledge or "
    "assumptions about what probably happened. Rules:\n"
    "- Only events that actually OCCURRED between {created} and {today} settle the forecast.\n"
    "- Guidance, targets, projections, plans, schedules, or other people's forecasts "
    "are NOT outcomes.\n"
    "- Evidence dated before {created} cannot confirm the forecast — it describes the "
    "situation the forecast was made from.\n"
    "- If the evidence does not clearly settle whether the forecast came true, reply "
    "unresolvable.\n"
    "Reply JSON only: {{\"verdict\": \"hit\"|\"miss\"|\"unresolvable\", "
    "\"evidence_date\": \"YYYY-MM-DD\" or null, "
    "\"reason\": \"one sentence pointing to the evidence\"}}."
)

_RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["hit", "miss", "unresolvable"]},
        "evidence_date": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

_REL_DATE_RE = re.compile(r"(\d+)\s*(minute|hour|day|week|month)s?\s+ago", re.I)


def _parse_evidence_date(value: str) -> datetime | None:
    """Best-effort date from a search engine's 'published' field."""
    if not value:
        return None
    s = str(value).strip()
    m = _REL_DATE_RE.search(s)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        delta = {"minute": timedelta(minutes=n), "hour": timedelta(hours=n),
                 "day": timedelta(days=n), "week": timedelta(weeks=n),
                 "month": timedelta(days=30 * n)}[unit]
        return _now() - delta
    ts = _parse_ts(s)
    if ts:
        return ts
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[:20].strip(), fmt)
        except ValueError:
            continue
    return None


def _filter_evidence(results, created_at) -> tuple[list, int, int]:
    """Drop results published before the forecast and low-credibility hosts.
    Returns (kept, dropped_old, dropped_junk)."""
    created = _parse_ts(created_at)
    try:
        from app.core.source_authority import authority
    except Exception:  # pragma: no cover
        authority = None
    kept, old, junk = [], 0, 0
    for r in results:
        url = (getattr(r, "url", "") or "").strip()
        host = (urlparse(url).netloc or "").replace("www.", "") if url else ""
        if authority is not None and host and authority(host) < 0.3:
            junk += 1
            continue
        pub = _parse_evidence_date(getattr(r, "published_date", "") or "")
        if created is not None and pub is not None and pub.date() < created.date():
            old += 1
            continue
        kept.append((r, pub))
    return kept, old, junk


async def _gather_evidence(claim: str, *, created_at=None, max_results: int = 8) -> str:
    """Live web evidence for grading a forecast — recent news, falling back to
    general results, minus anything published before the forecast was made and
    minus low-credibility hosts. Returns a compact titled-snippet block, or ''
    if nothing usable (in which case the caller must NOT let the model guess)."""
    try:
        from app.tools import native_search
        results = await native_search.search(claim, max_results=max_results, mode="news")
        if len(results) < 3:
            results = list(results) + await native_search.search(
                claim, max_results=max_results, mode="general")
    except Exception as e:
        logger.debug("[Forecast] evidence search failed: %s", e)
        return ""
    kept, old, junk = _filter_evidence(results, created_at)
    if old or junk:
        logger.info("[Forecast] evidence filter: kept %d, dropped %d pre-forecast, %d low-credibility",
                    len(kept), old, junk)
    seen: set[str] = set()
    lines: list[str] = []
    for r, pub in kept:
        url = (getattr(r, "url", "") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = (getattr(r, "title", "") or "").strip()
        snippet = (getattr(r, "snippet", "") or "").strip()
        host = (urlparse(url).netloc or "").replace("www.", "")
        date = pub.date().isoformat() if pub else "undated"
        entry = f"- {title} ({date})" + (f" [{host}]" if host else "")
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


def _hit_is_grounded(verdict: str, evidence_date, created_at, evidence: str) -> tuple[bool, str]:
    """A HIT must point at dated evidence inside the window whenever the
    evidence block carries dates at all (a snippet dated before the forecast
    is the pre-window-confirmation failure the audit measured at 8/99)."""
    if verdict != "hit":
        return True, ""
    created = _parse_ts(created_at)
    dated_lines = [ln for ln in (evidence or "").splitlines()
                   if re.search(r"\((20\d{2}-\d{2}-\d{2})\)", ln)]
    ev = _parse_ts(str(evidence_date)) if evidence_date else None
    if ev is None:
        if dated_lines:
            return False, "hit without an evidence date although dated evidence was available"
        return True, "undated evidence"
    if created is not None and ev.date() < created.date():
        return False, f"evidence dated {ev.date().isoformat()} predates the forecast"
    return True, ""


async def resolve_one(db, fc: dict) -> str:
    """Grade one due forecast against LIVE web evidence. Returns the verdict.

    The grade is grounded: we web-search the claim first and force the model to
    judge only from those results. If no evidence is found, we DEFER (count the
    attempt) rather than let a frozen-cutoff 9B invent a verdict — an honest
    'couldn't verify yet' beats a fabricated track record."""
    evidence = await _gather_evidence(fc["claim"], created_at=fc.get("created_at"))
    if not evidence:
        logger.info("[Forecast] no web evidence to grade %r — deferring", fc["claim"][:80])
        return _bump_attempts(db, fc)
    created = str(fc.get("created_at", "?"))[:10]
    due = str(fc.get("resolves_at", "?"))[:10]
    prompt = _RESOLVE_PROMPT.format(
        created=created, due=due, today=_now().date().isoformat(),
        claim=fc["claim"], evidence=evidence)
    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True, json_schema=_RESOLVE_SCHEMA, max_tokens=220, temperature=0.1,
        )
        data = llm.extract_json_object(raw) if raw else {}
    except Exception as e:
        logger.warning("[Forecast] resolve LLM failed: %s", e)
        return _bump_attempts(db, fc)
    if not isinstance(data, dict):
        return _bump_attempts(db, fc)
    verdict = str(data.get("verdict", "")).lower().strip()
    if verdict not in ("hit", "miss", "unresolvable"):
        return _bump_attempts(db, fc)  # count it; retire if chronically ungradeable
    evidence_date = data.get("evidence_date")
    ok, why = _hit_is_grounded(verdict, evidence_date, fc.get("created_at"), evidence)
    if not ok:
        logger.info("[Forecast] #%s hit rejected (%s) — deferring", fc.get("id"), why)
        return _bump_attempts(db, fc)
    reason = str(data.get("reason", ""))[:300]
    stamp = ""
    if evidence_date:
        ev = _parse_ts(str(evidence_date))
        if ev:
            stamp = f"[{ev.date().isoformat()}] "
    elif why:
        stamp = f"[{why}] "
    try:
        # to_thread (2026-08-29): sync UPDATE on the event-loop thread from an
        # async resolver — takes the write lock on the loop (54h-freeze class).
        await asyncio.to_thread(
            db.execute,
            "UPDATE forecasts SET status = ?, resolution = ?, resolved_at = datetime('now') WHERE id = ?",
            (verdict, (stamp + reason)[:300], fc["id"]),
        )
    except Exception as e:
        logger.warning("[Forecast] resolve update failed: %s", e)
    return verdict


# ---------------------------------------------------------------------------
# Track record and calibration
# ---------------------------------------------------------------------------

def _regime_clause(regime: str | None) -> tuple[str, tuple]:
    """SQL fragment scoping a query to one scoring regime.

    Rows minted before the column existed carry NULL, so the legacy regime is
    matched by date as well as by value — a backfill is not required for the
    numbers to be right.
    """
    if regime is None:
        return "", ()
    if regime == REGIME_LEGACY:
        return (" AND (regime = ? OR (regime IS NULL AND created_at < ?))",
                (REGIME_LEGACY, REGIME_CUTOVER))
    return (" AND (regime = ? OR (regime IS NULL AND created_at >= ?))",
            (regime, REGIME_CUTOVER))


def accuracy(db, *, regime: str | None = REGIME) -> dict:
    """Rolling track record over resolved (hit/miss) forecasts.

    Scoped to the current scoring regime by default (see REGIME): a record
    produced by code that no longer runs is history, not a measurement of how
    Nova forecasts now. Pass regime=None to pool everything.
    """
    clause, args = _regime_clause(regime)
    try:
        rows = db.fetchall(
            "SELECT status FROM forecasts WHERE status IN ('hit','miss')" + clause, args)
    except Exception:
        return {"resolved": 0, "hits": 0, "rate": None, "regime": regime}
    hits = sum(1 for r in rows if r["status"] == "hit")
    n = len(rows)
    return {"resolved": n, "hits": hits,
            "rate": round(hits / n, 2) if n else None, "regime": regime}


def calibration(db, *, key_prefix: str | None = None, min_n: int = 1,
                regime: str | None = REGIME) -> dict | None:
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
    clause, rargs = _regime_clause(regime)
    where += clause
    args = args + rargs
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


def calibration_buckets(db, *, min_bucket: int = 5) -> list[tuple[float, float, int]]:
    """(stated confidence bucket, delivered hit rate, n) for buckets with
    at least `min_bucket` resolved forecasts."""
    try:
        rows = db.fetchall(
            "SELECT confidence, status FROM forecasts WHERE status IN ('hit','miss')")
    except Exception:
        return []
    buckets: dict[float, list[int]] = {}
    for r in rows:
        b = round(float(r["confidence"] or 0.55), 1)
        hits, n = buckets.setdefault(b, [0, 0])
        buckets[b][0] = hits + (1 if r["status"] == "hit" else 0)
        buckets[b][1] = n + 1
    out = [(b, h / n, n) for b, (h, n) in sorted(buckets.items()) if n >= min_bucket]
    return out


def global_calibration_note(db, *, min_n: int = 20) -> str | None:
    """A short record of how Nova's stated confidence has actually delivered,
    for every mint prompt (dossier and storyline). The per-family note only
    reaches families with ≥5 resolutions (4 of 29 live); everything else kept
    minting at 0.7/0.8 with a Brier at coin-flip. None below `min_n`."""
    cal = calibration(db, min_n=min_n)
    if not cal:
        return None
    parts = [f"{b:.1f}->{rate:.0%} (n={n})" for b, rate, n in calibration_buckets(db)]
    bucket_text = ", ".join(parts) if parts else "not enough per-bucket data"
    return (
        f"CALIBRATION RECORD (all Nova forecasts, n={cal['n']}): stated confidence -> "
        f"delivered hit rate: {bucket_text}. Overall stated {cal['mean_conf']:.0%} vs "
        f"delivered {cal['hit_rate']:.0%} (Brier {cal['brier']:.2f}; 0.25 = coin flip). "
        f"State the confidence you would actually bet at: where 0.8 has delivered "
        f"~{next((rate for b, rate, _ in calibration_buckets(db) if b == 0.8), cal['hit_rate']):.0%}, "
        f"a claim you feel is 0.8 belongs at that number. Prefer a dated, checkable "
        f"claim at 0.6 over a vague one at 0.9."
    )


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
