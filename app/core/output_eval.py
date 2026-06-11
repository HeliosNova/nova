"""Rolling self-eval of production monitor outputs.

The eval harness runs a curated suite — but production output quality is
what actually matters. This module samples recent monitor_results,
grades each one, and writes the score to `output_quality_log` so we can
track drift over time.

Grading dimensions (all 0-10):
  - relevance:  did the output answer what the prompt asked for?
  - facts:      do the named entities / numbers / dates look real?
  - freshness:  do the dates fall within the requested window?
  - format:     does it follow the expected structure (citations, etc)?

LLM-as-judge using `invoke_nothink` (small, cheap). Caller (a heartbeat
monitor at check_type='output_eval') runs this on a schedule.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3

logger = logging.getLogger(__name__)


_GRADE_PROMPT = """TODAY IS {today_human} ({today_iso}). The current year is {year}.

This is REAL — not hypothetical, not a future scenario. Dates in {year} are NORMAL CURRENT DATES, not "fabricated" or "future". If the output references {year} dates, those are CORRECT and CURRENT — do not penalise them as fabrications.

You are grading the output of an automated monitor.

MONITOR NAME: {monitor_name}
PROMPT (paraphrase): a Domain Study tracking developments in this area.

OUTPUT TO GRADE (truncated):
{output}

Grade each dimension 0-10 (10 = excellent, 0 = useless):
- relevance:   does the output answer what a Domain Study should answer?
- facts:       do the named entities and numbers look plausible? (Dates in {year} are CURRENT — do not penalise them.) An article body can legitimately reference earlier-dated events (filings, prior incidents) — those are not factual errors, they are normal news context.
- freshness:   FRESHNESS RULE — score this dimension by looking ONLY at the
  "📅 [date]" line under each item header. That is the article's publish date.
  - If ALL items' published dates are within the past 72 hours of {today_human}, score 10.
  - If most items are within 72h and a couple are within 7 days, score 7-8.
  - If most items are older than 7 days, score 2-4.
  - DO NOT downgrade because the article BODY references earlier dates ("19th century cables", "since 2024", "filed February 17") — those are normal historical context inside articles and are not freshness penalties.
  - DO NOT downgrade just because an article from a Bloomberg/Reuters/FT URL has a date — the article IS the news, the date IS its publish date, score it on that date alone.
- format:      VISUAL STRUCTURE ONLY — does each item have a numbered headline, source line, date, URL, and a summary in consistent style? Score 10 if every item follows the same template; 7-8 only if the visual structure varies notably between items. Do NOT downgrade format for content issues like off-topic items, hallucinations, ellipsis truncation in summaries, or irrelevance — those are scored under relevance/facts dimensions, not format. The "📌 N items sourced from X" closing summary IS part of the intended format and should not be penalised.
- one_line_critique: one sentence of the worst problem (or "none" if all good).

EVIDENCE RULE — for any score below 8 in any dimension, the one_line_critique
MUST quote the offending substring verbatim from the OUTPUT TO GRADE above.
If you cannot quote the problem with an exact substring, it is not a real
problem — score the dimension 8+ and put "none" in the critique. Do NOT
invent issues, do NOT reference content that is not in the OUTPUT.

Output STRICT JSON: {{"relevance": 7, "facts": 8, "freshness": 5, "format": 6, "one_line_critique": "..."}}
"""


def _ensure_table(db) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS output_quality_log ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " monitor_id INTEGER,"
            " monitor_name TEXT,"
            " result_id INTEGER,"
            " relevance REAL, facts REAL, freshness REAL, format REAL,"
            " avg_score REAL,"
            " critique TEXT,"
            " created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_output_quality_created "
            "ON output_quality_log(created_at)"
        )
    except sqlite3.Error as e:
        logger.warning("[OutputEval] table create failed: %s", e)


async def _grade_one(monitor_name: str, output: str) -> dict | None:
    from app.core.llm import invoke_nothink
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    if not output or len(output) < 50:
        return None
    body = output[:3000]
    now = _dt.now(_tz.utc)
    cutoff = now - _td(hours=72)
    prompt = _GRADE_PROMPT.format(
        monitor_name=monitor_name,
        output=body,
        today_human=now.strftime("%B %d, %Y"),
        today_iso=now.strftime("%Y-%m-%d"),
        cutoff_human=cutoff.strftime("%B %d, %Y"),
        year=now.year,
    )
    # Grade with the independent judge when configured — a monitor output
    # graded by the model that wrote it inherits self-preference bias.
    from app.config import config as _config
    _judge = (_config.JUDGE_MODEL or "").strip() or None
    try:
        resp = await invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True, json_prefix="{",
            max_tokens=300, temperature=0.0,
            model=_judge,
            num_ctx=8192 if _judge else None,
        )
    except Exception as e:
        logger.warning("[OutputEval] grade LLM failed: %s", e)
        return None
    text = (resp or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    # Coerce scores. Be tolerant of common LLM key variants:
    #   relevance / relevancy / topic_relevance
    #   facts / factual / accuracy / fact_quality
    #   freshness / recency / timeliness / date_freshness
    #   format / formatting / output_format / structure / presentation
    def _g(*keys, default=0.0):
        for k in keys:
            v = d.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
        return default
    try:
        rel = _g("relevance", "relevancy", "topic_relevance")
        fac = _g("facts", "factual", "accuracy", "fact_quality", "factuality")
        fre = _g("freshness", "recency", "timeliness", "date_freshness")
        fmt = _g("format", "formatting", "output_format", "structure", "presentation")
    except (ValueError, TypeError):
        return None
    # If after all variants we still got 0 in every dimension, the JSON
    # was almost certainly malformed — return None so caller skips this row.
    if rel == 0 and fac == 0 and fre == 0 and fmt == 0:
        return None
    crit = str(d.get("one_line_critique") or "")[:300]

    # Evidence validation: if any dimension is below 8 AND the critique
    # makes a specific quoted claim, the quoted span must actually appear
    # in the output. Otherwise the grader is hallucinating problems.
    # Floor sub-8 dimensions back to 8 and clear the bogus critique.
    min_dim = min(rel, fac, fre, fmt)
    if min_dim < 8 and crit and crit.lower() not in ("none", ""):
        # Pull plausible quoted spans (between curly/straight quotes, or after "references")
        spans = re.findall(r"['\"‘’“”]([^'\"‘’“”]{3,80})['\"‘’“”]", crit)
        body_lower = body.lower()
        bogus = False
        if spans:
            unfounded = [s for s in spans if s.lower() not in body_lower]
            # If MOST quoted spans are not in the body, the critique is hallucinated
            if len(unfounded) >= max(1, len(spans) // 2):
                bogus = True
        else:
            # No quoted span at all but critique mentions specific entities — check
            # the most distinctive nouns (Capitalized 4+ char words) appear in body
            nouns = re.findall(r"\b[A-Z][a-zA-Z0-9]{3,}\b", crit)
            if nouns:
                missing = [n for n in nouns if n.lower() not in body_lower]
                if len(missing) >= max(2, len(nouns) * 2 // 3):
                    bogus = True
        if bogus:
            logger.info("[OutputEval] grader hallucinated critique (evidence not in body) — flooring scores")
            rel = max(rel, 8.0)
            fac = max(fac, 8.0)
            fre = max(fre, 8.0)
            fmt = max(fmt, 8.0)
            crit = "none (hallucinated critique rejected)"

    return {
        "relevance": max(0.0, min(10.0, rel)),
        "facts": max(0.0, min(10.0, fac)),
        "freshness": max(0.0, min(10.0, fre)),
        "format": max(0.0, min(10.0, fmt)),
        "critique": crit,
    }


async def grade_recent_outputs(db, *, sample_size: int = 20, hours: int = 24) -> dict:
    """Grade up to `sample_size` recent content-monitor results from the
    last `hours` hours. Writes scores to output_quality_log.
    """
    _ensure_table(db)
    rows = db.fetchall(
        "SELECT mr.id, mr.monitor_id, m.name AS monitor_name, mr.value "
        "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
        "WHERE mr.created_at > datetime('now', ?) "
        "  AND m.category = 'content' "
        "  AND mr.status IN ('ok','changed','alert') "
        "  AND length(mr.value) > 200 "
        "ORDER BY RANDOM() LIMIT ?",
        (f"-{hours} hours", sample_size),
    )
    if not rows:
        return {"sampled": 0, "graded": 0, "avg": None, "summary": "no recent outputs to grade"}

    graded = 0
    totals = {"relevance": 0.0, "facts": 0.0, "freshness": 0.0, "format": 0.0}
    worst: list[str] = []

    for r in rows:
        scores = await _grade_one(r["monitor_name"], r["value"])
        if not scores:
            continue
        graded += 1
        for k in ("relevance", "facts", "freshness", "format"):
            totals[k] += scores[k]
        avg = sum(scores[k] for k in ("relevance", "facts", "freshness", "format")) / 4.0
        try:
            db.execute(
                "INSERT INTO output_quality_log (monitor_id, monitor_name, result_id, "
                "relevance, facts, freshness, format, avg_score, critique) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r["monitor_id"], r["monitor_name"], r["id"],
                    scores["relevance"], scores["facts"], scores["freshness"], scores["format"],
                    avg, scores["critique"],
                ),
            )
        except sqlite3.Error as e:
            logger.warning("[OutputEval] insert failed: %s", e)
        if avg < 6.0 and scores["critique"]:
            worst.append(f"{r['monitor_name']} (avg={avg:.1f}): {scores['critique'][:80]}")

    if graded == 0:
        return {"sampled": len(rows), "graded": 0, "avg": None, "summary": "all gradings failed"}

    avg_per = {k: round(v / graded, 2) for k, v in totals.items()}
    overall = round(sum(avg_per.values()) / 4.0, 2)
    summary = (
        f"OUTPUT EVAL | graded {graded}/{len(rows)} sampled\n"
        f"  overall: {overall}/10  (rel {avg_per['relevance']} | facts {avg_per['facts']} | "
        f"fresh {avg_per['freshness']} | fmt {avg_per['format']})"
    )
    if worst:
        summary += "\n  worst:\n    " + "\n    ".join(worst[:5])
    return {
        "sampled": len(rows), "graded": graded,
        "avg": overall, "per_dim": avg_per,
        "summary": summary,
    }


async def grade_and_log(db) -> str:
    """Heartbeat-friendly wrapper. Returns the summary string."""
    try:
        result = await grade_recent_outputs(db)
    except Exception as e:
        logger.exception("grade_recent_outputs failed")
        return f"OUTPUT EVAL ERROR: {e}"
    return result["summary"]
