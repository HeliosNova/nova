"""Resolve cross-dossier tensions from Nova's OWN records (plan 2026-10-09, C1).

A tension is two current dossiers asserting materially different values for
what reads as the same quantity ("Finance says CPI 3.4% but Economics says
2.9%"). Web research cannot settle which of Nova's records is right — the
curiosity path scored 0 resolved of 26 such items — so this resolver never
searches. It reads the last three stored digests of each domain, locates the
sentence(s) carrying each figure, hands ONLY those dated spans to the judge
(temperature 0) and asks: same quantity? which is current? A second,
independent judge sees the same spans and the verdict and must agree before
the resolution is admitted; otherwise the tension stays open with both
reasons stored. (MemDream/MemCodex: consolidation from one's own spans.)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

VERDICTS = ("a", "b", "both", "unresolvable")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\-•\d\"'(])")


@dataclass
class Tension:
    title_a: str
    dkey_a: str
    val_a: float
    head_a: str
    title_b: str
    dkey_b: str
    val_b: float
    head_b: str
    unit: str
    text: str = ""
    spans_a: list[dict] = field(default_factory=list)
    spans_b: list[dict] = field(default_factory=list)


def _figure_forms(val: float, unit: str) -> list[str]:
    """Spellings a digest may use for a value: '5.501%', '5.501 percent',
    '2 billion', '2bn', '2,000,000,000', '$2 billion'."""
    u = (unit or "").lower()
    g = f"{val:g}"
    forms = [g]
    if "." not in g and abs(val) >= 1000:
        forms.append(f"{val:,.0f}")
    out: list[str] = []
    for f in forms:
        if u == "%":
            out += [f"{f}%", f"{f} %", f"{f} percent", f"{f} per cent"]
        elif u == "bp":
            out += [f"{f}bp", f"{f} bp", f"{f} bps", f"{f} basis points"]
        elif u in ("million", "billion", "trillion"):
            short = {"million": "mn", "billion": "bn", "trillion": "tn"}[u]
            out += [f"{f} {u}", f"{f}{u}", f"{f}{short}", f"{f} {short}", f"{f}{short[0].upper()}"]
        else:
            out.append(f)
    return out


def locate_spans(digests: list[tuple[str, str]], val: float, unit: str, *, limit: int = 3) -> list[dict]:
    """Sentences in `digests` [(date, text)] that carry the figure — exact form
    first. Newest digest first; at most `limit` spans."""
    forms = _figure_forms(val, unit)
    found: list[dict] = []
    for date, text in sorted(digests, key=lambda d: d[0], reverse=True):
        for sent in _SENT_RE.split((text or "").replace("\n", " ")):
            s = sent.strip()
            if not s:
                continue
            low = s.lower()
            if any(f.lower() in low for f in forms):
                found.append({"date": (date or "")[:10], "sentence": s[:400]})
                if len(found) >= limit:
                    return found
    return found


def domain_digests(db, title: str, *, limit: int = 3) -> list[tuple[str, str]]:
    """Last `limit` stored digests of the domain monitor behind a dossier title."""
    from app.core.dossiers import _DOMAIN_PREFIX
    name = title if title.startswith(_DOMAIN_PREFIX) else f"{_DOMAIN_PREFIX}{title}"
    try:
        rows = db.fetchall(
            "SELECT mr.value AS value, mr.created_at AS created_at "
            "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
            "WHERE m.name = ? AND mr.value IS NOT NULL AND length(mr.value) > 400 "
            "ORDER BY mr.created_at DESC LIMIT ?",
            (name, limit),
        )
    except Exception as e:
        logger.debug("[Tensions] digests unavailable for %r: %s", title, e)
        return []
    return [(r["created_at"] or "", r["value"] or "") for r in rows]


def _span_pack(domain: str, spans: list[dict]) -> str:
    if not spans:
        return f"[{domain}] (no digest sentence carries this figure)"
    return "\n".join(f"[{domain} · {s['date'] or 'n/d'}] {s['sentence']}" for s in spans)


_JUDGE_PROMPT = (
    "Today is {today}. Two of my own stored briefings state different figures.\n"
    "A: {title_a} says {fig_a} — \"{head_a}\"\n"
    "B: {title_b} says {fig_b} — \"{head_b}\"\n\n"
    "Below are the ONLY sentences from the stored briefings that carry each figure, "
    "with the briefing's date. Use nothing else.\n\n{pack_a}\n\n{pack_b}\n\n"
    "Decide: are A and B the same quantity? If NOT (different metric, period, "
    "instrument, region), verdict is \"both\". If they are the same quantity, which "
    "figure is current — by date, and by wording (revised, updated, latest)? Answer "
    "\"a\" or \"b\"; \"unresolvable\" only when the spans cannot tell.\n"
    "Return JSON: {{\"verdict\": \"a|b|both|unresolvable\", \"reasoning\": \"at most 40 words\", "
    "\"winner_date\": \"YYYY-MM-DD or null\"}}"
)

_VERIFIER_PROMPT = (
    "A judge read the sentences below and ruled: verdict={verdict} — {reasoning}\n\n"
    "{pack_a}\n\n{pack_b}\n\n"
    "A: {title_a} says {fig_a}. B: {title_b} says {fig_b}.\n"
    "Is that ruling justified by these sentences alone? Answer yes or no, then one sentence why."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "reasoning": {"type": "string"},
        "winner_date": {"type": ["string", "null"]},
    },
    "required": ["verdict", "reasoning"],
}


async def resolve_tension(db, t: Tension, *, judge_model: str | None = None,
                          verifier_model: str | None = None) -> dict:
    """Judge + independent verifier over the tension's own spans. Stores a
    tension_resolutions row either way; `admitted` is True only when the
    verifier agrees with a decisive verdict (a or b)."""
    from app.core import llm
    from app.core.dossiers import _fmt_qty

    t.spans_a = locate_spans(domain_digests(db, t.title_a), t.val_a, t.unit)
    t.spans_b = locate_spans(domain_digests(db, t.title_b), t.val_b, t.unit)
    # Phase 4: when the digests have aged out, the spans kept at consolidation
    # (fact_spans) still carry the sentence and its date.
    try:
        from app.core.fact_spans import spans_for
        if not t.spans_a:
            t.spans_a = spans_for(db, t.dkey_a, t.val_a, t.unit)
        if not t.spans_b:
            t.spans_b = spans_for(db, t.dkey_b, t.val_b, t.unit)
    except Exception as e:
        logger.debug("[Tensions] fact_spans fallback skipped: %s", e)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fig_a, fig_b = _fmt_qty(t.val_a, t.unit), _fmt_qty(t.val_b, t.unit)
    pack_a, pack_b = _span_pack(t.title_a, t.spans_a), _span_pack(t.title_b, t.spans_b)
    verdict, reasoning, winner_date = "unresolvable", "", None
    verifier_agrees = False
    verifier_reason = ""
    if not t.spans_a and not t.spans_b:
        # Replay 2026-10-11 (56 stored tensions): 30 had no digest sentence
        # carrying either figure — the digests had aged past the 3-per-domain
        # window — and every one of those cost two judge calls to reach
        # "unresolvable". No spans means no evidence: answer without asking.
        reasoning = "no stored digest sentence carries either figure"
        return _store(db, t, fig_a, fig_b, pack_a, pack_b, verdict, reasoning, None,
                      judge_model, verifier_model, False, "")
    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": _JUDGE_PROMPT.format(
                today=today, title_a=t.title_a, title_b=t.title_b, fig_a=fig_a, fig_b=fig_b,
                head_a=t.head_a, head_b=t.head_b, pack_a=pack_a, pack_b=pack_b)}],
            json_mode=True, json_schema=_SCHEMA, max_tokens=500, temperature=0.0,
            model=judge_model, num_ctx=8192)
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        v = str(data.get("verdict", "")).strip().lower()
        verdict = v if v in VERDICTS else "unresolvable"
        reasoning = str(data.get("reasoning", "")).strip()[:600]
        wd = data.get("winner_date")
        winner_date = str(wd)[:10] if wd else None
    except Exception as e:
        logger.warning("[Tensions] judge failed for %r: %s", t.text[:80], e)
        reasoning = f"judge failed: {e}"[:600]
    if verdict in ("a", "b", "both"):
        try:
            raw2 = await llm.invoke_nothink(
                [{"role": "user", "content": _VERIFIER_PROMPT.format(
                    verdict=verdict, reasoning=reasoning, pack_a=pack_a, pack_b=pack_b,
                    title_a=t.title_a, title_b=t.title_b, fig_a=fig_a, fig_b=fig_b)}],
                max_tokens=80, temperature=0.0, model=verifier_model, num_ctx=8192)
            txt = (raw2 or "").strip()
            verifier_agrees = txt.lower().startswith("y")
            verifier_reason = txt[:300]
        except Exception as e:
            logger.warning("[Tensions] verifier failed for %r: %s", t.text[:80], e)
            verifier_reason = f"verifier failed: {e}"[:300]
    return _store(db, t, fig_a, fig_b, pack_a, pack_b, verdict, reasoning, winner_date,
                  judge_model, verifier_model, verifier_agrees, verifier_reason)


def _store(db, t: Tension, fig_a: str, fig_b: str, pack_a: str, pack_b: str, verdict: str,
           reasoning: str, winner_date: str | None, judge_model: str | None,
           verifier_model: str | None, verifier_agrees: bool, verifier_reason: str) -> dict:
    admitted = bool(verifier_agrees and verdict in ("a", "b", "both"))
    try:
        cur = db.execute(
            "INSERT INTO tension_resolutions (tension_text, dkey_a, dkey_b, figure_a, figure_b, "
            "span_a, span_b, verdict, reasoning, judge_model, verifier_model, verifier_agrees, "
            "winner_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (t.text[:600], t.dkey_a, t.dkey_b, fig_a, fig_b, pack_a[:2000], pack_b[:2000],
             verdict, (reasoning + (f" | verifier: {verifier_reason}" if verifier_reason else ""))[:900],
             judge_model or "", verifier_model or "", 1 if verifier_agrees else 0, winner_date),
        )
        row_id = cur.lastrowid
    except Exception as e:
        logger.warning("[Tensions] resolution row not stored: %s", e)
        row_id = None
    logger.info("[Tensions] %s -> %s (verifier %s)%s", t.text[:90], verdict,
                "agrees" if verifier_agrees else "disagrees", " ADMITTED" if admitted else "")
    return {"id": row_id, "verdict": verdict, "reasoning": reasoning, "winner_date": winner_date,
            "verifier_agrees": verifier_agrees, "admitted": admitted, "figure_a": fig_a, "figure_b": fig_b}


def apply_resolution(db, t: Tension, res: dict) -> bool:
    """Write an admitted a/b resolution into the LOSING dossier's Key facts
    bullet and record a belief revision. 'both' needs no rewrite. Returns
    True when a dossier changed."""
    if not res.get("admitted") or res.get("verdict") not in ("a", "b"):
        return False
    from app.core.questions import record_revisions
    win_is_a = res["verdict"] == "a"
    lose_title, lose_dkey, lose_head = (t.title_b, t.dkey_b, t.head_b) if win_is_a else (t.title_a, t.dkey_a, t.head_a)
    win_title, win_fig = (t.title_a, res["figure_a"]) if win_is_a else (t.title_b, res["figure_b"])
    row = db.fetchone("SELECT id, body FROM dossiers WHERE kind = 'domain' AND dkey = ?", (lose_dkey,))
    if not row:
        return False
    body = row["body"] or ""
    head = (lose_head or "").strip().rstrip("…").strip()[:60]
    if not head:
        return False
    lines = body.split("\n")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    note = (f" — superseded: {win_title} records {win_fig}"
            f"{' as of ' + res['winner_date'] if res.get('winner_date') else ''} "
            f"(tension resolved from stored digests, {stamp})")
    changed = False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith(("*", "-")) and head.lower() in s.lower() and "superseded:" not in s:
            lines[i] = ln.rstrip() + note
            changed = True
            break
    if not changed:
        return False
    db.execute("UPDATE dossiers SET body = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
               ("\n".join(lines), row["id"]))
    record_revisions(db, row["id"], lose_dkey,
                     f"REVISED: {lose_title} — {head} superseded by {win_title}'s {win_fig}")
    return True


def list_resolutions(db, *, limit: int = 50) -> list[dict]:
    try:
        rows = db.fetchall(
            "SELECT id, tension_text, dkey_a, dkey_b, figure_a, figure_b, verdict, reasoning, "
            "judge_model, verifier_model, verifier_agrees, winner_date, created_at "
            "FROM tension_resolutions ORDER BY id DESC LIMIT ?", (limit,))
    except Exception:
        return []
    return [dict(r) for r in rows]
