"""Fact spans: the sentences that carried each Key-fact figure, kept at
consolidation (phase 4 of the 2026-10-12 forward program; STALE/TOKI).

Why: the tension resolver judges from the dated sentences that carry each
figure, but it read them from the last three stored digests per domain, and
in the 2026-10-11 replay 30 of 56 tensions had no such sentence left — the
digests had aged out. Keeping the span at the moment the dossier asserts the
fact makes a later tension judgeable from Nova's own record, whenever it
surfaces. Rows are small (one sentence each) and pruned per dossier.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_KEEP_PER_DKEY = 300
_SPANS_PER_FIGURE = 2
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\-•\d\"'(])")
_DIGEST_HDR_RE = re.compile(r"\[digest (\d{4}-\d{2}-\d{2})[^\]]*\]")
_KEY_FACTS_RE = re.compile(r"(?is)## Key facts[^\n]*\n(.+?)(?=\n## |\Z)")
_SUPERSEDED_RE = re.compile(r"(?i)\bsuperseded:")


def _figure_forms(raw: str) -> list[str]:
    """Spellings a digest may use for a bullet's figure ('3.4%', '2 billion', '$1,250')."""
    raw = raw.strip()
    forms = {raw, raw.replace(",", "")}
    m = re.match(r"\$?([\d,]+(?:\.\d+)?)\s*(%|percent|bps?|basis points?|trillion|billion|million|bn|mn)?", raw, re.I)
    if m:
        num, unit = m.group(1), (m.group(2) or "").lower()
        n = num.replace(",", "")
        if unit in ("%", "percent"):
            forms |= {f"{n}%", f"{n} percent", f"{n} per cent"}
        elif unit in ("billion", "bn"):
            forms |= {f"{n} billion", f"{n}bn", f"{n} bn"}
        elif unit in ("million", "mn"):
            forms |= {f"{n} million", f"{n}mn", f"{n} mn"}
        elif unit in ("trillion",):
            forms |= {f"{n} trillion", f"{n}tn"}
        elif unit.startswith("b"):
            forms |= {f"{n} bp", f"{n}bp", f"{n} bps", f"{n} basis points"}
        else:
            forms.add(n)
    return sorted(f for f in forms if f)


def _dated_sentences(sources: str) -> list[tuple[str, str]]:
    """[(date, sentence)] from the consolidation's source text, which carries
    '[digest YYYY-MM-DD ...]' headers between digests (dossiers._domain_sources)."""
    out: list[tuple[str, str]] = []
    date = ""
    for chunk in re.split(r"(\[digest [^\]]*\])", sources or ""):
        m = _DIGEST_HDR_RE.match(chunk)
        if m:
            date = m.group(1)
            continue
        for s in _SENT_RE.split(chunk.replace("\n", " ")):
            s = s.strip()
            if len(s) >= 25:
                out.append((date, s))
    return out


def record_fact_spans(db, kind: str, dkey: str, body: str, sources: str) -> int:
    """For each Key-facts bullet with a figure, store up to two source
    sentences that carry it. Idempotent on (dkey, figure, sentence). Returns rows added."""
    from app.core.dossiers import _NUM_UNIT_RE
    m = _KEY_FACTS_RE.search(body or "")
    if not m or not sources:
        return 0
    sents = _dated_sentences(sources)
    if not sents:
        return 0
    added = 0
    for line in m.group(1).splitlines():
        s = line.strip()
        if not s.startswith(("*", "-")) or _SUPERSEDED_RE.search(s):
            continue
        head = re.sub(r"\([^)]*\)", "", s.lstrip("*- ")).strip()[:120]
        for fm in _NUM_UNIT_RE.finditer(s):
            raw = fm.group(0).strip()
            try:
                val = float(fm.group(1).replace(",", ""))
            except ValueError:
                continue
            unit = (fm.group(2) or "").lower()
            if not unit and (val > 1900 or (val == int(val) and val < 32)):
                continue    # years and small counts are not metrics
            forms = _figure_forms(raw)
            found = 0
            for date, sent in sents:
                low = sent.lower()
                if any(f.lower() in low for f in forms):
                    try:
                        cur = db.execute(
                            "INSERT OR IGNORE INTO fact_spans (kind, dkey, figure, bullet_head, sentence, source_date) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (kind, dkey, raw, head, sent[:500], date or None),
                        )
                        added += int(getattr(cur, "rowcount", 0) or 0)
                    except Exception as e:
                        logger.debug("[FactSpans] insert skipped: %s", e)
                    found += 1
                    if found >= _SPANS_PER_FIGURE:
                        break
    if added:
        try:
            db.execute(
                "DELETE FROM fact_spans WHERE dkey = ? AND id NOT IN ("
                "  SELECT id FROM fact_spans WHERE dkey = ? ORDER BY id DESC LIMIT ?)",
                (dkey, dkey, _KEEP_PER_DKEY),
            )
        except Exception as e:
            logger.debug("[FactSpans] prune skipped: %s", e)
        logger.info("[FactSpans] %s: %d span(s) kept for Key facts", dkey, added)
    return added


def spans_for(db, dkey: str, val: float, unit: str, *, limit: int = 3) -> list[dict]:
    """Stored sentences carrying this figure for a dossier, newest first —
    the tension resolver's fallback when the digests have aged out."""
    from app.core.dossiers import _fmt_qty
    forms = {f.lower() for f in _figure_forms(_fmt_qty(val, unit))}
    try:
        rows = db.fetchall(
            "SELECT figure, sentence, source_date FROM fact_spans WHERE dkey = ? "
            "ORDER BY COALESCE(source_date, '') DESC, id DESC LIMIT 400", (dkey,))
    except Exception:
        return []
    out: list[dict] = []
    for r in rows:
        low = (r["sentence"] or "").lower()
        if any(f in low for f in forms):
            out.append({"date": (r["source_date"] or "")[:10], "sentence": r["sentence"]})
            if len(out) >= limit:
                break
    return out


def drop_superseded(text: str) -> str:
    """Remove bullets the tension resolver marked superseded (STALE: a
    retrieved-but-stale fact must not carry authority in generation)."""
    if not text or "superseded:" not in text.lower():
        return text
    kept = [ln for ln in text.split("\n") if not _SUPERSEDED_RE.search(ln)]
    return "\n".join(kept)
