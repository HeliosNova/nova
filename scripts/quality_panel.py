"""Day-by-day panel of what Nova actually PRODUCED. Run before and after every deploy.

Why this exists: on 2026-09-04 the owner spotted digests citing "(deep
analysis)". The full test suite was green, because the suite asserts code
structure and cannot see a prompt change degrade the product. The artifact had
been climbing for three days — about 10% of digests before the 09-02 deploy,
then 34%, 45%, 55% — and nothing in the pipeline was looking at the output.

Everything here is read from stored artifacts and computed deterministically:
no model, no network, seconds to run. A step change lined up with a deploy is
the signal; the absolute numbers matter less than the shape.

    docker exec nova-app python /app/scripts/quality_panel.py [--days 14]

READ IT WITH ONE RULE: never compare across a change to the measuring code.
The judge-score columns come from output_quality_log, and that judge changed on
2026-09-02 (it used to read 3,000 chars and floor scores to 8), so scores
either side of that date are different instruments, not different quality.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict

# A short parenthetical naming "analysis" with no domain token is the digest
# citing its own reasoning — mirrors deep_research._strip_pseudo_citations.
_PAREN = re.compile(r"\(([^)]{0,80})\)")
_ANALYSIS = re.compile(r"(?i)\banalys[ei]s\b")
_DOMAINISH = re.compile(r"[a-z0-9-]+\.[a-z]{2,}")
_CITE = re.compile(r"\(([a-z0-9-]+\.[a-z]{2,})\)")
_DATED = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                    r"[a-z]*\s+\d{1,2}\b")
# Scaffolding the model is never supposed to ship.
_LEAKS = (
    re.compile(r"(?i)\bas an ai\b"),
    re.compile(r"(?i)\b(?:step|stage) \d+/\d+\b"),
    re.compile(r"(?i)\bnot specified here\b"),
    re.compile(r"(?i)\bsearch results?\b"),
    re.compile(r"</?tool_call>"),
    re.compile(r"(?i)\bI (?:cannot|can't) (?:access|browse)\b"),
)


def panel(db_path: str, days: int) -> None:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT mr.created_at, mr.value FROM monitor_results mr "
        "JOIN monitors m ON m.id = mr.monitor_id "
        "WHERE m.category = 'content' AND mr.created_at > datetime('now', ?) "
        "AND mr.value IS NOT NULL AND LENGTH(mr.value) > 400", (f"-{days} days",)).fetchall()

    day: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "chars": 0, "cites": 0, "pseudo": 0, "dated": 0,
                 "linkonly": 0, "leaks": 0, "thin": 0})
    for r in rows:
        v = r["value"]
        s = day[r["created_at"][:10]]
        s["n"] += 1
        s["chars"] += len(v)
        s["cites"] += len(_CITE.findall(v))
        s["pseudo"] += sum(1 for m in _PAREN.finditer(v)
                           if _ANALYSIS.search(m.group(1)) and not _DOMAINISH.search(m.group(1)))
        s["dated"] += len(_DATED.findall(v))
        s["leaks"] += sum(1 for rx in _LEAKS if rx.search(v))
        if len(v) < 600 and "http" in v:
            s["linkonly"] += 1
        if len(v) < 2500:
            s["thin"] += 1

    print("PRODUCT — what the digests look like (deterministic, no model involved)")
    print(f"{'day':<12}{'digests':>8}{'avg chars':>10}{'cites':>7}{'PSEUDO':>8}"
          f"{'dates':>7}{'thin':>6}{'link':>6}{'leaks':>7}")
    for d in sorted(day):
        s = day[d]
        n = max(s["n"], 1)
        print(f"{d:<12}{s['n']:>8}{s['chars'] // n:>10}{s['cites'] / n:>7.1f}"
              f"{s['pseudo'] / n:>8.2f}{s['dated'] / n:>7.1f}{s['thin']:>6}"
              f"{s['linkonly']:>6}{s['leaks']:>7}")
    print("  PSEUDO = self-citations per digest; any rise is a prompt/guard mismatch.")
    print("  thin   = digests under 2,500 chars; link = under 600 chars with a URL.")

    print("\nPIPELINE — what the machinery did")
    print(f"{'day':<12}{'kg_new':>8}{'kg_live':>9}{'survive':>9}{'fc_mint':>9}"
          f"{'fc_resolv':>11}{'curio_res':>11}{'lessons':>9}")
    for d in sorted(day):
        q = lambda s: con.execute(s, (d,)).fetchone()[0]          # noqa: E731
        new = q("SELECT COUNT(*) FROM kg_facts WHERE substr(created_at,1,10)=?")
        live = q("SELECT COUNT(*) FROM kg_facts WHERE substr(created_at,1,10)=? "
                 "AND superseded_at IS NULL")
        print(f"{d:<12}{new:>8}{live:>9}{(live / new if new else 0):>9.0%}"
              f"{q('SELECT COUNT(*) FROM forecasts WHERE substr(created_at,1,10)=?'):>9}"
              f"{q('SELECT COUNT(*) FROM forecasts WHERE substr(resolved_at,1,10)=?'):>11}"
              f"{q('SELECT COUNT(*) FROM curiosity_queue WHERE substr(resolved_at,1,10)=?'):>11}"
              f"{q('SELECT COUNT(*) FROM lessons WHERE substr(created_at,1,10)=?'):>9}")
    print("  survive = share of that day's facts still live; the KG ring buffer used to")
    print("  evict the newest, so this sat near 20%.")

    print("\nJUDGE — output_quality_log  (INSTRUMENT CHANGED 2026-09-02: do not compare across it)")
    print(f"{'day':<12}{'n':>5}{'avg':>7}{'relev':>7}{'facts':>7}{'fresh':>7}{'format':>8}{'novelty':>9}")
    for r in con.execute(
            "SELECT substr(created_at,1,10) d, COUNT(*) n, ROUND(AVG(avg_score),2) a, "
            "ROUND(AVG(relevance),2) r, ROUND(AVG(facts),2) f, ROUND(AVG(freshness),2) fr, "
            "ROUND(AVG(format),2) fo, ROUND(AVG(novelty),2) nv FROM output_quality_log "
            "WHERE created_at > datetime('now', ?) GROUP BY d ORDER BY d", (f"-{days} days",)):
        nv = "-" if r["nv"] is None else f"{r['nv']:.2f}"
        print(f"{r['d']:<12}{r['n']:>5}{r['a']:>7}{r['r']:>7}{r['f']:>7}"
              f"{r['fr']:>7}{r['fo']:>8}{nv:>9}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--db", default="/data/nova.db")
    a = ap.parse_args()
    panel(a.db, a.days)
