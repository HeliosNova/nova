"""RACE/FACT-style grader for deep-research reports (DeepResearch Bench rubric, adapted).

Two orthogonal scores so a change can be proven, not eyeballed:
  • RACE  — one LLM-judge pass scoring comprehensiveness / depth / on-mission / readability
            (report quality). 1-5 each.
  • FACT  — DETERMINISTIC citation-support: of the factual sentences, how many carry an
            inline (outlet) citation, and how many of those cite a host that was actually
            among the READ sources (a fabricated-attribution catches here). No model, no
            network — cheap, stable, regression-safe.

Used as a before/after regression harness for the engine roadmap (Phase 0). Grades any
report string; the read-source hosts are parsed from the digest's own header line, so a
report is self-describing.
"""
from __future__ import annotations

import re

from app.core import llm

# Citations appear inside parentheticals and may carry a reliability tag:
# (reuters.com), (theglobeandmail.com · primary-doc), or (fool.com · single; ap.org · wire).
# So find EVERY host token inside ANY (...) group — not just a bare (host).
_PAREN_RE = re.compile(r"\(([^)]{3,160})\)")
_HOST_RE = re.compile(r"\b([a-z0-9][a-z0-9.-]*\.[a-z]{2,})\b", re.IGNORECASE)


def _cites_in(text: str) -> list[str]:
    """All source hosts cited inside parentheticals of `text` (handles the reliability-tag
    and multi-host forms). A parenthetical with no host (e.g. '(33%)') yields nothing."""
    hosts: list[str] = []
    for m in _PAREN_RE.finditer(text or ""):
        hosts.extend(h.group(1).lower() for h in _HOST_RE.finditer(m.group(1)))
    return hosts
# Split on a sentence terminator followed by whitespace + a new-sentence start, OR a
# newline. The whitespace requirement means "3.5%"/"$4.2B" decimals don't split a
# sentence (no space after the dot), which a naive [.!?] split gets wrong.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•*\-])|\n+")
# the digest header: "_read 6 sources: apnews.com, cnbc.com, fool.com · 6 facts …_"
_HOSTS_RE = re.compile(r"read\s+\d+\s+sources?:\s*([^_·\n]+)", re.IGNORECASE)


def _hosts_from_report(report: str) -> list[str]:
    m = _HOSTS_RE.search(report or "")
    if not m:
        return []
    return [h.strip().lower().replace("www.", "")
            for h in re.split(r"[,\s]+", m.group(1)) if "." in h]


def _host_matches(cite: str, hosts: set[str]) -> bool:
    cite = cite.lower()
    return any(cite == h or cite.endswith("." + h) or h.endswith("." + cite) for h in hosts)


def fact_score(report: str, read_hosts: list[str] | None = None) -> dict:
    """Deterministic citation-support metrics. `read_hosts` defaults to the hosts named
    in the report's own header. Returns support (valid-cited / factual), citation_rate
    (cited / factual), and fabricated (cited-but-host-not-read / cited)."""
    hosts = {h.lower().replace("www.", "") for h in (read_hosts or _hosts_from_report(report))}
    # When read-hosts come from the digest header and it was truncated ("+more"), host
    # validity can't be verified — citation_rate is then the reliable FACT signal.
    truncated = read_hosts is None and "+more" in (report or "")
    # Drop metadata lines: the "## header", the "_read N sources …_" line, dividers/footers,
    # the 💡 insight — none are claims to grade.
    # ⚠/🔎 lines are the fresh-check advisory annotations (Lever A) — appended
    # meta-commentary quoting a flagged claim, not claims themselves; grading
    # them as "uncited factual sentences" undercounts support (found 2026-07-06).
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(report or "")
             if len(s.strip()) > 40 and not _HOSTS_RE.search(s)
             and not s.lstrip().startswith(("#", "_", "─", "📌", "🛰", "💡", "⚠", "🔎"))]
    # a "factual" sentence makes a checkable claim: has a number, a citation, or a
    # mid-sentence proper-noun pair (named entity) — not pure connective prose.
    factual = [s for s in sents
               if any(c.isdigit() for c in s) or _cites_in(s)
               or re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+", s)]
    cited = valid = fabricated = 0
    for s in factual:
        cites = _cites_in(s)
        if not cites:
            continue
        cited += 1
        if any(_host_matches(c, hosts) for c in cites):
            valid += 1
        else:
            fabricated += 1
    n = len(factual) or 1
    citation_rate = round(cited / n, 3)          # fraction of claims that cite anything
    if truncated:      # partial header → can't verify host validity; citation_rate is the signal
        return {"n_factual": len(factual), "citation_rate": citation_rate,
                "support": citation_rate, "fabricated_rate": 0.0}
    return {
        "n_factual": len(factual),
        "citation_rate": citation_rate,
        "support": round(valid / n, 3),                # fraction of claims cited to a READ host
        "fabricated_rate": round(fabricated / max(cited, 1), 3),  # cited-but-not-read (bad)
    }


# Named-entity anchor: a capitalized token, optionally extended into a 2-4 token
# proper-noun phrase (bridging lowercase connectives like "Bank of England").
# Single capitalized tokens ARE captured (Nvidia, Apple, OpenAI) — critical for
# tech/finance where the key players are one word — with the stop-set + length
# filter below removing sentence-initial common words. Internal &/-/. keep
# "S&P", "AT&T", "U.S." intact.
_ANCHOR_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9.&'-]+(?:\s+(?:of|the|and|for|de|von|&)\s+"
    r"|\s+)?(?:[A-Z][A-Za-z0-9.&'-]+)?(?:\s+[A-Z][A-Za-z0-9.&'-]+){0,2})\b"
)
# Common sentence-initial / connective words that get capitalized without being
# entities. A single-token anchor matching one of these is dropped.
_ANCHOR_STOP = frozenset({
    "The", "This", "That", "These", "Those", "It", "Its", "In", "On", "At", "As",
    "But", "And", "Or", "For", "With", "While", "However", "Meanwhile", "Today",
    "A", "An", "Of", "To", "By", "If", "So", "Yet", "Now", "Then", "Thus", "Also",
    "After", "Before", "During", "Since", "Until", "When", "Where", "Why", "How",
    "What", "Who", "Which", "They", "He", "She", "We", "You", "I", "His", "Her",
    "Their", "Our", "One", "Two", "Three", "Both", "Some", "Many", "Most", "New",
    "More", "Other", "Another", "Such", "First", "Second", "Last", "Next", "Per",
    "According", "Amid", "Despite", "Following", "Instead", "Overall", "Still",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
})


def _anchors(text: str) -> set[str]:
    """Distinct named-entity anchors in `text`, normalized to lowercase. These
    are the concrete stories/players a report should cover — proper nouns are
    preserved near-verbatim across paraphrase, so presence is a reliable
    coverage signal (numbers get rounded, entities don't)."""
    out: set[str] = set()
    for m in _ANCHOR_RE.finditer(text or ""):
        toks = m.group(1).strip().split()
        # Peel leading stop words (sentence-initial "In"/"The", or a month/day
        # the capitalization rule caught) until the phrase starts with a real
        # entity token.
        while toks and toks[0] in _ANCHOR_STOP:
            toks = toks[1:]
        if not toks:
            continue
        phrase = " ".join(toks)
        # A lone remaining token that is itself a stop word (e.g. "January"
        # after peeling "In") is noise, not an entity.
        if len(toks) == 1 and toks[0] in _ANCHOR_STOP:
            continue
        if len(phrase) < 4 or (" " not in phrase and len(phrase) < 6):
            continue
        out.add(phrase.lower())
    return out


def coverage_score(report: str, findings: list) -> dict:
    """Deterministic recall proxy (task #65, 2026-07-08): of the distinct
    named-entity anchors surfaced during GATHER (from the extracted findings),
    what fraction does the final report actually mention?

    Separates two recall failures:
      • pool_anchors  — gather breadth (how many distinct stories we surfaced)
      • coverage      — synthesis recall (of surfaced, how much made the report)
    A low coverage with a high pool = synthesis dropping found material; a low
    pool = gather missing sources. Judge-free, so it's a clean before/after
    signal for the multi-angle sweep + gap-loop levers.

    `findings` is [(title, url, finding_text), ...] as _findings returns.
    """
    # Per-finding anchor sets so we can count how many DISTINCT findings each
    # anchor appears in — a "core" story (multiply-sourced) vs a one-off mention.
    per_finding = [
        _anchors(f"{(t or '')} {(f or '')}")
        for t, _u, f in (findings or []) if (f or t)
    ]
    pool: dict[str, int] = {}
    for aset in per_finding:
        for a in aset:
            pool[a] = pool.get(a, 0) + 1
    if not pool:
        return {"pool_anchors": 0, "covered": 0, "coverage": 1.0,
                "core_anchors": 0, "core_covered": 0, "core_coverage": 1.0, "missed": []}
    rep = _anchors(report or "")

    def _in_report(a: str) -> bool:
        # token-superset/subset match: "Nvidia" covers "Nvidia Corp" and vice-versa
        at = set(a.split())
        return any(at & set(r.split()) for r in rep)

    covered = missed = 0
    core = [a for a, c in pool.items() if c >= 2]   # multiply-sourced = a real story
    core_covered = 0
    core_missed: list[str] = []
    for a in pool:
        hit = _in_report(a)
        covered += 1 if hit else 0
        missed += 0 if hit else 1
    for a in core:
        if _in_report(a):
            core_covered += 1
        else:
            core_missed.append(a)
    return {
        "pool_anchors": len(pool),
        "covered": covered,
        "coverage": round(covered / len(pool), 3),
        # CORE coverage is the recall signal: dropping a multiply-sourced story
        # is a real miss; omitting a once-named entity is editorial selection.
        "core_anchors": len(core),
        "core_covered": core_covered,
        "core_coverage": round(core_covered / len(core), 3) if core else 1.0,
        "missed": sorted(core_missed)[:20],   # missed CORE stories = what to worry about
    }


_RACE_PROMPT = (
    "You are a strict editor grading a {label} intelligence report. Score each dimension "
    "1-5 (5 = excellent, 1 = poor). Return JSON ONLY:\n"
    '{{"comprehensiveness":N,"depth":N,"on_mission":N,"readability":N}}\n'
    "- comprehensiveness: covers the consequential developments, not just one.\n"
    "- depth: concrete numbers, named players, mechanism, and second-order implications — "
    "not a surface skim.\n"
    "- on_mission: the developments are ABOUT the {label} domain. News, politics, economics, "
    "science, or analysis ON THE TOPIC are all on-mission — do NOT penalize political vs "
    "financial framing. Score low ONLY for genuinely unrelated tangents, promotional fluff, "
    "or evergreen filler.\n"
    "- readability: clear structure and prose.\n\n"
    "REPORT:\n{report}"
)


async def grade_report(report: str, label: str, *, model: str | None = None) -> dict:
    """Full grade = deterministic FACT + LLM-judged RACE. RACE failures degrade to zeros
    (never raise) so the harness always returns a comparable row. RACE is judged by the
    bigger synthesis model when configured — a 9B judge mis-scores (e.g. on_mission=0 on
    an on-mission report)."""
    if model is None:
        from app.config import config as _c
        model = (getattr(_c, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None
    out = {"fact": fact_score(report),
           "race": {"comprehensiveness": 0.0, "depth": 0.0, "on_mission": 0.0, "readability": 0.0}}
    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": _RACE_PROMPT.format(label=label, report=(report or "")[:6000])}],
            json_mode=True, json_prefix='{"', max_tokens=120, temperature=0.0, model=model,
            # explicit num_ctx: a no-ctx call loads the model at the OLLAMA_NUM_CTX
            # env default (was 32768 → the 27B ballooned to 24 GB, 185 MiB free on
            # the whole GPU, 2026-07-06). The report is capped at 6000 chars.
            num_ctx=8192)
        d = llm.extract_json_object(raw) or {}
        for k in out["race"]:
            try:
                out["race"][k] = max(0.0, min(5.0, float(d.get(k, 0) or 0)))
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    out["race_avg"] = round(sum(out["race"].values()) / 4, 2)
    # single headline number: RACE quality (0-1) tempered by citation support
    out["overall"] = round((out["race_avg"] / 5) * 0.7 + out["fact"]["support"] * 0.3, 3)
    return out


async def grade_recent(n: int = 8, *, model: str | None = None) -> dict:
    """Grade the latest digest of each of the N most-recent content monitors and return
    a mean RACE/FACT snapshot + per-report rows — the before/after regression signal for
    engine changes. FACT is deterministic; RACE uses `model` (default = the config LLM)."""
    from app.database import get_db
    rows = get_db().execute(
        "SELECT m.name, r.value, MAX(r.created_at) FROM monitor_results r "
        "JOIN monitors m ON m.id = r.monitor_id "
        "WHERE m.category = 'content' AND length(r.value) > 800 "
        "GROUP BY m.id ORDER BY MAX(r.created_at) DESC LIMIT ?", (n,)).fetchall()
    graded = []
    for name, body, _ in rows:
        try:
            g = await grade_report(body, str(name).replace("Domain Study:", "").strip(), model=model)
            graded.append({"monitor": name, **g})
        except Exception:
            pass
    agg = {"n": len(graded)}
    if graded:
        agg["overall"] = round(sum(x["overall"] for x in graded) / len(graded), 3)
        agg["race_avg"] = round(sum(x["race_avg"] for x in graded) / len(graded), 3)
        agg["support"] = round(sum(x["fact"]["support"] for x in graded) / len(graded), 3)
        agg["citation_rate"] = round(sum(x["fact"]["citation_rate"] for x in graded) / len(graded), 3)
        agg["fabricated_rate"] = round(sum(x["fact"]["fabricated_rate"] for x in graded) / len(graded), 3)
    return {"aggregate": agg, "reports": graded}
