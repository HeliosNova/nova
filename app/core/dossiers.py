"""Living dossiers — the KNOWING tier (2026-08-12).

Nova gathers elite intelligence (digests) and stores verified facts (bitemporal
KG), but its richest thinking — the analyses, connections, "why it matters" —
lives in `monitor_results` under a 30-day retention and evaporates. Each digest
is written from a 48h window: comprehension without accumulation.

This module closes that gap. A dossier is DURABLE, REVISABLE understanding:
one per watched domain and one per mature storyline, distilled from the
already-8-layer-verified digests BEFORE they expire, superseded-not-deleted
(`dossier_revisions` keeps every prior body → "what did Nova understand about
X on date D" is queryable, same philosophy as kg_facts bitemporality).

Consumers:
  - heartbeat "Knowledge Consolidation" monitor (check_type="consolidation",
    daily) → `consolidate_dossiers` — the distillation cycle, bounded.
  - deep_research `_synthesize_from_evidence` → `get_domain_dossier` — digests
    are written AGAINST prior understanding (lead with what's NEW, flag
    contradictions) instead of amnesiac re-reporting.
  - brain chat context → `get_relevant_dossiers` — questions get answered FROM
    understanding (dossier prose retrieves into the prompt where raw triples
    were denied by the 9B, forward-queue 2026-07-07).

Reuses llm.invoke_nothink (27B via MONITOR_SYNTHESIS_MODEL when set — this is
background/latency-tolerant, quality wins). Every LLM call passes an explicit
num_ctx (num_ctx discipline, 2026-08-11: background calls that can exceed ~12k chars must size their own context).
"""

from __future__ import annotations

import asyncio
import logging
import re

from app.core import llm
from app.core.text_utils import STOP_WORDS

logger = logging.getLogger(__name__)


async def _bg_invoke(*args, **kwargs):
    """llm.invoke_nothink with the GPU-yield checkpoint (parity with
    deep_research._invoke_bg, added 2026-08-19): this module's background
    chains make many sequential 27B calls — without the checkpoint an owner
    chat arriving mid-chain contends for the GPU for the whole run (probe:
    350 tokens at ~2 tok/s during a consolidation cycle)."""
    try:
        from app.core.llm import wait_for_interactive_quiet as _w
        waited = await _w(max_wait_s=240.0)
        if waited:
            logger.info("[gpu-yield] %s yielded to chat for %.0fs", __name__, waited)
    except Exception:
        pass
    return await llm.invoke_nothink(*args, **kwargs)


# Bounded cycle: at most this many dossier updates per consolidation run.
_MAX_UPDATES_PER_CYCLE = 8
# Dossier body ceiling (chars) — bounded at a sentence boundary, never mid-thought.
_BODY_CAP = 9000
# New-material budget per update (chars of recent digests fed to the LLM).
_SOURCE_CAP = 18000
# Prior-body budget fed back for revision.
_PRIOR_CAP = 8000
# Storylines graduate to a dossier once they've moved this many times.
_STORYLINE_MIN_UPDATES = 3

_DOMAIN_PREFIX = "Domain Study: "

# Function words excluded from the ≥2-char retrieval tokens (get_relevant_dossiers).
_SHORT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "had", "has", "how", "i", "if", "in", "is", "it", "its", "no",
    "not", "now", "of", "on", "or", "our", "out", "per", "so", "the", "their",
    "they", "this", "that", "to", "up", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
})


def _slug(text: str) -> str:
    """Stable slug key (mirrors storylines._story_key)."""
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:80]


def _bound(text: str, cap: int = _BODY_CAP) -> str:
    """Trim to cap at the last sentence boundary — a dossier must never end
    mid-thought (the truncation lesson, 2026-08-11)."""
    text = (text or "").strip()
    if len(text) <= cap:
        return text
    cut = text[:cap]
    # Prefer the last sentence end; fall back to last newline; then hard cut.
    m = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("!\n"), cut.rfind("?\n"))
    if m > cap // 2:
        return cut[: m + 1].rstrip()
    nl = cut.rfind("\n")
    return (cut[:nl] if nl > cap // 2 else cut).rstrip()


_UPDATE_PROMPT = (
    "You maintain Nova's PERMANENT UNDERSTANDING of: {title}\n"
    "Below is the PRIOR DOSSIER (what was understood before), then NEW MATERIAL "
    "(verified intelligence digests written since the last consolidation — more recent).\n\n"
    "Rewrite the dossier to reflect the CURRENT state of understanding. Rules:\n"
    "- NEW MATERIAL wins where it conflicts with the prior dossier: describe the "
    "latest reality; when a prior belief is OVERTAKEN, replace it and record the "
    "shift in 'How we got here'.\n"
    "- If new material CONTRADICTS prior understanding, flag it explicitly with a "
    "line starting 'REVISED:' (what was believed → what is now understood → why).\n"
    "- Keep only what has DURABLE value — drop day-trivia and routine scheduling. "
    "Keep EXACT numbers, names, dates, and inline (outlet.com) citations exactly as "
    "they appear in the material. Invent NOTHING beyond the prior dossier + material.\n"
    "- Structure (markdown, keep these exact headings):\n"
    "## Current understanding\n"
    "(400-600 words — the state of things and why it matters)\n"
    "## How we got here\n"
    "(5-10 dated bullets of the major shifts, oldest→newest)\n"
    "## Key facts & figures\n"
    "(bullets, each with its citation)\n"
    "## Open questions\n"
    "(3-6 questions answerable TODAY by research, in the present tense — what is "
    "X's current…, who holds…, how much…; NOT future events)\n"
    "- Then one line 'Watch for: <expected future events, comma-separated>' "
    "(or 'Watch for: none') — futures live there, not among the questions.\n"
    "- End with exactly one line 'CHANGED: <what moved in this consolidation>' "
    "(or 'CHANGED: initial dossier' for a first version).\n"
    "- End with one FINAL line, always present: "
    "'FORECAST: <specific testable claim> | resolves YYYY-MM-DD | <0.x confidence>' "
    "when the material supports a clear falsifiable expectation, else exactly "
    "'FORECAST: none'. The date is when the outcome becomes knowable (up to a "
    "year out); put any deadline inside the claim too, make the claim settleable "
    "by a news search on that date, and never forecast a target, guidance figure "
    "or plan — only a realized outcome. A dated event or a scheduled decision in "
    "the material usually supports one.\n"
    "Write the COMPLETE dossier and finish cleanly — no preamble, no meta-commentary.\n\n"
    "PRIOR DOSSIER:\n{prior}\n\n"
    "NEW MATERIAL:\n{sources}"
)

# Cross-domain capstone (state-of-the-world): consolidation ACROSS the domain
# dossiers — the top of the knowing pyramid. Same output contract as
# _UPDATE_PROMPT so persistence/forecast/strip logic is shared.
_WORLD_PROMPT = (
    "You maintain Nova's PERMANENT CROSS-DOMAIN UNDERSTANDING: {title}\n"
    "Below is the PRIOR STATE-OF-THE-WORLD DOSSIER, then the CURRENT PER-DOMAIN "
    "UNDERSTANDINGS (each already consolidated from verified intelligence).\n\n"
    "Rewrite the state-of-the-world dossier. This is NOT a domain-by-domain list — "
    "it is the MACRO picture: the throughlines CONNECTING domains, the tensions "
    "BETWEEN them, and what the world is becoming. Rules:\n"
    "- Newer per-domain understanding wins over the prior dossier where they conflict.\n"
    "- Flag genuine cross-domain contradictions with a 'REVISED:' line.\n"
    "- Keep EXACT numbers, names, dates, and inline (outlet.com) citations from the "
    "material. Invent NOTHING.\n"
    "- Structure (markdown, keep these exact headings):\n"
    "## Current understanding\n"
    "(500-700 words — the macro state: connect at least three domains explicitly)\n"
    "## How we got here\n"
    "(5-10 dated bullets of the biggest cross-domain shifts)\n"
    "## Key facts & figures\n"
    "(only the cross-cutting ones, each with its citation)\n"
    "## Open questions\n"
    "(3-6 macro unknowns with real stakes, phrased as questions answerable TODAY by "
    "research — present tense, NOT future events)\n"
    "- Then one line 'Watch for: <expected future events, comma-separated>' "
    "(or 'Watch for: none').\n"
    "- End with exactly one line 'CHANGED: <what moved at the macro level>'.\n"
    "- End with one FINAL line, always present: "
    "'FORECAST: <specific testable claim> | resolves YYYY-MM-DD | <0.x confidence>' "
    "when the material supports a clear falsifiable expectation, else exactly "
    "'FORECAST: none'. The date is when the outcome becomes knowable (up to a "
    "year out); put any deadline inside the claim too, make the claim settleable "
    "by a news search on that date, and never forecast a target, guidance figure "
    "or plan — only a realized outcome. A dated event or a scheduled decision in "
    "the material usually supports one.\n"
    "Write the COMPLETE dossier and finish cleanly — no preamble, no meta-commentary.\n\n"
    "PRIOR DOSSIER:\n{prior}\n\n"
    "NEW MATERIAL:\n{sources}"
)

_CHANGED_RE = re.compile(r"(?im)^CHANGED:\s*(.+?)\s*$")
_FORECAST_RE = re.compile(r"(?im)^FORECAST:\s*.+$")
_OPEN_Q_SECTION_RE = re.compile(r"(?is)## Open questions\s*(.+?)(?=\n## |\Z)")
# Future-shaped questions are forecast material, not research targets — web
# search cannot resolve "Will X happen?" and each one burns an hourly research
# attempt forever (52-pending/0-resolved queue, found 2026-08-14).
_FUTURE_LEAD_RE = re.compile(
    r"(?i)^\s*(?:will|won'?t|should|would|whether|how will|how might|how soon|"
    r"is it likely|what will|when will|by when)\b")
_FUTURE_MARKER_RE = re.compile(
    r"(?i)\b(?:will|won'?t|by 20\d\d|next (?:year|month|quarter|week)|"
    r"in the (?:coming|next)|going to)\b")


def _is_future_question(q: str) -> bool:
    """True when a question is FORECAST material (unresolvable by web search now)
    rather than a researchable knowledge gap. Leading future modals qualify;
    'can/could/might' qualify ONLY with an explicit future marker (so 'Can X run
    on a 3090?' stays researchable while 'Could X reach $1T next year?' routes to
    forecast); and an embedded 'will' near the front catches 'What reforms WILL
    Congress pass?' — the shape that leaked into the hourly queue (2026-08-14)."""
    q = (q or "").strip()
    if not q:
        return False
    if _FUTURE_LEAD_RE.match(q):
        return True
    head = " ".join(q.split()[:8])
    if re.search(r"(?i)\bwill\b|\bwon'?t\b", head):
        return True
    if re.match(r"(?i)^\s*(?:can|could|might)\b", q) and _FUTURE_MARKER_RE.search(q):
        return True
    return False
_HOST_TOKEN_RE = re.compile(r"[a-z0-9.-]+\.[a-z]{2,}")


# Number allows comma grouping ("$5,300 million") — the old \d+(?:\.\d+)? stopped
# at the comma and mis-parsed "5,300 million" as "300 million" (2026-08-14 audit).
_NUM_UNIT_RE = re.compile(
    r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)\s*(%|percent|bps?|basis points?|trillion|billion|million|bn|mn)?",
    re.I)
_TENSION_STOP = frozenset({
    "with", "from", "this", "that", "than", "over", "into", "amid", "after",
    "while", "which", "their", "annually", "monthly", "month", "year", "week",
    "roughly", "about", "approximately", "record", "total", "level", "levels",
})
# Generic finance / magnitude / descriptor filler. These are ≥6-char words that
# co-occur across UNRELATED money facts, so the old "≥2 shared tokens incl. one
# ≥6 chars" rule counted them as evidence two bullets were the SAME quantity and
# minted a ~100%-false-positive tension stream ("Odin Mining $1.7B" vs "SpaceX
# $2B" share {billion, capital}; a 60% Ebola fatality vs a 7% dev stat share a
# unit) that the daemon burned ~124 brain.think()/day on (2026-08-18 audit). A
# genuine same-metric tension shares the ENTITY/METRIC name (e.g. "inflation",
# "headline") which is NOT in this set, so filtering filler preserves recall.
_TENSION_GENERIC = frozenset({
    "billion", "million", "trillion", "thousand", "dollars", "dollar",
    "capital", "funding", "raised", "raises", "raise", "revenue", "revenues",
    "profit", "profits", "valuation", "value", "worth", "sales", "price", "prices",
    "costs", "spending", "investment", "investments", "deal", "deals", "round",
    "series", "stake", "shares", "share", "market", "markets", "global", "growth",
    "sector", "company", "companies", "report", "reports", "quarter", "quarterly",
    "guidance", "estimate", "estimates", "percent", "rate", "rates", "increase",
    "decrease", "number", "figure", "figures", "average", "target", "range",
})


def _wtrim(s: str, cap: int) -> str:
    """Whole-word cap (2026-08-31). Tension excerpts and the curiosity topic
    were hard-sliced ([:80]/[:180]) mid-word — 33 stored dossier_tension
    topics at exactly len 203 opened and closed on word fragments ('enue was
    **$22.8 billion**…'), and those topics become live search queries."""
    s = (s or "").strip()
    if len(s) <= cap:
        return s
    cut = s[:cap]
    ws = cut.rfind(" ")
    return (cut[:ws].rstrip(" ,;:—-") + "…") if ws > 0 else cut


def _fmt_qty(val: float, unit: str) -> str:
    """'2billion' → '2 billion' (2026-08-31): symbol units (%, bp, x) attach
    directly; word units get a space. The old f'{val:g}{unit}' mangled every
    word-unit tension string."""
    u = unit or ""
    sep = "" if u in ("%", "bp", "x", "") else " "
    return f"{val:g}{sep}{u}"


def _numeric_tensions(db, *, max_report: int = 3) -> list[str]:
    """Thinking rung, brick one (2026-08-14): notice when two CURRENT dossiers
    assert materially different values for what reads as the SAME quantity
    (e.g. Finance says CPI 3.4% while Economics says 2.9%). Deterministic —
    Key-facts bullets only (the factual layer), matched by rare-token context
    overlap + unit class, flagged at >5%% relative divergence. Surfaces the
    tension for investigation; never auto-resolves."""
    try:
        rows = db.fetchall(
            "SELECT title, body FROM dossiers WHERE kind IN ('domain', 'meta')")
    except Exception:
        return []
    facts = []   # (dossier_title, context_tokens, value, unit, bullet_head)
    for r in rows:
        in_facts = False
        for ln in (r["body"] or "").split("\n"):
            s = ln.strip()
            if s.startswith("## "):
                in_facts = "key facts" in s.lower()
                continue
            if not in_facts or not s.startswith(("*", "-")) or "(" not in s:
                continue
            # citation hosts must not count as shared context (two unrelated
            # reuters-cited facts are not the same quantity)
            s_nocite = re.sub(r"\([^)]*\)", "", s)
            toks = frozenset(
                t for t in re.findall(r"[a-z]{4,}", s_nocite.lower())
                if t not in _TENSION_STOP and t not in STOP_WORDS
            ) - {"low", "authority", "sourcing"}
            for m in _NUM_UNIT_RE.finditer(s):
                val = float(m.group(1).replace(",", ""))
                unit = (m.group(2) or "").lower().rstrip("s")
                if unit in ("percent",):
                    unit = "%"
                if unit in ("basis point", "bp"):
                    unit = "bp"
                # years and bare small ints are noise, not metrics
                if not unit and (val > 1900 or val == int(val) and val < 32):
                    continue
                facts.append((r["title"], toks, val, unit, _wtrim(s.lstrip("*- "), 80)))
    tensions: list[str] = []
    seen_pairs: set[frozenset] = set()
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            a, b = facts[i], facts[j]
            if a[0] == b[0] or a[3] != b[3]:
                continue                       # same dossier / different unit
            shared = a[1] & b[1]
            union = a[1] | b[1]
            # same quantity = ≥2 shared DISTINCTIVE tokens (entity/metric names,
            # one ≥6 chars) AND ≥25% Jaccard overlap of the informative vocabulary.
            # The old rule ("Jaccard deliberately NOT required — recall beats
            # precision") counted GENERIC finance/magnitude filler as distinctive
            # AND had no overlap floor, so it paired wholly unrelated money facts
            # ("Odin Mining $1.7B" vs "SpaceX $2B" share {billion, capital};
            # "Bundibugyo 60% fatality" vs a 7% dev stat) — a ~100%-false-positive
            # stream the daemon burned ~124 brain.think()/day on (2026-08-18 live
            # audit). Both gates together are calibrated (live sweep) so the genuine
            # same-metric case (a CPI "3.4% headline inflation" vs "2.9% headline
            # inflation" pair, Jaccard ≈0.29, distinctive {headline, inflation})
            # clears the bar while the filler-only pairs (Jaccard <0.2 or distinctive
            # <2 after filtering) do not. Result on live dossiers: 16 → 6, ~83%
            # now genuine same-entity tensions. Recall does NOT beat precision at ~0.
            distinctive = shared - _TENSION_GENERIC
            if (len(distinctive) < 2 or not any(len(t) >= 6 for t in distinctive)
                    or (len(union) and len(shared) / len(union) < 0.25)):
                continue                       # not the same quantity
            hi, lo = max(a[2], b[2]), min(a[2], b[2])
            if hi == 0 or (hi - lo) / hi <= 0.05:
                continue                       # agreement within tolerance
            pair = frozenset((a[0], b[0], a[3], round(lo, 4), round(hi, 4)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            tensions.append(
                f"{a[0]} says {_fmt_qty(a[2], a[3])} but {b[0]} says {_fmt_qty(b[2], b[3])} — "
                f"\"{a[4]}\" vs \"{b[4]}\"")
            if len(tensions) >= max_report:
                return tensions
    return tensions


def _flag_weak_citations(body: str) -> str:
    """Key-facts bullets supported ONLY by dataset-confirmed junk-tier hosts
    (authority < 0.3; unknown hosts score a neutral ~0.5 and never trip this)
    get a visible low-authority tag. Nothing is deleted — the knowing tier
    marks weak sourcing rather than silently erasing it. Audit 2026-08-13
    found turkiyetoday-class hosts surviving into consolidated citations
    because authority ranking ran only at digest level, never here."""
    try:
        from app.core.source_authority import authority
    except Exception:
        return body
    out, in_facts = [], False
    for ln in body.split("\n"):
        s = ln.strip()
        if s.startswith("## "):
            in_facts = "key facts" in s.lower()
        elif in_facts and s.startswith(("*", "-")) and "⚠" not in ln and "(" in ln:
            hosts = [h for grp in re.findall(r"\(([^)]{4,120})\)", s.lower())
                     for h in _HOST_TOKEN_RE.findall(grp)]
            if hosts and max(authority(h) for h in hosts) < 0.3:
                ln = ln.rstrip() + "  ⚠ low-authority sourcing"
        out.append(ln)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

def get_dossier(db, kind: str, dkey: str):
    """Current dossier row for (kind, dkey) or None."""
    try:
        return db.fetchone(
            "SELECT * FROM dossiers WHERE kind = ? AND dkey = ?", (kind, dkey)
        )
    except Exception:
        return None


def resolve_domain_dkey(label: str, monitor_name: str | None = None) -> str:
    """Dossier key for a deep_research feed label or a monitor name.

    Consolidation keys domain dossiers by slug(monitor name minus
    'Domain Study: '). deep_research only knows the SHORT profile label
    ('AI/ML', 'open source', 'Europe + EU'), whose slug matches the key for
    just 13 of 39 domains (live check 2026-09-01) — the other 26 digests ran
    unprimed. A monitor name resolves directly; a bare label is mapped back
    through the profile table, falling back to its own slug.
    """
    if monitor_name:
        base = monitor_name.strip()
        if base.startswith(_DOMAIN_PREFIX):
            base = base[len(_DOMAIN_PREFIX):]
        return _slug(base)
    direct = _slug(label)
    try:
        from app.monitors.domain_study_runner import _DOMAIN_PROFILES
    except Exception:
        return direct
    lab = (label or "").strip().lower()
    for key, prof in _DOMAIN_PROFILES.items():
        if str(prof[1]).strip().lower() == lab:
            return _slug(key)
    return direct


def get_domain_dossier(db, label: str, monitor_name: str | None = None):
    """Domain dossier for a deep_research feed label (e.g. 'finance') or,
    preferably, the monitor name it runs under. Tries the monitor-name key,
    then the label's own slug, then the profile alias (see resolve_domain_dkey).
    """
    keys: list[str] = []
    if monitor_name:
        keys.append(resolve_domain_dkey(label, monitor_name))
    for k in (_slug(label), resolve_domain_dkey(label)):
        if k and k not in keys:
            keys.append(k)
    for k in keys:
        row = get_dossier(db, "domain", k)
        if row:
            return row
    return None


_CU_SECTION_RE = re.compile(r"(?is)## Current understanding\s*(.+?)(?=\n## |\Z)")


def priming_excerpt(body: str, *, cu_cap: int = 2500, oq_cap: int = 1200) -> str:
    """The part of a dossier worth priming a digest with: the current
    understanding (what is already known) plus the open questions (what the
    digest should try to answer). The old body[:2500] slice stopped before
    '## Open questions' in every one of the 39 domain dossiers (offsets
    5,387-8,907), so digests never saw the frontier they were meant to push."""
    body = body or ""
    m = _CU_SECTION_RE.search(body)
    cu = m.group(1).strip() if m else body.strip()
    out = "## Current understanding\n" + _bound(cu, cu_cap)
    m2 = _OPEN_Q_SECTION_RE.search(body)
    if m2 and m2.group(1).strip():
        out += "\n\n## Open questions\n" + _bound(m2.group(1).strip(), oq_cap)
    return out


_ASKS_GAPS_RE = re.compile(
    r"(?i)\b(?:what (?:do|don'?t|dont) you know|what (?:are you|is nova) (?:unsure|uncertain)|"
    r"unsure|uncertain|open questions?|don'?t (?:you )?know|not (?:yet )?know|unknowns?|gaps?|"
    r"what (?:is|are) (?:still )?(?:missing|unclear|unresolved))\b")


def get_relevant_dossiers(db, query: str, *, limit: int = 1, open_questions: bool | None = None) -> list[dict]:
    """Cheap keyword match of a chat query to dossiers (no LLM, no embedder).

    Mirrors storylines.get_relevant_storylines: overlap >= 2 significant tokens;
    returns [{'title', 'excerpt'}] with the excerpt cut from 'Current
    understanding' so chat injects prose the model answers from naturally.
    Returns [] on any miss — the chat path must never break on this.
    """
    # ≥2-char tokens with a function-word stoplist: the ≥4 floor used elsewhere
    # made the 'AI and ML' title contribute ZERO tokens (ai/ml are 2 chars) —
    # short-acronym dossiers were unmatchable (seen live 2026-08-12).
    _stop = _SHORT_STOPWORDS
    def _toks(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]{2,}", (text or "").lower()) if t not in _stop}
    q_toks = _toks(query)
    if not q_toks:
        return []
    try:
        # Every dossier (2026-09-01): the old LIMIT 60 made the 29 oldest —
        # entity dossiers like Anthropic and Microsoft — unreachable by
        # construction. 89 rows is trivial to scan.
        rows = db.fetchall(
            "SELECT title, body FROM dossiers WHERE body != '' ORDER BY updated_at DESC"
        )
    except Exception:
        return []
    scored = []
    for r in rows:
        title_toks = _toks(r["title"])
        body_toks = _toks(r["body"][:2000])
        overlap = len(q_toks & (title_toks | body_toks))
        # A single TITLE hit qualifies (titles are curated + specific: 'biotech
        # advances lately?' must match 'Biotech and Genetics'); body-only matches
        # still need >=2 tokens so incidental mentions don't retrieve.
        if overlap >= 2 or (q_toks & title_toks):
            # Title hits weigh 3× — "what changed at DeepMind?" must prefer the
            # 'AI and ML' dossier over a broad one that merely mentions AI
            # (mis-rank seen live on the first retrieval check, 2026-08-12).
            scored.append((overlap + 2 * len(q_toks & title_toks), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    want_gaps = _ASKS_GAPS_RE.search(query or "") is not None if open_questions is None else open_questions
    for _, r in scored[:limit]:
        body = r["body"]
        m = re.search(r"(?is)## Current understanding\s*(.+?)(?=\n## |\Z)", body)
        # 2,000 chars (was 900: ~71% of a median 3,141-char section was cut).
        excerpt = _bound((m.group(1) if m else body).strip(), 2000)
        if want_gaps:
            m2 = _OPEN_Q_SECTION_RE.search(body)
            if m2 and m2.group(1).strip():
                excerpt += "\n\nOpen questions (what Nova does not yet know):\n" + _bound(m2.group(1).strip(), 600)
        out.append({"title": r["title"], "excerpt": excerpt})
    return out


# ---------------------------------------------------------------------------
# Consolidation cycle
# ---------------------------------------------------------------------------

def _domains_needing_update(db) -> list[dict]:
    """Domain-study monitors with digests newer than their dossier (or with no
    dossier yet). Staleness-first: the longest-unconsolidated domain goes first,
    so partial cycles still rotate coverage instead of starving the tail."""
    try:
        rows = db.fetchall(
            "SELECT m.name AS name, MAX(mr.created_at) AS latest "
            "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
            "WHERE m.name LIKE ? AND mr.status IN ('ok','changed','alert') "
            "  AND mr.value IS NOT NULL AND length(mr.value) > 400 "
            "GROUP BY m.name",
            (_DOMAIN_PREFIX + "%",),
        )
    except Exception:
        return []
    out = []
    for r in rows:
        label = r["name"][len(_DOMAIN_PREFIX):]
        dkey = _slug(label)
        d = get_dossier(db, "domain", dkey)
        if d is None or (d["updated_at"] or "") < (r["latest"] or ""):
            out.append({
                "kind": "domain", "dkey": dkey, "title": label,
                "monitor_name": r["name"],
                "since": d["updated_at"] if d else None,
                "staleness": d["updated_at"] if d else "",   # '' sorts first = never consolidated
            })
    out.sort(key=lambda x: x["staleness"])
    return out


def _storylines_needing_update(db) -> list[dict]:
    """Mature active storylines (moved >= _STORYLINE_MIN_UPDATES times) whose
    events outrun their dossier."""
    try:
        rows = db.fetchall(
            "SELECT s.id, s.story_key, s.title, s.last_updated "
            "FROM storylines s WHERE s.status = 'active' AND s.update_count >= ? "
            "ORDER BY s.last_updated DESC LIMIT 40",
            (_STORYLINE_MIN_UPDATES,),
        )
    except Exception:
        return []
    out = []
    for r in rows:
        d = get_dossier(db, "storyline", r["story_key"])
        if d is None or (d["updated_at"] or "") < (r["last_updated"] or ""):
            out.append({
                "kind": "storyline", "dkey": r["story_key"], "title": r["title"],
                "storyline_id": r["id"],
                "since": d["updated_at"] if d else None,
                "staleness": d["updated_at"] if d else "",
            })
    out.sort(key=lambda x: x["staleness"])
    return out


def _domain_sources(db, monitor_name: str, since: str | None) -> str:
    """New digests for a domain since the last consolidation (newest last so the
    model reads chronologically), bounded to _SOURCE_CAP."""
    try:
        if since:
            rows = db.fetchall(
                "SELECT mr.value AS value, mr.created_at AS created_at "
                "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
                "WHERE m.name = ? AND mr.created_at > ? AND mr.value IS NOT NULL "
                "  AND length(mr.value) > 400 ORDER BY mr.created_at DESC LIMIT 3",
                (monitor_name, since),
            )
        else:
            rows = db.fetchall(
                "SELECT mr.value AS value, mr.created_at AS created_at "
                "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id "
                "WHERE m.name = ? AND mr.value IS NOT NULL AND length(mr.value) > 400 "
                "ORDER BY mr.created_at DESC LIMIT 3",
                (monitor_name,),
            )
    except Exception:
        return ""
    parts = [f"[digest {r['created_at']}]\n{r['value']}" for r in reversed(rows)]
    # Keep the NEWEST digests intact: parts are oldest→newest, so over cap drop
    # whole OLDEST parts from the front — an earlier tail-cut amputated the newest
    # digest, inverting the "new material wins" contract (2026-08-14 audit).
    while len(parts) > 1 and len("\n\n".join(parts)) > _SOURCE_CAP:
        parts.pop(0)
    # If the single newest digest still exceeds the cap, keep its HEAD (lead +
    # primary developments) rather than truncating from the newest end.
    return "\n\n".join(parts)[:_SOURCE_CAP]


_ENTITY_MIN_FACTS = 8       # recent live KG facts required to earn a dossier
_ENTITY_MAX_PER_CYCLE = 2   # entity updates never crowd out the domain backlog
_ENTITY_STOP = frozenset({
    "united states", "china", "russia", "europe", "world", "government",
    "market", "markets", "economy", "technology", "company", "companies",
    "report", "reports", "study", "president", "the united states",
})


def _entities_needing_update(db) -> list[dict]:
    """Consequential ENTITIES that have earned a dossier: subjects with >=
    _ENTITY_MIN_FACTS live KG facts added in the last 14 days (excluding
    generic actors that would produce mush), whose facts outrun their dossier.
    Capped hard so the domain/storyline backlog always keeps priority; new
    entities sort LAST ('~' staleness) and fill only spare cycle capacity."""
    try:
        rows = db.fetchall(
            "SELECT subject, COUNT(*) n, MAX(created_at) latest FROM kg_facts "
            "WHERE superseded_at IS NULL AND created_at > datetime('now','-14 day') "
            "GROUP BY subject HAVING n >= ? ORDER BY n DESC LIMIT 20",
            (_ENTITY_MIN_FACTS,),
        )
    except Exception:
        return []
    out = []
    for r in rows:
        subj = (r["subject"] or "").strip()
        if len(subj) < 4 or subj.lower() in _ENTITY_STOP or "/" in subj:
            continue
        d = get_dossier(db, "entity", _slug(subj))
        if d is not None and (d["updated_at"] or "") >= (r["latest"] or ""):
            continue
        out.append({
            "kind": "entity", "dkey": _slug(subj), "title": subj,
            "subject": subj,
            "since": d["updated_at"] if d else None,
            # New entities yield to the domain backlog ('~' sorts after any
            # timestamp); existing ones compete on real staleness.
            "staleness": d["updated_at"] if d else "~new",
        })
        if len(out) >= _ENTITY_MAX_PER_CYCLE:
            break
    return out


def _entity_sources(db, subject: str, since: str | None) -> str:
    """An entity's consolidation material: a DATED timeline, its verified KG
    facts, and recent digest passages that mention it (bounded).

    The timeline was added 2026-09-03. This prompt asks for "5-10 dated bullets
    of the major shifts, oldest→newest" while the material below carries no
    dates and drops every superseded fact, so the model had to invent the
    history it was being graded on. The KG has been bitemporal since 2026-05-16
    and nothing read the trail. (Timelines are deliberately absent from digest
    synthesis — see _synthesize_from_evidence — because prior context there
    measurably costs grounding. Here prior understanding IS the product.)
    """
    lines = []
    try:
        from app.core.timelines import timeline_block
        block = timeline_block(db, subject, cap=4000)
        if block:
            lines.append(block)
            lines.append("")
    except Exception as e:
        logger.debug("[Knowing] timeline unavailable for %r: %s", subject, e)
    try:
        facts = db.fetchall(
            "SELECT predicate, object, confidence FROM kg_facts "
            "WHERE subject = ? AND superseded_at IS NULL "
            "ORDER BY created_at DESC LIMIT 40", (subject,))
        if facts:
            lines.append("KG FACTS (verified, most recent first):")
            lines += [f"- {subject} {f['predicate']} {f['object']} (conf {f['confidence']})"
                      for f in facts]
    except Exception:
        pass
    try:
        rows = db.fetchall(
            "SELECT mr.value v FROM monitor_results mr "
            "WHERE mr.created_at > datetime('now','-7 day') AND mr.value LIKE ? "
            "ORDER BY mr.created_at DESC LIMIT 6", (f"%{subject}%",))
        if rows:
            lines.append("\nRECENT DIGEST MENTIONS:")
            for r in rows:
                v = r["v"] or ""
                i = v.find(subject)
                if i >= 0:
                    lines.append("… " + v[max(0, i - 160):i + 320].replace("\n", " ").strip() + " …")
    except Exception:
        pass
    return "\n".join(lines)[:_SOURCE_CAP]


def _storyline_sources(db, storyline_id: int, since: str | None) -> str:
    """Storyline's current summary + its events since the last consolidation."""
    try:
        # Meta-annotation exclusion (2026-08-31): QA notes appended to
        # storyline_events ("Fresh-check could NOT be confirmed", sourcing
        # notes) were filtered on the API timeline but flowed into dossier
        # consolidation prompts here as if they were narrative events.
        from app.core.storylines import EVENT_META_EXCL_SQL
        s = db.fetchone("SELECT title, summary FROM storylines WHERE id = ?", (storyline_id,))
        if since:
            evs = db.fetchall(
                "SELECT summary, created_at FROM storyline_events "
                f"WHERE storyline_id = ? AND created_at > ? {EVENT_META_EXCL_SQL} "
                "ORDER BY created_at LIMIT 40",
                (storyline_id, since),
            )
        else:
            evs = db.fetchall(
                "SELECT summary, created_at FROM storyline_events "
                f"WHERE storyline_id = ? {EVENT_META_EXCL_SQL} "
                "ORDER BY created_at LIMIT 60",
                (storyline_id,),
            )
    except Exception:
        return ""
    lines = [f"[tracked thread state] {s['summary']}"] if s and s["summary"] else []
    lines += [f"- ({e['created_at']}) {e['summary']}" for e in evs]
    return "\n".join(lines)[:_SOURCE_CAP]


async def _update_dossier(db, cand: dict, sources: str, syn_model: str | None,
                          prompt_template: str = _UPDATE_PROMPT) -> dict | None:
    """One consolidation: prior body + new material -> revised understanding.
    Persists the revision trail; mints an optional FORECAST (gated). Returns
    {'title','changed'} on change, else None."""
    if len(sources) < 300:
        return None   # nothing substantive to consolidate
    prior_row = await asyncio.to_thread(get_dossier, db, cand["kind"], cand["dkey"])
    prior = (prior_row["body"] if prior_row else "")[:_PRIOR_CAP] or "(none — first consolidation)"

    # Judgment feedback (2026-08-14): when this dossier family has a resolved
    # forecast track record, the model sees its own calibration before stating
    # the next confidence — stated numbers should be WORTH their number.
    try:
        from app.core.forecasts import calibration
        cal = await asyncio.to_thread(
            calibration, db, key_prefix=f"dossier:{cand['dkey'][:60]}", min_n=5)
        if cal and abs(cal["gap"]) > 0.05:
            direction = "overconfident" if cal["gap"] > 0 else "underconfident"
            sources = (sources + f"\n\nCALIBRATION NOTE: your last {cal['n']} forecasts "
                       f"here delivered {cal['hit_rate']:.0%} against a stated "
                       f"{cal['mean_conf']:.0%} — you ran {direction}; weight the next "
                       f"FORECAST confidence accordingly.")
    except Exception:
        pass
    # Global calibration record (2026-09-01): the per-family note above only
    # reached 4 of 29 families; every mint now sees how stated confidence has
    # actually delivered across all of Nova's forecasts.
    try:
        from app.core.forecasts import global_calibration_note
        _gnote = await asyncio.to_thread(global_calibration_note, db)
        if _gnote:
            sources = sources + "\n\n" + _gnote
    except Exception:
        pass

    prompt = prompt_template.format(title=cand["title"], prior=prior, sources=sources)
    try:
        out = await _bg_invoke(
            [{"role": "user", "content": prompt}],
            # 3600 (was 2600→1600): 2026-08-17 the truncation tripwire caught a
            # dense-domain dossier hitting the 2600 cap (eval=2600, ~10.5k chars,
            # cut mid-generation — losing the trailing 'Open questions' + CHANGED
            # line). The 27B writes ~6-7k chars for a typical dossier but 10k+ for
            # dense domains; 3600 tokens (~14k chars) lets those finish cleanly.
            max_tokens=3600, temperature=0.2, model=syn_model,
            # prior ≤8k + sources ≤18k + instructions ≈ 27.5k chars ≈ ~7.5k tokens
            # + 3.6k generation → 16384 holds with headroom (num_ctx discipline;
            # 16384 is measured GPU-safe in the deep_research lattice).
            num_ctx=16384,
        )
    except Exception as e:
        logger.warning("[Knowing] dossier LLM failed for %r: %s", cand["title"], e)
        return None
    out = (out or "").strip()
    if len(out) < 200 or "## Current understanding" not in out:
        # Un-silenced: a malformed consolidation must be loud, not a quiet skip.
        logger.warning("[Knowing] dossier update for %r returned malformed body "
                       "(%d chars) — head: %r", cand["title"], len(out), out[:160])
        return None

    m = _CHANGED_RE.search(out)
    changed = m.group(1).strip() if m else ("initial dossier" if not prior_row else "updated")
    # Normalize citations: the model sometimes embellishes bare (outlet.com)
    # cites into markdown links with INVENTED URLs (seen live 2026-08-12:
    # technologyreview.com → a fabricated technologyresearch.com URL). Keep the
    # visible outlet text, drop the un-verifiable link target.
    out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)

    # Forecast minting (knowing tier rung 3, 2026-08-12): consolidations end
    # with a mandatory FORECAST line ('FORECAST: none' when nothing falsifiable
    # — the v1 optional ask was NEVER taken by the 27B at temp 0.2; zero mints
    # across 60+ consolidations, probe-confirmed 2026-08-13). Parsed into the
    # self-grading forecast loop; failures logged, never fatal.
    try:
        from app.config import config as _cfg
        if getattr(_cfg, "ENABLE_FORECASTS", True):
            from app.core.forecasts import parse_and_store_forecast
            fid = await asyncio.to_thread(parse_and_store_forecast, db, out,
                                          storyline_key=f"dossier:{cand['dkey'][:60]}",
                                          source_monitor="Knowledge Consolidation")
            if fid:
                logger.info("[Knowing] forecast minted (#%s) from %r", fid, cand["title"])
            elif "FORECAST:" in out.upper() and "FORECAST: NONE" not in out.upper():
                # a FORECAST line was emitted but didn't parse/store — the exact
                # silent seam that zeroed minting for weeks pre-2026-08-13.
                logger.warning("[Knowing] FORECAST line present but not stored for %r "
                               "— mint format drift?", cand["title"])
    except Exception as e:
        logger.warning("[Knowing] forecast minting failed for %r: %s", cand["title"], e)

    body = _bound(_flag_weak_citations(
        _FORECAST_RE.sub("", _CHANGED_RE.sub("", out)).strip()))

    if prior_row:
        await asyncio.to_thread(
            db.execute,
            "INSERT INTO dossier_revisions (dossier_id, body, valid_from) VALUES (?, ?, ?)",
            (prior_row["id"], prior_row["body"], prior_row["updated_at"]),
        )
        await asyncio.to_thread(
            db.execute,
            "UPDATE dossiers SET body = ?, changed_note = ?, title = ?, "
            "update_count = update_count + 1, updated_at = datetime('now') WHERE id = ?",
            (body, changed, cand["title"], prior_row["id"]),
        )
    else:
        await asyncio.to_thread(
            db.execute,
            "INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (cand["kind"], cand["dkey"], cand["title"], body, changed),
        )
    logger.info("[Knowing] %s dossier %r consolidated (%s)",
                cand["kind"], cand["title"], changed[:100])
    return {"title": cand["title"], "changed": changed}


def _extract_open_questions(body: str, *, limit: int = 2) -> list[str]:
    """Pull the bullet/numbered lines out of a dossier's Open questions section."""
    m = _OPEN_Q_SECTION_RE.search(body or "")
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        q = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
        if len(q) >= 20:
            out.append(q[:300])
        if len(out) >= limit:
            break
    return out


async def _update_state_of_world(db, syn_model: str | None) -> dict | None:
    """The capstone: consolidation ACROSS domain dossiers — kind='meta',
    dkey='state-of-the-world'. Sources = the freshest per-domain Current-
    understanding sections; the prompt demands throughlines and tensions
    BETWEEN domains, not a per-domain list."""
    rows = await asyncio.to_thread(
        db.fetchall,
        "SELECT title, body, updated_at FROM dossiers WHERE kind = 'domain' AND body != '' "
        "ORDER BY updated_at DESC LIMIT 12",
    )
    parts = []
    for r in rows:
        m = re.search(r"(?is)## Current understanding\s*(.+?)(?=\n## |\Z)", r["body"])
        head = (m.group(1) if m else r["body"]).strip()[:1100]
        parts.append(f"[{r['title']} — as of {r['updated_at']}]\n{head}")
    if len(parts) < 3:
        return None   # a world-view from <3 domains would be a caricature
    sources = "\n\n".join(parts)[:_SOURCE_CAP]
    cand = {"kind": "meta", "dkey": "state-of-the-world", "title": "State of the World"}
    return await _update_dossier(db, cand, sources, syn_model, prompt_template=_WORLD_PROMPT)


async def consolidate_dossiers(db) -> str:
    """Full cycle: find domains/storylines that outran their dossiers, distill
    the newest material into revised understanding — staleness-first, bounded to
    _MAX_UPDATES_PER_CYCLE. Sequential on purpose: these are big-model calls on
    one GPU; the heartbeat is latency-tolerant.

    After the per-dossier updates (2026-08-12, rungs 2-4 of the knowing program):
      - each updated dossier's top Open Question feeds the curiosity queue
        (capped) — the epistemic frontier becomes targeted research;
      - when >=2 domain dossiers changed, the cross-domain state-of-the-world
        dossier is re-consolidated (the capstone);
      - falsifiable FORECAST lines are minted inside _update_dossier."""
    from app.config import config as _cfg
    syn_model = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None

    candidates = await asyncio.to_thread(
        lambda: _domains_needing_update(db) + _storylines_needing_update(db)
        + _entities_needing_update(db))
    candidates.sort(key=lambda x: x["staleness"])
    if not candidates:
        return "KNOWING | all dossiers current — nothing to consolidate"

    picks = candidates[:_MAX_UPDATES_PER_CYCLE]
    # Entity cold-start guarantee (audit 2026-08-13): '~new' staleness sorts
    # after every real timestamp, and with ~37 domains regenerating staleness
    # daily the backlog never drains — so no entity could ever earn its FIRST
    # dossier (zero existed after 60+ consolidations). Reserve one slot per
    # cycle for the top entity candidate when none makes the cut naturally;
    # once seeded, entities compete on real staleness like everyone else.
    if not any(c["kind"] == "entity" for c in picks):
        ent = next((c for c in candidates if c["kind"] == "entity"), None)
        if ent is not None:
            picks = picks[:_MAX_UPDATES_PER_CYCLE - 1] + [ent]

    updated, attempted, domain_updates = [], 0, 0
    for cand in picks:
        attempted += 1
        if cand["kind"] == "domain":
            sources = await asyncio.to_thread(_domain_sources, db, cand["monitor_name"], cand["since"])
        elif cand["kind"] == "entity":
            sources = await asyncio.to_thread(_entity_sources, db, cand["subject"], cand["since"])
        else:
            sources = await asyncio.to_thread(_storyline_sources, db, cand["storyline_id"], cand["since"])
        try:
            res = await _update_dossier(db, cand, sources, syn_model)
        except Exception as e:
            logger.warning("[Knowing] consolidation failed for %r: %s", cand["title"], e)
            continue
        if res:
            updated.append(res)
            # Open-questions ledger (2026-09-02): reconcile what the new body
            # asks against what it asked before; record REVISED: lines.
            try:
                from app.core.questions import sync_after_consolidation
                await asyncio.to_thread(sync_after_consolidation, db, cand["kind"], cand["dkey"])
            except Exception as e:
                logger.debug("[Knowing] question ledger sync failed for %r: %s", cand["title"], e)
            if cand["kind"] == "domain":
                domain_updates += 1
            # Curiosity <- Open questions (rung 2): the dossier's own stated
            # unknowns become research targets. Top question per dossier,
            # <=3 adds per cycle; CuriosityQueue.add dedups internally.
            # RESEARCHABLE shapes only (2026-08-14): dossier questions are
            # often futures ("Will X…?", "How will Y adapt…?") which web
            # research can structurally never resolve — they clogged the
            # queue to 52 pending / 0 resolved with the researcher burning
            # an attempt per hour on them. Futures belong to the FORECAST
            # line; curiosity gets present-tense unknowns.
            if len([u for u in updated if u.get("questions_fed")]) < 3:
                try:
                    row = await asyncio.to_thread(get_dossier, db, cand["kind"], cand["dkey"])
                    qs = _extract_open_questions(row["body"] if row else "", limit=3)
                    qs = [q for q in qs if not _is_future_question(q)
                          and not q.lower().startswith("watch for")][:1]
                    if qs:
                        from app.core.curiosity import CuriosityQueue
                        # Bounded title prefix: a storyline title can be long
                        # enough to bury the question in the search query
                        # ("DeFi Institutionalization & Regulatory Bifurcation
                        # (Morpho/Aave/Ripple): What is the current status of…").
                        _topic = f"{cand['title'][:60].rstrip()}: {qs[0]}"
                        _cid = await asyncio.to_thread(
                            CuriosityQueue(db).add, _topic,
                            "dossier_open_question", 0.6)
                        res["questions_fed"] = _cid > 0
                        if _cid > 0:
                            try:
                                from app.core.questions import mark_queued
                                await asyncio.to_thread(mark_queued, db, cand["dkey"], qs[0], _cid)
                            except Exception as e:
                                logger.debug("[Knowing] ledger mark_queued failed: %s", e)
                        else:
                            logger.info("[Knowing] %r question not queued (curiosity queue "
                                        "at capacity) — stays open in the ledger", cand["title"])
                        logger.info("[Knowing] curiosity fed from %r: %s", cand["title"], qs[0][:90])
                except Exception as e:
                    logger.debug("[Knowing] curiosity feed failed for %r: %s", cand["title"], e)

    # Capstone (rung 4): re-consolidate the state of the world when the domain
    # picture moved. Does not count against the per-cycle cap.
    # Ledger hygiene (2026-09-02): questions of closed storyline threads are
    # no longer the frontier — retire them so the counts stay honest.
    try:
        from app.core.questions import retire_orphaned
        _n_retired = await asyncio.to_thread(retire_orphaned, db)
        if _n_retired:
            logger.info("[Knowing] question ledger: %d question(s) retired with their closed storylines",
                        _n_retired)
    except Exception as e:
        logger.debug("[Knowing] ledger orphan sweep failed: %s", e)

    world_note = ""
    if domain_updates >= 2:
        try:
            w = await _update_state_of_world(db, syn_model)
            if w:
                world_note = f"\n- 🌍 **State of the World** — {w['changed']}"
                logger.info("[Knowing] state-of-the-world consolidated (%s)", w["changed"][:100])
        except Exception as e:
            logger.warning("[Knowing] state-of-the-world failed: %s", e)

    # Thinking rung, brick one (2026-08-14): after consolidation, look for
    # dossiers asserting materially different values for the same quantity.
    # A noticed tension is surfaced AND becomes targeted research — the
    # knowing tier investigating its own disagreements.
    tension_note = ""
    try:
        tensions = await asyncio.to_thread(_numeric_tensions, db)
        if tensions:
            tension_note = "\n- ⚡ **Tension:** " + "\n- ⚡ **Tension:** ".join(tensions)
            logger.info("[Knowing] %d cross-dossier tension(s): %s",
                        len(tensions), tensions[0][:120])
            from app.core.curiosity import CuriosityQueue
            # Urgency 0.5 (was 0.7): a tension is worth a NORMAL-priority research
            # pass, not the daemon's CRITICAL loop (fires every idle ~5-min tick
            # whenever any pending item is ≥0.7). At 0.7 the false-tension stream
            # drove ~124 unresolvable brain.think()/day (2026-08-18 audit). Below
            # 0.7 it still gets researched via get_next and self-limits after
            # MAX_CURIOSITY_ATTEMPTS, but never pins critical_curiosity>0.
            await asyncio.to_thread(
                CuriosityQueue(db).add,
                f"Resolve contradiction: {_wtrim(tensions[0], 180)}",
                "dossier_tension", 0.5)
    except Exception as e:
        logger.debug("[Knowing] tension scan failed: %s", e)

    backlog = max(0, len(candidates) - _MAX_UPDATES_PER_CYCLE)
    if not updated:
        return (f"KNOWING | {attempted} candidate(s) checked, no dossier changed"
                + (f" ({backlog} queued)" if backlog else ""))
    lines = [f"## 📚 KNOWING — {len(updated)} dossier(s) consolidated"
             + (f" ({backlog} queued for next cycle)" if backlog else "")]
    for u in updated:
        lines.append(f"- **{u['title']}** — {u['changed']}")
    return "\n".join(lines) + world_note + tension_note
