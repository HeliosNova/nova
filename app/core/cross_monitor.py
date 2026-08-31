"""Cross-monitor synthesis — find patterns no single monitor sees.

Each monitor runs in isolation: Crypto sees crypto, Geopolitics sees
geopolitics, Cybersecurity sees breaches. Reality is correlated — a
cyber incident at an exchange shows up in all three streams as
fragments. This module reads the last 24-48h of monitor_results,
finds entities/themes recurring across DIFFERENT monitor categories,
and asks the LLM to write a synthesis that names the cross-cutting
pattern.

Output is written:
  - As a `monitor_results` row for the synthesis monitor itself
  - As KG facts with `provenance='cross_synthesis'` so the patterns
    survive into Nova's working knowledge

Called from heartbeat_loop.py via check_type='synthesis'.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass

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
    from app.core.llm import invoke_nothink as _invoke
    return await _invoke(*args, **kwargs)



# Stopwords + low-signal tokens we don't want as cluster keys.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "as", "from", "this", "that", "these", "those", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "can",
    "what", "which", "when", "where", "how", "why", "who", "you", "your",
    "we", "our", "they", "them", "their", "i", "me", "my", "it", "its",
    "more", "most", "less", "many", "much", "some", "any", "all", "no", "not",
    "than", "then", "also", "just", "only", "very", "well", "still", "now",
    "today", "yesterday", "week", "year", "month", "day", "days", "hour",
    "hours", "ago", "new", "news", "report", "reports", "according", "said",
    "says", "yet", "while", "after", "before", "during", "since", "over",
    "under", "between", "into", "out", "up", "down", "off", "about",
    "see", "show", "shows", "showed", "find", "found", "make", "made",
    "first", "second", "third", "last", "next", "one", "two", "three",
    "monitor", "result", "alert", "summary", "update", "fetch", "check",
    "data", "info", "details", "items", "list", "links", "story", "stories",
    "article", "articles", "headline", "headlines", "rumor", "rumors",
    # Extremely generic monitor noun-phrases:
    "current", "events", "watch", "tracking", "highlights", "developments",
    # Filler/meta tokens that surfaced as themes in the first live run
    # because every monitor result mentions them — useless cluster keys.
    "across", "within", "around", "based", "related", "regarding",
    "intelligence", "source", "sources", "date", "dates", "significant",
    "notable", "major", "key", "important", "specific", "general",
    "overall", "total", "average", "various", "several", "multiple",
    "include", "includes", "including", "involve", "involves", "involving",
    "appears", "appeared", "showing", "indicating", "according", "per",
    "such", "each", "every", "another", "other", "others",
    # Months — never a useful cluster key on their own (every article has
    # a date). If a month matters cross-monitor, the LLM synthesis already
    # contains it.
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # Domain-noise that recurs but isn't a real entity
    "company", "companies", "country", "countries", "people", "person",
    "team", "teams", "group", "market", "markets", "industry", "industries",
    "sector", "sectors", "system", "systems", "service", "services",
    "product", "products", "platform", "platforms", "technology",
    "technologies", "tool", "tools",
    # Verbs / adjectives that describe what monitors report (not what's
    # being reported on). Surfaced in 2nd live run.
    "matters", "announced", "represents", "occurred", "results", "reports",
    "released", "launched", "unveiled", "revealed", "confirmed", "stated",
    "expected", "anticipated", "described", "discussed", "considered",
    "introduced", "developed", "implemented", "established", "demonstrated",
    "addressed", "achieved", "completed", "continued", "remained",
    "ongoing", "pending", "planned", "proposed", "scheduled",
    # SearXNG/results scaffolding
    "google", "search", "searches", "query", "queries", "snippet", "result",
    "website", "websites", "page", "pages",
    # Even more abstract nouns the LLM keeps flagging as noise
    "activity", "activities", "reported", "critical", "development",
    "infrastructure", "global", "international", "national", "regional",
    "potential", "potentially", "likely", "expected", "estimated",
    "operations", "operation", "operating",
    # Long prepositions/conjunctions that bypassed the short-word check
    "through", "without", "within", "between", "across", "among",
    "during", "before", "after", "above", "below", "beyond",
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "although", "despite", "whereas", "regardless",
    "according", "including", "involving", "regarding", "concerning",
    "following", "preceding", "remaining", "continuing",
    # Frequently-recurring abstract adjectives
    "recent", "current", "various", "additional", "particular",
    "specific", "general", "broad", "narrow", "common", "typical",
    "primary", "secondary", "tertiary", "official", "unofficial",
    # Digest FORMATTING metadata — never a real cross-monitor theme. These leak
    # from the "✓ Confirmed by N outlets / Primary source:" digest scaffolding
    # (2026-06-21: "outlets" was surfacing as a top false theme).
    "outlets", "outlet", "confirmed", "corroborated", "headline", "headlines",
    "story", "stories", "update", "updates", "developments",
    # More digest scaffolding leaking as false themes (live-caught 2026-06-21):
    # the "💡 Insight" analysis line and "N cross-confirmed by multiple outlets"
    # footer in domain_study_runner recur across every domain digest.
    "insight", "insights", "cross-confirmed", "recurrence", "recurring",
    "sourced", "convergence", "intelligence",
    # deep_research.py digest scaffolding (2026-06-23): these template words
    # surfaced as the TOP false themes ("[learned] across 45 monitors",
    # "[overview]", "[million]") because every overview header/section uses them.
    "learned", "overview", "domain", "briefing", "connections", "throughline",
    "bottom", "lead", "million", "billion", "trillion", "percent", "facts",
    "fact", "weekly", "researched", "verified", "corroborated",
})

# Match "real" content tokens: 4+ chars, lowercase letters or numbers
_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9-]{3,}\b")
# Proper-noun-ish: 2+ capitalised tokens together (e.g. "United States",
# "Federal Reserve", "Open AI"). Single capitalised tokens (Apple, Russia)
# also count.
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}(?:\s+[A-Z][a-zA-Z0-9]+){0,3}\b")


@dataclass
class ThemeCluster:
    """A keyword/phrase that appeared across multiple monitors."""

    key: str
    monitors: set[str]
    snippets: list[tuple[str, str]]  # (monitor_name, snippet) pairs

    @property
    def breadth(self) -> int:
        return len(self.monitors)


def _extract_signals(text: str) -> set[str]:
    """Extract candidate cluster keys from a monitor result.

    Strategy: prefer multi-word proper-noun phrases (high specificity).
    Fall back to single tokens only if they're long AND not stopwords —
    short tokens like "april", "across", "source" are useless cluster keys
    because they appear in every monitor result.
    """
    if not text:
        return set()
    out: set[str] = set()

    # Proper-noun phrases — highest signal cluster keys (multi-word capitalised)
    for m in _PROPER_RE.findall(text):
        norm = m.strip().lower()
        if 4 <= len(norm) <= 60:
            words = norm.split()
            # Drop if every word is a stopword (e.g. "And The")
            substantive = [w for w in words if w not in _STOPWORDS]
            if not substantive:
                continue
            # Multi-word phrase OR single word ≥ 6 chars
            if len(words) >= 2 or len(words[0]) >= 6:
                out.add(norm)

    # Single lowercase tokens — much stricter: ≥7 chars, not a stopword,
    # not a pure number/year. The first synthesis run found "april",
    # "across", "within", "source" — all 5-6 char filler. 7+ excludes them.
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS or len(tok) < 7:
            continue
        if tok.isdigit():
            continue
        # Skip year-like tokens (4 digits + letters: "2026q1")
        if re.match(r"^\d{4}", tok):
            continue
        out.add(tok)

    return out


def _gather_recent_outputs(
    db, *, hours: int, max_per_monitor: int
) -> dict[str, list[str]]:
    """Group recent monitor result `value`s by monitor name.

    Excludes system/health monitors (their content is about Nova's internal
    state — not cross-cuttable real-world signal).
    """
    rows = db.fetchall(
        "SELECT m.name AS name, m.category AS category, m.check_type AS check_type, "
        "       mr.value AS value, mr.created_at AS created_at "
        "FROM monitor_results mr "
        "JOIN monitors m ON m.id = mr.monitor_id "
        "WHERE mr.created_at > datetime('now', ?) "
        "  AND mr.status IN ('ok','changed','alert') "
        "  AND m.category = 'content' "
        # Exclude meta-monitors that synthesize OVER monitor output — feeding
        # their narrative/synthesis back in creates a loop where their own
        # scaffolding ("insight", "cross-confirmed") recurs as false themes.
        "  AND m.check_type NOT IN ('storyline', 'synthesis', 'forecast_resolve') "
        "  AND mr.value IS NOT NULL AND length(mr.value) > 80 "
        "ORDER BY mr.created_at DESC LIMIT 2000",
        (f"-{hours} hours",),
    )

    grouped: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        name = r["name"]
        if len(grouped[name]) >= max_per_monitor:
            continue
        grouped[name].append(r["value"])
    return grouped


def _build_clusters(
    grouped_outputs: dict[str, list[str]],
    *,
    min_breadth: int,
    max_clusters: int,
) -> list[ThemeCluster]:
    """Find tokens/phrases appearing across ≥ min_breadth distinct monitors."""
    # signal -> {monitor_name: [snippet, ...]}
    by_signal: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for monitor_name, results in grouped_outputs.items():
        for raw in results:
            signals = _extract_signals(raw)
            if not signals:
                continue
            # Pull a focused excerpt — first 200 chars after the first signal hit
            # if possible, else just the head. Keeps the synthesis prompt small.
            head = (raw[:280]).replace("\n", " ").strip()
            for sig in signals:
                # Cap snippets per (signal, monitor) to avoid one monitor
                # dominating a cluster's evidence
                if len(by_signal[sig][monitor_name]) >= 2:
                    continue
                by_signal[sig][monitor_name].append(head)

    clusters: list[ThemeCluster] = []
    for sig, per_monitor in by_signal.items():
        if len(per_monitor) < min_breadth:
            continue
        snippets: list[tuple[str, str]] = []
        for mname, snips in per_monitor.items():
            for s in snips:
                snippets.append((mname, s))
        clusters.append(
            ThemeCluster(
                key=sig,
                monitors=set(per_monitor.keys()),
                snippets=snippets,
            )
        )
    # Most cross-cutting first; tiebreak: more evidence
    clusters.sort(key=lambda c: (c.breadth, len(c.snippets)), reverse=True)
    return clusters[:max_clusters]


_SYNTHESIS_PROMPT = (
    "You are reading isolated outputs from {n} different monitors that all "
    "happened to mention '{theme}' in the last {hours} hours.\n\n"
    "Your job: in 2–4 sentences, name the cross-cutting pattern these "
    "monitors are seeing. Do NOT just list each monitor — identify the "
    "underlying real-world event, trend, or correlation that explains why "
    "this term recurs across these specific domains. If there is no real "
    "underlying pattern (the term is generic or coincidental), say so "
    "explicitly.\n\n"
    "Monitors and excerpts:\n{evidence}\n\n"
    "Write the synthesis as a single paragraph, no preamble, no headers, no "
    "lists. Start with the noun phrase of the pattern."
)


_KEY_VALIDATION_PROMPT = """You will see a list of recurring tokens that appeared across multiple intelligence-feed outputs. For each token, judge whether it is:

  - SUBSTANTIVE: a real entity, place, person, technology, event, or topic that could plausibly explain why multiple monitors mention it (e.g. "tesla", "ukraine war", "bitcoin etf", "openai")
  - FILLER: a generic word, abstract noun, time word, or scaffolding that recurs trivially across any monitor outputs (e.g. "without", "million", "research", "across", "infrastructure", "current")

Be ruthless. If in doubt, mark FILLER. The goal is to surface only tokens that name something concrete the world is actually doing.

Tokens to classify:
{tokens}

Respond with STRICT JSON: an object whose keys are the tokens and values are either "SUBSTANTIVE" or "FILLER". No preamble."""


async def _validate_cluster_keys(keys: list[str]) -> set[str]:
    """Ask the LLM to classify cluster keys; return the substantive subset.

    One LLM call gates all clusters before the expensive synthesis pass.
    On error we fail-open (return all keys) rather than dropping work.
    """
    if not keys:
        return set()
    from app.core.llm import invoke_nothink
    prompt = _KEY_VALIDATION_PROMPT.format(
        tokens="\n".join(f"- {k}" for k in keys)
    )
    try:
        text = await _bg_invoke(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            json_prefix="{",
            max_tokens=600,
            temperature=0.0,
        )
    except Exception as e:
        logger.warning("[Synthesis] cluster-key validation failed: %s", e)
        return set(keys)  # fail-open

    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        import json as _json
        parsed = _json.loads(text)
    except (ValueError, TypeError):
        # Log the fail-open (2026-08-18): a silently-dead validator lets every
        # candidate cluster through, so the expensive synthesis + causal-probe pass
        # runs on filler the validator was supposed to gate — with no signal.
        logger.warning("[Synthesis] cluster-key validation unparseable — failing open (%d keys)", len(keys))
        return set(keys)  # fail-open

    if not isinstance(parsed, dict):
        logger.warning("[Synthesis] cluster-key validation returned non-dict %s — failing open",
                       type(parsed).__name__)
        return set(keys)

    keep: set[str] = set()
    for k in keys:
        verdict = parsed.get(k) or parsed.get(k.lower()) or ""
        if isinstance(verdict, str) and "substantive" in verdict.lower():
            keep.add(k)
    if not keep:
        # If LLM rejected everything, fall back to top-by-breadth so we don't
        # blank the synthesis output entirely on a model glitch
        logger.info("[Synthesis] LLM dropped all %d cluster keys — keeping top 2 anyway", len(keys))
        keep = set(keys[:2])
    return keep


_CAUSAL_PROMPT = (
    "These excerpts about '{theme}' come from {n} different intelligence domains "
    "({domains}). Identify the single most-likely CAUSAL CHAIN connecting them — "
    "how a development in one domain drives another (e.g. 'export controls' → "
    "'chip revenue' → 'AI buildout').\n\n"
    "{evidence}\n\n"
    'Return JSON only: {{"chain": [{{"cause": "short entity", "effect": "short entity"}}], '
    '"confidence": 0.0-0.85}}. Use SHORT entity names (1-4 words), never sentences. '
    "Evidence that genuinely spans domains almost always carries at least one "
    "causal link — extract the STRONGEST one. Return an empty chain ONLY when "
    "the overlap is purely coincidental keyword usage with no plausible mechanism."
)


async def _causal_probe(cluster: ThemeCluster, *, hours: int) -> list[dict]:
    """Ask the LLM for a cross-domain causal chain. Returns [{cause, effect, confidence}].

    Cross-domain by construction (clusters span >= min_breadth distinct monitors).
    Short entity names so the chain lands as real `caused_by` KG facts, not noise.
    """
    from app.core.llm import invoke_nothink, extract_json_object
    domains = ", ".join(sorted({re.sub(r"^Domain Study:\s*", "", m) for m in cluster.monitors})[:6])
    evidence = "\n".join(f"- [{m}] {s}" for m, s in cluster.snippets[:8])
    prompt = _CAUSAL_PROMPT.format(theme=cluster.key, n=cluster.breadth, domains=domains, evidence=evidence)
    # Route to the synthesis 27B (2026-08-14): the 9B returned {"chain": []} on
    # a probe with a TEXTBOOK causal chain (Iran threat → oil → equities) — the
    # KG-banking arm was silently dead since Jun 23 while the narrative shipped.
    # Same remedy as KG extraction (4.6× grounded yield on the 27B).
    from app.config import config as _cfg
    _syn = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None
    try:
        raw = await _bg_invoke(
            [{"role": "user", "content": prompt}],
            json_mode=True, json_prefix="{", max_tokens=300, temperature=0.1,
            model=_syn,
        )
    except Exception as e:
        logger.warning("[Synthesis] causal probe failed for '%s': %s", cluster.key, e)
        return []
    if not raw:
        logger.info("[Synthesis] causal probe returned nothing for '%s'", cluster.key)
        return []
    try:
        # `json` was never imported in this module (2026-08-29) — the only
        # import is `import json as _json` inside a DIFFERENT function, so this
        # line raised NameError on every call. The bug was invisible because the
        # except below caught it and extract_json_object() quietly produced the
        # same answer: the intended fast path simply never ran, and every causal
        # probe paid an exception. Delete the fallback and this breaks outright.
        import json as _json_mod
        data = _json_mod.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        data = extract_json_object(raw)
    if not isinstance(data, dict):
        return []
    conf = data.get("confidence", 0.6)
    try:
        conf = max(0.3, min(0.85, float(conf)))
    except (TypeError, ValueError):
        conf = 0.6
    out = []
    for link in (data.get("chain") or [])[:4]:
        if not isinstance(link, dict):
            continue
        cause = str(link.get("cause", "")).strip().lower()
        effect = str(link.get("effect", "")).strip().lower()
        # Guard: short entity names only (the KG garbage gate also enforces this).
        if cause and effect and cause != effect and len(cause.split()) <= 5 and len(effect.split()) <= 5:
            out.append({"cause": cause, "effect": effect, "confidence": conf})
    return out


async def _synthesize_cluster(cluster: ThemeCluster, *, hours: int) -> str:
    """Ask the LLM to name the cross-cutting pattern for one cluster."""
    from app.core.llm import invoke_nothink

    evidence_lines = []
    for mname, snip in cluster.snippets[:8]:  # cap evidence
        evidence_lines.append(f"- [{mname}] {snip}")
    evidence = "\n".join(evidence_lines)

    prompt = _SYNTHESIS_PROMPT.format(
        theme=cluster.key,
        n=cluster.breadth,
        hours=hours,
        evidence=evidence,
    )

    try:
        text = await _bg_invoke(
            [{"role": "user", "content": prompt}],
            max_tokens=320,
            temperature=0.2,
        )
    except Exception as e:
        logger.warning("[Synthesis] LLM call failed for '%s': %s", cluster.key, e)
        return ""

    text = (text or "").strip()
    # Reject obvious non-answers
    low = text.lower()
    if not text or len(text) < 40:
        return ""
    if low.startswith(("i cannot", "i'm sorry", "as an ai", "[error", "no underlying")):
        # "no underlying" is acceptable if the model genuinely judged it noise
        if "no underlying" in low:
            return text
        return ""
    # Whole-word bound (2026-08-31). This was `text[:1200]` — a hard mid-word
    # cut — while the fix for exactly that bug sat ONE FUNCTION below
    # (_preview, built 2026-08-15 after a cross-monitor preview shipped
    # "…Leinweber Foundat…") and was never applied here. max_tokens=320 above
    # generates ~1200-1500 chars, so the cap bit MOST outputs: 299 stored
    # digest inserts measured ending mid-word at exactly length 1200. Prefer
    # the last sentence boundary in the tail; fall back to the last whole
    # word + ellipsis.
    if len(text) <= 1200:
        return text
    cut = text[:1200]
    tail = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if tail >= 1000:
        return cut[:tail + 1]
    ws = cut.rfind(" ")
    return (cut[:ws].rstrip() + "…") if ws > 0 else cut


def _preview(text: str, cap: int = 280) -> str:
    """A short digest preview that never cuts mid-word — the 2026-08-15 overnight
    watch caught a cross-monitor preview shipping '…Leinweber Foundat…'. Trims to
    the last whole word within `cap` and appends an ellipsis only when it actually
    truncated."""
    t = (text or "").strip()
    if len(t) <= cap:
        return t
    head = t[:cap]
    if " " in head:
        head = head.rsplit(None, 1)[0]
    return head.rstrip(" ,;:—-") + "…"


async def synthesize_across_monitors(
    db,
    kg,
    *,
    hours: int = 36,
    min_breadth: int = 3,
    max_clusters: int = 8,
    max_per_monitor: int = 3,
) -> dict:
    """Top-level entry — read recent monitor outputs, find cross-cutting
    themes, write syntheses to KG, return a summary dict.
    """
    # to_thread: this fetchall scans up to 2000 monitor_results rows — a
    # sync DB read on the event loop (self-flagged by _warn_if_event_loop
    # live on 2026-08-24) blocks every coroutine while SQLite pages.
    grouped = await asyncio.to_thread(
        _gather_recent_outputs, db, hours=hours, max_per_monitor=max_per_monitor
    )
    if not grouped:
        return {
            "summary": "CROSS-SYNTHESIS | no recent content monitor results",
            "themes": 0,
            "kg_writes": 0,
        }

    clusters = _build_clusters(
        grouped, min_breadth=min_breadth, max_clusters=max_clusters
    )
    if not clusters:
        return {
            "summary": (
                f"CROSS-SYNTHESIS | scanned {len(grouped)} monitors / "
                f"{sum(len(v) for v in grouped.values())} results — "
                f"no themes recurring across {min_breadth}+ monitors"
            ),
            "themes": 0,
            "kg_writes": 0,
        }

    # LLM-validate cluster keys before spending tokens on full synthesis. The
    # stopword list catches the obvious filler ("april", "across") but the
    # LLM catches the abstract-noun filler the regex misses ("research",
    # "infrastructure") so we don't burn 6 expensive synthesis calls on noise.
    candidate_keys = [c.key for c in clusters]
    substantive = await _validate_cluster_keys(candidate_keys)
    pre_filter = len(clusters)
    clusters = [c for c in clusters if c.key in substantive]
    logger.info(
        "[Synthesis] cluster-key validation: %d → %d clusters",
        pre_filter, len(clusters),
    )
    if not clusters:
        return {
            "summary": (
                f"CROSS-SYNTHESIS | scanned {len(grouped)} monitors, "
                f"{pre_filter} candidate clusters all classified as filler"
            ),
            "themes": 0,
            "kg_writes": 0,
        }

    summaries: list[str] = [
        f"CROSS-SYNTHESIS | scanned {len(grouped)} monitors, "
        f"{len(clusters)} substantive themes after LLM validation "
        f"({pre_filter - len(clusters)} dropped as filler) (last {hours}h):"
    ]
    kg_writes = 0
    rich_themes = 0

    for c in clusters:
        synth = await _synthesize_cluster(c, hours=hours)
        if not synth:
            continue
        rich_themes += 1
        monitors_label = ", ".join(sorted(c.monitors)[:5])
        if len(c.monitors) > 5:
            monitors_label += f" (+{len(c.monitors)-5} more)"
        summaries.append(
            f"  • [{c.key}] across {c.breadth} monitors ({monitors_label}):\n"
            f"    {_preview(synth, 280)}"
        )

        # Reject meta-commentary syntheses where the LLM concluded "no real
        # pattern" — those are anti-facts and pollute the KG. The LLM is
        # supposed to say "no pattern" by writing "none" / nothing — when it
        # writes a long paragraph explaining WHY there is no pattern, the
        # garbage filter below catches it.
        _meta_markers = (
            "coincidental", "not driven by", "algorithmic noise",
            "linguistic usage", "no real pattern", "no underlying",
            "rather than reflecting", "generic keyword", "not an underlying",
            "is not driven by", "not reflective of",
        )
        synth_low = synth.lower()
        if any(m in synth_low for m in _meta_markers):
            logger.info("[Synthesis] rejecting meta-commentary for '%s'", c.key)
            continue

        # Cross-domain CAUSAL synthesis (Phase B): probe for a causal chain
        # connecting the domains and store it as real `caused_by` entity triples
        # (durable, queryable). Replaces the old `cross_pattern:X recurs_across
        # <paragraph>` meta-noise. Each link is run through is_garbage_triple at
        # write time — add_fact does NOT gate, so without this an LLM-emitted
        # fragment ("chip sales fell") could persist until daily curation.
        from app.core.kg import is_garbage_triple
        chain = await _causal_probe(c, hours=hours)
        chain = [lk for lk in chain
                 if not is_garbage_triple(lk["effect"], "caused_by", lk["cause"])]
        if chain:
            arrow = " → ".join(
                [chain[0]["cause"]] + [lk["effect"] for lk in chain]
            )
            summaries.append(f"    ⮑ causal chain: {arrow}")
        if kg is not None:
            for lk in chain:
                try:
                    ok = await kg.add_fact(
                        subject=lk["effect"][:80],
                        predicate="caused_by",
                        object_=lk["cause"][:80],
                        confidence=lk["confidence"],
                        source="cross_synthesis",
                        provenance=f"cross_synthesis_causal:{c.breadth}_monitors:{hours}h",
                    )
                    if ok:
                        kg_writes += 1
                except Exception as e:
                    logger.warning(
                        "[Synthesis] causal add_fact failed for '%s': %s", c.key, e
                    )

    if rich_themes == 0:
        summaries[0] = (
            f"CROSS-SYNTHESIS | scanned {len(grouped)} monitors, "
            f"{len(clusters)} cluster keys but no LLM synthesis worth keeping"
        )

    return {
        "summary": "\n".join(summaries),
        "themes": rich_themes,
        "kg_writes": kg_writes,
    }


_LEAD_RE = re.compile(
    r"(?is)lead\s+develop(?:ment)?s?\s*[:*#\n]*\s*(.+?)"
    r"(?=\n\s*(?:\*\*|#{1,3}\s|secondary|connections|bottom\s+line|━|─|📌)|\Z)")


def _extract_lead(text: str) -> str:
    """Pull a monitor digest's actual LEAD story, stripping header/scaffold lines.
    This is what makes cross-synthesis work — clustering on the raw digest picks up
    template words ('overview', 'learned'); the lead is the real content."""
    if not text:
        return ""
    m = _LEAD_RE.search(text)
    if m:
        lead = m.group(1)
    else:
        # fallback: first substantive prose lines (skip headers/source/footer lines)
        good = []
        for ln in text.split("\n"):
            s = ln.strip()
            low = s.lower()
            if (not s or s.startswith(("#", "_", "##", "**`", "↳", "📌", "━", "─", "💡", "🌐"))
                    or "domain overview" in low or low.startswith("read ") or "sources:" in low):
                continue
            good.append(s)
            if len(" ".join(good)) > 400:
                break
        lead = " ".join(good)
    return re.sub(r"\s+", " ", lead).strip()[:520]


async def meta_synthesis(db, *, hours: int = 36) -> str:
    """TODAY'S BIG PICTURE — one narrative pass over every monitor's lead story to
    surface the 2-3 threads that span MULTIPLE domains (the meta-story a per-domain
    digest can't see). Operates on the leads, not raw digests, so it can't be fooled
    by template words. This is the user-facing cross-domain intelligence."""
    from app.core.llm import invoke_nothink
    grouped = await asyncio.to_thread(
        _gather_recent_outputs, db, hours=hours, max_per_monitor=1)
    leads = []
    for name, vals in grouped.items():
        lead = _extract_lead(vals[0]) if vals else ""
        if lead and len(lead) > 60:
            label = name.replace("Domain Study:", "").strip()
            leads.append((label, lead))
    if len(leads) < 4:
        return ""
    blob = "\n\n".join(f"[{lbl}] {ld}" for lbl, ld in leads)[:13000]
    try:
        out = await _bg_invoke([{"role": "user", "content":
            f"Below are today's LEAD developments from {len(leads)} domain-intelligence monitors "
            "(each tagged with its domain).\n"
            "Find the 2-3 dominant THREADS that connect MULTIPLE domains — a single force or event "
            "driving stories across several monitors (e.g. one geopolitical event moving energy, "
            "markets, trade, and inflation at once). For EACH thread write a tight paragraph: name the "
            "thread, the domains it spans (in [brackets]), and why it matters. Lead with the single "
            "biggest cross-cutting thread. IGNORE stories confined to one domain. Use ONLY what's in "
            "the leads — invent nothing. No preamble, no restating the list.\n\n"
            f"LEADS:\n{blob}"}],
            max_tokens=750, temperature=0.3, num_ctx=8192)
        out = (out or "").strip()
    except Exception as e:
        logger.warning("[MetaSynthesis] failed: %s", e)
        return ""
    if len(out) < 100:
        return ""
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    return (f"## 🧭 **TODAY'S BIG PICTURE** · {today}\n"
            f"_threads connecting {len(leads)} monitors_\n\n{out}")


async def synthesize_and_log(db, kg) -> str:
    """Monitor-friendly wrapper. Leads with the narrative TODAY'S BIG PICTURE
    (user-facing), then still runs the causal cluster pass to bank `caused_by`
    KG facts in the background."""
    big_picture = ""
    try:
        big_picture = await meta_synthesis(db)
    except Exception:
        logger.exception("meta_synthesis failed")
    try:
        result = await synthesize_across_monitors(db, kg)
        causal = result["summary"]
    except Exception as e:
        logger.exception("synthesize_across_monitors failed")
        causal = f"CROSS-SYNTHESIS ERROR: {e}"
    if big_picture:
        return big_picture + "\n\n---\n" + causal
    return causal
