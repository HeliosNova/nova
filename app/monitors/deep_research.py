"""Deep research engine — monitor the world the way an analyst actually does.

Replaces the headline-skim (fetch RSS, grab the top item, write 2 sentences from
the blurb). For a domain this:
  1. finds the single most-COVERED current story (most coverage = best-documented),
  2. searches that specific story from several facets, across engines,
  3. reads only CREDIBLE sources in full — social media / forums / SEO are blocked,
  4. extracts concrete findings from each body (off-topic articles dropped),
  5. drafts a cross-source briefing, then runs a VERIFICATION pass that strips any
     claim not supported by the findings (kills the 9B's confident fabrications),
  6. banks grounded, garbage-gated facts into the KG so Nova knows more each cycle.

If there are no readable credible sources it says so — it never fabricates a
briefing from nothing. Heavy by design (many search + fetch + LLM calls); runs on
the background monitor schedule. Quality over speed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, quote

from app.core import llm
from app.core.source_authority import authority as _sa_authority

logger = logging.getLogger(__name__)


def _syn_model() -> str | None:
    """The configured synthesis model, or None for the provider default.
    Every digest-chain stage uses it (2026-09-01): running angles, findings,
    gap follow-up and the fresh-check on the 9B forced 3-4 weight reloads per
    digest on a card that cannot hold both models."""
    from app.config import config as _cfg_
    return (getattr(_cfg_, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None


async def _invoke_bg(*args, **kwargs):
    """llm.invoke_nothink with a GPU-yield checkpoint.

    A digest run makes 15+ sequential LLM calls over many minutes; the
    heartbeat's cycle-start defer can't help once the run has started, so a
    chat arriving mid-digest used to contend for the GPU the whole time
    (audit 2026-07-08). Before each background LLM call, wait (capped) while
    the owner is actively chatting. Quiet case costs one monotonic read.
    """
    waited = await llm.wait_for_interactive_quiet(max_wait_s=240.0)
    if waited:
        logger.info("[deep-research] yielded GPU to chat for %.0fs", waited)
    return await llm.invoke_nothink(*args, **kwargs)


def _NOW() -> datetime:
    return datetime.now(timezone.utc)


def _json_array(raw) -> list:
    """Parse a JSON array from a possibly-messy LLM string. Robust to the 9B's
    truncation/preamble: strict load, then balanced-bracket salvage, then pull
    individual {...} objects. Never raises — [] is the floor so a fragile
    extraction never silently zeroes out a pipeline."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    s = raw.strip()
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except Exception:
        pass
    i = s.find("[")
    if i >= 0:
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(s[i:j + 1])
                        return v if isinstance(v, list) else []
                    except Exception:
                        break
    out = []
    for m in re.finditer(r"\{[^{}]*\}", s):
        try:
            out.append(json.loads(m.group(0)))
        except Exception:
            pass
    return out


# Grammar-constrained array schemas (Ollama `format`) — replace the json_prefix="[{"
# prefill that broke on Ollama 0.32.13 (both models returned a bare {object}, the
# provider re-prepend corrupted it to "[{{…" → parse fail). See
# brain_kg._KG_TRIPLES_SCHEMA for the full root-cause note. Two array-shaped calls
# in this file were on the broken prefill: _learn_facts (grounded KG banking — a
# SECOND silent-zero-facts vector) and _deep_analyze's story clustering (which
# only degraded to "one big story" via its fallback, but still lost the per-story
# analysis layer). Fixed 2026-08-18.
_FACT_TRIPLES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["subject", "predicate", "object"],
    },
}
_STORY_GROUPS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["title", "items"],
    },
}
# Flat array-of-strings (search queries / subject rewrites). The json_prefix='["'
# callers degraded to [] on 0.32.13 (the model object-wrapped the array, the
# re-prepend corrupted it, _json_array salvage returned dicts → the isinstance(str)
# filter dropped them → empty). This lost the Search-o1 gap-followup loop, facet
# expansion, and the semantic SKIP/rewrite pass (all fell back to regex/deterministic
# backstops). A string-array `format` guarantees the shape. Fixed 2026-08-18.
_STRING_ARRAY_SCHEMA = {"type": "array", "items": {"type": "string"}}


# --- Source-quality tiers -------------------------------------------------
# ".edu" dropped 2026-09-01: academia.edu, alumni pages and course notes ranked
# with the wires; papers still score via _PRIMARY_URL_RE.
_TIER1_SUFFIX = (".gov", ".int", ".mil")
_TIER1_HOSTS = {
    "reuters.com", "apnews.com", "bloomberg.com", "ft.com", "wsj.com",
    "economist.com", "nature.com", "science.org", "arxiv.org",
    # Primary research at the source: preprint servers + working-paper repos. The
    # actual paper, not reporting about it — rank with the wires for grounding.
    "biorxiv.org", "medrxiv.org", "ssrn.com", "papers.ssrn.com", "nber.org",
}
_TIER2_HOSTS = {
    "cnbc.com", "bbc.com", "bbc.co.uk", "theguardian.com", "nytimes.com",
    "washingtonpost.com", "brookings.edu", "cfr.org", "carnegieendowment.org",
    "technologyreview.com", "arstechnica.com", "wired.com", "theverge.com",
    "semianalysis.com", "tomshardware.com", "anandtech.com", "stratechery.com",
    "lawfaremedia.org", "foreignpolicy.com", "thediplomat.com", "defensenews.com",
    "fiercebiotech.com", "statnews.com", "coindesk.com", "cointelegraph.com",
    "fortune.com", "axios.com", "politico.com", "schneier.com", "krebsonsecurity.com",
    "bleepingcomputer.com", "thehackernews.com", "darkreading.com", "securityweek.com",
    "techcrunch.com", "theblock.co", "asahi.com", "scmp.com",
    # Quality international news, server-rendered (read via http) or JS-rendered
    # but browser-readable. Tier-2 ranks them up + makes JS ones browser-eligible.
    # NOTE: aljazeera.com is deliberately NOT here — it blocks headless renders
    # (confirmed fail), so browser-escalating it just burns the budget.
    "dw.com", "france24.com", "timesofisrael.com", "al-monitor.com",
    "channelnewsasia.com", "middleeastmonitor.com", "atlanticcouncil.org",
    "reuters.com", "apnews.com", "thehindu.com", "kyivpost.com",
    "defenseone.com", "balkaninsight.com",
    # High-volume India outlets (verified parse+read) — tier-2 ranks them up so
    # India draws on real breadth instead of falling back.
    "timesofindia.indiatimes.com", "economictimes.indiatimes.com",
    "news18.com", "moneycontrol.com",
    # Thin-domain lifts (verified parse+read 2026-06-23): semiconductors, defense,
    # us-policy, energy — browser-eligible so they're read even if http fails.
    "wccftech.com", "datacenterdynamics.com", "militarytimes.com",
    "rollcall.com", "canarymedia.com", "pv-magazine.com",
    # Corporate press wires — a company's OWN announcement is the primary source
    # for what it said (earnings, M&A, product). Promotional, so tier-2 not tier-1,
    # but ranked above secondary reporting that paraphrases them.
    "prnewswire.com", "businesswire.com", "globenewswire.com",
    # Thin-domain lifts round 2 (browser-verified read 2026-06-24): these are
    # already curated feeds for Europe/Biotech/China/Research but were NOT tier-2,
    # so their JS-rendered pages failed the http read and weren't browser-escalated
    # — the gather fell back to weaker search hosts (biggo, tellers, france24).
    # Tier-2 makes them browser-eligible so the EU-native / domain-native voices win.
    "politico.eu", "euractiv.com",          # Europe + EU
    "biopharmadive.com",                      # Biotech
    "technode.com",                           # China tech (the hard domain)
    "quantamagazine.org",                     # Research frontiers + physics
}
_JUNK_HOSTS_RE = re.compile(
    r"(blogspot|wordpress\.com|androguider|\.blog$|examplenews|contentfarm|/amp/|"
    r"starworksglobal|starpedia|nftevening|"
    # SEO stock/crypto-signal mills + content-farm patterns (catch siblings of the
    # specific hosts blocked below).
    r"marketsdaily|tickerreport|livetradingnews|stockstotrade|tradingview-news|"
    r"\.onl$|newkerala|prokerala)", re.IGNORECASE
)
# HARD-BLOCK: social, forums, aggregators, known SEO/junk. Never read — they carry
# no reportable primary content and bait the synthesizer into fabrication. The
# 2026-06-22 additions are offenders observed across a full 45-monitor digest run.
_BLOCKED_HOSTS = {
    "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "reddit.com", "stocktwits.com", "tiktok.com", "pinterest.com", "youtube.com",
    "quora.com", "medium.com", "substack.com", "threads.net", "tumblr.com",
    "bitcointalk.org", "discord.com", "telegram.org", "mcplato.com",
    "navitasorganics.com", "roic.ai", "simplywall.st", "stockanalysis.com",
    "stocktitan.net", "marketbeat.com", "zacks.com", "danelfin.com",
    "intellectia.ai", "investtech.com", "alphaspread.com",
    # stock/crypto SEO mills + exchange-promo pages (not journalism)
    "livetradingnews.com", "tickerreport.com", "themarketsdaily.com", "kavout.com",
    "mexc.co", "lbank.com", "kucoin.com", "yellow.com",
    # gossip / SEO aggregators / syndication farms
    "bollyspice.com", "cubaheadlines.com", "prokerala.com", "newkerala.com",
    "iraqsun.com", "mexicostar.com", "meridia.news", "tmcnet.com",
    "asiatechreview.com", "awesomeagents.ai", "innovationopenlab.com",
    # fringe / SEO blogs / junk TLDs / off-topic-evergreen
    "moonofalabama.org", "memeburn.com", "londonmercury.com", "startupfortune.com",
    "americastrikes.com", "working-ref.com", "marble.onl", "evermx.com",
    "teachmecoolstuff.com", "nhindustryjournal.com", "captiveinsurancetimes.com",
    "biography.com", "newsforkids.net", "city.udn.com",
    # "World News Network" auto-syndication farm (reprints wire copy across
    # hundreds of <place><star|sun|times|news> domains — a curated list, since a
    # regex would falsely catch legit torontostar/baltimoresun/nytimes/latimes).
    "birminghamstar.com", "britainnews.net", "cambodiantimes.com", "arayonews.com",
    "peopledaily.digital",
    # same WNN family, caught leaking opinion/PR as fact in the 2026-06-27 audit
    # (e.g. kenyastar.com syndicating an RT/Pravda 'Pax Silica' op-ed as straight intel)
    "kenyastar.com", "australiannews.net", "philippinetimes.com", "vietnamtribune.com",
    # more SEO/stock/crypto-signal mills observed in the 2026-06-23 full run
    "candede.com", "inforcapital.com", "robotomated.com", "bullbear.news",
    "ancilar.com", "itbiznews.com", "sahmcapital.com",
    # stragglers observed across the full-45 runs
    "arabherald.com", "compuserve.com", "bulletproofservers.hk",
    "techpulseglobe.com", "comparos.in", "marketscale.com",
    # UK/PK local-news syndication farms that reprint one wire story across many
    # mastheads — they GAMED the corroboration-ranked selection (a reprinted UK
    # fiscal-politics story became the "lead" of Research Frontiers 2026-06-24).
    # Curated, not regex: legit National-World mastheads (scotsman, yorkshirepost)
    # must still pass; these specific small reprinters are the offenders.
    "dissexpress.co.uk", "lynnnews.co.uk", "dunyanews.tv", "educationtimes.com",
    # SOURCE-QUALITY TIGHTENING (2026-06-24, after the 3rd verification pass found
    # weak sources propagate semantic MIS-FRAMINGS). The research-grounded authority
    # dataset (Lin et al. PC1) can't catch these — they're all at its 0.50 'unknown'
    # default, indistinguishable from good niche sources, so this stays CURATED.
    # (a) SEO / aggregator / AI-content / market-research mills — no editorial value:
    "aihaberleri.org", "tellers.ai", "towardshealthcare.com", "showsbee.com",
    "catalystalert.io", "thecentwise.com", "clinicalmetric.com", "artiverse.ca",
    "influencematters.asia", "biggo.com", "bignewsnetwork.com",
    # (b) Legit US-LOCAL outlets that publish AI-syndicated OFF-DOMAIN filler on
    # topics they don't cover — azcentral PROVEN to mis-frame 'Creative Fabrica' as
    # Alibaba HappyHorse's release channel (it was only an early-access partner). The
    # engine's domains are national/international/tech, so these add ~no real value.
    "azcentral.com", "oklahoman.com", "ktvl.com", "hitsfm.net",
    # Low editorial reliability flagged in the 2026-06-29 output audit: zerohedge
    # carried the core Iran/market claims (cited ×12 in Geopolitics) that propagated
    # misframings; coinalertnews pushes crypto presale shill copy as "whale" intel.
    "zerohedge.com", "coinalertnews.com",
    # 2026-06-30 LIVE-digest audit: creati.ai is an AI-content mill whose synthesized
    # narrative (a fabricated codename/conspiracy story) anchored an Open
    # Source lead; counterfire.org is an activist outlet that anchored a Commodities lead
    # ("largest oil supply shock on record") as a lone weak source. Neither is intel.
    "creati.ai", "counterfire.org",
    # 2026-07-09 full-system exploration: the AI/ML digest LED with a fabricated
    # "OpenAI GPT-5.6 Sol/Terra/Luna" product line (invented tiers + pricing) sourced
    # entirely to these AI-content farms. The grounding stack made every claim trace
    # to a read host (fabricated_rate=0) but the hosts themselves manufacture the
    # "facts" — provenance without credibility. Hard-block the observed offenders;
    # the _is_content_farm heuristic below catches their unseen siblings generically.
    "neobitdaily.com", "aitoolly.com", "zglg.work", "n1n.ai", "marketerintel.com",
    "buildfastwithai.com", "deepusecase.com", "applyingai.com", "banthebots.org",
}

# Generic AI-content-farm detector — the curated block-list can't keep up with the
# infinite supply of SEO/AI-generated "news" mills, so this catches their STRUCTURE.
# Conservative + high-precision: never fires on a tier-1/2 host, and only DOWN-WEIGHTS
# (0.4 in _source_quality, labelled LOW-CREDIBILITY) rather than hard-blocking, so a
# false positive merely deprioritises — it never silently drops a real source.
_FARM_AI_EDGE = ("ai", "gpt", "llm", "genai", "chatgpt")   # AI token at a name edge
_FARM_CONTENT = ("tool", "usecase", "buildfast", "applying", "mastery", "automate",
                 "hub", "guide", "insights", "intel", "marketer", "prompt", "bots",
                 "hacks", "wizard", "genie", "guru", "hype", "daily")
_FARM_TLDS = (".work", ".online", ".site", ".xyz", ".info", ".click", ".icu",
              ".top", ".live", ".shop", ".fun", ".biz")


def _is_content_farm(url_or_host: str) -> bool:
    h = url_or_host if ("." in url_or_host and "/" not in url_or_host) else _host(url_or_host)
    if h in _TIER1_HOSTS or h in _TIER2_HOSTS or any(h.endswith(s) for s in _TIER1_SUFFIX):
        return False
    parts = h.split(".")
    core = parts[-2] if len(parts) >= 2 else parts[0]      # registrable name
    ai_edge = any(core.startswith(a) or core.endswith(a) for a in _FARM_AI_EDGE)
    content = any(c in core for c in _FARM_CONTENT)
    if ai_edge and content:                                 # aitoolly, applyingai, buildfastwithai
        return True
    if any(h.endswith(t) for t in _FARM_TLDS):              # cheap farm TLD + weak SLD signal
        if ai_edge or content or any(ch.isdigit() for ch in core) or len(core) <= 5:
            return True
    return False


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")


def _blocked(url: str) -> bool:
    h = _host(url)
    return any(h == b or h.endswith("." + b) for b in _BLOCKED_HOSTS)


# A URL that IS the primary artifact — an SEC filing, a regulator release, a
# company newsroom post, a preprint/DOI — not an article reporting about it. These
# patterns let an otherwise-unknown host carry primary-source weight, so the gather
# prefers reading the source over the secondary write-up.
_PRIMARY_URL_RE = re.compile(
    r"(?i)("
    r"sec\.gov/archives|/cgi-bin/browse-edgar|"            # EDGAR filings
    r"arxiv\.org/abs/|/doi/|doi\.org/|/pmc/articles/|"     # papers / DOIs
    r"/press-?releases?/|/news-?releases?/|/press-?announcements?/|"
    r"/newsroom/|/investor-?relations?/|/investors?/news|"
    r"/media/press|/sec-filings?/|/8-?k|/10-?[kq]|/form-?4"
    r")")


# Reference/explainer sites: fine as background, never a "development" and
# never a lead anchor (2026-09-01: wikipedia 0.834 / investopedia 0.926 in the
# authority dataset made them "quality" reads that produced evergreen bullets).
_REFERENCE_HOSTS = frozenset({
    "wikipedia.org", "wikimedia.org", "britannica.com", "investopedia.com",
    "howtogeek.com", "wikihow.com", "wikiwand.com", "encyclopedia.com",
    "dictionary.com", "merriam-webster.com", "techtarget.com", "geeksforgeeks.org",
    "w3schools.com", "tutorialspoint.com", "simple.wikipedia.org", "wiktionary.org",
})


def _is_reference_host(host: str) -> bool:
    h = (host or "").lower().replace("www.", "")
    return any(h == r or h.endswith("." + r) for r in _REFERENCE_HOSTS)


# Publish dates for read articles (URL path or engine-supplied), keyed by URL.
_PUB_DATES: dict[str, str] = {}
_URL_DATE_RE = re.compile(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:[/-]|$|\b)")


def _url_date(url: str) -> str | None:
    m = _URL_DATE_RE.search(url or "")
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _record_pub_date(r) -> None:
    """Remember the best known publish date for a read result."""
    url = getattr(r, "url", "") or ""
    if not url:
        return
    if len(_PUB_DATES) > 5000:
        _PUB_DATES.clear()
    date = (getattr(r, "published_date", "") or "").strip() or _url_date(url) or ""
    if date:
        _PUB_DATES[url] = date


def _pub_dt(value: str):
    for fmt in _DT_FMTS:
        try:
            d = datetime.strptime(value.strip(), fmt)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        d = datetime.strptime(value.strip()[:10], "%Y-%m-%d")
        return d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _source_quality(url: str) -> float:
    if _blocked(url):
        return 0.0
    h = _host(url)
    if _is_reference_host(h):
        return 1.0
    if any(h.endswith(s) for s in _TIER1_SUFFIX) or h in _TIER1_HOSTS:
        return 3.0
    if h in _TIER2_HOSTS:
        return 2.0
    # Primary-artifact URL on an otherwise-unranked host: the actual filing/release/
    # paper beats secondary reporting. Lifts above the 1.0 generic floor, below the
    # curated tier-1 wires, and never rescues a junk/blocked host (handled above).
    if _PRIMARY_URL_RE.search(url):
        return 2.5
    # Generic AI-content-farm (structural, catches unblocked siblings): below the
    # 0.5 junk floor so it loses the read budget to real sources and is labelled
    # LOW-CREDIBILITY — the synthesizer must not lead with or state it as fact.
    if _is_content_farm(h):
        return 0.4
    if _JUNK_HOSTS_RE.search(url):
        return 0.5
    # Dataset-informed floor (2026-08-12): consult the 11,520-domain source-
    # authority ratings so SELECTION and GATING see them, not just the evidence
    # tags — before this, a dataset-rated farm (authority <0.3) enjoyed the same
    # generic 1.0 floor as any unknown host, and a dataset-reputable outlet
    # (>=0.8) could never anchor a lead. Hand-curated tiers above always win.
    a = _sa_authority(h)
    if a >= 0.8:
        return 2.0
    if a < 0.3:
        return 0.4
    return 1.0


def _tier_label(url: str) -> str:
    """Human reliability label for a source, so the synthesizer can CALIBRATE — lead
    with confirmed material, caveat lone weak-source claims (an intelligence brief must
    not state a single-blog rumor with the same authority as a wire-confirmed fact)."""
    q = _source_quality(url)
    if q >= 3.0:
        return "wire/primary"
    if q >= 2.5:
        return "primary-doc"
    if q >= 2.0:
        return "quality"
    if q <= 0.5:
        return "LOW-CREDIBILITY"
    return "single/unverified"


def _annotated_evidence(findings: list, host_clusters: dict | None = None) -> str:
    """Each finding tagged '(outlet · reliability · corroboration)' so the synthesizer
    weights by evidence strength instead of treating every claim as equal fact.
    Corroboration = distinct hosts among findings sharing ≥2 significant title tokens
    (a cheap same-story proxy — no extra LLM/clustering call).
    `host_clusters` (independence, 2026-08-12): when the caller provides the
    mirror map, same-cluster hosts count ONCE — a syndication network can no
    longer inflate the corroboration tag the synthesizer picks its lead on."""
    toks = [{w for w in _key_terms(t) if w not in _HEADLINE_STOP and len(w) >= 3}
            for t, _u, _f in findings]
    hc = host_clusters or {}
    blocks = []
    for i, (t, u, f) in enumerate(findings):
        hosts = {_host(findings[j][1]) for j in range(len(findings)) if len(toks[i] & toks[j]) >= 2}
        n_indep = len({hc.get(h, h) for h in hosts})
        corro = f"{n_indep} sources" if n_indep >= 2 else "1 source"
        blocks.append(f"[{t}] ({_host(u)} · {_tier_label(u)} · {corro})\n{f}")
    return "\n\n".join(blocks)


_CSS_TOKENS = re.compile(
    r"(\{[^}]{0,40}:[^}]{0,40}\}|@media|font-family|margin:|padding:|rgba?\(|px;|<style)",
    re.IGNORECASE)


# Crypto-shill / presale promotion — never news. A body with ≥2 of these in its head
# is marketing copy (the Whale-Watch "Pepeto presale crossed $10M ahead of rumored
# Binance listings" class the 2026-06-29 audit flagged), not whale activity.
_PROMO_RE = re.compile(r"(?i)\b("
    r"presale|pre-sale|token sale|airdrop|whitelist|"
    r"\d{2,4}x\s+(?:potential|gains?|returns?|profit)|to the moon|"
    r"next\s+(?:100x|moonshot|big\s+gem|crypto\s+gem|bitcoin|ethereum)|"
    r"guaranteed\s+(?:returns?|profits?)|don'?t\s+miss\s+(?:out|this)|"
    r"limited[-\s]time\s+(?:offer|opportunity)|join\s+the\s+(?:presale|whitelist)|"
    r"get\s+in\s+early|early\s+investors?|rumored\s+(?:to\s+be\s+)?listing|"
    r"buy\s+(?:now\s+)?before|huge\s+(?:gains?|upside)"
    r")\b")
# Evergreen / contact-page / how-to boilerplate — not 'past 24-48h' news (the FDA
# toll-free number, IRS 'how to file Form 9465', agency mission statements the audit
# found padding breadth). High-precision; checked only in the HEAD where such pages lead.
_FILLER_RE = re.compile(r"(?i)("
    r"\b1-?8(?:00|66|77|88)-?\d{3}-?\d{4}\b|1-?888-?info|"
    r"is\s+responsible\s+for\s+protecting\s+(?:the\s+)?public\s+health|"
    r"our\s+mission\s+is\s+to\s+(?:protect|ensure|promote|advance)|"
    r"\bstep-by-step\s+(?:guide|instructions)\b|"
    r"\bhow\s+to\s+(?:file|apply\s+for|submit|enroll\s+in|complete)\s+(?:a\s+|an\s+|your\s+)?\w)")


def _is_promo_or_filler(b: str) -> bool:
    """True if the body is crypto-shill promotion or evergreen/contact-page filler —
    deterministic, drops it before extraction so the shill numbers never enter the
    grounding corpus and the filler never pads breadth."""
    head = b[:1200]
    if len(_PROMO_RE.findall(head)) >= 2:
        return True
    return bool(_FILLER_RE.search(head[:600]))


def _junk_body(b: str) -> bool:
    if not b or len(b) < 400:
        return True
    if _is_promo_or_filler(b):   # crypto-shill presale + evergreen/contact-page filler
        return True
    head = b[:240].lower()
    if ("minimal readable" in head or "%pdf" in head
            or "endstream" in b[:600].lower() or b.count(" obj") > 5):
        return True
    # Anti-bot / error interstitials (Akamai "Access Denied" on war.gov is 432
    # chars — passes the length gate and was treated as a valid article body).
    if any(p in head for p in (
        "access denied", "you don't have permission", "pardon our interruption",
        "request unsuccessful", "attention required", "just a moment",
        "client challenge",
    )):
        return True
    sample = b[:3000]
    if len(_CSS_TOKENS.findall(sample)) >= 8 and sample.count(". ") < 5:
        return True
    # Nav-chrome / link-list (homepage, section page, .gov landing, wiki index):
    # lots of short Capitalized link tokens, almost no sentence prose. These pass
    # the CSS check (browser innerText has no CSS) but carry no story content, so
    # extraction yields nothing — drop them as bodies.
    sentences = sample.count(". ")
    words = sample.split()
    if len(words) >= 30:
        cap_short = sum(1 for w in words[:90] if w[:1].isupper() and len(w) <= 14)
        if cap_short > 30 and sentences < 4:
            return True
    if len(sample) > 900 and sentences < 3:   # long but barely any prose
        return True
    return False


_STOP = frozenset({"the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
                   "with", "by", "from", "as", "at", "is", "are", "was", "its"})


def _key_terms(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9.\-]{2,}", (s or "").lower())
            if w not in _STOP}


# --- Prompts --------------------------------------------------------------
_FACT_PROMPT = (
    "From the research briefing below on '{topic}', extract durable factual "
    "(subject, predicate, object) triples — real entities and relationships, not "
    "events or opinions. Short entity names. Max 10.\n"
    "PREDICATE must be a specific relation (acquired, leads, launched, owns, regulates, "
    "sanctioned, partnered_with, located_in, price_of) — NEVER a vague 'related_to'.\n"
    "SUBJECT must be a NAMED entity (a specific company/person/place/product) — NEVER a "
    "generic category word ('developer', 'the market', 'a study', 'the team', 'users'); "
    "if you can't name the actor, SKIP the fact.\n"
    "OBJECT must be a concrete entity or a quantity WITH its meaning ('$160 billion in "
    "daily clearing volume', not '$160 billion') — NEVER a bare date, month, duration, "
    "clock time, or standalone measurement ('July 5', 'eight years', '1:00 AM UTC', "
    "'20 hours daily', '10,000 repetitions' are useless objects; put the thing that happened).\n"
    "Return JSON array: [{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}}]\n\n{evidence}"
)


async def _headlines(label: str) -> list[str]:
    """Recent headlines for the domain, across engines/modes, blocked hosts
    removed. Searches run concurrently — each can sit on the 30s engine timeout,
    so serial would cost minutes."""
    from app.tools import native_search
    year = _NOW().strftime("%Y")

    async def _s(q, mode):
        try:
            return await native_search.search(q, max_results=12, mode=mode)
        except Exception:
            return []

    queries = [(f"{label} news {year}", "news"),
               (f"{label} biggest news this week", "news"),
               (f"{label} latest developments", "general")]
    results = await asyncio.gather(*[_s(q, m) for q, m in queries])
    heads = []
    for rs in results:
        for r in rs:
            if r.title and not _blocked(r.url) and not _too_old(getattr(r, "published_date", "")):
                heads.append(r.title)
    return list(dict.fromkeys(heads))  # dedup, keep order


# Outlet-homepage / feed-channel "headlines" that describe the OUTLET rather than an
# event ("Associated Press News publishes breaking headlines", "BBC Home reports world
# news and business updates", "<Outlet> Today covers latest developments in <region>").
# They leak from homepage search results into the coverage ranking and become vacuous
# "stories" the engine then researches into vague generic copy (2026-06-29 owner flag).
_GENERIC_FEED_TITLE_RE = re.compile(r"(?i)("
    r"\b(?:reports?|publishes?|covers?|brings?|delivers?|provides?|features?)\s+"
    r"(?:you\s+)?(?:the\s+|its\s+)?(?:latest|breaking|top|daily|world|live)\s+"
    r"(?:news|headlines|stories|developments|updates|coverage)\b|"
    r"\b(?:news|home|today|online|daily|live)\s+(?:latest|breaking|top|live)\s+"
    r"(?:news|updates|headlines|stories|coverage|developments)\b|"   # "BBC News latest updates" (no verb)
    r"\bworld\s+news\s+(?:and|&)\s+business\b|"
    r"\bbreaking\s+news,?\s+latest\s+headlines\b|"
    r"\blatest\s+developments\s+in\s+\w+\s+region\b"
    r")")


_NEWS_META_WORDS = frozenset({
    "news", "latest", "breaking", "updates", "update", "headlines", "headline",
    "world", "international", "live", "daily", "top", "stories", "story", "coverage",
    "home", "homepage", "today", "online", "report", "reports", "section", "video",
    "videos", "photos", "analysis", "opinion", "global", "us", "uk", "politics",
    "business", "the", "and", "your", "from",
})


def _is_generic_feed_title(title: str) -> bool:
    """True for outlet-homepage / section / feed-channel titles that name no event —
    'World News - The New York Times International', 'World | Latest News & Updates',
    'BBC News latest updates'. Two signals: (1) the meta-phrase regex; (2) robustly —
    after stripping a trailing '- Outlet' / '| Outlet', the remainder is ALL news-
    structure words with no substantive content word, which a real story always has."""
    t = (title or "").strip()
    if not t:
        return True
    if _GENERIC_FEED_TITLE_RE.search(t):
        return True
    core = re.split(r"\s[|\-–—]\s", t)[0]                 # drop a trailing "- Outlet" / "| Outlet"
    words = re.findall(r"[A-Za-z][A-Za-z0-9'&]+", core.lower())
    if not words:
        return True
    return all(w in _NEWS_META_WORDS or len(w) < 3 for w in words)


async def _headlines_with_hosts(label: str) -> list[tuple[str, str]]:
    """Recent headlines as (title, host) pairs across engines — the raw material for
    coverage ranking (how many independent outlets carry each story)."""
    from app.tools import native_search
    year = _NOW().strftime("%Y")

    async def _s(q, mode):
        try:
            return await native_search.search(q, max_results=14, mode=mode)
        except Exception:
            return []

    queries = [(f"{label} news {year}", "news"),
               (f"{label} biggest news this week", "news"),
               (f"{label} top stories", "news"),
               (f"{label} latest developments", "general")]
    results = await asyncio.gather(*[_s(q, m) for q, m in queries])
    seen, out = set(), []
    for rs in results:
        for r in rs:
            if not (r.title and r.url) or _blocked(r.url) or _too_old(getattr(r, "published_date", "")):
                continue
            if _article_score(r.url) <= -0.5 or _is_generic_feed_title(r.title):
                continue   # homepage/section root or outlet self-description — not a story
            key = r.title.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((r.title.strip(), _host(r.url)))
    return out


_HEADLINE_STOP = _STOP | {
    "news", "latest", "update", "updates", "says", "say", "report", "reports",
    "new", "amid", "after", "before", "could", "will", "would", "how", "why",
    "what", "this", "week", "today", "set", "may", "can", "get", "big", "top",
}

# SEO / evergreen / listicle headlines — NOT news. These syndicate widely (high
# false "coverage") and would hijack the coverage ranking with course/guide spam.
_SEO_HEADLINE_RE = re.compile(
    r"(?i)(\btop\s+\d+\b|\bbest\s+\d*\s*\w|\b\d+\s+best\b|\bcourses?\b|\btutorials?\b|"
    r"\bguide\b|\bhow\s+to\b|\bfor\s+beginners\b|\bcheat\s*sheet\b|\bexplained\b|"
    r"\broundup\b|\bvs\.?\b|\breview\b|\bprograms?\b|\bcertification|\bmagic\s+quadrant\b|"
    r"\bwhat\s+is\b|\blists?\s+of\b|\bultimate\b|\bstep[\s-]by[\s-]step\b|"
    # listicle patterns: leading number-word, "N <plural>", "trends/tips/ways to"
    r"^\s*(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+\w|"
    r"\b\d+\s+\w+\s+(trends|tips|ways|things|tools|reasons|examples|skills|jobs)\b|"
    # financial listicles the above missed: "8 Commodity ETFs for …", "5 stocks to buy".
    # Require a listicle CONTINUATION after the finance-plural noun so real news with a
    # count ("2 China funds halt redemptions") is NOT caught — only list-headers are.
    r"\b\d+\s+[\w-]+(?:\s+[\w-]+){0,2}\s+(etfs?|stocks?|shares?|funds?|coins?|tokens?|"
    r"picks?|plays?|reits?)\s+(to\s+(buy|watch|consider|own|hold|avoid)|for\b|you\b|"
    r"that\b|worth\b|under\s+\$|right\s+now|now\b|today\b|in\s+20)|"
    r"\b\d+\s+(?:reasons|ways|things)\s+to\b|"
    r"\btrends?\s+(for|to\s+watch|in\s+20|reshaping)\b|\bthings?\s+(you|to)\b|\bways?\s+to\b)")

# Promotional / pump-and-dump / advertorial headlines — advertising dressed as news.
# They carry NO reportable event, are pure sentiment bait, and (worse) push the
# synthesizer toward shill framing. Distinct from SEO: these are stock/crypto promo.
_PROMO_HEADLINE_RE = re.compile(
    r"(?i)("
    r"get in on the ground floor|ground[\s-]?floor|off (wall street'?s?|the|its) radar|"
    r"under[\s-]the[\s-]radar|hidden gem|emerging giant|the next \w+|next big|"
    r"set to (soar|explode|skyrocket|surge|rocket|jump|double|triple)|"
    r"could (make you|turn|soar|explode|double|triple)|"
    r"millionaire[\s-]?maker|make you (a millionaire|rich)|"
    r"screaming buy|table[\s-]pounding|strong buy alert|"
    r"\b(stocks?|coins?|cryptos?|shares?|etfs?|plays?) to buy (now|today|right now)|"
    r"buy (now|before|this)|before (it'?s too late|the|it)|"
    r"you (need to|should|must|have to) (buy|own|know|watch)|"
    r"\b\d+x\b|\b\d+00%\b|moonshot|to the moon|"
    r"why (i'?m|you should) (buy|bought)|my top (pick|stock|buy)|"
    r"presale|price prediction|will \w+ (hit|reach|explode|soar))")


def _is_seo_headline(title: str) -> bool:
    t = title or ""
    return bool(_SEO_HEADLINE_RE.search(t) or _PROMO_HEADLINE_RE.search(t))


# A "subject" that is really a SECTION / INDEX / market-wrap / meta page, not a story —
# these produce empty deep-research gathers ("Reuters publishes latest finance news",
# "ft.com automobiles section", "Bloomberg Markets update", "FedEx shipment tracking").
_GENERIC_SUBJECT_RE = re.compile(
    r"(?i)("
    r"\bpublish(es|ed)\b.{0,30}\b(news|headlines)\b|"
    r"\blatest\b.{0,25}\b(news|headlines)\b|"
    r"\b(news|headlines)\s+(today|now|this\s+(week|morning))\b|"
    r"\bmarkets?\s+(update|movements?|wrap|roundup|report|recap|moves|coverage)\b|"
    r"\bfutures\s+and\s+commodities\b|"
    r"\bsection\s*$|\bhomepage\b|\bfront\s+page\b|"
    r"\blive\s+quotes?\b|\bprice\s+data\b|\breal[\s-]?time\s+(quote|price|data|tracking|market)|"
    r"\b(shipment|package|order|parcel)\s+tracking\b|\btracking\s+services\b|"
    r"\bweekly\s+media\s+briefing\b|\bat\s+\d{1,2}:\d{2}\s*[ap]\.?m\b|"    # schedule/listing pages
    r"\bweekly\s+(technical\s+)?outlook\b)")


def _is_generic_subject(s: str) -> bool:
    return bool(_GENERIC_SUBJECT_RE.search(s or ""))


def _cluster_headlines(headlines: list[tuple[str, str]]) -> list[dict]:
    """Greedy-cluster headlines about the same story (≥2 shared significant tokens),
    tracking the set of distinct HOSTS = how many independent outlets cover it."""
    clusters: list[dict] = []
    for title, host in headlines:
        toks = {w for w in _key_terms(title) if w not in _HEADLINE_STOP and len(w) >= 3}
        if len(toks) < 2:
            continue
        hit = None
        for c in clusters:
            if len(toks & c["toks"]) >= 2:
                hit = c
                break
        if hit:
            hit["toks"] |= toks
            hit["titles"].append(title)
            hit["hosts"].add(host)
        else:
            clusters.append({"toks": set(toks), "titles": [title], "hosts": {host}})
    return clusters


# Corporate press wires: one release syndicated across all of them is ONE source, not
# many — don't let that inflate the coverage count.
_PR_WIRE_HOSTS = {
    "prnewswire.com", "businesswire.com", "globenewswire.com", "newswire.com",
    "accesswire.com", "prweb.com", "einnews.com", "24-7pressrelease.com",
    "prlog.org", "openpr.com", "prunderground.com", "einpresswire.com",
}


def _coverage_score(cluster: dict) -> float:
    """Quality-weighted cross-source coverage. Each INDEPENDENT outlet adds its tier
    weight (reuters/AP 3.0, tier-2 2.0, unknown 1.0); all PR-wire hosts collapse to a
    single +1.0 (same release), so a press release on 5 wires no longer outranks a real
    story on 3 independent desks. This is the importance signal `_focus_subjects` ranks."""
    hosts = cluster["hosts"]
    wires = {h for h in hosts if h in _PR_WIRE_HOSTS}
    score = sum(_source_quality("https://" + h + "/") for h in (hosts - wires))
    return score + (1.0 if wires else 0.0)


async def _feed_headlines(feed_key: str) -> list[tuple[str, str]]:
    """Headlines from the domain's CURATED feeds — real news (not SEO), so coverage
    counted here is trustworthy. (title, host) pairs."""
    try:
        from app.monitors.rss_feeds import fetch_recent_items
        items = await fetch_recent_items(feed_key, hours=72, max_total=30)
    except Exception:
        return []
    return [(it.title.strip(), it.source_host) for it in items
            if it.title and it.url and not _blocked(it.url)
            and not _is_generic_feed_title(it.title)   # drop feed-channel/outlet self-description titles
            and _article_score(it.url) > -0.5]


async def _focus_subjects(label: str, feed_key: str | None = None, n: int = 5) -> list[str]:
    """Top-N stories ranked by CROSS-SOURCE COVERAGE — the story the most independent
    outlets carry is the most important, so heavily-covered stories can't be missed
    (the old LLM 'pick the biggest' could, e.g. Shazeer→OpenAI). Candidates come from
    the CURATED feeds (real news) + search, with SEO/listicle headlines filtered so
    course/guide spam can't hijack the coverage count. Ranking is deterministic; one
    LLM call only cleans the phrasing, preserving the order."""
    today = _NOW().strftime("%B %d, %Y")
    year = _NOW().strftime("%Y")
    feed_heads, search_heads = await asyncio.gather(
        _feed_headlines(feed_key or label), _headlines_with_hosts(label))
    headlines = [(t, h) for t, h in (feed_heads + search_heads) if not _is_seo_headline(t)]
    if len(headlines) < 4:
        return [f"{label} latest developments {year}"]
    clusters = _cluster_headlines(headlines)
    # Rank by QUALITY-WEIGHTED coverage (PR-wire syndication collapsed), then raw breadth.
    clusters.sort(key=lambda c: (_coverage_score(c), len(c["hosts"]), len(c["titles"])), reverse=True)
    top = clusters[: n + 3]           # buffer so generic/meta drops below can backfill
    if not top:
        return [f"{label} latest developments {year}"]
    reps = [max(c["titles"], key=len) for c in top]
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(reps))
    from app.config import config as _cfg
    syn = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None  # 27B judges on-mission/evergreen better
    try:
        raw = await _invoke_bg([{"role": "user", "content":
            f"Today is {today}. Below are candidate {label} headlines, ranked by importance. For EACH, "
            f"in the SAME order and count, output one JSON string:\n"
            f"- If it is a SPECIFIC recent {label} news EVENT, rewrite it as one concrete {year} subject "
            f"phrase (key entities + what happened; no outlet name, no clickbait, no question mark).\n"
            f"- Otherwise output exactly \"SKIP\" — use SKIP when it is evergreen/analysis with no dated "
            f"event, a section/index/market-wrap/quote page, local trivia, a listicle or advert, or "
            f"OFF-TOPIC for {label}.\n" + listing +
            f"\n\nReturn a JSON array of exactly {len(reps)} strings (each a rewrite or \"SKIP\"), same order."}],
            json_mode=True, json_schema=_STRING_ARRAY_SCHEMA, max_tokens=420, temperature=0.2, model=syn)
        subs = [str(s).strip().rstrip("?") for s in _json_array(raw)
                if isinstance(s, str) and len(str(s).strip()) > 10
                and not str(s).strip().upper().startswith("SKIP")]
    except Exception as e:
        logger.warning("[DeepResearch] subject-rewrite LLM failed (%s) — falling back to "
                       "deterministic headline cleanup: %s", label, e)
        subs = []
    if not subs:  # LLM failed entirely → clean the representative headlines deterministically
        subs = [re.sub(r"\s*[-–|:]\s*[A-Z][\w. '&]{2,30}$", "", t).strip() for t in reps]
    # Deterministic backstops for anything the semantic SKIP missed: drop section/wrap/
    # promo garbage, plus bare-label ("Geopolitics") and sub-3-word fragments that carry
    # no concrete event.
    lab = label.strip().lower()
    subs = [s for s in subs if not _is_generic_subject(s) and not _is_seo_headline(s)
            and s.strip().lower() != lab and len(s.split()) >= 3]
    return subs[:n] or [f"{label} latest developments {year}"]


async def _focus_subject(label: str) -> str:
    """The single most-COVERED current story (coverage-ranked top-1)."""
    subs = await _focus_subjects(label, n=1)
    return subs[0] if subs else f"{label} latest developments {_NOW().strftime('%Y')}"


# Limit concurrent headless-browser renders (chromium contexts are heavy).
_BROWSER_SEM = asyncio.Semaphore(3)


_JINA_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_PAYWALL_STUB_RE = re.compile(
    r"(?i)(subscribe to (unlock|read|continue)|try unlimited access|to read this article|"
    r"sign in to (read|continue)|register to (read|continue)|this (article|content) is for "
    r"subscribers|\$\d+ for \d+ weeks|create (a )?free account to)")


def _reader_main_content(md: str) -> str:
    """Extract the ARTICLE prose from a reader-proxy markdown (which wraps nav chrome +
    related-article link-lists + footer around the body). Keep only SUBSTANTIAL prose
    paragraphs (≥40 words, ≥2 sentences, not a bullet/link list); return '' if there's
    no real article — so a paywall that yields only nav scraps cleanly fails, not feeds
    the synthesizer a menu."""
    good = []
    for p in re.split(r"\n\s*\n", md):
        p = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", p)     # drop images
        p = _JINA_LINK_RE.sub(r"\1", p)                 # markdown link → its text
        p = re.sub(r"\s+", " ", p).strip()
        sents = p.count(". ") + p.count("? ") + p.count("! ")
        if len(p.split()) >= 40 and sents >= 2 and p.count(" * ") < 2 and not p.startswith(("*", "#", ">")):
            good.append(p)
    art = "\n\n".join(good)
    return art if len(art) >= 500 else ""


# Article-body cap. The entailment gate (MiniCheck) needs enough of the cited
# page to locate the claim; a 5000-char cap here starved the gate's window
# SELECTION POOL — windows are chosen from the fetched body, so nothing beyond
# 5000 chars could ever be selected. (Note: observed doc_len tops out at ~5607
# regardless of this cap — that is the gate's own windowing arithmetic, 2
# articles × 2 windows × 1400 chars + separators, NOT a body truncation. See
# _windows/_doc_for. Audit 2026-08-24 re-derived this.) The synthesis path
# re-caps bodies to 900/28000 of its own accord, so raising this only improves
# grounding (audit 2026-08-22). Any _invoke_bg caller embedding a full body
# MUST pass num_ctx sized for ~12k chars (the 4096 default truncates —
# _findings hit exactly that on 2026-08-23/24, ~95×/day).
_BODY_MAX_CHARS = 12000


async def _fetch_via_jina(url: str, max_len: int = _BODY_MAX_CHARS) -> str | None:
    """Bypass a paywall/verification wall on a QUALITY source via Jina Reader
    (r.jina.ai) — a reader proxy that fetches + extracts clean article text (verified
    to return full FT/Bloomberg articles the sovereign fetch gets a 403 on). External
    service (gated by ENABLE_PAYWALL_BYPASS; deliberate exception to local-only, public
    news only). Optional JINA_API_KEY lifts the rate limit."""
    import os
    import httpx
    headers = {"User-Agent": "Mozilla/5.0", "X-Return-Format": "markdown"}
    key = os.getenv("JINA_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as c:
            r = await c.get("https://r.jina.ai/" + url)
        if r.status_code >= 400 or len(r.text) < 400:
            return None
        raw = r.text
        m = re.search(r"(?is)Markdown Content:\s*(.+)$", raw)     # strip the reader preamble
        raw = m.group(1) if m else raw
        if _PAYWALL_STUB_RE.search(raw[:2000]):                   # hit the wall, not the article
            return None
        body = _reader_main_content(raw)                          # article prose only (nav/related dropped)
        if body and not _stale_body(body):
            # Sanitize like every other fetch path (2026-08-29). The two primary
            # readers wrap their output — HttpFetchTool at http_fetch.py:485 and
            # BrowserTool at browser.py:631/751 — but this bypass returned raw
            # reader text straight to the synthesis and KG-extraction prompts.
            # It is the path used for exactly the content we trust least to be
            # clean: third-party-rendered paywalled pages, plus every fallback
            # once the browser budget is spent. Untrusted page text reaching a
            # fact-extraction prompt unwrapped is the memory-poisoning vector.
            from app.config import config as _icfg
            if getattr(_icfg, "ENABLE_INJECTION_DETECTION", True):
                from app.core.injection import sanitize_content
                body = sanitize_content(body, context="paywall bypass")
            logger.info("[DeepResearch] paywall bypass (jina) read %s (%d chars)", _host(url), len(body))
            return body[:max_len]
    except Exception as e:
        logger.debug("[DeepResearch] jina bypass failed %s: %s", _host(url), e)
    return None


async def _fetch_body(url: str, *, browser_budget: list[int] | None = None,
                      allow_bypass: bool = True, max_len: int = _BODY_MAX_CHARS) -> str | None:
    """Read an article body. Fast path: http_fetch (15s). Fallback: headless browser
    (22s) — most quality news (BBC/CNBC/Economist) is JS-rendered. Final fallback for
    QUALITY sources that hit a paywall/verification wall: the Jina reader bypass (gated
    by ENABLE_PAYWALL_BYPASS) — so FT/Bloomberg/WSJ are READ, not skipped. The headless
    browser is stealth-hardened + residential-IP'd, so it clears Cloudflare interstitials
    and reads metered/client-side paywalls (NYT/WaPo/BI) from a fresh cookieless context.
    `browser_budget` (mutable [n]) caps renders; only HARD server-side paywall hosts skip
    the browser (renders only the wall) and go straight to the bypass."""
    from app.tools.http_fetch import HttpFetchTool
    try:
        res = await asyncio.wait_for(HttpFetchTool().execute(url=url), timeout=15)
        body = (res.output or "") if getattr(res, "success", False) else ""
        if not _junk_body(body):
            return body[:max_len]
    except Exception:
        pass
    from app.config import config as _cfg
    can_bypass = (allow_bypass and getattr(_cfg, "ENABLE_PAYWALL_BYPASS", True)
                  and _source_quality(url) >= 2.0)
    # Hard (server-side) paywall: even past Cloudflare the browser renders only the wall,
    # so bypass directly (no render cost). Metered/client-side hosts fall through to the
    # stealth browser below — a fresh cookieless context resets the meter.
    if _host(url) in _HARD_PAYWALL_HOSTS:
        return await _fetch_via_jina(url, max_len) if can_bypass else None
    # JS-rendered → render with the browser, but only within budget.
    if browser_budget is not None:
        if browser_budget[0] <= 0:
            return await _fetch_via_jina(url, max_len) if can_bypass else None
        browser_budget[0] -= 1
    from app.tools.browser import BrowserTool
    async with _BROWSER_SEM:
        try:
            r = await asyncio.wait_for(
                BrowserTool().execute(action="navigate", url=url), timeout=22)
            body = (r.output or "") if getattr(r, "success", False) else ""
            if body and not _junk_body(body):
                return body[:max_len]
        except Exception:
            pass
    # Browser blocked (consent/soft-paywall) → reader bypass for quality sources.
    return await _fetch_via_jina(url, max_len) if can_bypass else None


_OLD_YEARS = frozenset(str(y) for y in range(2010, _NOW().year))


def _stale_body(body: str) -> bool:
    """True if an article body reads as OLD: the current year is absent from the
    lede while a prior year is repeated. Kills the failure where a search engine
    surfaces a years-old article as 'today's news'."""
    head = (body or "")[:1500]
    if not head:
        return False
    year = _NOW().strftime("%Y")
    if year in head:
        return False
    return any(head.count(y) >= 2 for y in _OLD_YEARS)


_DT_FMTS = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d", "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S")


def _too_old(pub: str, *, days: int = 45) -> bool:
    """True ONLY if the result carries a parseable date older than `days`. Undated
    results pass (most engines omit dates) — we kill only what we can PROVE is
    stale. The search engines here are flaky and sometimes surface years-old
    wikinews/wiki pages; this stops those from being picked as 'today's' story."""
    if not pub:
        return False
    pub = pub.strip()
    for fmt in _DT_FMTS:
        try:
            d = datetime.strptime(pub, fmt)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d < _NOW() - timedelta(days=days)
        except ValueError:
            continue
    return False


_SECTION_SLUGS = {"news", "world", "business", "markets", "market", "politics",
                  "sport", "sports", "technology", "tech", "money", "economy",
                  "opinion", "finance", "us", "uk", "live", "latest", "video"}


def _article_score(url: str) -> float:
    """Heuristic: how likely a URL is a specific ARTICLE vs a homepage/section
    index. A domain root or bare section page (bbc.com, cfr.org/world) carries no
    story to read; an article has a deep path with a long slug / date / id. The
    gather kept reading landing pages from good domains — this fixes that."""
    p = urlparse(url)
    path = p.path.strip("/")
    if not path:
        return -1.0  # domain root
    segs = path.split("/")
    last = segs[-1].lower()
    score = 0.0
    if len(segs) >= 2:
        score += 0.5
    if len(last) > 14 or last.count("-") >= 2 or any(c.isdigit() for c in last):
        score += 0.6  # long slug / dated / id'd → article-like
    if len(segs) == 1 and last in _SECTION_SLUGS:
        score -= 0.8  # bare section page
    return score


async def _search_candidates(angles: list[str], *, want: int) -> list:
    """Concurrent search over angles × (news, general); credible, host-capped
    picks (≤2 per host), proven-stale dropped, ARTICLE urls preferred over
    homepages/section indexes. Concurrency matters: each search can sit on the
    30s engine timeout, so never run serially."""
    from app.tools import native_search

    async def _s(q, mode):
        try:
            return await native_search.search(q, max_results=16, mode=mode)
        except Exception as e:
            logger.debug("[DeepResearch] search failed %r (%s): %s", q, mode, e)
            return []

    results = await asyncio.gather(*[_s(q, m) for q in angles for m in ("news", "general")])
    seen, pool = set(), []
    for rs in results:
        for r in rs:
            if (r.url and r.url not in seen and not r.url.lower().endswith(".pdf")
                    and not _blocked(r.url) and not _too_old(getattr(r, "published_date", ""))
                    and _article_score(r.url) > -0.5):   # drop domain roots / bare sections
                seen.add(r.url)
                pool.append(r)
    # Rank by source authority + article-likeness so we read real stories.
    credible = [r for r in pool if _source_quality(r.url) >= 1.0]
    credible.sort(key=lambda r: _source_quality(r.url) + _article_score(r.url), reverse=True)
    host_ct, picks = {}, []
    for r in credible:
        h = _host(r.url)
        if host_ct.get(h, 0) < 2:
            host_ct[h] = host_ct.get(h, 0) + 1
            picks.append(r)
        if len(picks) >= want:
            break
    return picks


# Hard, server-side paywalls: the server sends only a stub to non-subscribers, so a
# browser — even past Cloudflare — renders just the wall. Route straight to the reader
# bypass; spend no render budget.
_HARD_PAYWALL_HOSTS = {
    "bloomberg.com", "ft.com", "wsj.com", "economist.com", "barrons.com",
}
# Metered / client-side paywalls: the article IS in the DOM and the meter lives in
# cookies, so a fresh cookieless STEALTH browser context (which also clears the
# Cloudflare interstitial) reads them for free — they escalate to the browser, not
# the bypass. Proven live: businessinsider full article, no wall.
_METERED_HOSTS = {
    "nytimes.com", "washingtonpost.com", "newyorker.com", "theatlantic.com",
    "businessinsider.com", "seekingalpha.com",
}
_PAYWALL_HOSTS = _HARD_PAYWALL_HOSTS | _METERED_HOSTS


async def _read_bodies(picks: list, *, read_target: int, browser_budget: int) -> list:
    """Read article bodies cheaply and reliably. Phase 1: http_fetch ALL picks in
    parallel (fast, no browser). Phase 2: escalate to the headless browser for
    only a few HIGH-VALUE (tier ≥ 2) misses — the browser resists asyncio
    cancellation and a single dead URL can burn 60s, so we minimize and bound it.
    Stale bodies dropped throughout."""
    async def _http(r):
        body = await _fetch_body(r.url, browser_budget=[0], allow_bypass=False)  # pure http
        return (r, body)

    good, misses = [], []
    for r, body in await asyncio.gather(*[_http(r) for r in picks]):
        if body and not _stale_body(body):
            _record_pub_date(r)
            good.append((r.title, r.url, body))
        else:
            misses.append(r)

    if len(good) < read_target:
        # Escalate HIGH-VALUE misses. HARD server-side paywalls (FT/Bloomberg/WSJ) route
        # to the reader BYPASS (no render cost); everything else — incl. metered hosts the
        # stealth browser can crack — escalates to the headless browser, capped by budget.
        hv = [r for r in misses if _source_quality(r.url) >= 2.0]
        pay = [r for r in hv if _host(r.url) in _HARD_PAYWALL_HOSTS][:12]
        js = [r for r in hv if _host(r.url) not in _HARD_PAYWALL_HOSTS][:browser_budget]
        budget = [browser_budget]

        async def _esc(r):
            body = await _fetch_body(r.url, browser_budget=budget)
            if body and not _stale_body(body):
                _record_pub_date(r)
                return (r.title, r.url, body)
            return None

        good += [a for a in await asyncio.gather(*[_esc(r) for r in (pay + js)]) if a]
    return good[:read_target]


async def _gather_sources(subject: str, *, read_target: int, browser_budget: int = 6) -> list:
    """Deep single-story gather: facet-expand the subject, search, read."""
    year = _NOW().strftime("%Y")
    raw = await _invoke_bg([{"role": "user", "content":
        f"3 web-search queries digging into this {year} story from different facets "
        f"(what happened, numbers/who, reactions/analysis): '{subject}'. JSON array of 3."}],
        json_mode=True, json_schema=_STRING_ARRAY_SCHEMA, max_tokens=160)
    extra = [a for a in _json_array(raw) if isinstance(a, str) and len(a) > 8][:3]
    angles = [subject, f"{subject} {year}"] + extra
    search_picks, aux_picks = await asyncio.gather(
        _search_candidates(angles, want=read_target + 8),
        _aux_news([subject]),
    )
    # dedup aux + search (aux = GDELT/Bing, search-independent); ≤2 per host.
    seen, host_ct, picks = set(), {}, []
    for r in aux_picks + search_picks:
        u = r.url.split("#")[0].rstrip("/")
        if u in seen:
            continue
        seen.add(u)
        h = _host(r.url)
        if host_ct.get(h, 0) >= 2:
            continue
        host_ct[h] = host_ct.get(h, 0) + 1
        picks.append(r)
        if len(picks) >= read_target + 8:
            break
    return await _read_bodies(picks, read_target=read_target, browser_budget=browser_budget)


async def _overview_angles(subjects: list[str], *, model: str | None = None) -> list[str]:
    """Facet-expand the stories into article-finding queries. The bare subject
    phrase tends to surface landing/SEO pages; a 'what happened / numbers' facet
    surfaces actual articles. ONE LLM call, and the angle count is capped — we get
    'more sources' by reading deeper into each query's results (max_results=16),
    NOT by firing many queries, which is what rate-limits the search engines."""
    year = _NOW().strftime("%Y")
    angles = list(subjects)
    try:
        # Multi-MODE sweep (#66, 2026-07-08): at the SAME query budget, ask for
        # queries spanning DISTINCT retrieval modes — developments, key players,
        # and risks/opposition — instead of one "what happened" facet per story.
        # Each mode surfaces sources the others miss (different desks cover the
        # deal vs the criticism vs the numbers), so the read pool gets broader
        # WITHOUT more queries (query volume, not depth, is what rate-limits the
        # engines — the residential-IP reputation constraint from the ops audit).
        # Query VOLUME is what rate-limits the engines (residential-IP guardrail,
        # ops audit), so the multi-mode spread is applied only to the SINGLE most
        # consequential story — three angles that hit different desks (facts /
        # players / criticism) — while every other story gets one query. ~1.5×
        # the old volume, not 3×; measured 2026-07-08 after 3× pushed candidates
        # 8→40 and query load ~2.5-3× with no proven core-coverage gain yet.
        raw = await _invoke_bg([{"role": "user", "content":
            f"For these {year} news stories, write focused web-search queries that surface actual "
            f"news ARTICLES (not homepages). For the FIRST (most consequential) story, give THREE "
            f"queries, one per angle: (1) what happened + key numbers, (2) the named "
            f"players/companies/people involved, (3) risks, criticism, or opposition. For every "
            f"OTHER story, ONE 'what happened' query.\n" +
            "\n".join(f"- {s}" for s in subjects) +
            "\nReturn one flat JSON array of all the query strings."}],
            json_mode=True, json_schema=_STRING_ARRAY_SCHEMA, max_tokens=360, temperature=0.2,
            model=model if model is not None else _syn_model())
        angles += [a for a in _json_array(raw) if isinstance(a, str) and len(a) > 8]
    except Exception:
        pass
    return list(dict.fromkeys(angles))[:16]   # cap angles → bounded search load (query volume is the rate-limit lever)


class _Cand:
    """Minimal candidate (url/title/published_date) so RSS-feed articles and
    search results flow through the same read/selection path."""
    __slots__ = ("url", "title", "published_date")

    def __init__(self, url: str, title: str = "", published_date: str = ""):
        self.url, self.title, self.published_date = url, title, published_date


async def _feed_candidates(feed_key: str) -> list:
    """Article candidates from the domain's CURATED RSS feeds — recent, credible,
    and INDEPENDENT of the rate-limited search engines. This is the reliability
    lever: it gives every domain the strong dedicated sources that Cybersecurity
    gets for free, so depth no longer hinges on flaky general search."""
    try:
        from app.monitors.rss_feeds import fetch_recent_items
        items = await fetch_recent_items(feed_key, hours=72, max_total=24)
    except Exception as e:
        logger.debug("[DeepResearch] feed fetch failed for %r: %s", feed_key, e)
        return []
    out = []
    for it in items:
        if it.url and not _blocked(it.url) and _article_score(it.url) > -0.5:
            out.append(_Cand(it.url, it.title or "", ""))
    return out


async def _gdelt(query: str, *, max_records: int = 20) -> list:
    """GDELT DOC API — keyless global news search returning REAL article URLs +
    dates. English-only, recent. Independent of SearXNG, so it survives when the
    aggregator's engines are CAPTCHA-throttled. Rate limit ~1 req/5s, so callers
    fire it ONCE per gather (never per-angle)."""
    import httpx
    params = {
        "query": f"{query} sourcelang:eng", "mode": "ArtList", "format": "json",
        "maxrecords": str(max_records), "timespan": "5d", "sort": "DateDesc",
    }
    try:
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "Mozilla/5.0"}) as c:
            r = await c.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
        data = r.json()
    except Exception as e:
        logger.debug("[DeepResearch] gdelt failed %r: %s", query, e)
        return []
    out = []
    for a in (data.get("articles") or []):
        u = a.get("url")
        if u and not _blocked(u) and _article_score(u) > -0.5:
            out.append(_Cand(u, a.get("title", "") or "", ""))
    return out


_RSS_ITEM_RE = re.compile(r"<item>(.*?)</item>", re.DOTALL | re.IGNORECASE)
_RSS_LINK_RE = re.compile(r"<link>\s*(https?://[^<\s]+)\s*</link>", re.IGNORECASE)
_RSS_TITLE_RE = re.compile(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", re.DOTALL | re.IGNORECASE)


async def _bing_news(query: str, *, max_items: int = 15) -> list:
    """Bing News RSS — clean article URLs (no Google-style redirect encoding),
    recent. A second SearXNG-independent channel."""
    import httpx
    url = f"https://www.bing.com/news/search?q={quote(query)}&format=rss"
    try:
        async with httpx.AsyncClient(timeout=15, headers={"User-Agent": "Mozilla/5.0"},
                                     follow_redirects=True) as c:
            txt = (await c.get(url)).text
    except Exception as e:
        logger.debug("[DeepResearch] bing-news failed %r: %s", query, e)
        return []
    out = []
    for m in _RSS_ITEM_RE.finditer(txt):
        block = m.group(1)
        lk = _RSS_LINK_RE.search(block)
        if not lk:
            continue
        u = lk.group(1).strip()
        if "bing.com" in u or _blocked(u) or _article_score(u) <= -0.5:
            continue
        ti = _RSS_TITLE_RE.search(block)
        out.append(_Cand(u, (ti.group(1).strip() if ti else ""), ""))
        if len(out) >= max_items:
            break
    return out


def _short_query(s: str) -> str:
    """Reduce a long subject phrase to its key entities — GDELT/Bing return 0 for
    long natural-language phrases but work well on a few proper nouns/keywords."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]+", s or "")
    sig = [w for w in words if w.lower() not in _STOP and len(w) > 2]
    caps = [w for w in sig if w[0].isupper()]
    pick = (caps or sig)[:4]
    return " ".join(pick) or (s or "")


async def _aux_news(queries: list[str]) -> list:
    """Auxiliary news candidates from the SearXNG-INDEPENDENT channels: Bing News
    RSS + GDELT (first query only, rate limit). Queries are shortened to key
    entities — long phrases return 0 from both engines. Resilience layer: when the
    SearXNG engines are throttled, these still return real recent article URLs."""
    short = [_short_query(q) for q in queries if q][:3]
    if not short:
        return []
    tasks = [_bing_news(q) for q in short]
    tasks.append(_gdelt(short[0]))   # one GDELT call only
    results = await asyncio.gather(*tasks)
    return [c for lst in results for c in lst]


async def _title_relevance(picks: list, subjects: list[str]) -> dict:
    """Map pick.url -> topic-relevance score in [0,1] via the resident CPU
    bge-m3 embedder (#38 reranker, 2026-07-08).

    The read-set selection in _gather_overview ranks candidates by REPUTABILITY
    and article-shape only — a strong wire on a tangential story outranks an
    on-topic domain specialist when the read budget binds. This adds the missing
    topic-relevance signal: cosine(subjects, title). ORDERING-ONLY (host caps +
    budget still apply downstream), so recall is preserved — nothing is dropped
    by a low score, it just reads later. Fails open (empty dict -> no effect) if
    the embedder is unreachable, matching the CPU-embed degradation pattern.
    """
    titles = [(getattr(r, "url", None), (getattr(r, "title", "") or "").strip()) for r in picks]
    titles = [(u, t) for u, t in titles if u and t]
    if not titles or not subjects:
        return {}
    try:
        from app.core.embedding import get_embedding_function
        ef = get_embedding_function()
        if ef is None:
            return {}
        query = " / ".join(subjects[:5])[:400]
        texts = [query] + [t for _, t in titles]
        vecs = await asyncio.to_thread(ef, texts)
        if not vecs or len(vecs) != len(texts):
            return {}
        import math
        qv = vecs[0]
        qn = math.sqrt(sum(x * x for x in qv)) or 1.0
        out: dict = {}
        for (u, _t), tv in zip(titles, vecs[1:]):
            tn = math.sqrt(sum(x * x for x in tv)) or 1.0
            cos = sum(a * b for a, b in zip(qv, tv)) / (qn * tn)
            out[u] = max(0.0, cos)  # negatives = off-topic → no bonus
        return out
    except Exception as e:
        logger.warning("[DeepResearch] title-relevance rerank skipped: %s", e)
        return {}


async def _gather_overview(subjects: list[str], feed_key: str, *,
                           read_target: int, browser_budget: int) -> list:
    """Pooled breadth gather: merge CURATED FEED articles (reliable, search-
    independent) with facet-expanded SEARCH results, then read bodies http-fast-
    path first. Feeds come first so a domain's strong dedicated sources anchor the
    read set even when general search is throttled. read_target is large for
    breadth: extra reads are cheap http GETs, so we read widely and let synthesis
    pick. Not N parallel deep gathers — those contend for the browser semaphore."""
    angles = await _overview_angles(subjects)
    feed_picks, search_picks, aux_picks = await asyncio.gather(
        _feed_candidates(feed_key),
        _search_candidates(angles, want=read_target + 12),
        _aux_news(subjects[:3]),
    )
    # Spend the read budget on the STRONGEST readable sources, not just whatever pooled
    # first: sort the pool by source quality + article-likeness (curated feeds get a
    # small on-mission bonus so a domain specialist isn't buried under a generic wire).
    # Weak single-source blogs (techgrid.media-class) get read only if nothing better
    # remains — the input must be strong, not only the output-calibration.
    feed_urls = {r.url for r in feed_picks}
    pool = feed_picks + aux_picks + search_picks
    # Topic-relevance rerank (#38): blend cosine(subjects, title) into the
    # reputability prior so on-topic sources win the read budget. Scaled to
    # ~+1.0 max — meaningful but below a full source-tier gap, so a tier-1 wire
    # still outranks a tangential blog unless the blog is much more on-topic.
    _rel = await _title_relevance(pool, subjects)
    pool.sort(key=lambda r: _source_quality(r.url) + (0.4 if r.url in feed_urls else 0.0)
                            + _article_score(r.url) + 1.5 * _rel.get(r.url, 0.0),
              reverse=True)
    if _rel:
        logger.info("[DeepResearch] title-relevance rerank applied to %d candidate(s)", len(_rel))
    seen, host_ct, picks = set(), {}, []
    for r in pool:
        u = r.url.split("#")[0].rstrip("/")
        if u in seen:
            continue
        seen.add(u)
        h = _host(r.url)
        if host_ct.get(h, 0) >= 3:
            continue
        host_ct[h] = host_ct.get(h, 0) + 1
        picks.append(r)
        if len(picks) >= read_target + 16:
            break
    logger.info("[DeepResearch] overview candidates: %d feed + %d aux + %d search → %d picks",
                len(feed_picks), len(aux_picks), len(search_picks), len(picks))
    return await _read_bodies(picks, read_target=read_target, browser_budget=browser_budget)


async def _gap_followup(findings: list, label: str, *, model: str | None = None) -> list[str]:
    """Reflect on the first-pass findings and return targeted follow-up SEARCH QUERIES for
    the consequential sub-questions still UNANSWERED or thinly sourced — the engine of the
    iterative loop (Search-o1 / Jina DeepSearch). Conditioning later queries on what the
    first read revealed is exactly what single-pass gather can't do. Empty if pool is thin."""
    if len(findings) < 3:
        return []
    have = "\n".join(f"- {t} ({_host(u)})" for t, u, _ in findings[:20])
    try:
        raw = await _invoke_bg([{"role": "user", "content":
            f"Today is {_NOW():%B %d, %Y}. We are compiling a {label} intelligence brief and have "
            f"gathered these stories so far:\n{have}\n\n"
            f"Which CONSEQUENTIAL {label} sub-questions are still UNANSWERED or only thinly sourced "
            f"(a single outlet)? Output 3-5 specific web SEARCH QUERIES that would fill the biggest "
            f"gaps — missing corroboration, the 'so what' details (exact numbers, named players, "
            f"outcomes), or a consequential angle not yet covered. Return a JSON array of query "
            f"strings, most important first; NO commentary."}],
            json_mode=True, json_schema=_STRING_ARRAY_SCHEMA, max_tokens=220, temperature=0.3,
            model=model if model is not None else _syn_model())
        return [str(q).strip() for q in _json_array(raw)
                if isinstance(q, str) and 8 < len(str(q).strip()) < 120][:5]
    except Exception as e:
        logger.warning("[DeepResearch] gap-followup query generation failed (%s) — "
                       "iterative loop skips expansion this cycle: %s", label, e)
        return []


async def _gather_gap(queries: list[str], *, read_target: int = 10, browser_budget: int = 8) -> list:
    """Search + read the follow-up gap queries (no feeds — these are targeted angles, not a
    domain sweep). Returns [(title, url, body)] like _gather_overview so results merge in."""
    if not queries:
        return []
    picks = await _search_candidates(queries, want=read_target + 6)
    if not picks:
        return []
    return await _read_bodies(picks, read_target=read_target, browser_budget=browser_budget)


_NO_CONTENT_RE = re.compile(
    r"\b(irrelevant|no (?:concrete |specific |relevant |substantive )?(?:information|findings|"
    r"content|facts|details|data)|contains no|no (?:actual )?article|boilerplate|"
    r"navigation (?:menu|bar|chrome|links)|cannot extract|nothing (?:to extract|reportable))",
    re.IGNORECASE)


# A line that ends cleanly: sentence punctuation, optionally followed by a
# closing quote/bracket/citation paren, OR a self-contained short label line.
_CLEAN_TAIL_RE = re.compile(r"""[.!?…][)"'”’\]]*\s*$""")


def _trim_dangling_tail(text: str) -> str:
    """Drop a final line that was cut mid-sentence by the max_tokens cap.

    Only the LAST line is ever considered (bullet lists keep every complete
    bullet), and only when it plainly lacks a sentence boundary. If trimming
    would gut the text, the original is returned — downstream gates decide."""
    if not text:
        return text
    lines = text.rstrip().split("\n")
    last = lines[-1].strip()
    if not last or _CLEAN_TAIL_RE.search(last):
        return text
    # numeric/label endings like "…up 34%" or "…$1.2B" are complete thoughts
    # magnitude suffixes count: "$1.2B)" / "40M" are complete numeric tails
    if re.search(r'[%\d][KMBTkmbt]?[)"”’\]]*\s*$', last):
        return text
    trimmed = "\n".join(lines[:-1]).rstrip()
    if len(trimmed) >= 40:
        return trimmed
    # Single-line (or nearly) text cut mid-word: dropping the whole line would
    # gut it, but the line may still contain complete sentences — keep those.
    # "TSMC confirmed the expansion. Construction begins in Octo" ->
    # "TSMC confirmed the expansion."
    # a sentence ender must precede whitespace/end — "." inside "$1.2" is a
    # decimal point, not a boundary (it mangled "$1.2B)" to "$1." before)
    m = list(re.finditer(r"""[.!?…][)"'”’\]]*(?=\s|$)""", last))
    if m:
        intra = last[:m[-1].end()].rstrip()
        rebuilt = "\n".join(lines[:-1] + [intra]).strip()
        # Floor is LOWER here than the caller's 40-char gate on purpose:
        # the trimmer's job is to return clean sentences; the caller's gate
        # then decides whether what remains is substantial enough to keep.
        # A repaired-but-thin result gets dropped THERE — which is the right
        # outcome for a finding reduced to six words — whereas returning the
        # raw fragment would put a mid-sentence tail into the evidence pool.
        if len(rebuilt) >= 20:
            return rebuilt
    return text


async def _findings(articles: list, subject: str, *, model: str | None = None) -> list[tuple[str, str, str]]:
    async def _one(title, url, body):
        try:
            f = await _invoke_bg([{"role": "user", "content":
                f"Extract 2-3 concrete findings (facts, numbers, named events) from this article "
                f"relevant to '{subject}'. Reply IRRELEVANT only if the article is about a COMPLETELY "
                f"different field. Only what is stated.\n\nTITLE: {title}\n\n{body}"}],
                # 512 (was 240 → 320), 2026-08-26: the tripwire still caught
                # ~20 truncations/day at 320 — "2-3 concrete findings" from a
                # dense 12k-char body legitimately runs 1.3-1.6k chars, and a
                # mid-sentence cut puts a dangling fragment into the evidence
                # pool. max_tokens only CAPS: short findings pay nothing.
                # num_ctx 8192 (2026-08-25): the 08-22 body-cap raise
                # (5k→12k chars ≈ up to ~4k tokens) overflowed the 4096
                # model default — prompt_eval+eval hit exactly 4096 and
                # findings truncated mid-sentence ~95×/day, re-opening the
                # 08-14 defect from the prompt side.
                # 800 (was 512), 2026-09-01: on the 27B the extraction is worth
                # the room — 5 of 6 truncation warnings in a day came from here.
                max_tokens=800, temperature=0.2, num_ctx=8192,
                model=model if model is not None else _syn_model())
            f = (f or "").strip()
            # Dangling-fragment repair (2026-08-31). The cap history here is
            # 240 -> 320 -> 512 and the tripwire STILL catches an occasional
            # cut (eval == max_tokens, mid-generation) — raising the cap
            # shrinks the tail but never kills it, and a mid-sentence fragment
            # lands in the EVIDENCE POOL where synthesis may quote it. Repair
            # deterministically instead of raising again: if the last line
            # does not end at a sentence/citation boundary, drop that line.
            # Same philosophy as the digest-side _bound_and_clean repair.
            f = _trim_dangling_tail(f)
            # Drop empties, relevance-rejects, and "the page has no real content"
            # outputs (nav/boilerplate pages that slipped the body gate) — these
            # otherwise pollute the evidence with non-findings.
            if not f or len(f) < 40 or _NO_CONTENT_RE.search(f[:160]):
                return None
            return (title, url, f)
        except Exception:
            return None
    return [r for r in await asyncio.gather(*[_one(t, u, b) for t, u, b in articles]) if r]


_PAREN_CITE_RE = re.compile(r"\([^()]*\b[a-z0-9][a-z0-9\-]*\.[a-z]{2,}[^()]*\)")
_DOMAIN_TOKEN_RE = re.compile(r"\b[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.[a-z]{2,}\b")


def _cite_tokens(hosts) -> set[str]:
    """Valid citation tokens = each read host plus its registrable domain, so a
    '(slashdot.org)' citation matches a read host 'tech.slashdot.org'."""
    toks: set[str] = set()
    for h in hosts:
        h = h.lower()
        toks.add(h)
        parts = h.split(".")
        if len(parts) >= 2:
            toks.add(".".join(parts[-2:]))
    return toks


def _strip_fake_citations(text: str, hosts) -> str:
    """Delete any line that carries an inline (outlet.tld) citation where NONE of
    the cited outlets were actually read. This kills the 9B's fabricated-attribution
    failure (e.g. citing '(cnbc.com)' for a story pulled from memory, not sources)
    — the LLM verify pass can't be trusted to catch it, so we enforce it in code.
    Lines without a domain citation (headers, connective prose) are kept."""
    valid = _cite_tokens(hosts)
    if not valid:
        return text
    kept = []
    for line in text.split("\n"):
        if _PAREN_CITE_RE.search(line):
            cited = set(_DOMAIN_TOKEN_RE.findall(line.lower()))
            if cited and not any(
                c == v or c.endswith("." + v) or v.endswith("." + c)
                for c in cited for v in valid
            ):
                continue  # every cited outlet is unread → fabricated attribution → drop
        kept.append(line)
    return "\n".join(kept)


# --- Numeric grounding: external verification of every figure against sources ---
# The 9B reliably FINDS a number in a given text but unreliably REMEMBERS it across
# a long synthesis (it inflated FortiBleed to 110M vs the real 86,644). So we don't
# trust the model's memory: we verify every magnitude figure in the briefing against
# the raw source bodies, correct/drop what doesn't trace. Accuracy then depends on
# the source text, not the model — which is how you beat the size limit, not accept it.
_MAGNITUDE_RE = re.compile(
    r"(?P<cur>[$€£])?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?"
    r"(?P<unit>million|billion|trillion|percent|per cent|%|bps|basis points)?",
    re.IGNORECASE)
_MULT = {"million": 1_000_000, "billion": 1_000_000_000, "trillion": 1_000_000_000_000}
# A year-range number directly before a count-noun is a COUNT, not a year — used to
# catch the '20-30 employees'→'2030 employees' mis-join that the bare-value check
# misses (the digits also appear as a year in the sources).
_COUNT_NOUN_RE = (r"(?:employees?|people|workers?|staff|users?|customers?|patients?|"
                  r"members?|residents?|households?|devices?|firewalls?|servers?|accounts?)")
_YEAR_COUNT_RE = re.compile(r"\b(19\d\d|20\d\d)\s+(" + _COUNT_NOUN_RE + r")\b", re.IGNORECASE)
# A 3-digit value (100–999) before a physical MEASUREMENT unit is checkable but slips
# the magnitude gate (val<1000, unit isn't a money/scale word) — e.g. an altitude
# '185 miles' that should be 370. Targeted at the hundreds; smaller values are too
# collision-prone to flag safely.
_MEASURE_UNIT_RE = re.compile(
    r"\b(\d{3})\s+(?:miles?|kilometers?|kilometres?|km|meters?|metres?|feet|ft|"
    r"light-?years?|mph|knots?|acres?|hectares?|tonnes?|tons?)\b", re.IGNORECASE)


def _num_variants(num: str, unit: str | None, cur: str | None) -> set[str]:
    """All string forms a figure could take in source text, so '110 million' matches
    '110 million' / '110,000,000' / '110000000'."""
    num_nc = num.replace(",", "")
    out = {num.lower(), num_nc.lower()}
    u = (unit or "").lower()
    try:
        val = float(num_nc)
    except ValueError:
        return out
    if u in _MULT:
        full = int(val * _MULT[u])
        out |= {f"{num.lower()} {u}", f"{num_nc} {u}", str(full), f"{full:,}"}
    if u in ("percent", "per cent", "%"):
        out |= {f"{num_nc}%", f"{num_nc} percent", f"{num_nc} per cent"}
    if cur:
        out |= {f"{cur}{num.lower()}", f"{cur}{num_nc}"}
    return {v for v in out if v}


def _unverified_numbers(text: str, corpus: str, corpus_nc: str) -> list[str]:
    """Magnitude figures in `text` whose value appears NOWHERE in the source corpus.
    Only flags clear absences (counts ≥1000, $/€/£ amounts, %, or N million/billion) —
    small bare integers and years are ignored to avoid false positives."""
    bad = []
    for m in _MAGNITUDE_RE.finditer(text):
        raw = m.group(0).strip()
        num, unit, cur = m.group("num"), m.group("unit"), m.group("cur")
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        u = (unit or "").lower()
        magnitude = ("," in num) or (u in _MULT) or (cur is not None) \
            or (u in ("percent", "per cent", "%")) or val >= 1000
        if not magnitude:
            continue
        variants = _num_variants(num, unit, cur)
        if not any(v in corpus or v in corpus_nc for v in variants):
            bad.append(raw)
    # Year-vs-count collision: re-check a year-range number used as a COUNT with its
    # noun, since the bare digits pass above by matching a YEAR mention. Flag only if
    # neither 'NNNN noun' nor its comma form 'N,NNN noun' is in the sources.
    for m in _YEAR_COUNT_RE.finditer(text):
        yr, noun = m.group(1), m.group(2).lower()
        if not any(f in corpus for f in (f"{yr} {noun}", f"{int(yr):,} {noun}")):
            bad.append(m.group(0).strip())
    # Measurement figures in the hundreds (e.g. '185 miles' vs a real 370): flag when
    # the bare value is absent from every source. Conservative — only fires if the
    # digits appear NOWHERE, so a value present in any context is left alone.
    for m in _MEASURE_UNIT_RE.finditer(text):
        val_s = m.group(1)
        if val_s not in corpus and val_s not in corpus_nc:
            bad.append(m.group(0).strip())
    # dedupe, keep order
    return list(dict.fromkeys(bad))


def _drop_sentences_with(text: str, bad: set[str]) -> str:
    """Deterministic backstop: remove sentences that STILL carry an unverified figure
    after correction. Operates line-by-line so markdown headers/bullets survive."""
    out_lines = []
    for line in text.split("\n"):
        stripped = line.lstrip("*-• ").strip()
        if stripped.startswith("#") or stripped.startswith("**") or not stripped:
            out_lines.append(line)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", line)
        kept = [s for s in sentences if not any(b in s for b in bad)]
        if kept:
            out_lines.append(" ".join(kept))
        # if every sentence in the line carried a bad figure, drop the line
    return "\n".join(out_lines)


# Placeholder tokens the correction model writes INSTEAD of removing a sentence
# whose figure it can't find — bracketed ('[FIGURE NOT IN SOURCES]') or an empty
# labeled parenthetical ('(Political Rights:; Civil Liberties:)'). These reached
# Discord verbatim (2026-08-14 Whale Watch shipped six); the numeric backstop
# only re-checks DIGITS so placeholder prose sailed through.
_PLACEHOLDER_RE = re.compile(
    r"\[[^\]]*(?:figure|number|value|data|amount|not in source|not provided|"
    r"not stated|not specified|no figure|not available|n/?a)[^\]]*\]", re.I)
_EMPTY_LABEL_RE = re.compile(r"\([^()]*\b\w[\w ]*:\s*(?:;|\))")


def _strip_placeholders(text: str) -> str:
    """Drop any sentence carrying a placeholder token (see _PLACEHOLDER_RE /
    _EMPTY_LABEL_RE). Line-by-line so headers/bullets survive; a bare label line
    is kept even if its content sentence dies."""
    if not text or ("[" not in text and ":" not in text):
        return text
    out_lines = []
    dropped = 0
    for line in text.split("\n"):
        stripped = line.lstrip("*-• ").strip()
        if stripped.startswith("#") or not stripped:
            out_lines.append(line)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", line)
        kept = [s for s in sentences
                if not _PLACEHOLDER_RE.search(s) and not _EMPTY_LABEL_RE.search(s)]
        dropped += len(sentences) - len(kept)
        if kept:
            out_lines.append(" ".join(kept))
        elif _is_label_line(line):
            out_lines.append(line)
    if dropped:
        logger.info("[DeepResearch] dropped %d sentence(s) carrying a placeholder token", dropped)
    return "\n".join(out_lines)


_SCAFFOLD_RE = re.compile(
    r"(?i)(rewritten draft|provided source text|provided text block|the sources? section|"
    r"based on the (provided|sources)|corrected to match|have been (removed|adjusted)|"
    r"strictly speaking|the prompt|the draft'?s?\b|in the draft|to ensure (strict )?adherence|"
    r"here('?s| is) the (corrected|rewritten|updated))")


def _strip_correction_scaffold(text: str) -> str:
    """Remove meta-commentary the correction LLM sometimes prepends/injects ('Based
    on the provided source text… here is the rewritten draft…', inline '*Note:*'
    explanations). Keeps only the briefing itself."""
    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            kept.append(ln)
            continue
        low = s.lower()
        if s.startswith(("*Note", "Note:", "(Note", "*(", "> ")):
            continue
        # scaffold line, but never drop an actual heading/bold-lead line
        if _SCAFFOLD_RE.search(low) and not s.startswith(("**", "##", "#", "*   ", "* ", "-")):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


# A dangling "titled '<generic>'" clause — the synthesis model writes a PLACEHOLDER
# title ('…published in a new article titled "new article"…') when it never got the
# real one, and it leaked to Discord (2026-08-16 Space digest). Anchored so the
# quote's ENTIRE content is a generic document-word (optionally 'new/the/a …'), so a
# real title ('titled "The Immune Response to X"') never matches — the closing quote
# won't align. Strips only the clause; the surrounding finding survives.
_PLACEHOLDER_TITLE_RE = re.compile(
    r"\s*,?\s*(?:titled|entitled|called)\s+"
    r"[\"'“‘]\s*(?:"
    r"(?:new|the|this|a|an)?\s*"
    r"(?:article|study|paper|report|preprint|analysis|manuscript|publication|piece)"
    r"|untitled|unnamed|unknown|n/?a|tbd|none"       # bare placeholder as the whole title
    r")\s*[\"'”’]",
    re.I,
)


def _tidy_citations(text: str) -> str:
    """Cosmetic cleanup of synthesis artifacts (2026-06-24 verification pass): the
    model occasionally wraps a citation in stray '$' ('($quiverquant.com$)'), drops a
    truncated source TITLE into a citation slot ('(… starts making noise…)'), emits a
    placeholder article title ('titled "new article"'), or leaves a trailing space
    inside emphasis when a token is dropped ('*Artemis *'). Deterministic + tightly
    scoped, so real figures ('$9.2m'), real titles, bold labels, and bullets are
    untouched."""
    if not text:
        return text
    # 1) strip stray '$' inside any parenthetical that contains a domain (a citation).
    def _fix(m):
        inner = re.sub(r"\$+", "", m.group(1)).strip()
        return "(" + re.sub(r"\s{2,}", " ", inner) + ")"
    text = re.sub(r"\(([^()]*\b[a-z0-9][a-z0-9.\-]*\.[a-z]{2,}[^()]*)\)", _fix, text)
    # 2) drop a parenthetical with NO domain that ends in '…' — a leaked source title.
    def _drop(m):
        return m.group(0) if re.search(r"\b[a-z0-9.\-]+\.[a-z]{2,}\b", m.group(1)) else ""
    text = re.sub(r"\s*\(([^()]*\.\.\.)\)", _drop, text)
    # 3) drop a dangling placeholder-title clause ('titled "new article"') the model
    #    emits when it lacks the real title (2026-08-16). Clause-level, so the real
    #    sentence survives.
    text = _PLACEHOLDER_TITLE_RE.sub("", text)
    # 4) collapse single-asterisk emphasis wrapping a trailing space ('*Artemis *' ->
    #    '*Artemis*') — the artifact left when a token is dropped inside italics
    #    (qwen3.8 dropped 'II' from '*Artemis II*'). Single '*' only (never '**bold**'),
    #    requires a letter in the span (skips '3 * 4 *' arithmetic), and the opening
    #    '*' must be mid-line (skips '* ' bullets).
    text = re.sub(r"(?<=[^\n*])\*([^*\n]*[A-Za-z][^*\n]*?) +\*(?!\*)", r"*\1*", text)
    # 5) trim a trailing space left inside a BOLD span wrapping a FIGURE whose unit
    #    the model dropped ('**13,931 **' -> '**13,931**'; qwen3.x sometimes drops
    #    the unit token e.g. 'BTC' the same way it dropped 'II' from '*Artemis II*').
    #    The unit itself is unrecoverable, but the dangling '** **' reads as broken.
    #    Scoped to a number-ending bold span (requires a digit run before the trailing
    #    space) so bold labels and any '**a** **b**' sequence are untouched (2026-08-18).
    text = re.sub(r"(\*\*[^*\n]*\d[\d,.]*) +\*\*", r"\1**", text)
    return text


def _is_label_line(s: str) -> bool:
    """A line that is ONLY a section header / bold lead-label (scaffolding, no
    inline content): '## Foo', '**Secondary Developments**'. A bold label with
    text after it ('**Lead** — the paragraph…') is NOT a bare label."""
    s = s.strip()
    if not s:
        return False
    if s.startswith("#"):
        return True
    return s.startswith("**") and s.endswith("**") and s.count("**") == 2


# Synthesis-prompt INSTRUCTION text the model occasionally ECHOES verbatim after a
# section header instead of replacing it with content (seen live in a Physics digest:
# "**Lead Development** — THE single most consequential … second-order implications.").
# These are our EXACT prompt fingerprints, so they never occur in real synthesis — strip
# them; a header left empty is then dropped by _drop_orphan_headers.
_PROMPT_LEAK_RE = re.compile(
    r"(?is)(?:\s*—)?\s*(?:"
    r"THE single most consequential.*?second-order implications\.?"
    r"|each OTHER genuinely consequential.*?padded with filler\.?"
    r"|(?:\d[-\s]?\d?\s*sentences?:?\s*)?the throughline across the stories.*?in this section\.?"
    r")")


def _strip_prompt_leak(text: str) -> str:
    """Drop synthesis-prompt instruction blocks the model echoed instead of following."""
    return _PROMPT_LEAK_RE.sub("", text or "").strip()


def _drop_orphan_headers(text: str) -> str:
    """Remove a section header / bold-label line that has NO content beneath it.
    Every correction pass deliberately KEEPS header lines but can drop all the prose
    under them (`_drop_sentences_with`), leaving a dangling '**Secondary Developments**'
    with nothing after it — the visible digest misfire. A label is orphaned when the
    next non-empty line is another label or end-of-text."""
    lines = text.split("\n")
    keep = [True] * len(lines)
    for i, ln in enumerate(lines):
        if not _is_label_line(ln):
            continue
        has_body = False
        for j in range(i + 1, len(lines)):
            nxt = lines[j].strip()
            if not nxt:
                continue
            has_body = not _is_label_line(lines[j])   # body iff next non-empty isn't a label
            break
        if not has_body:
            keep[i] = False
    return "\n".join(l for k, l in zip(keep, lines) if k)


def _accept_correction(original: str, fixed: str) -> bool:
    """Accept an LLM correction pass's output only if it (a) kept the briefing's
    structure and (b) did NOT collapse its length. A faithful 'change only X, keep
    everything else' edit stays close in size; a drastic shrink means the pass gutted
    real content — so fall back to the original. This bounds the erosion risk of the
    strip stack running downstream of the (stronger) synthesis model."""
    if not fixed or ("**" not in fixed and "##" not in fixed):
        return False
    # Reject a leaked reasoning monologue outright: think tags, or the tell-tale
    # chain-of-thought preamble a correction pass must never emit. A think block
    # can contain markdown ** and be long enough to pass the size checks below,
    # so it slipped through and replaced a good digest (Science incident,
    # 2026-07-08). Belt-and-suspenders with the invoke_nothink + strip fixes.
    fl = fixed.lower()
    if "<think>" in fl or "</think>" in fl or "here's a thinking process" in fl \
            or "analyze user input" in fl:
        return False
    o, f = len(original.strip()), len(fixed.strip())
    if o >= 400 and f < 0.5 * o:   # lost >half of a substantial briefing → over-stripped
        return False
    return True


# High-precision signatures of a sentence left grammatically BROKEN when a strip/
# correction pass (or the model) deleted a load-bearing word. Each is something that
# is essentially never valid prose, so dropping the carrying sentence is safe.
_BROKEN_RE = re.compile(
    r"\bthe\s+(?:by|and|of|to|with)\b(?!-)"                          # noun deleted after an article: "letting down the by failing", "the and on March 12"
    r"|\b(?:up to|nearly|about|around|roughly|as much as)\s+times\b"  # number deleted before "times": "up to times the legal limit"
    r"|\bU\.?S\.?\s+of\s+State\b",                                    # title dropped: "US of State confirmed" (was "US Secretary of State")
    re.IGNORECASE)


# A preposition immediately followed by a comma = its object was excised by an
# upstream strip pass ("implications for, requiring" — the contamination-guard
# backstop that removed a grafted country name, 2026-07-08). A preposition with
# NOTHING between it and the comma is essentially never valid prose, so removing
# the orphaned "<prep>," recovers the sentence ("implications requiring"). The
# object-bearing case ("for X, Y") never matches — there's text before the comma.
_DANGLING_PREP_RE = re.compile(
    r"\b(?:for|in|on|at|to|with|by|of|from|into|onto|over|under|about|against|"
    r"including|involving|regarding|concerning|amid|amidst|across|between|among|"
    r"through|during|despite|toward|towards|upon|within|without|per|via)\s*,",
    re.IGNORECASE,
)
# A possessive "'s" with no owner in front (start of line/clause) — the entity
# name was stripped, leaving "'s small-molecule agonist" (2026-07-08).
_ORPHAN_POSSESSIVE_RE = re.compile(r"(^|[.!?]\s+|[:;]\s+|\(\s*)'s\s+", re.IGNORECASE)


def _repair_dangling_fragments(text: str) -> str:
    """Recover sentences left grammatically broken when a strip/correction pass
    deleted a NAMED ENTITY but not its surrounding scaffolding. Runs BEFORE
    `_repair_broken_sentences` so a repaired sentence isn't needlessly dropped.
    Two high-precision patterns (both essentially never valid prose):
      • "<preposition> ,"  → the prep's object was excised → drop the orphan prep
      • orphan "'s "       → the possessor was excised → drop the orphan "'s"
    """
    if not text:
        return text
    out = _DANGLING_PREP_RE.sub(",", text)     # "implications for, requiring" → "implications, requiring"
    out = _ORPHAN_POSSESSIVE_RE.sub(r"\1", out)  # "'s small-molecule" → "small-molecule"
    # tidy the double punctuation / spacing the excision can leave
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out


# Sentence-ending: terminal punctuation, optionally followed by a closing
# quote/paren/citation, at a whitespace or string boundary.
_SENT_END_RE = re.compile(r"[.!?][)\"'”’]*(?=\s|$)")


def _bound_and_clean(text: str, max_chars: int = 11000) -> str:
    """DETERMINISTIC guarantee that a digest ends on a complete sentence and is
    length-bounded — the final backstop against BOTH failure modes that dogged
    the digests (2026-07-08/09): (a) generation hitting a max_tokens cap and
    ending mid-word (dense text hits 5000 tokens at only ~8.5k chars), and
    (b) the model ignoring the soft "~8500 char" prompt bound and ballooning
    past the 12000 storage cap. Chasing token/storage caps was an arms race
    against variable content density; this ends it. Trims to the last complete
    sentence within max_chars (< the 12000 storage cap). Applied to the RETURN
    value so the Discord post AND the stored copy are both clean + bounded.
    """
    if not text:
        return text
    t = text if len(text) <= max_chars else text[:max_chars]
    st = t.rstrip()
    if not st or st[-1] in ".!?)\"'”’":
        return st                      # already ends cleanly
    matches = list(_SENT_END_RE.finditer(st))
    if matches and matches[-1].end() > len(st) * 0.5:
        return st[:matches[-1].end()]  # drop the trailing incomplete sentence
    return st                          # no usable boundary (rare) — leave as-is


def _repair_broken_sentences(text: str) -> str:
    """Deterministic LAST-pass cleanup: drop any sentence left grammatically BROKEN by
    the upstream strip/correction passes deleting a load-bearing word — 'letting down
    the by failing', 'China adopted the and on March 12', 'up to times the legal limit',
    'US of State confirmed'. A shipped briefing with a mangled fragment reads as
    untrustworthy; better to omit the sentence. Conservative + line-by-line so headers,
    bold labels, and the other sentences in a bullet survive (mirrors
    `_drop_sentences_with`)."""
    out_lines = []
    for line in text.split("\n"):
        stripped = line.lstrip("*-•# ").strip()
        if not stripped or _is_label_line(line) or stripped.startswith(("#", "**", ">")):
            out_lines.append(line)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", line)
        kept = [s for s in sentences if not _BROKEN_RE.search(s)]
        if len(kept) == len(sentences):
            out_lines.append(line)                  # nothing broken → untouched
        elif kept:
            out_lines.append(" ".join(kept))        # drop only the broken sentence(s)
        # else: every sentence broken → drop the whole line
    return "\n".join(out_lines)


_BOTTOMLINE_RE = re.compile(r"(?is)(\*\*\s*connections?\b.*?bottom line\s*\*\*.*)$")


def _strip_novel_bottomline_figures(text: str) -> str:
    """The 'Connections & bottom line' must only REFERENCE figures already stated in the
    developments above — a currency/scale figure that appears ONLY in the bottom line is
    introduced-in-conclusion, usually invented or a scope error (the audit's US-Policy
    '$67 billion' vs the body's '$87.6B/$21B'). Deterministically drop the carrying
    sentence. FP-safe: a real summary adds no new numbers, and bare percentages (often
    forward-looking thresholds) are deliberately exempt."""
    m = _BOTTOMLINE_RE.search(text)
    if not m:
        return text
    above, bottom = text[:m.start()], m.group(1)
    # separate the "**Connections & bottom line**" label (usually inline) from its prose
    lm = re.match(r"(?is)(\*\*[^*]+\*\*[\s—:\-]*)(.*)", bottom)
    if not lm:
        return text
    label, content = lm.group(1), lm.group(2)
    above_low = above.lower()
    above_nc = above_low.replace(",", "")
    novel = []
    for fm in _MAGNITUDE_RE.finditer(content):
        num, unit, cur = fm.group("num"), fm.group("unit"), fm.group("cur")
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        u = (unit or "").lower()
        if u in ("percent", "per cent", "%"):
            continue   # exempt bare percentages (forward-looking thresholds)
        if not ((cur is not None) or ("," in num) or (u in _MULT) or val >= 1000):
            continue   # only meaningful currency/count magnitudes
        variants = _num_variants(num, unit, cur)
        if not any(v in above_low or v in above_nc for v in variants):
            novel.append(fm.group(0).strip())
    if not novel:
        return text
    sentences = _SENT_SPLIT_RE.split(content)
    kept = [s for s in sentences if not any(nv in s for nv in novel)]
    logger.info("[DeepResearch] bottom-line: dropped %d sentence(s) with a novel figure (%s)",
                len(sentences) - len(kept), ", ".join(novel[:4]))
    if not kept:
        return above.rstrip()                        # whole bottom-line was novel → drop it
    return above + label + " ".join(kept)


async def _ground_numbers(text: str, bodies: list[str], *, model: str | None = None) -> tuple[str, int]:
    """Verify every figure in `text` against the raw source bodies. Unsupported
    figures get one focused, source-grounded correction pass (the 9B IS good at
    'find the real number in this text'), then a deterministic backstop drops any
    sentence whose figure still doesn't trace. Returns (text, n_unverified)."""
    bodies = [b for b in bodies if b]
    if not text or not bodies:
        return text, 0
    corpus = " ".join(bodies).lower()
    corpus_nc = corpus.replace(",", "")
    unver = _unverified_numbers(text, corpus, corpus_nc)
    if not unver:
        return text, 0
    src = "\n\n".join(b[:4000] for b in bodies)[:14000]
    try:
        fixed = await _invoke_bg([{"role": "user", "content":
            "SOURCE TEXTS and a DRAFT briefing are below. These figures in the draft do NOT appear "
            "in the sources and were likely misread: " + "; ".join(unver[:12]) + ".\n"
            "Rewrite the draft so EVERY number matches the sources exactly: replace each listed "
            "figure with the value actually stated in the sources, or REMOVE THE ENTIRE SENTENCE "
            "if the sources give no figure. Never write a placeholder such as '[FIGURE NOT IN "
            "SOURCES]' and never leave an empty '(Label:)' — omit the whole sentence instead. "
            "Change nothing else — keep all wording, structure, and citations.\n"
            "OUTPUT RULES: return ONLY the corrected briefing itself, starting at its first heading "
            "or bold line. Do NOT add any preamble, sign-posting, or notes explaining what you "
            "changed (no 'Based on the sources…', no 'here is the rewritten draft', no '*Note:*').\n\n"
            "SOURCES:\n" + src + "\n\nDRAFT:\n" + text}],
            max_tokens=5000, temperature=0.0, num_ctx=16384, model=model)
        fixed = _strip_correction_scaffold((fixed or "").strip())
    except Exception as e:
        logger.warning("[DeepResearch] numeric grounding pass failed: %s", e)
        fixed = ""
    # Use the correction only if it kept the structure AND didn't gut the briefing —
    # else fall back to the original draft.
    out = fixed if _accept_correction(text, fixed) else text
    still = set(_unverified_numbers(out, corpus, corpus_nc))
    if still:
        out = _drop_sentences_with(out, still)
    out = _strip_placeholders(out)  # backstop: model wrote a placeholder, not a removal
    logger.info("[DeepResearch] numeric grounding: %d unverified figure(s) corrected/stripped",
                len(unver))
    return out, len(unver)


# Ubiquitous acronyms / caps tokens that are fine even when a specific source body
# didn't happen to contain them — flagging these would be noise, not fabrication.
_COMMON_ACRONYMS = frozenset({
    "US", "USA", "UK", "EU", "UN", "AI", "ML", "IT", "HR", "PR", "CEO", "CFO", "CTO",
    "COO", "GDP", "CPI", "PPI", "FBI", "CIA", "NSA", "FDA", "SEC", "DOJ", "DOD", "DHS",
    "FOMC", "NATO", "OPEC", "IPO", "ETF", "GPU", "CPU", "API", "OS", "PC", "TV", "EV",
    "NYC", "LA", "DC", "ID", "OK", "AM", "PM", "ET", "PT", "UTC", "GMT", "NAV", "ESG",
    "CVE", "SQL", "HTTP", "URL", "VPN", "LLM", "LLMS", "WHO", "IMF", "ECB", "RBI", "NIH",
    "Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY", "YOY", "CO2", "5G", "6G", "EU", "AWS",
    "GB", "TB", "MB", "KW", "MW", "GW", "RMB", "USD", "EUR", "GBP", "NASA", "EVS",
})
# A real acronym embedded in a normal sentence (LSD, GLUT5, PURL, IBIT) — distinctive
# and hard to paraphrase, so absence from the sources is a strong fabrication signal.
_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,5})\b")
# 2–4 consecutive Capitalized words: a distinctive named entity/phrase.
_PROPER_PHRASE_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+){1,3})\b")
# A single CamelCase / capitalized-compound coinage ('FortiBleed', 'ParyonUSD',
# 'OpenAI') — distinctive, hard to paraphrase, and the class the acronym + multi-word
# patterns BOTH miss. A coined compound absent from every source is a strong garble/
# fabrication signal (e.g. the 9B writing 'ParyonUSD' for the real 'pUSD').
_COMPOUND_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:[A-Z][a-z]+|[A-Z]{2,})+)\b")
# Ubiquitous CamelCase brands/terms — fine even if a given source body didn't happen
# to spell them out; flagging these would be noise, not fabrication.
_COMMON_COMPOUNDS = frozenset({
    "GitHub", "GitLab", "OpenAI", "DeepMind", "DeepSeek", "JavaScript", "TypeScript",
    "PostgreSQL", "MongoDB", "PayPal", "YouTube", "LinkedIn", "WhatsApp", "PowerShell",
    "TikTok", "SpaceX", "OnePlus", "PlayStation", "MacBook", "ChatGPT", "BlackRock",
    "JPMorgan", "MicroStrategy", "CoinDesk", "CoinGecko", "CoinGlass", "PeckShield",
    "ByteDance", "WeChat", "AirPods", "iCloud", "iPhone", "iPad", "macOS",
})


def _split_compound(tok: str) -> list[str]:
    """Split a CamelCase compound into its parts ('ParyonUSD' -> ['Paryon','USD'])."""
    return re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+", tok)


def _orphan_terms(text: str, corpus: str) -> list[str]:
    """Distinctive CHECKABLE terms in `text` that appear NOWHERE in the source corpus
    — the qualitative analogue of `_unverified_numbers`. Catches fabricated specifics
    that numeric grounding can't see (e.g. an invented 'LSD therapy' rationale grafted
    onto a real acquisition). Conservative by design: ACRONYMS, multi-word PROPER-NOUN
    phrases, and single CamelCase COMPOUND coinages (all high-distinctiveness, hard to
    paraphrase). Plain single names are skipped — a body-extraction miss on one name
    would falsely strip real content."""
    orphans = []
    for line in text.split("\n"):
        s = line.strip()
        # skip headings and bold-only lead labels (scaffolding, not claims)
        if not s or s.startswith(("#", ">")) or (s.startswith("**") and s.endswith("**")):
            continue
        # strip bold spans — the model's OWN synthesized sub-labels ('**Strategic
        # Threat Scenarios:**'), not source-grounded claims; scanning them only adds
        # false positives. Fabricated specifics live in the plain prose, still scanned.
        line = re.sub(r"\*\*[^*]+?\*\*", " ", line)
        for m in _ACRONYM_RE.finditer(line):
            tok = m.group(1)
            if tok in _COMMON_ACRONYMS or tok.rstrip("S") in _COMMON_ACRONYMS:
                continue
            low = tok.lower()
            if low not in corpus and low.rstrip("s") not in corpus:
                orphans.append(tok)
        for m in _PROPER_PHRASE_RE.finditer(line):
            phrase = m.group(1)
            if phrase.lower() in corpus:
                continue
            # If every content word (≥4 chars) already appears in the sources, this is
            # a reorder/paraphrase of a real entity, not a fabrication — leave it.
            words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", phrase)]
            if words and all(w in corpus for w in words):
                continue
            orphans.append(phrase)
        for m in _COMPOUND_RE.finditer(line):
            tok = m.group(1)
            if tok in _COMMON_COMPOUNDS:
                continue
            low = tok.lower()
            if low in corpus or low.rstrip("s") in corpus:
                continue
            # Reprieve: if every distinctive part (≥3 chars) is already in the sources,
            # this is a reformat of a real term ('Black Rock'->'BlackRock'), not a
            # coinage. Only a part that's truly absent (e.g. 'paryon') signals a garble.
            parts = [p.lower() for p in _split_compound(tok) if len(p) >= 3]
            if parts and all(p in corpus for p in parts):
                continue
            orphans.append(tok)
    return list(dict.fromkeys(orphans))


async def _ground_claims(text: str, bodies: list[str], *, model: str | None = None) -> tuple[str, int]:
    """Qualitative grounding: verify distinctive claim-terms against the source bodies,
    not just numbers. Same proven shape as `_ground_numbers` — deterministic detection
    of terms absent from every source, ONE constrained source-grounded correction, then
    a deterministic backstop that drops any sentence still carrying an orphan term.
    Targets the fabricated-detail-inside-a-real-story failure (verification pass
    2026-06-24: AbbVie's invented 'LSD depression therapy' rationale). Returns
    (text, n_orphans)."""
    bodies = [b for b in bodies if b]
    if not text or not bodies:
        return text, 0
    corpus = " ".join(bodies).lower()
    if len(corpus) < 500:   # too little source text to judge fairly
        return text, 0
    orphans = _orphan_terms(text, corpus)
    if not orphans:
        return text, 0
    src = "\n\n".join(b[:4000] for b in bodies)[:14000]
    try:
        fixed = await _invoke_bg([{"role": "user", "content":
            "SOURCE TEXTS and a DRAFT briefing are below. These specific terms in the draft appear "
            "NOWHERE in the sources and may be fabricated or grafted from an unrelated story: "
            + "; ".join(orphans[:14]) + ".\n"
            "For EACH listed term: delete the specific claim it is part of UNLESS the sources clearly "
            "support that claim (possibly under different wording), in which case keep it. Change "
            "nothing else — keep all supported wording, structure, and citations exactly.\n"
            "OUTPUT RULES: return ONLY the corrected briefing itself, starting at its first heading or "
            "bold line. No preamble, no sign-posting, no notes explaining changes (no 'Based on the "
            "sources…', no 'here is the rewritten draft', no '*Note:*').\n\n"
            "SOURCES:\n" + src + "\n\nDRAFT:\n" + text}],
            max_tokens=5000, temperature=0.0, num_ctx=16384, model=model)
        fixed = _strip_correction_scaffold((fixed or "").strip())
    except Exception as e:
        logger.warning("[DeepResearch] claim grounding pass failed: %s", e)
        fixed = ""
    out = fixed if _accept_correction(text, fixed) else text
    # Backstop: drop any sentence that STILL carries an orphan term after correction.
    still = set(_orphan_terms(out, corpus))
    if still:
        out = _drop_sentences_with(out, still)
    logger.info("[DeepResearch] claim grounding: %d orphan term(s) corrected/stripped (%s)",
                len(orphans), ", ".join(orphans[:8]))
    return out, len(orphans)


def _reg_domain(host: str) -> str:
    """Registrable-ish domain (last two labels) so a citation token 'indiatimes.com'
    matches a read host 'economictimes.indiatimes.com'."""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


# Abbreviation-safe (2026-08-14): the plain (?<=[.!?]) split broke "Staff
# Sgt. Benjamin Pennington" at the title period — the gate then processed the
# halves separately and a live digest led with a decapitated name fragment.
# Common title/rank/entity abbreviations must not end a sentence.
_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])"
    r"(?<!\bSgt\.)(?<!\bGen\.)(?<!\bLt\.)(?<!\bCol\.)(?<!\bCapt\.)(?<!\bMaj\.)"
    r"(?<!\bAdm\.)(?<!\bBrig\.)(?<!\bCmdr\.)(?<!\bSpc\.)(?<!\bCpl\.)(?<!\bPvt\.)"
    r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bProf\.)(?<!\bRev\.)"
    r"(?<!\bSen\.)(?<!\bRep\.)(?<!\bGov\.)(?<!\bPres\.)(?<!\bSec\.)(?<!\bAmb\.)"
    r"(?<!\bSt\.)(?<!\bMt\.)(?<!\bFt\.)(?<!\bvs\.)(?<!\bInc\.)(?<!\bCorp\.)"
    r"(?<!\bLtd\.)(?<!\bCo\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\bNo\.)(?<!\bU\.S\.)"
    r"(?<!\bU\.K\.)(?<!\bU\.N\.)(?<!\bE\.U\.)"
    r"\s+(?=[A-Z(\"'])"
)


def _cited_regs(sent: str) -> set:
    """Registrable domains cited inline in a sentence, e.g. '(thehackernews.com)'."""
    regs = set()
    for cm in _PAREN_CITE_RE.finditer(sent):
        for tok in _DOMAIN_TOKEN_RE.findall(cm.group(0)):
            regs.add(_reg_domain(tok.lower()))
    return regs


async def _check_contamination(text: str, articles: list, *, model: str | None = None) -> tuple[str, int]:
    """Cross-claim contamination guard: a distinctive term cited to a source that does
    NOT contain it, but which IS explained by a DIFFERENT read source — a detail
    grafted from another story onto this claim. This catches the FortiBleed
    'Moscow Time geofencing' distortion that whole-corpus grounding (`_ground_claims`)
    misses, because the words live elsewhere in the corpus so the reprieve spares them.

    Citation-anchored: checks each SINGLE-cited sentence's distinctive terms against
    its OWN cited source only. A term flagged against the cited source but NOT against
    the whole corpus = present elsewhere = grafted/mis-attributed (vs a fabrication,
    which `_ground_claims` already handled). Correction-only, no hard sentence drop —
    a graft carries real (mis-cited) info, so we trim the detail, never nuke the line.
    Returns (text, n_grafts)."""
    arts = [(t, u, b) for (t, u, b) in articles if b]
    if not text or not arts:
        return text, 0
    host_corpus: dict[str, list] = {}
    for t, u, b in arts:
        host_corpus.setdefault(_reg_domain(_host(u)), []).append(f"{t} {b}".lower())
    host_corpus = {r: " ".join(v) for r, v in host_corpus.items()}
    full_corpus = " ".join(host_corpus.values())
    if len(full_corpus) < 500 or len(host_corpus) < 2:   # need ≥2 sources to graft between
        return text, 0
    grafts = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith(("#", ">")) or (s.startswith("**") and s.endswith("**")):
            continue
        for sent in _SENT_SPLIT_RE.split(line):
            cited = _cited_regs(sent) & set(host_corpus)
            if len(cited) != 1:          # exactly one resolvable cited source = clean attribution
                continue
            cited_orphans = _orphan_terms(sent, host_corpus[next(iter(cited))])
            if not cited_orphans:
                continue
            full_orphans = set(_orphan_terms(sent, full_corpus))
            grafts.extend(t for t in cited_orphans if t not in full_orphans)
    grafts = list(dict.fromkeys(grafts))
    if not grafts:
        return text, 0
    src = "\n\n".join(b[:4000] for _, _, b in arts)[:14000]
    try:
        fixed = await _invoke_bg([{"role": "user", "content":
            "A DRAFT briefing is below, with its SOURCE TEXTS. Each of these specific details is cited to "
            "a source that does NOT contain it — each was grafted from a different story and must be "
            "removed: " + "; ".join(grafts[:14]) + ".\n"
            "Rewrite the briefing DELETING each listed detail: drop the phrase and any clause that exists "
            "only to state it, while keeping the surrounding sentence grammatical and every other claim, "
            "number, and citation intact. Example: 'It harvested tokens using a filter set to Moscow Time "
            "to evade detection (x.com)' becomes 'It harvested tokens to evade detection (x.com)'. Remove "
            "nothing that is not on the list.\n"
            "OUTPUT RULES: return ONLY the corrected briefing itself, starting at its first heading or bold "
            "line. No preamble, no sign-posting, no notes explaining changes.\n\n"
            "SOURCES:\n" + src + "\n\nDRAFT:\n" + text}],
            max_tokens=5000, temperature=0.0, num_ctx=16384, model=model)
        fixed = _strip_correction_scaffold((fixed or "").strip())
    except Exception as e:
        logger.warning("[DeepResearch] contamination guard failed: %s", e)
        fixed = ""
    out = fixed if _accept_correction(text, fixed) else text
    # Deterministic backstop: the 9B is unreliable at surgical clause removal, so
    # guarantee any graft phrase it left behind is excised (drop the phrase + a
    # leading comma/space, then tidy stray whitespace/punctuation). The detail is
    # what matters; slightly terser grammar beats a surviving fabrication.
    survivors = [g for g in grafts if g.lower() in out.lower()]
    for g in survivors:
        out = re.sub(r"\s*,?\s+" + re.escape(g) + r"(?=\b)", " ", out, flags=re.IGNORECASE)
    if survivors:
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"\s+([.,;)])", r"\1", out)
    logger.info("[DeepResearch] contamination guard: %d grafted detail(s) removed (%s)",
                len(grafts), ", ".join(grafts[:8]))
    return out, len(grafts)


def _numeric_grafts(text: str, articles: list) -> list[str]:
    """Distinctive figures in a SINGLE-cited sentence that are absent from the CITED
    source's body but present ELSEWHERE in the corpus — a real number mis-attributed
    to the wrong source (the '$107.8B' that should be '$104.3B', where 107.8 is a
    different/earlier value living in another body). Whole-corpus grounding can't see
    it because the digit string exists somewhere. Pure detection (numeric analogue of
    `_orphan_terms` used by `_check_contamination`). Conservative: only DISTINCTIVE
    figures (decimal-scale, comma-grouped, or ≥10000) — round numbers collide."""
    arts = [(t, u, b) for (t, u, b) in articles if b]
    if not text or len(arts) < 2:
        return []
    host_corpus: dict[str, str] = {}
    for t, u, b in arts:
        h = _reg_domain(_host(u))
        host_corpus[h] = host_corpus.get(h, "") + " " + f"{t} {b}".lower()
    host_nc = {h: c.replace(",", "") for h, c in host_corpus.items()}
    full = " ".join(host_corpus.values())
    full_nc = full.replace(",", "")
    suspects: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith(("#", ">")) or (s.startswith("**") and s.endswith("**")):
            continue
        for sent in _SENT_SPLIT_RE.split(line):
            cited = _cited_regs(sent) & set(host_corpus)
            if len(cited) != 1:                 # exactly one resolvable cited source
                continue
            ch = next(iter(cited))
            for m in _MAGNITUDE_RE.finditer(sent):
                num, unit, cur = m.group("num"), m.group("unit"), m.group("cur")
                try:
                    val = float(num.replace(",", ""))
                except ValueError:
                    continue
                u = (unit or "").lower()
                distinctive = ("," in num) or (u in _MULT and "." in num) or val >= 10000
                if not distinctive:
                    continue
                variants = _num_variants(num, unit, cur)
                in_cited = any(v in host_corpus[ch] or v in host_nc[ch] for v in variants)
                in_full = any(v in full or v in full_nc for v in variants)
                if (not in_cited) and in_full:   # present elsewhere, not in cited src
                    suspects.append(m.group(0).strip())
    return list(dict.fromkeys(suspects))


async def _check_numeric_attribution(text: str, articles: list, *, model: str | None = None) -> tuple[str, int]:
    """Cited-source-anchored numeric correction. Detects figures mis-attributed to a
    source that doesn't state them (`_numeric_grafts`), then ONE constrained pass that
    replaces each with the value the cited source actually gives — or drops that
    specific claim. Correction-only with fallback-to-original (the figure names a real
    quantity, so we fix it, never nuke the line). Returns (text, n_misattributed)."""
    arts = [(t, u, b) for (t, u, b) in articles if b]
    if not text or len(arts) < 2:
        return text, 0
    if len(" ".join(b for _, _, b in arts)) < 500:   # too little source text to judge
        return text, 0
    grafts = _numeric_grafts(text, arts)
    if not grafts:
        return text, 0
    src = "\n\n".join(b[:4000] for _, _, b in arts)[:14000]
    try:
        fixed = await _invoke_bg([{"role": "user", "content":
            "SOURCE TEXTS and a DRAFT briefing are below. Each of these figures is cited to a source "
            "that does NOT state it — the number was taken from a different story and mis-attributed: "
            + "; ".join(grafts[:12]) + ".\n"
            "For EACH figure: replace it with the value the CITED source actually states for that exact "
            "quantity, or delete that specific claim if the cited source gives no such figure. Change "
            "nothing else — keep every correct number, wording, structure, and citation intact.\n"
            "OUTPUT RULES: return ONLY the corrected briefing, starting at its first heading or bold "
            "line. No preamble, no sign-posting, no notes.\n\n"
            "SOURCES:\n" + src + "\n\nDRAFT:\n" + text}],
            max_tokens=5000, temperature=0.0, num_ctx=16384, model=model)
        fixed = _strip_correction_scaffold((fixed or "").strip())
    except Exception as e:
        logger.warning("[DeepResearch] numeric attribution pass failed: %s", e)
        fixed = ""
    out = fixed if _accept_correction(text, fixed) else text
    logger.info("[DeepResearch] numeric attribution: %d mis-attributed figure(s) (%s)",
                len(grafts), ", ".join(grafts[:6]))
    return out, len(grafts)


# ---------------------------------------------------------------------------
# Source INDEPENDENCE (2026-08-12) — the anti-laundering layer.
#
# Every corroboration count in this module used to equate "distinct hosts" with
# "independent sources". A mirror/syndication network (N domains republishing
# the same text) therefore manufactured corroboration: it passed the ≥2-source
# lead gate, inflated the evidence-pack corroboration tags the synthesizer
# calibrates on, and ✓-confirmed figures so Lever A skipped fresh-verifying
# them. This was the #1-ranked residual risk of the 2026-07-09 full-system
# exploration ("credibility-not-provenance").
#
# Fix: cluster articles by near-duplicate BODY text (5-word shingles, Jaccard)
# — union-find, deterministic, no LLM — and count CLUSTERS wherever
# independence matters. Two outlets that both merely reprinted the same wire
# copy are ONE source; an outlet that also wrote its own analysis is a second.
#
# Honest limit: text similarity cannot see PARAPHRASE laundering (an LLM-farm
# rewriting the same claim). For figures that gap is narrowed by the authority
# floor in _corroborate_numbers (two junk-tier clusters never confirm); full
# paraphrase-network detection needs temporal host-pair co-occurrence history
# (designed, not yet built — see memory knowing-tier-2026-08-12).
# ---------------------------------------------------------------------------

_SHINGLE_WORDS = 5
_MIRROR_JACCARD = 0.55


def _shingles(body: str) -> set[int]:
    """Hashed 5-word shingles of a normalized body head (bounded for speed)."""
    words = re.findall(r"[a-z0-9]+", (body or "").lower()[:4000])
    return {hash(" ".join(words[i:i + _SHINGLE_WORDS]))
            for i in range(max(0, len(words) - _SHINGLE_WORDS + 1))}


def _independence_clusters(articles: list) -> tuple[list[int], dict[str, int]]:
    """(article_cluster_ids, host→cluster_id): near-duplicate bodies share a
    cluster. Article-level ids drive figure corroboration (an outlet's ORIGINAL
    analysis stays independent of the wire copy it also ran); the host map is
    the coarse view for lead-gate/evidence-tag counting — hosts joined by ANY
    mirrored pair collapse, which is correct there: if two outlets' only overlap
    is the same reprint, the lead is NOT independently corroborated."""
    n = len(articles)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    shingle_sets = [_shingles(b) for _t, _u, b in articles]
    for i in range(n):
        if not shingle_sets[i]:
            continue
        for j in range(i + 1, n):
            if not shingle_sets[j]:
                continue
            inter = len(shingle_sets[i] & shingle_sets[j])
            if not inter:
                continue
            union = len(shingle_sets[i] | shingle_sets[j])
            if inter / union >= _MIRROR_JACCARD:
                parent[find(i)] = find(j)

    art_cluster = [find(i) for i in range(n)]
    host_cluster: dict[str, int] = {}
    # Hosts sharing any mirrored article pair merge (transitively via cluster id).
    cluster_alias: dict[int, int] = {}
    for i, (_t, url, _b) in enumerate(articles):
        h = _host(url)
        c = cluster_alias.setdefault(art_cluster[i], art_cluster[i])
        if h in host_cluster and host_cluster[h] != c:
            # Same host in two clusters — alias the clusters together.
            old = host_cluster[h]
            for k, v in list(cluster_alias.items()):
                if v == old:
                    cluster_alias[k] = c
            for k, v in list(host_cluster.items()):
                if v == old:
                    host_cluster[k] = c
        host_cluster[h] = c
    n_mirrored = n - len(set(art_cluster))
    if n_mirrored:
        logger.info("[Independence] %d of %d articles are mirrors/syndication (%d independent clusters)",
                    n_mirrored, n, len(set(art_cluster)))
    return art_cluster, host_cluster


# --- Temporal co-occurrence (paraphrase-network detection, 2026-08-12) -----
# Text similarity catches mirrors; it cannot catch an LLM-farm REWRITING the
# same content across its domains. Those networks have a temporal fingerprint:
# junk-tier hosts that appear together in nearly every digest either one
# appears in. Collection is always-on and cheap (one upsert batch per digest);
# detection self-arms once counts accumulate and feeds the SAME host-cluster
# map the mirror detector uses. Reputable hosts are never merged — major
# outlets legitimately co-occur constantly.

_NETWORK_MIN_COOCCUR = 8       # pair must co-appear in >= this many digests
_NETWORK_MIN_RATIO = 0.8       # ... and in >= 80% of the rarer host's digests


def _record_host_cooccurrence(db, hosts: list) -> None:
    """Upsert this digest's host set into the co-occurrence counts (bounded)."""
    hs = sorted({h for h in hosts if h and "." in h})[:30]
    for h in hs:
        db.execute(
            "INSERT INTO host_digest_counts (host, n_digests) VALUES (?, 1) "
            "ON CONFLICT(host) DO UPDATE SET n_digests = n_digests + 1, "
            "last_seen = CURRENT_TIMESTAMP", (h,))
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            db.execute(
                "INSERT INTO host_cooccurrence (host_a, host_b) VALUES (?, ?) "
                "ON CONFLICT(host_a, host_b) DO UPDATE SET n_cooccur = n_cooccur + 1, "
                "last_seen = CURRENT_TIMESTAMP", (hs[i], hs[j]))


def _network_pairs(db) -> set:
    """Flagged (host_a, host_b) pairs: junk-tier hosts whose co-occurrence
    ratio marks them as one syndication/farm network. Reputable hosts
    (hand tier >= 2.0 or dataset authority >= 0.8) are never flagged."""
    try:
        rows = db.fetchall(
            "SELECT c.host_a, c.host_b, c.n_cooccur, ha.n_digests na, hb.n_digests nb "
            "FROM host_cooccurrence c "
            "JOIN host_digest_counts ha ON ha.host = c.host_a "
            "JOIN host_digest_counts hb ON hb.host = c.host_b "
            "WHERE c.n_cooccur >= ?", (_NETWORK_MIN_COOCCUR,))
    except Exception:
        return set()
    out = set()
    for r in rows:
        rarer = min(r["na"], r["nb"])
        if rarer < _NETWORK_MIN_COOCCUR or r["n_cooccur"] / rarer < _NETWORK_MIN_RATIO:
            continue
        if any(_source_quality("http://" + h) >= 2.0 or _sa_authority(h) >= 0.8
               for h in (r["host_a"], r["host_b"])):
            continue
        out.add((r["host_a"], r["host_b"]))
    return out


def _apply_network_pairs(db, host_clusters: dict) -> dict:
    """Merge temporally-flagged network pairs into the mirror cluster map."""
    try:
        pairs = _network_pairs(db)
    except Exception:
        return host_clusters
    if not pairs:
        return host_clusters
    merged = 0
    for a, b in pairs:
        ca = host_clusters.setdefault(a, hash(a))
        cb = host_clusters.setdefault(b, hash(b))
        if ca != cb:
            for k, v in list(host_clusters.items()):
                if v == cb:
                    host_clusters[k] = ca
            merged += 1
    if merged:
        logger.info("[Independence] %d temporal network pair(s) merged into clusters", merged)
    return host_clusters


def _figure_support(text: str, articles: list) -> dict[str, tuple[int, set]]:
    """Deterministic cross-source support: for each magnitude figure in `text`,
    how many INDEPENDENT sources state that exact value (matched with comma/
    expanded/word variants). Pure value-agreement — no semantic judgment, so unlike
    the 9B's contradiction-guessing it can't be wrong about 'same quantity'.
    Independence (2026-08-12): support is counted in near-duplicate CLUSTERS,
    not hosts — N mirror domains stating a figure are ONE source."""
    art_cluster, _ = _independence_clusters(articles)
    corpora = []
    for i, (_t, url, body) in enumerate(articles):
        c = (body or "").lower()
        corpora.append((art_cluster[i], _host(url), c, c.replace(",", "")))
    out: dict[str, tuple[int, set]] = {}
    for m in _MAGNITUDE_RE.finditer(text):
        num, unit, cur = m.group("num"), m.group("unit"), m.group("cur")
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        u = (unit or "").lower()
        # DISTINCTIVE values only — collision-improbable, so a 2-source match really
        # means the same fact. Excludes percentages and round numbers (8%, $20B),
        # which collide across unrelated stories and would create false "corroboration".
        distinctive = ("," in num) or (u in _MULT and "." in num) or val >= 10000
        if not distinctive:
            continue
        variants = _num_variants(num, unit, cur)
        clusters, hosts = set(), set()
        for cid, h, c, cnc in corpora:
            if any(v in c or v in cnc for v in variants):
                clusters.add(cid)
                hosts.add(h)
        raw = m.group(0).strip()
        prev_n, prev_hosts = out.get(raw, (0, set()))
        out[raw] = (max(len(clusters), prev_n), hosts | prev_hosts)
    return out


# --- Currency-unit grounding: catch a NON-USD figure stamped with "$" -------------
# The deepest residual class (2026-06-29 audit): synthesis prints "$1.185 trillion" for
# Alibaba revenue the sources actually state in CNY ("1.185 trillion yuan") — the VALUE
# traces (so `_ground_numbers` passes) but the CURRENCY is wrong, an ~8× error. Purely
# DETERMINISTIC (no model — the LLM contradiction-judgment approach was proven backwards
# 2026-06-23): if a distinctive $-figure's value appears in the sources ONLY beside a
# non-USD currency marker (never USD), re-stamp it to the source currency.
_CUR_WORDS = [
    ("renminbi", "CNY"), ("yuan", "CNY"), ("rmb", "CNY"), ("cny", "CNY"),
    ("hk$", "HK$"), ("hkd", "HK$"), ("hong kong dollar", "HK$"),
    ("euros", "EUR"), ("euro", "EUR"), ("eur", "EUR"), ("€", "EUR"),
    ("pounds", "GBP"), ("pound", "GBP"), ("gbp", "GBP"), ("£", "GBP"),
    ("yen", "JPY"), ("jpy", "JPY"),
    ("rupees", "INR"), ("rupee", "INR"), ("inr", "INR"), ("₹", "INR"),
    ("won", "KRW"), ("krw", "KRW"), ("₩", "KRW"),
]
_CUR_PREFIX = {"CNY": "CNY ", "HK$": "HK$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹", "KRW": "₩"}


def _nearest_currency(window: str) -> str | None:
    """Currency implied by the text around a value. Explicit words/codes first
    (unambiguous), bare '¥' last (yuan/yen — default CNY since 'yen' is caught above),
    USD only on a $/dollar marker."""
    for marker, code in _CUR_WORDS:
        if marker in window:
            return code
    if "¥" in window:
        return "CNY"
    if "$" in window or "usd" in window or "dollar" in window:
        return "USD"
    return None


def _currency_mislabels(text: str, articles: list) -> list[tuple[str, str]]:
    """$-stamped DISTINCTIVE figures whose value appears in the sources ONLY in a
    non-USD currency (never USD) → list of (figure, currency_code)."""
    corpus = " ".join(f"{t} {b}" for t, _u, b in articles if b).lower()
    if len(corpus) < 400:
        return []
    out, seen = [], set()
    for m in _MAGNITUDE_RE.finditer(text):
        if m.group("cur") != "$":
            continue
        cs = m.start("cur")
        if cs > 0 and text[cs - 1].isalnum():   # already a qualified currency (HK$, US$, A$, C$) → skip
            continue
        num, unit = m.group("num"), m.group("unit")
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        u = (unit or "").lower()
        if not (("," in num) or (u in _MULT and "." in num) or val >= 10000):   # distinctive only
            continue
        fig = m.group(0).strip()
        if fig in seen:
            continue
        # bare value forms (no $), keep only the distinctive ones (with a unit word or
        # ≥7 digits) — a bare "1.185" is too collision-prone to window on.
        variants = [v for v in _num_variants(num, unit, None)
                    if " " in v or len(v.replace(",", "")) >= 7]
        nonusd: dict[str, int] = {}
        usd = 0
        for v in variants:
            i = corpus.find(v)
            while i >= 0:
                code = _nearest_currency(corpus[max(0, i - 20):i + len(v) + 20])
                if code == "USD":
                    usd += 1
                elif code:
                    nonusd[code] = nonusd.get(code, 0) + 1
                i = corpus.find(v, i + len(v))
        if nonusd and usd == 0:
            seen.add(fig)
            out.append((fig, max(nonusd, key=nonusd.get)))
    return out


def _correct_currency_mislabels(text: str, articles: list) -> str:
    """Re-stamp $-figures the sources state in a non-USD currency. Deterministic."""
    arts = [(t, u, b) for (t, u, b) in articles if b]
    if not text or not arts:
        return text
    mis = _currency_mislabels(text, arts)
    for fig, code in mis:
        if fig.startswith("$") and fig in text:
            text = text.replace(fig, _CUR_PREFIX.get(code, code + " ") + fig[1:])
    if mis:
        logger.info("[DeepResearch] currency mislabel: re-stamped %d $-figure(s) to source currency (%s)",
                    len(mis), ", ".join(f"{f}→{c}" for f, c in mis[:6]))
    return text


async def _corroborate_numbers(text: str, articles: list) -> tuple[str, set]:
    """Cross-source numeric corroboration — DETERMINISTIC. Counts how many independent
    sources state each distinctive figure's value (the LLM-judgment approach did this
    BACKWARDS — 2026-06-23 — so we never ask the model). Returns the SET of figures
    that ≥2 sources confirm, WITHOUT mutating the text.

    It USED to badge each corroborated figure with an inline ' ✓'. A 2026-06-29 output
    audit (4 independent reads) found that badge (a) read as scaffold cruft in the
    shipped prose, and worse (b) rubber-stamped MISFRAMED figures that two sources
    happened to share (e.g. a CNY value both outlets printed with a '$') — cross-source
    AGREEMENT is not correctness, so the ✓ laundered false confidence. The badge is
    gone; the set is still returned so Lever A can skip fresh-checking figures that are
    already corroborated (drift hides in SINGLE-source numbers)."""
    if not text or len(articles) < 2:
        return text, set()
    support = _figure_support(text, articles)
    # Independence + authority floor (2026-08-12): ≥2 INDEPENDENT clusters must
    # agree (mirrors count once), and agreement among only junk-tier hosts never
    # confirms (a paraphrase-farm pair evades text-dedup but not this) — unless
    # three independent clusters agree. Unconfirmed figures fail TOWARD safety:
    # Lever A fresh-verifies exactly the figures not in this set.
    confirmed = {
        f for f, (c, hosts) in support.items()
        if c >= 2 and (c >= 3 or max((_sa_authority(h) for h in hosts), default=0.0) >= 0.5)
    }
    if confirmed:
        logger.info("[DeepResearch] numeric corroboration: %d figure(s) confirmed by ≥2 independent sources", len(confirmed))
    return text, confirmed


_COMMON_ANALYSIS_CAP = 30000   # chars of deep analyses in the shared block
_COMMON_EVIDENCE_CAP = 9000    # chars of findings in the shared block
# ANALYSIS_CAP 18000→30000 (2026-08-11): 7 stories × ~4.3k chars of self-bounded
# 27B analysis ≈ 30k — at 18k, stories 4-7's analyses were silently dropped from
# every synthesis. Worst case COMMON ≈ 39k chars ≈ ~10k tokens; the chain stages'
# num_ctx were re-sized for it (candidates/merge 24576, enrich/verify 20480,
# pick 16384) — the prefix cache (#34) makes the bigger shared pack near-free
# after the first stage. If `ollama ps` ever shows CPU spill on the 27B, dial
# the 24576 stages down before touching the caps.


# Knowing tier (2026-08-12): the synthesis is asked to count today's developments
# against the PRIOR UNDERSTANDING dossier; the line is logged (instrumentation of
# knowing) and stripped before the digest is stored/posted.
_KNOWN_VS_NEW_RE = re.compile(r"(?im)^\s*KNOWN-VS-NEW:\s*(.+?)\s*$")


def _common_context(today: str, label: str, analysis_block: str, evidence: str) -> str:
    """The invariant evidence pack shared as the BYTE-IDENTICAL PREFIX of every
    synthesis-chain call (synthesis → judge → enrich → verify). llama.cpp reuses
    the KV cache of the longest common prefix across consecutive requests on a
    slot, so every stage after the first skips re-processing the (large) evidence
    — only the per-stage instructions + draft that come AFTER this block are new
    tokens. Byte-identity is the contract: same caps, same wording, built once.
    (Phase 1 #34; MAX_CONCURRENT_LLM_MONITORS=1 keeps the chain uninterleaved.)

    The domain dossier used to ride inside this prefix as PRIOR UNDERSTANDING.
    A paired A/B on 16 frozen topics (2026-09-03) measured that injection as a
    net negative — see _synthesize_from_evidence — so the pack is evidence
    only: nothing in it is admissible that today's sources do not support."""
    return (f"Today is {today}. {label} intelligence — evidence pack. The DEEP ANALYSES "
            "and SOURCE FINDINGS below are the ONLY admissible facts for every task that "
            "follows.\n\n"
            + analysis_block[:_COMMON_ANALYSIS_CAP]
            + "SOURCE FINDINGS (cite these inline):\n" + evidence[:_COMMON_EVIDENCE_CAP]
            + "\n\n")


async def _best_synthesis(prompt: str, evidence: str, *, n: int = 2,
                          temps: tuple = (0.2, 0.55), max_tokens: int = 5000,
                          model: str | None = None,
                          context_block: str | None = None) -> str:
    """Parallel best-of-N synthesis: generate N full candidate briefings (varied
    temperature → different framings), then an external grounded JUDGE picks the
    sharpest. This is the research-backed way to get more analytical depth from a
    small model — N independent shots + external selection, NOT self-refine (which
    degrades the 9B) and NOT decomposition (which proved lossy here). One judge call.
    `model` (Lever C) routes the candidate generation to a bigger synthesis model.
    `context_block` (prefix-cache #34): the caller's shared evidence prefix — the
    judge prompt starts with the same bytes so its KV cache is reused."""
    temps = list(temps)[:max(1, n)]
    # num_ctx 12288→24576 (2026-08-11): the synthesis prompt = common pack (~10k
    # tokens after the ANALYSIS_CAP raise) + evidence overflow + checklist +
    # instructions (~5-6k) + 5000 gen — 12288 was silently head-truncating the
    # evidence pack on big domains. 27B KV at 24576 extrapolates to ~19.4GB total
    # on the 3090 (measured 18.6GB at 16384) — verified post-deploy via ollama ps.
    raw = await asyncio.gather(*[
        _invoke_bg([{"role": "user", "content": prompt}],
                           max_tokens=max_tokens, temperature=t, num_ctx=24576, model=model)
        for t in temps], return_exceptions=True)
    cands = [(c or "").strip() for c in raw if isinstance(c, str) and (c or "").strip()]
    if len(cands) <= 1:
        return cands[0] if cands else ""
    judge_ctx = context_block if context_block else f"SOURCE FINDINGS:\n{evidence[:12000]}\n\n"

    # AGGREGATION over selection (2026-07-08, task #62): a same-model judge
    # PICKING one candidate plateaus near 55% selection accuracy (CMU, arXiv
    # 2602.18998 — even a frontier external verifier underperformed); treating
    # the N drafts as SOURCES and merging them measured +10.3% on deep-research
    # tasks (AggAgent, arXiv 2604.11753; Tongyi's Heavy Mode is the same shape:
    # parallel drafts → one synthesis pass, HLE 32.9→38.3). The merge sees the
    # same evidence prefix (KV-cache reuse) and the full drafts — the judge's
    # 3800-char preview cap would amputate what the merge must keep.
    listing = "\n\n".join(f"=== DRAFT {i + 1} ===\n{c[:6000]}" for i, c in enumerate(cands))
    try:
        merged = (await _invoke_bg([{"role": "user", "content":
            judge_ctx +
            f"Below are {len(cands)} DRAFT briefings independently written from the "
            "SOURCE FINDINGS above. Write the single FINAL briefing by aggregating "
            "them: cover every distinct story any draft covers; where drafts overlap, "
            "keep the version with the more specific numbers, names, and dates; keep "
            "inline (outlet.tld) citations exactly as written next to the claims they "
            "support; keep the sharpest analysis and 'so what'. Do NOT introduce any "
            "claim that is not in the SOURCE FINDINGS. Output ONLY the final briefing "
            "in the same format as the drafts.\n\n" + listing}],
            # 12288→24576 (2026-08-11): common (~10k tok) + 4×6k-char drafts (~6k
            # tok) + 5000 gen ≈ 21k — the FINAL-briefing writer was running with a
            # silently amputated prompt at 12288.
            max_tokens=max_tokens, temperature=0.15, num_ctx=24576, model=model) or "").strip()
        # A valid merge must be briefing-sized; a refusal/fragment falls back.
        if len(merged) >= 0.5 * max(len(c) for c in cands):
            return merged
        logger.warning("[DeepResearch] aggregation merge too short (%d chars) — falling back to judge-pick",
                       len(merged))
    except Exception as e:
        logger.warning("[DeepResearch] aggregation merge failed (%s) — falling back to judge-pick", e)

    # Fallback: legacy judge-pick.
    pick_listing = "\n\n".join(f"=== CANDIDATE {i + 1} ===\n{c[:3800]}" for i, c in enumerate(cands))
    try:
        pick = await _invoke_bg([{"role": "user", "content":
            judge_ctx +
            "Candidate briefings are below. Pick the SINGLE best candidate — the "
            "one that is (a) best grounded in the findings (nothing beyond them), (b) most specific "
            "(concrete numbers, names, dates), (c) sharpest on analysis and the 'so what', (d) covers "
            "the most distinct stories, (e) cleanest structure. Reply with ONLY the candidate number.\n\n"
            + pick_listing}],
            # 12288→16384 (2026-08-11): common + 4×3800-char previews overflowed
            # 12288 (the 5× "max_tokens (6)" truncation warns were this call
            # generating against a truncated prompt).
            max_tokens=6, temperature=0.0, num_ctx=16384, model=model)
        m = re.search(r"\d+", pick or "")
        idx = (int(m.group()) - 1) if m else 0
    except Exception:
        idx = 0
    return cands[idx] if 0 <= idx < len(cands) else cands[0]


async def _learn_facts(topic: str, brief: str, findings: list, kg, *,
                       articles: list | None = None, model: str | None = None) -> int:
    """Bank grounded facts from the verified briefing — kept only if they trace to
    what was actually read (anti-hallucination), garbage-gated.

    Grounding corpus fix (2026-07-06): the gate used to check only the FINDINGS
    text, but the synthesis elaborates from the FULL article bodies — so facts
    from abstract domains (geopolitics, AI/ML) failed the substring gate and
    banked 0 while concrete domains (crypto) passed. Ground against the bodies
    the synthesis actually read (`articles`), findings as fallback.
    `model`: route extraction to the synthesis model — the default-model call
    forced a 27B→9B swap mid-chain (VRAM can't hold both), costing a reload."""
    if kg is None or not brief or len(findings) < 1:
        return 0
    from app.core.kg import is_garbage_triple
    try:
        raw = await _invoke_bg(
            [{"role": "user", "content": _FACT_PROMPT.format(topic=topic, evidence=brief[:4500])}],
            json_mode=True, json_schema=_FACT_TRIPLES_SCHEMA, max_tokens=600, num_ctx=8192, model=model)
        cands = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(cands, dict):
            cands = cands.get("facts") or cands.get("triples") or []
    except Exception as e:
        # Loud, not silent (2026-08-18): this is the grounded-KG banking path —
        # the exact silent-zero class the Ollama-0.32 JSON regression caused. A
        # swallowed 0 here is indistinguishable from a legitimately fact-free
        # digest, so a broken extraction can hide for days. Log it.
        logger.warning("[DeepResearch] fact extraction/parse failed (topic=%r): %s", topic, e)
        return 0
    if not isinstance(cands, list):
        logger.warning("[DeepResearch] fact extraction returned non-list %s (topic=%r)",
                       type(cands).__name__, topic)
        return 0
    bodies_blob = " ".join((b or "").lower() for _, _, b in (articles or []))
    combined = (bodies_blob + " " + " ".join(f.lower() for _, _, f in findings)).strip()
    stored = gate_rejected = garbage = banking_errors = 0
    for c in cands[:10]:
        if not isinstance(c, dict):
            continue
        s, p, o = (str(c.get("subject", "")).strip(), str(c.get("predicate", "")).strip(),
                   str(c.get("object", "")).strip())
        if not (s and p and o) or is_garbage_triple(s, p, o):
            garbage += 1
            continue
        st, ot = _key_terms(s), _key_terms(o)
        if not st or not ot:
            gate_rejected += 1
            continue
        if not (any(t in combined for t in st) and any(t in combined for t in ot)):
            gate_rejected += 1
            continue
        # A fact is trustworthy if it is CORROBORATED (≥2 findings) OR sourced
        # from a CREDIBLE outlet (tier ≥2 — reuters/AP/BBC/etc). Keying trust on
        # corroboration COUNT alone false-positived on obviously-true single-
        # source facts ("Zelensky leads Ukraine" from a grounded current-events
        # digest) and quarantined them out of chat — regressing the #49 payoff
        # (found 2026-07-08). Source credibility is the right poisoning signal:
        # a single reputable outlet is not the AgentPoison attack surface; a
        # single unknown blog is. Only the latter stays quarantined-until-
        # corroborated (and now also auto-promotes by age — kg.promote_aged).
        support = 0
        credible = False
        for _t, _u, _f in findings:
            fl = (_f or "").lower()
            if any(t in fl for t in st) and any(t in fl for t in ot):
                support += 1
                if _source_quality(_u) >= 2.0:
                    credible = True
        try:
            if await kg.add_fact(s, p, o, confidence=min(0.9, 0.6 + 0.1 * max(1, support)),
                                 source="researched", provenance=f"deep_research:{topic[:60]}",
                                 trust=0.8 if (support >= 2 or credible) else 0.5):
                stored += 1
        except Exception as e:
            # This is the knowing tier's core write. Swallowed, the summary line
            # below still reports "N candidates -> 0 stored" and attributes the
            # loss to nothing at all — a banking outage would read exactly like
            # a fact-free digest (2026-09-04).
            banking_errors += 1
            logger.warning("[DeepResearch] fact banking failed for %r: %r", (s, p, o), e)
    logger.info("[DeepResearch] fact banking (%s): %d candidate(s) → %d stored, %d gate-rejected, %d garbage, %d failed",
                topic[:40], len(cands) if isinstance(cands, list) else 0, stored, gate_rejected, garbage, banking_errors)
    return stored


async def _ensure_citations(text: str, findings: list, *, model: str | None = None) -> tuple[str, int]:
    """Deterministic citation-coverage backstop. The synthesis + verify prompts ASK
    for an inline (outlet.tld) citation on every claim, but the 9B sometimes ignores
    that and ships an ENTIRELY uncited briefing (observed live: World Awareness,
    Finance — ~5% of synthesized digests). Citation enforcement can't be left to the
    model. When density is near-zero on a substantial digest, run ONE focused
    re-citation pass that adds the correct outlet to each sentence by matching it to
    the source findings (or drops a sentence matching none). Fallback-to-original, and
    used ONLY if it actually adds citations — so it can never make a good digest worse,
    and the deterministic gate means it's a no-op on the 95% that are already cited."""
    if not text or not findings:
        return text, 0
    body = "\n".join(l for l in text.split("\n") if not l.strip().startswith(("#", "_")))
    n_cites = len(_PAREN_CITE_RE.findall(body))
    if n_cites >= 2 or len(body.strip()) < 400:   # already cited, or too thin to bother
        return text, 0
    src = "\n\n".join(f"[{t}] ({_host(u)})\n{(b or '')[:900]}" for t, u, b in findings)[:12000]
    try:
        fixed = await _invoke_bg([{"role": "user", "content":
            "A DRAFT briefing below is MISSING its inline source citations. Using the SOURCE FINDINGS "
            "(each tagged with its outlet), add the correct '(outlet.tld)' citation at the end of EACH "
            "sentence and bullet, matched to the finding that supports it. If a sentence matches NO "
            "finding, DELETE it. Change nothing else — keep all supported wording, numbers, and "
            "structure; only ADD citations (or drop unsupported sentences).\n"
            "OUTPUT RULES: return ONLY the corrected briefing, starting at its first heading or bold "
            "line. No preamble, no notes.\n\n"
            "SOURCE FINDINGS:\n" + src + "\n\nDRAFT:\n" + text}],
            max_tokens=5000, temperature=0.0, num_ctx=16384, model=model)
        fixed = _strip_correction_scaffold((fixed or "").strip())
    except Exception as e:
        logger.warning("[DeepResearch] citation backstop failed: %s", e)
        return text, 0
    # NOTE: no length-floor guard here (unlike the correction passes) — re-citing a
    # near-uncited digest legitimately DROPS the unsupported sentences, so a large
    # shrink is expected. The cite-increase check below is the real guard.
    if fixed and ("**" in fixed or "##" in fixed):
        new_cites = len(_PAREN_CITE_RE.findall(fixed))
        if new_cites > n_cites:
            logger.info("[DeepResearch] citation backstop: re-cited an uncited digest (%d→%d)", n_cites, new_cites)
            return fixed, new_cites - n_cites
    return text, 0


# --- Deterministic per-sentence citation attribution ------------------------------
# Autopsy 2026-07-06 (full-length digests, post storage-cap fix): ~28% of factual
# sentences carry NO inline citation — multi-sentence bullets cite the fact-densest
# sentence while sibling sentences of the SAME story go uncited (bullet openers,
# elaborations, the lead's first sentence). The synthesis/verify prompts already
# demand per-sentence cites and demonstrably don't deliver them — so, like the rest
# of the grounding stack, enforce it in code: attribute each factual-uncited
# sentence to the ONE read source whose body actually contains its distinctive
# tokens, and append that host's citation. No match / ambiguous match → leave the
# sentence untouched (never guess an attribution).

_ATTR_STOPWORDS = frozenset({
    "the", "this", "that", "these", "those", "while", "after", "before", "following",
    "meanwhile", "notably", "however", "although", "despite", "beyond", "across",
    "western", "eastern", "northern", "southern", "global", "new", "for", "with",
})


def _attribution_tokens(sent: str) -> set[str]:
    """Distinctive, source-checkable tokens of a sentence: magnitude figures and
    proper-noun phrases (ubiquitous acronyms + capitalized connectives excluded).
    These are what must appear in a source body to credit it as the citation."""
    toks: set[str] = set()
    for m in re.finditer(r"\$?\d[\d,.]*\s?(?:%|billion|million|trillion)?", sent):
        t = m.group(0).strip().rstrip(".,").lower()
        if any(ch.isdigit() for ch in t):
            toks.add(t)
    for m in re.finditer(r"\b[A-Z][a-zA-Z0-9&'’\-]{2,}(?:\s+[A-Z][a-zA-Z0-9&'’\-]{2,}){0,3}", sent):
        t = m.group(0)
        if t.upper() in _COMMON_ACRONYMS or t.lower() in _ATTR_STOPWORDS:
            continue
        toks.add(t.lower())
    return toks


def _tok_in(tok: str, blob: str, blob_nc: str) -> bool:
    return tok in blob or tok.replace(",", "") in blob_nc


_CLAUSE_SPLIT_RE = re.compile(
    r"[;—–]|:\s+|,\s+(?:and|but|while|which|whereas|though|with)\s+"
    # bare coordinators (no comma) are strong clause boundaries in the house
    # style's dense compounds ("softened to 2.5% annually BUT rose 0.2%…");
    # bare "and" stays out — it joins noun phrases ("food and energy").
    r"|\s+(?:but|while|whereas|although)\s+"
    # participial trailers carry the analytic gloss; the head is the fact
    # ("…increase, ACCOUNTING for roughly two-thirds of…").
    r"|,\s+(?=(?:accounting|remaining|marking|reflecting|signaling|suggesting|"
    r"indicating|leaving|driving|bringing|pushing|raising|underscoring|highlighting)\b)"
)
_COPULA_TAIL_RE = re.compile(
    r"\b(?:is|was|are|were|remains?|represents?|marks?|signals?|constitutes?|reveals?)\b\s+(?:that\s+)?(.{40,})",
    re.IGNORECASE,
)


_ANALYTICAL_RE = re.compile(
    r"(?i)\b(?:second-order implication|the (?:broader|deeper|real|second-order) "
    r"(?:implication|question|risk|significance|lesson)|the implication is|"
    r"this (?:consolidation|move|shift|strategy|finding|trend|divergence|tension|"
    r"approach|incident|development|rupture|split|decision|deal|ruling|breach|episode|"
    r"pattern|escalation|dynamic|outcome|fracture|pivot|reversal) "
    r"(?:mirrors|suggests|reflects|attempts|crystallizes|positions|forces|signals|"
    r"underscores|implies|marks|serves|reveals|highlights|threatens|raises|exposes|"
    r"confirms|creates|sets|tests|erodes|accelerates|cements|complicates)|"
    r"mirrors similar|crystallizes the|reflects? a broader|suggesting (?:a|an|sector)|"
    r"is moving from|marks a (?:shift|critical|turning)|represents a (?:broader|"
    r"fundamental|critical)|serves as a critical|potentially (?:preempting|shifting|"
    r"fragment)|is not merely|not merely a|stress test|reassessment|"
    r"raises? the (?:question|prospect|risk|stakes)|sets? the stage|points? to a|"
    r"what to watch|signals? that|underscores? (?:the|a|how)|highlights? (?:the|a|how)|"
    r"the (?:throughline|takeaway) )")


def _is_analytical(claim: str) -> bool:
    """Analysis-shaped sentence: implication/synthesis verbs and no hard
    numbers (years excluded). These are the digest's OWN reasoning — the
    48h [entail-drop] corpus showed ~half of all final drops were this shape.
    A citation on them is decoration no source can entail, but the sentence
    itself is the product: gate v4 DE-CITES them instead of deleting them.
    Number-bearing sentences never qualify — an unsupported figure must
    still drop."""
    # Figures are facts; years (2026) and alphanumeric names (G20, H100, 5G)
    # are not figures.
    scrub = re.sub(r"\b(?:19|20)\d\d\b", "", claim)
    scrub = re.sub(r"\b[A-Za-z]+\d+\w*\b|\b\d+[A-Za-z]+\b", "", scrub)
    if re.search(r"\d", scrub):
        return False
    return bool(_ANALYTICAL_RE.search(claim))


# How often the NARROW document alone entails a claim, per (monitor, call site).
#
# The cascade below is only worth running while this is high, and it varies far
# more by topic than any single sample suggested: a 60-pair offline sample sat at
# 70% supported, while two live digests on 2026-09-04 ran 83% and 87%
# UNSUPPORTED — topics whose cited hosts had no matching article, where both
# widths fail and a narrow pass is pure tax. So the rate is measured per call
# site and remembered, and a site known to fail skips the narrow pass entirely
# rather than paying a probe every cycle. In-process on purpose: it is a cost
# heuristic, not a fact, and a restart should re-measure rather than trust a
# stale number.
_NARROW_RATE: dict[str, float] = {}
_NARROW_SKIPS: dict[str, int] = {}
_NARROW_MIN_RATE = 0.30      # break-even is ~0.275 (2.42s narrow vs 8.79s full)
_NARROW_PROBE_N = 8
_NARROW_REPROBE_EVERY = 20   # a known-bad site gets re-measured this often


def _narrow_worth_it(key: str) -> bool:
    """Unknown sites are probed; known-bad ones are re-probed occasionally.

    The first version of this was a one-way latch: once a call site measured
    below the bar it never measured again for the life of the process. Restarts
    reset it, so it was never going to be loud — which is exactly the shape of
    thing that ends up silently off for weeks here. A monitor whose sources
    improve deserves to be found out, so every _NARROW_REPROBE_EVERY skips buys
    one probe: ~5% of the saving, in exchange for the decision staying current.
    """
    prior = _NARROW_RATE.get(key)
    if prior is None or prior >= _NARROW_MIN_RATE:
        _NARROW_SKIPS.pop(key, None)
        return True
    n = _NARROW_SKIPS.get(key, 0) + 1
    if n >= _NARROW_REPROBE_EVERY:
        _NARROW_SKIPS[key] = 0
        return True
    _NARROW_SKIPS[key] = n
    return False


def _record_narrow_rate(key: str, rate: float) -> None:
    prior = _NARROW_RATE.get(key)
    _NARROW_RATE[key] = rate if prior is None else (prior + rate) / 2.0


def _sub_claims(claim: str) -> list[str]:
    """Informative sub-claims of a synthesis sentence: its clauses plus the
    copula/appraisal tail ("the most transformative element IS <fact>" → the
    fact). Only fragments meaningfully shorter than the whole claim qualify —
    a sentence with no such decomposition yields [] and gets no rescue pass."""
    subs: list[str] = []
    for p in _CLAUSE_SPLIT_RE.split(claim):
        p = p.strip(" ,.*")
        if 30 <= len(p) <= len(claim) - 10 and len(re.findall(r"[a-z0-9]{4,}", p.lower())) >= 4:
            subs.append(p)
    m = _COPULA_TAIL_RE.search(claim)
    if m:
        tail = m.group(1).strip(" ,.*")
        if 40 <= len(tail) <= len(claim) - 10:
            subs.append(tail)
    seen: set[str] = set()
    out: list[str] = []
    for s in subs:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:4]


# Player/paywall/nav boilerplate that jina-fetched pages carry as their own
# lines. It poisons evidence windows (live 2026-08-12: nbcnews video-player
# blocks — "This video file cannot be played. (Error Code: 102630)" — WON the
# window scoring for CPI claims). Substring match, short-line guard, and \xa0
# normalization; prose lines are far longer than these fragments.
_CHROME_SNIPPETS = (
    "this video file cannot be played", "error code:", "create a free profile",
    "add us on google", "subscribe now", "sign up for our", "cookie settings",
    "cookies policy",
)
# generic words that are chrome only when they ARE the line ("Advertisement",
# "3 min read") — prose mentioning them is far longer
_CHROME_SHORT = ("advertisement", "min read")
_CHROME_TIME_RE = re.compile(r"(?:\d{1,2}:\d{2}\s*)+|1x")
# A navigation menu is a RUN of short unpunctuated lines. Four, not three:
# three short lines in a row happen in real prose (a pull quote, a byline
# block); six in a row are a section menu.
_NAV_LINE_MAX = 45
_NAV_RUN_MIN = 4
_SENT_END_CHARS = frozenset('.!?:;"\'”)')


def _nav_run_indices(lines: list[str]) -> set[int]:
    """Indices belonging to a run of at least _NAV_RUN_MIN nav-shaped lines.

    Structural, because a blocklist of site strings loses to the next site.
    Live 2026-09-04: a claim about rate expectations was scored against
    "Video / Big Business / So Expensive / View From Above / Small Business /
    Authorized Account" — CNBC's section menu. Six consecutive short unpunctuated
    lines beat the real article on IDF overlap and became the evidence window.

    One short line is a heading and stays; several in a row are a menu. Digits
    are the escape hatch: a run of short numeric lines is a table or a ticker
    list, which is content.
    """
    nav = [i for i, ln in enumerate(lines)
           if 0 < len(ln.strip()) < _NAV_LINE_MAX
           and ln.strip()[-1] not in _SENT_END_CHARS
           and not any(ch.isdigit() for ch in ln)]
    out: set[int] = set()
    run: list[int] = []
    for i in nav:
        if run and i == run[-1] + 1:
            run.append(i)
            continue
        if len(run) >= _NAV_RUN_MIN:
            out.update(run)
        run = [i]
    if len(run) >= _NAV_RUN_MIN:
        out.update(run)
    return out


def _scrub_chrome(body: str) -> str:
    lines = body.split("\n")
    nav = _nav_run_indices(lines)
    keep = []
    for i, ln in enumerate(lines):
        low = ln.strip().lower().replace("\xa0", " ")
        if not low:
            continue
        if i in nav:
            continue
        # A short ALL-CAPS line is a masthead or a control, never prose
        # ("FREE ACCOUNT", "SUBSCRIBE", "LOG IN"). Live 2026-09-04: "FREE
        # ACCOUNT" opened the evidence window for a market claim.
        bare = ln.strip()
        if len(bare) < 30 and bare.isupper() and any(c.isalpha() for c in bare):
            continue
        if len(low) < 160 and any(s in low for s in _CHROME_SNIPPETS):
            continue
        if len(low) < 30 and any(s in low for s in _CHROME_SHORT):
            continue
        if _CHROME_TIME_RE.fullmatch(low):
            continue
        keep.append(ln)
    return "\n".join(keep)


_CLAIM_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")
_CLAIM_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _claim_tokens(claim: str) -> set[str]:
    """Tokens that participate in IDF evidence selection (entail gate).

    The base pattern keeps alphanumeric runs of ≥4 chars, which silently
    discarded SHORT numbers — "surging 68.5% to $44 million" contributed
    no numeric anchor, so number-bearing claims selected their evidence
    window on prose words alone and missed it in long articles (the
    dominant share of the gate's ~51%/day cited-sentence drops, audit
    2026-08-24). Digit tokens of ≥2 chars now participate; IDF still
    zeroes the ubiquitous ones (years, day-of-month), so only the
    distinctive figures steer selection.
    """
    low = claim.lower()
    toks = set(_CLAIM_TOKEN_RE.findall(low))
    toks.update(t for t in _CLAIM_NUM_RE.findall(low) if len(t) >= 2)
    return toks


async def _entailment_gate(text: str, arts: list, *, max_checks: int = 48,
                           label: str = "") -> tuple[str, int]:
    """MiniCheck entailment gate (#48, gated by ENABLE_MINICHECK, fail-open).

    v3 semantics (2026-08-12, from live [entail-miss] forensics): a cited
    sentence passes when its cited source entails the WHOLE sentence, or at
    least one informative CLAUSE of it (anchored-in-source). The house
    synthesis style opens stories with analytic framing ("the most
    structurally transformative element is …") that whole-sentence strict
    entailment can never pass even when every fact inside is source-backed —
    the v2 gate deleted 20-23 of 24 GOOD sentences per live digest while the
    MiniCheck model itself probe-verified perfectly calibrated. A sentence
    NONE of whose sub-claims its source entails — the actual
    fabricated-attribution case — still re-cites or drops.

    v3 evidence (same forensics): per-ARTICLE, not per-host concatenation.
    The old per_host[:24000] cap amputated a busy host's later articles
    entirely (claims citing them had NO evidence at all), and raw claim-token
    window scoring kept landing on nav chrome, cookie banners, or the wrong
    article (dates/boilerplate tokens score everywhere). Articles are ranked
    per claim by rare-token (IDF-weighted) overlap and windows are scored the
    same way, so boilerplate carries ~zero weight.

    Bounded (max_checks primary, ≤64 clause pairs, ≤48 re-cite pairs) and
    fail-open: sidecar down → text unchanged."""
    from app.config import config as _cfg
    if not getattr(_cfg, "ENABLE_MINICHECK", False) or not text or not arts:
        return text, 0
    url = (getattr(_cfg, "MINICHECK_URL", "") or "").rstrip("/")
    if not url:
        return text, 0

    articles: list[tuple[str, str]] = []   # (host, "title body") — one entry PER ARTICLE
    for t, u, b in arts:
        if b:
            articles.append((_host(u), ((t or "") + " " + _scrub_chrome(b)).strip()[:24000]))
    if not articles:
        return text, 0
    art_lowers = [a[1].lower() for a in articles]
    host_set = {h for h, _ in articles}
    host_rds = {_reg_domain(h) for h in host_set}

    _df_cache: dict[str, int] = {}

    def _weights(claim: str) -> dict[str, float]:
        """IDF weight per claim token: a token found in most read articles
        (dates, site chrome, stock phrases) is worth ~nothing; a rare content
        token (a name, a technical term, an exact figure) dominates
        window/article selection."""
        wts: dict[str, float] = {}
        for tok in _claim_tokens(claim):
            if tok not in _df_cache:
                _df_cache[tok] = sum(1 for bl in art_lowers if tok in bl)
            wts[tok] = 1.0 / (1.0 + _df_cache[tok])
        return wts

    def _windows(wts: dict[str, float], idx: int, *, w: int = 1400, k: int = 2) -> str:
        """The k best claim-matching windows of ONE article — what the T5
        actually judges. Overlapping windows, IDF-weighted token scoring."""
        art, low = articles[idx][1], art_lowers[idx]
        if not wts or len(art) <= w * k:
            return art[:w * k]
        step = w // 2
        scored: list[tuple[float, int]] = []
        for start in range(0, len(art) - step, step):
            seg = low[start:start + w]
            scored.append((sum(wt for t, wt in wts.items() if t in seg), start))
        scored.sort(reverse=True)
        picks = sorted(s for _sc, s in scored[:k])
        return " … ".join(art[s:s + w] for s in picks)

    def _doc_for(hosts: list[str], claim: str, *, narrow: bool = False) -> str:
        """Evidence for a claim from the cited hosts: their best-matching
        ARTICLES (ranked by IDF overlap), windowed.

        `narrow` keeps only the single best article. Entailment cost scales
        steeply with document length — measured 2026-09-04 on 60 real pairs,
        5,508 chars took 8.79 s/pair and 2,754 took 2.42 — and entailment is
        64% of a digest's wall clock. See _check_cascade for why halving the
        document does not cost a verdict.
        """
        wts = _weights(claim)
        cand = [i for i, _sc in _ranked_cands(hosts, claim)]
        return " ".join(_windows(wts, i) for i in cand[:1 if narrow else 2])[:6000]

    def _ranked_cands(hosts: list[str], claim: str) -> list[tuple[int, float]]:
        """Every article the cited hosts supplied, best IDF match first.

        Split out of _doc_for so the miss forensics can say how many articles
        the host actually had and how well the ones we did NOT read matched.
        A gate that drops 83% of cited sentences is either reading the wrong
        two articles or catching a model that wrote past its sources, and the
        two look identical in the drop log without this.
        """
        wts = _weights(claim)
        rds = {_reg_domain(h) for h in hosts}
        scored = [(i, sum(wt for t, wt in wts.items() if t in art_lowers[i]))
                  for i, (h, _) in enumerate(articles)
                  if h in hosts or _reg_domain(h) in rds]
        scored.sort(key=lambda p: -p[1])
        return scored

    # collect (line_idx, sentence, cited_hosts) for cited factual sentences
    lines = text.split("\n")
    checks: list[tuple[int, str, list[str]]] = []
    for li, line in enumerate(lines):
        stripped = line.lstrip("*-• ").strip()
        if (not stripped or stripped.startswith(("#", "_"))
                or (stripped.startswith("**") and stripped.endswith("**"))):
            continue
        for seg in _SENT_SPLIT_RE.split(line):
            st = seg.strip()
            if len(st) <= 40 or len(checks) >= max_checks:
                continue
            cited = [h for h in _DOMAIN_TOKEN_RE.findall(" ".join(_PAREN_CITE_RE.findall(st)))
                     if h in host_set or _reg_domain(h) in host_rds]
            if cited:
                checks.append((li, st, cited))
    if not checks:
        return text, 0

    def _claim_of(st: str) -> str:
        return _PAREN_CITE_RE.sub("", st).strip()

    import httpx

    async def _check_pairs(check_pairs: list[dict], *, fail_open: bool) -> list[dict] | None:
        """Chunked posts (8 pairs/request): a partial verification beats an
        all-or-nothing timeout. fail_open=True marks failed chunks supported;
        fail_open=False returns None on failure (caller skips that rescue)."""
        results: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=240) as client:
                for i in range(0, len(check_pairs), 8):
                    try:
                        r = await client.post(f"{url}/check_batch", json={"pairs": check_pairs[i:i + 8]})
                        r.raise_for_status()
                        results.extend(r.json()["results"])
                    except Exception as e:
                        if not fail_open:
                            raise
                        logger.warning("[DeepResearch] entailment chunk %d failed (%r) — fail-open", i // 8, e)
                        results.extend({"supported": True, "prob": -1.0} for _ in check_pairs[i:i + 8])
        except Exception as e:
            logger.warning("[DeepResearch] entailment pass unavailable (%r)", e)
            return None
        return results

    async def _check_cascade(specs: list[tuple[list[str], str]], *,
                             fail_open: bool, site: str) -> list[dict] | None:
        """Score pairs against the NARROW document first, then re-check at full
        width only the ones it could not support.

        Measured on 60 real claim/document pairs (2026-09-04): the narrow
        document agrees with the full one on 93% of verdicts, and — this is what
        makes the cascade exact rather than a trade — NOTHING the narrow
        document supported was rejected at full width. So narrow-supported is a
        subset of full-supported, and re-checking only the narrow failures
        reproduces the full-width verdict set.

        Exact is not the same as cheap, and the first version of this said it
        was. A narrow pass costs 2.42 s a pair and a full one 8.79, so the
        cascade pays only while more than ~27% of pairs clear the narrow
        document. That sample sat at 70%; two live digests the same morning ran
        83% and 87% UNSUPPORTED, and there the cascade would have been ~10%
        SLOWER. So each call site measures its own rate (`_NARROW_RATE`) and a
        site known to fail goes straight to full width. The clause rescue and
        the alternate re-cite settle there quickly, which is right: both exist
        to re-examine claims that already failed once.
        """
        key = f"{label}:{site}"
        n = len(specs)

        def _full(idx):
            return [{"doc": _doc_for(specs[i][0], specs[i][1]), "claim": specs[i][1]}
                    for i in idx]

        def _narrow(idx):
            return [{"doc": _doc_for(specs[i][0], specs[i][1], narrow=True),
                     "claim": specs[i][1]} for i in idx]

        res: list[dict | None] = [None] * n
        rate = None
        if _narrow_worth_it(key):
            # Spread the probe rather than taking the head: a briefing's opening
            # sentences are its lead, and their support rate is not the rest's.
            probe_idx = (list(range(n)) if n <= _NARROW_PROBE_N else
                         sorted({round(i * n / _NARROW_PROBE_N)
                                 for i in range(_NARROW_PROBE_N)}))
            probe = await _check_pairs(_narrow(probe_idx), fail_open=fail_open)
            if probe is None:
                return None
            for i, r in zip(probe_idx, probe):
                res[i] = r
            rate = sum(1 for r in probe if r.get("supported")) / max(1, len(probe))
            _record_narrow_rate(key, rate)
            rest = [i for i in range(n) if res[i] is None]
            if rest and rate >= _NARROW_MIN_RATE:
                more = await _check_pairs(_narrow(rest), fail_open=fail_open)
                if more is None:
                    return None
                for i, r in zip(rest, more):
                    res[i] = r

        n_narrow = sum(1 for r in res if r is not None)
        todo = [i for i in range(n) if res[i] is None or not res[i].get("supported")]
        if todo:
            res_full = await _check_pairs(_full(todo), fail_open=fail_open)
            if res_full is None:
                if not fail_open or any(res[i] is None for i in todo):
                    return None
                return [r for r in res]          # narrow verdicts, all accepted
            for i, r2 in zip(todo, res_full):
                res[i] = r2
        logger.info("[entail-cascade] %s/%s: %d pair(s), %d scored narrow "
                    "(support %s), %d read at full width", label, site, n, n_narrow,
                    "unprobed" if rate is None else f"{rate:.0%}", len(todo))
        return [r for r in res]

    results = await _check_cascade([(hosts, _claim_of(st)) for _, st, hosts in checks],
                                   fail_open=True, site="gate")
    if results is None:
        return text, 0
    if all(r.get("prob") == -1.0 for r in results):
        logger.warning("[DeepResearch] entailment gate: every chunk failed — skipping")
        return text, 0

    unsupported = [(li, st, hosts) for (li, st, hosts), res in zip(checks, results)
                   if not res.get("supported")]
    # Show-your-work forensics (2026-08-12): the gate DELETES content — when it
    # fails a claim, the exact (claim, evidence-head, prob) must be inspectable.
    # These lines are what exposed the v2 root causes (chrome windows, amputated
    # articles, unentailable analytic lead-ins) after two blind fix attempts.
    for (li, st, hosts), res in list(zip(checks, results))[:24]:
        if not res.get("supported"):
            _c = _claim_of(st)
            _d = _doc_for(hosts, _c)
            _rk = _ranked_cands(hosts, _c)
            _read = [sc for _i, sc in _rk[:2]]
            _unread = [sc for _i, sc in _rk[2:]]
            logger.info("[entail-miss] p=%.3f hosts=%s arts=%d/%d read_worst=%.2f "
                        "unread_best=%.2f claim=%r doc_head=%r doc_len=%d",
                        res.get("prob", -1), hosts[:3], len(_read), len(_rk),
                        min(_read) if _read else 0.0, max(_unread) if _unread else 0.0,
                        _c[:110], _d[:90], len(_d))
    if not unsupported:
        logger.info("[DeepResearch] entailment gate: %d/%d cited sentences entailed by their source",
                    len(checks), len(checks))
        return text, 0

    # Anchored-in-source rescue: before re-citing/dropping, check the failed
    # sentence's informative sub-claims against the SAME cited evidence. Any
    # entailed sub-claim keeps the sentence and its citation — the source
    # demonstrably backs the sentence's factual core; the remainder is the
    # digest's own synthesis, which is the product, not a fabrication.
    _CLAUSE_TOTAL = 64
    clause_specs: list[tuple[list[str], str]] = []
    clause_meta: list[tuple[int, str]] = []
    for li, st, hosts in unsupported:
        if len(clause_specs) >= _CLAUSE_TOTAL:
            break
        claim = _claim_of(st)
        for sub in _sub_claims(claim):
            if len(clause_specs) >= _CLAUSE_TOTAL:
                break
            clause_specs.append((hosts, sub))
            clause_meta.append((li, st))
    anchored: set[tuple[int, str]] = set()
    if clause_specs:
        c_res = await _check_cascade(clause_specs, fail_open=False, site="clause") or []
        for key, res in zip(clause_meta, c_res):
            if res.get("supported"):
                anchored.add(key)

    still = [(li, st, hosts) for li, st, hosts in unsupported if (li, st) not in anchored]

    # second chance: does ANY other read source entail the claim? (re-cite, don't drop)
    # Bounded fan-out (2026-08-12): the v1 re-cite tried EVERY other read host
    # per failed claim — a 24-source overview with 12 failures queued ~276 CPU
    # entailment pairs (~30 min), stalling the whole digest chain. An IDF-ranked
    # prescreen keeps only the 4 most plausible alternates per claim, ≤48 total.
    _ALT_PER_CLAIM, _ALT_TOTAL = 4, 48
    alt_pairs: list[tuple[list[str], str]] = []
    alt_meta: list[tuple[int, str, str]] = []
    for li, st, hosts in still:
        if len(alt_pairs) >= _ALT_TOTAL:
            break
        claim = _claim_of(st)
        wts = _weights(claim)
        rds = {_reg_domain(h) for h in hosts}
        host_best: dict[str, tuple[int, float]] = {}   # host → (raw hits, idf score)
        for i, (h, _) in enumerate(articles):
            if h in hosts or _reg_domain(h) in rds:
                continue
            hits = sum(1 for t in wts if t in art_lowers[i])
            sc = sum(wt for t, wt in wts.items() if t in art_lowers[i])
            cur = host_best.get(h)
            if cur is None or sc > cur[1]:
                host_best[h] = (hits, sc)
        scored = sorted(((v[1], h) for h, v in host_best.items() if v[0] >= 2), reverse=True)
        for _sc, h in scored[:_ALT_PER_CLAIM]:
            if len(alt_pairs) >= _ALT_TOTAL:
                break
            alt_pairs.append(([h], claim))
            alt_meta.append((li, st, h))
    alt_results: list[dict] = []
    if alt_pairs:
        alt_results = await _check_cascade(alt_pairs, fail_open=False, site="alt") or []
    entailed_by: dict[tuple[int, str], str] = {}
    for (li, st, h), res in zip(alt_meta, alt_results):
        if res.get("supported") and (li, st) not in entailed_by:
            entailed_by[(li, st)] = h
    n_changed = 0
    decited = 0
    drop: list[tuple[int, str]] = []
    for li, st, hosts in still:
        if (li, st) in entailed_by:
            new_host = entailed_by[(li, st)]
            new_st = _PAREN_CITE_RE.sub("", st).rstrip()
            if new_st.endswith((".", "!", "?")):
                new_st = f"{new_st[:-1].rstrip()} ({new_host}){new_st[-1]}"
            else:
                new_st = f"{new_st} ({new_host})"
            lines[li] = lines[li].replace(st, new_st)
            n_changed += 1
        elif _is_analytical(_claim_of(st)):
            # gate v4 (2026-08-14): the digest's own reasoning keeps its
            # sentence but loses the citation no source could entail.
            new_st = re.sub(r"\s+([.!?])$", r"\1", _PAREN_CITE_RE.sub("", st).rstrip())
            lines[li] = lines[li].replace(st, new_st)
            decited += 1
            n_changed += 1
        else:
            drop.append((li, st))
    # v4.1 (2026-08-14): a bold-headline LEAD ("* **Headline:** …") is a SUMMARY
    # of its section — per-sentence entailment fails it for the same structural
    # reason as analysis sentences (it aggregates many sources). When the rest of
    # its line SURVIVED the gate, decapitating it leaves an incoherent body: keep
    # the lead, strip its citation. A lead whose whole section died still dies
    # with it. (regex, not lstrip: a charwise strip of "*-• " eats the bold
    # marker's own asterisks and never matches.)
    # TWO PHASES so a lead is judged against what ACTUALLY survives its section,
    # not its soon-to-die siblings: an earlier one-pass version evaluated the
    # lead against the pre-drop line, so a lead whose whole section was doomed
    # still saw ≥40 surviving chars (the doomed siblings) and wrongly survived.
    leads = [(li, st) for li, st in drop if re.match(r"\s*[*\-•]?\s*\*\*", st)]
    lead_set = {(li, st) for li, st in leads}
    # Phase 1: remove every NON-lead dropped sentence.
    for li, st in drop:
        if (li, st) in lead_set:
            continue
        lines[li] = lines[li].replace(st, "").rstrip()
        n_changed += 1
        # Post-rescue forensics: [entail-miss] shows what the PRIMARY check
        # rejected (rescue input); these lines show what actually DIED after
        # every rescue — the set the next gate iteration must be judged on.
        logger.info("[entail-drop] claim=%r", _claim_of(st)[:140])
    # Phase 2: keep a lead DE-CITED only if ≥40 chars of its own line survived.
    for li, st in leads:
        remaining = _PAREN_CITE_RE.sub("", lines[li].replace(st, "")).strip("*-• \t")
        if len(remaining) >= 40:
            new_st = re.sub(r"\s+([.!?])$", r"\1", _PAREN_CITE_RE.sub("", st).rstrip())
            lines[li] = lines[li].replace(st, new_st)
            decited += 1
            n_changed += 1
        else:
            lines[li] = lines[li].replace(st, "").rstrip()
            n_changed += 1
            logger.info("[entail-drop] claim=%r", _claim_of(st)[:140])
    # One greppable line per digest so drop-rate TRENDS are a metric, not
    # log archaeology (audit 2026-08-24: ~51%/day attrition went unnoticed
    # because only per-claim misses were logged).
    logger.info(
        "[entail-gate] %s: %d checked, %d unsupported → %d anchored (clause), "
        "%d re-cited, %d de-cited (analysis), %d dropped",
        label or "digest",
        len(checks), len(unsupported), len(anchored), len(entailed_by), decited, len(drop))
    out = "\n".join(l for l in lines)
    return out, n_changed


# Any short parenthetical that cites "analysis" instead of a source.
# WIDENED 2026-09-04 after the owner spotted it live. The v1 pattern required a
# COLON — it removed "(deep analysis: the Fed pivot)" but not "(deep analysis)".
# The same 2026-09-01 change that added this stripper also told the synthesis
# prompt never to write "(deep analysis: …)", naming the exact spelling. The
# model obeyed the letter and switched to the colon-less form, which sailed
# through: digests carrying the artifact went from ~10% before that deploy to
# 34% on 09-02, 45% on 09-03 and 55% on 09-04. A guard and a prompt written
# against each other, each looking correct alone.
# A parenthetical is a pseudo-citation when it mentions analysis and carries NO
# domain token — "(analysis by reuters.com)" is a real attribution and stays.
_PSEUDO_PAREN_RE = re.compile(r"\s*\(([^)]{0,80})\)")
_ANALYSIS_WORD_RE = re.compile(r"(?i)\banalys[ei]s\b")
_DOMAINISH_RE = re.compile(r"[a-z0-9-]+\.[a-z]{2,}")
# Kept for callers/tests that reference the original name.
_PSEUDO_CITE_RE = re.compile(r"\s*\((?:deep )?analysis:[^)]*\)")


def _strip_pseudo_citations(text: str) -> tuple[str, int]:
    """Remove '(deep analysis…)' markers: they carry no domain token, so
    _PAREN_CITE_RE never saw them and the sentences they decorated shipped
    unchecked (8 of 36 factual sentences in one Middle East digest, 2026-09-01).
    """
    if not text:
        return text, 0
    n = 0

    def _drop(m: "re.Match") -> str:
        nonlocal n
        inner = m.group(1)
        if _ANALYSIS_WORD_RE.search(inner) and not _DOMAINISH_RE.search(inner):
            n += 1
            return ""
        return m.group(0)

    return _PSEUDO_PAREN_RE.sub(_drop, text), n


def _decite_analysis(text: str) -> tuple[str, int]:
    """Strip outlet citations from the digest's OWN analysis sentences before
    the gate runs (2026-09-01): an analysis sentence with a citation is a
    fabricated attribution the entail gate can only drop — 52% of live drops
    were exactly this shape. The sentence is the product; the citation is not."""
    if not text:
        return text, 0
    n = 0
    out_lines = []
    for line in text.split("\n"):
        stripped = line.lstrip("*-• ").strip()
        if (not stripped or stripped.startswith(("#", "_"))
                or (stripped.startswith("**") and stripped.endswith("**"))):
            out_lines.append(line)
            continue
        segs = []
        for seg in _SENT_SPLIT_RE.split(line):
            st = seg.strip()
            if st and _PAREN_CITE_RE.search(st) and _is_analytical(_PAREN_CITE_RE.sub("", st)):
                cleaned = _PAREN_CITE_RE.sub("", seg)
                cleaned = re.sub(r"\s{2,}", " ", cleaned).replace(" .", ".").replace(" ,", ",")
                segs.append(cleaned)
                n += 1
            else:
                segs.append(seg)
        out_lines.append(" ".join(segs))
    return "\n".join(out_lines), n


_ARTIFACT_RES = (
    re.compile(r"(?i)\bthe (?:launched|announced|reported|is attempting|has announced|are attempting)\b"),
    re.compile(r"(?:^|\s)'s\s"),
    re.compile(r"(?i)\bthe (?:is|are|was|were) attempting\b"),
    re.compile(r"(?i)\b(?:a|an|the) the\b"),
    re.compile(r"\bof the \*\*\s*$"),
    re.compile(r"\bthe \*\*\s*$"),
)
_EMPTY_PAREN_RE = re.compile(r"\s*\((?:deep analysis:|analysis:)?\s*\)")


def _drop_artifact_sentences(text: str) -> tuple[str, int]:
    """Final canary (2026-09-01): a sentence the regex excision passes left
    mangled ('the launched a $10 million…', "solidifying 's position",
    'a debate the is attempting to mediate') is dropped whole rather than
    shipped; headlines lose the dangling tail; empty '(deep analysis:)'
    parentheses vanish. Returns (text, number of repairs)."""
    if not text:
        return text, 0
    n = 0
    text, k = _EMPTY_PAREN_RE.subn("", text)
    n += k
    out_lines = []
    for line in text.split("\n"):
        stripped = line.lstrip("*-• #").strip()
        is_head = stripped.startswith("**") or line.lstrip().startswith("#")
        if is_head:
            new = line
            for rx in _ARTIFACT_RES[-2:]:
                if rx.search(new):
                    new = rx.sub("**" if "**" in rx.pattern else "", new).rstrip()
                    n += 1
            out_lines.append(new)
            continue
        segs = []
        for seg in _SENT_SPLIT_RE.split(line):
            if seg.strip() and any(rx.search(seg) for rx in _ARTIFACT_RES[:4]):
                n += 1
                continue
            segs.append(seg)
        joined = " ".join(s_ for s_ in segs if s_ is not None).strip()
        if joined or not line.strip():
            out_lines.append(joined if line.strip() else line)
    if n:
        logger.info("[DeepResearch] artifact canary: %d mangled fragment(s) removed", n)
    return "\n".join(out_lines), n


def _cite_uncited_sentences(text: str, arts: list, *, max_added: int = 12) -> tuple[str, int]:
    """Append '(host)' to factual sentences that lack a citation, when EXACTLY ONE
    read source strongly contains the sentence's distinctive tokens (≥2 token hits,
    ≥60% of tokens, and a clear margin over the runner-up host). Deterministic —
    no model, no network; ambiguous or unmatched sentences are left alone."""
    per_host: dict[str, str] = {}
    for t, u, b in arts:
        if b:
            h = _host(u)
            per_host[h] = per_host.get(h, "") + " " + (t or "").lower() + " " + b.lower()
    if not text or not per_host:
        return text, 0
    host_nc = {h: blob.replace(",", "") for h, blob in per_host.items()}
    added = 0
    out_lines = []
    for line in text.split("\n"):
        stripped = line.lstrip("*-• ").strip()
        if (not stripped or stripped.startswith(("#", "_"))
                or (stripped.startswith("**") and stripped.endswith("**"))):
            out_lines.append(line)
            continue
        segs = _SENT_SPLIT_RE.split(line)
        new_segs = []
        for seg in segs:
            st = seg.strip()
            factual = (len(st) > 40 and (any(c.isdigit() for c in st)
                       or re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+", st))
                       and not _is_analytical(st))
            if added >= max_added or not factual or _PAREN_CITE_RE.search(st):
                new_segs.append(seg)
                continue
            toks = _attribution_tokens(st)
            if len(toks) < 2:
                new_segs.append(seg)
                continue
            scores = sorted(
                ((sum(1 for t in toks if _tok_in(t, blob, host_nc[h])) / len(toks), h)
                 for h, blob in per_host.items()), reverse=True)
            best, host = scores[0]
            second = scores[1][0] if len(scores) > 1 else 0.0
            hits = round(best * len(toks))
            if best >= 0.6 and hits >= 2 and (best - second >= 0.25 or best == 1.0):
                seg_r = seg.rstrip()
                tail = seg[len(seg_r):]
                if seg_r and seg_r[-1] in ".!?":
                    seg = f"{seg_r[:-1].rstrip()} ({host}){seg_r[-1]}{tail}"
                else:
                    seg = f"{seg_r} ({host}){tail}"
                added += 1
            new_segs.append(seg)
        out_lines.append(" ".join(new_segs))
    if added:
        logger.info("[DeepResearch] cite-uncited: attributed %d uncited sentence(s) to their source", added)
    return "\n".join(out_lines), added


# --- Lever A: bounded per-claim verification against a FRESH independent search ---
# The deep residual is a real fact mis-framed (wrong unit/scope/date) from a good
# source — invisible to corpus grounding. The only thing that catches it is a fresh,
# independent check of the specific claim, which is what manual verification does.
# Kept bounded (lead development's top numeric claims, one search each) so the GPU
# cost stays sane, and advisory-only (appends a caveat, never rewrites the body —
# the 9B is unreliable at prose rewriting). Gated by ENABLE_CLAIM_VERIFICATION.
# Matches the digest's LEAD section across the real header formats the synthesizer
# actually emits — "**Lead Development: <title>**", "### LEAD DEVELOPMENT: …",
# "**LEAD DEVELOPMENT …**" (case-insensitive). The OLD regex required the bold to
# close immediately after the word ("**Lead development**") which NO real digest
# does, so _lead_claims silently fell back to "first long block" and the fresh-
# check ran on the wrong text (full-system exploration 2026-07-09). Group 1 is the
# lead PROSE, ending right before the next section header (**Secondary… / ###…).
_LEAD_RE = re.compile(
    r"(?is)(?:\*{1,3}|#{1,4})\s*lead\s+development\b[^\n]*\n+"
    r"(.*?)(?=\n[ \t]*(?:\*\*[A-Z0-9]|#{1,4}\s)|\Z)")


def _lead_claims(text: str, *, corroborated=frozenset(), max_claims: int = 3) -> list[str]:
    """The lead development's most checkable claims — sentences carrying a DISTINCTIVE
    magnitude figure (highest-salience, most search-decisive). Bounded to the lead so
    fresh verification stays cheap. Pure/deterministic."""
    m = _LEAD_RE.search(text or "")
    lead = m.group(1).strip() if m else ""
    if not lead:
        for block in (text or "").split("\n\n"):
            b = block.strip()
            if len(b) > 200 and not b.startswith(("#", "_", "**Secondary", "**Connections")):
                lead = b
                break
    if not lead:
        return []
    claims: list[str] = []
    for sent in _SENT_SPLIT_RE.split(lead.replace("\n", " ")):
        sent = sent.strip()
        if len(sent) < 40:
            continue
        distinctive = False
        for mm in _MAGNITUDE_RE.finditer(sent):
            num, unit, cur = mm.group("num"), (mm.group("unit") or "").lower(), mm.group("cur")
            try:
                val = float(num.replace(",", ""))
            except ValueError:
                continue
            # Skip figures already corroborated by ≥2 sources — fresh-checking them is
            # wasted; drift hides in SINGLE-source figures (e.g. one stale widget).
            if mm.group(0).strip() in corroborated:
                continue
            if ("," in num) or (unit in _MULT and "." in num) or val >= 10000 or cur:
                distinctive = True
                break
        if not distinctive:
            continue
        c = re.sub(r"\([^()]*\.[a-z]{2,}[^()]*\)", "", sent)   # drop inline citations
        c = re.sub(r"[*_✓]", "", c).strip()                    # drop markdown/badges
        if c:
            claims.append(c)
        if len(claims) >= max_claims:
            break
    return claims


async def _verify_lead_claims(text: str, label: str, *, corroborated=frozenset(), max_claims: int = 3) -> tuple[str, int]:
    """Run a FRESH independent web search for each top lead claim and have the model
    judge — grounded ONLY in those fresh results — whether the claim's figure/framing
    holds. Contradicted claims get a concise '🔎 Fresh-check' caveat appended; the
    body is never rewritten. Returns (text, n_flagged)."""
    claims = _lead_claims(text, corroborated=corroborated, max_claims=max_claims)
    if not claims:
        return text, 0
    from app.tools import native_search
    flagged: list[tuple[str, str]] = []
    for claim in claims:
        try:
            results = await native_search.search(claim[:200], max_results=6, mode="news")
        except Exception as e:
            logger.debug("[DeepResearch] fresh-check search failed: %s", e)
            continue
        ev = []
        for r in (results or [])[:6]:
            t = (getattr(r, "title", "") or "").strip()
            sn = (getattr(r, "snippet", "") or "").strip()
            if t or sn:
                ev.append(f"- {t}: {sn[:240]}")
        if len(ev) < 2:
            continue   # too little fresh evidence to judge — do NOT guess
        try:
            raw = await _invoke_bg([{"role": "user", "content":
                "Judge a CLAIM strictly against FRESH web evidence — use ONLY the evidence, not prior "
                "knowledge.\n\n"
                f"CLAIM: {claim}\n\nFRESH EVIDENCE:\n" + "\n".join(ev) + "\n\n"
                "Does the fresh evidence CONTRADICT the claim's specific figure or framing (wrong value, "
                "wrong unit, wrong scope, or out of date)? Reply JSON only: "
                "{\"verdict\": \"supported\"|\"contradicted\"|\"unaddressed\", "
                "\"note\": \"<one short sentence giving the value the evidence states, ONLY if "
                "contradicted>\"}."}],
                json_mode=True, json_prefix="{", max_tokens=160, temperature=0.0,
                model=_syn_model())
            data = llm.extract_json_object(raw) if raw else {}
        except Exception as e:
            logger.debug("[DeepResearch] fresh-check judge failed: %s", e)
            continue
        if str(data.get("verdict", "")).lower() == "contradicted":
            flagged.append((claim, str(data.get("note", "")).strip()[:200]))
    if not flagged:
        return text, 0
    # Build the caveat block. ENFORCEMENT (full-system exploration 2026-07-09):
    # previously this was appended at the very BOTTOM as an "advisory" — so a
    # fabricated/unconfirmed figure LED the digest while its contradiction sat in
    # a footnote most readers never reach (the live AI/ML digest led with an
    # invented "GPT-5.6" price the fresh-check had flagged). Now the caveat is
    # spliced in DIRECTLY AFTER THE LEAD, with UNVERIFIED wording, so the flag is
    # impossible to miss at the point of the claim. Mid-text placement also
    # survives the trailing _bound_and_clean trim (a bottom footnote could be cut).
    cav = ["",
           f"⚠️ _Fresh-check: {len(flagged)} of {len(claims)} lead claim(s) could NOT be "
           f"confirmed by an independent search — treat the flagged figure(s) as UNVERIFIED:_"]
    for claim, note in flagged:
        snip = claim[:90] + ("…" if len(claim) > 90 else "")
        cav.append(f"  ⚠ \"{snip}\" — {note}" if note else
                   f"  ⚠ \"{snip}\" — not supported by current sources")
    caveat = "\n".join(cav)
    m = _LEAD_RE.search(text)
    if m:
        cut = m.end(1)
        out_text = text[:cut].rstrip() + "\n" + caveat + "\n" + text[cut:]
    else:
        out_text = text + "\n" + caveat   # no parseable lead → fall back to append
    logger.info("[DeepResearch] %s fresh-check: %d/%d lead claims flagged (inline)", label, len(flagged), len(claims))
    return out_text, len(flagged)


_LEAD_CITE_RE = re.compile(r"\(([a-z0-9][a-z0-9.\-]*\.[a-z]{2,})\)")


def _gate_lead_credibility(text: str, host_clusters: dict | None = None,
                           articles: list | None = None) -> tuple[str, bool]:
    """CORROBORATION-GATED LEAD: a headline that rests on a SINGLE lower-credibility
    source with no independent corroboration gets an inline sourcing caveat.

    This is the robust complement to the farm blocklist (full-system exploration
    2026-07-09): the blocklist can only stop farms we've named, but an UNKNOWN weak
    host (e.g. winzheng.com) can still anchor a lead. Rather than chase an endless
    denylist, we gate on EVIDENCE STRENGTH — the lead must be anchored by a credible
    outlet (tier ≥2) OR corroborated by ≥2 distinct sources. A lead standing on one
    unknown/weak source is flagged, not deleted (deterministic, low-harm). Returns
    (text, gated)."""
    m = _LEAD_RE.search(text or "")
    if not m:
        return text, False
    lead = m.group(1)
    hosts = {_host("http://" + h) for h in _LEAD_CITE_RE.findall(lead.lower())}
    hosts = {h for h in hosts if "." in h}
    if not hosts:
        return text, False                      # uncited lead — _ensure_citations' job
    # Independence (2026-08-12): mirror-network hosts collapse to one source
    # before the ≥2 check — N farm domains reprinting the same text can no
    # longer manufacture the corroboration that waves a lead through.
    hc = host_clusters or {}
    n_indep = len({hc.get(h, h) for h in hosts})
    anchors = [h for h in hosts if not _is_reference_host(h)]
    credible = bool(anchors) and max(_source_quality("http://" + h) for h in anchors) >= 2.0
    if credible or n_indep >= 2:
        return _freshness_note(text, m, hosts, articles), False   # credible anchor OR ≥2 INDEPENDENT sources
    caveat = ("\n⚠️ _Sourcing note: this lead rests on a single lower-credibility source "
              "and is not independently corroborated — treat as unconfirmed._")
    cut = m.end(1)
    return text[:cut].rstrip() + "\n" + caveat + "\n" + text[cut:], True


def _freshness_note(text: str, m, hosts: set[str], articles: list | None) -> str:
    """Append a freshness caveat when every cited source behind the lead has a
    KNOWN publish date older than 7 days (2026-09-01: September digests led
    with July events; no read article carried a date before this)."""
    if not articles or not hosts:
        return text
    newest = None
    known = 0
    rds = {_reg_domain(h) for h in hosts}
    for _t, u, _b in articles:
        hu = _host(u)
        if hu not in hosts and _reg_domain(hu) not in rds:
            continue
        raw = _PUB_DATES.get(u)
        if not raw:
            continue
        d = _pub_dt(raw)
        if d is None:
            continue
        known += 1
        if newest is None or d > newest:
            newest = d
    if newest is None:
        return text
    age_days = (_NOW() - newest).total_seconds() / 86400.0
    logger.info("[DeepResearch] lead freshness: newest cited source dated %s (%.1f d, %d dated)",
                newest.date().isoformat(), age_days, known)
    if age_days <= 7:
        return text
    note = ("\n⚠️ _Freshness note: the sources behind this lead are more than a week old "
            f"(newest {newest.date().isoformat()}) — this may be background, not today's development._")
    cut = m.end(1)
    return text[:cut].rstrip() + "\n" + note + "\n" + text[cut:]


def _count_content_sentences(text: str) -> int:
    """Content sentences in a briefing (headers/metadata lines excluded; same split as
    `_drop_sentences_with`, >40 chars so bold section labels don't count). The
    deterministic before/after unit for measuring what the verify pass deletes."""
    n = 0
    for line in (text or "").split("\n"):
        stripped = line.lstrip("*-• ").strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("_"):
            continue
        n += sum(1 for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", stripped)
                 if len(s.strip()) > 40)
    return n


def _verify_prompt(evidence: str, draft: str, *, overview: bool, rarr: bool,
                   common: str | None = None) -> str:
    """Build the verification-pass instruction.

    Legacy mode DELETES unsupported/uncited sentences — cheap but lossy: a claim
    whose core IS in the findings dies for one stray detail. RARR mode
    (ENABLE_RARR; Gao et al., ACL 2023) revises a partial-support claim down to
    its findings-supported core and deletes only claims with NO support. Safe to
    be lenient here: the deterministic grounding stack downstream re-checks every
    kept claim (figures, orphan terms, contamination, attribution).

    `common` (prefix-cache #34): when given, the prompt STARTS with the caller's
    shared evidence prefix (which already carries the findings) so the KV cache
    from the earlier chain stages is reused; instructions + draft follow it."""
    if rarr:
        core = (
            "1. For EVERY statement not fully supported by the findings: if the findings support "
            "its CORE claim, MINIMALLY EDIT the sentence so it asserts only what the findings "
            "state — fix or remove the unsupported number, name, or detail, keep the rest and its "
            "citation. DELETE a sentence ONLY if the findings contain nothing supporting its core "
            "claim.\n")
        cite_rule = (
            "2. Any sentence or bullet lacking an inline outlet citation like (cnbc.com): if the "
            "findings support it, append the correct outlet citation from the findings; if they "
            "don't, delete it.\n")
    else:
        core = (
            "1. Remove or correct EVERY statement not directly supported by the findings (invented "
            "numbers, names, events).\n")
        cite_rule = (
            "2. DELETE any sentence or bullet that lacks an inline outlet citation like (cnbc.com) — "
            "an uncited claim is where fabrication hides; cut it.\n")
    if overview:
        rules = (
            core + cite_rule +
            "3. Keep ALL supported, cited claims, their specifics, and the FULL structure — do NOT "
            "shorten, condense, merge, or drop any development; preserve every sourced detail.\n"
            "Output the corrected overview only.\n\n")
        if common:
            # prefix-cache layout: shared evidence pack first (carries the findings),
            # per-stage instructions + draft after.
            return (common +
                    "Below is a DRAFT domain overview. Verify it against the DEEP ANALYSES and "
                    "SOURCE FINDINGS above ('the findings'). Do all of:\n"
                    + rules + f"DRAFT:\n{draft}")
        return (
            "Below are SOURCE FINDINGS and a DRAFT domain overview. Do all of:\n"
            + rules + f"SOURCE FINDINGS:\n{evidence}\n\nDRAFT:\n{draft}")
    if rarr:
        return (
            "Below are SOURCE FINDINGS and a DRAFT briefing. For every statement not fully "
            "supported by the findings: if the findings support its CORE claim, MINIMALLY EDIT it "
            "to assert only what the findings state (fix or remove the unsupported number, name, "
            "or detail); DELETE it ONLY if the findings contain nothing supporting its core claim. "
            "Keep all supported claims and the structure. Output the corrected briefing only.\n\n"
            f"SOURCE FINDINGS:\n{evidence}\n\nDRAFT:\n{draft}")
    return (
        "Below are SOURCE FINDINGS and a DRAFT briefing. Remove or correct EVERY statement in the "
        "draft that is not directly supported by the findings (invented numbers, names, events). "
        "Keep only supported claims; preserve structure. Output the corrected briefing only.\n\n"
        f"SOURCE FINDINGS:\n{evidence}\n\nDRAFT:\n{draft}")


async def research_and_brief(label: str, topic: str | None = None, kg=None) -> str:
    """Full research cycle → a verified cross-source briefing + learned KG facts."""
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    subject = topic or await _focus_subject(label)
    logger.info("[DeepResearch] %s → subject: %s", label, subject)

    articles = await _gather_sources(subject, read_target=10)
    findings = await _findings(articles, subject) if articles else []
    if not findings:
        # Honest: never fabricate a briefing from nothing.
        return (f"## 🔬 {label} — researched briefing\n_subject: {subject} · "
                f"no credible readable sources today_\n\n"
                f"No readable credible sources were found on this story today; "
                f"nothing reported rather than speculation.")

    hosts = sorted({_host(u) for _, u, _ in findings})
    # Independence map (2026-08-12): near-duplicate bodies collapse for every
    # corroboration count in this chain — mirrors/syndication count once.
    _host_clusters = _independence_clusters(articles)[1] if articles else {}
    # Temporal layer: record this digest's host set (always-on, cheap) and merge
    # any flagged paraphrase-network pairs into the cluster map. Fail-open.
    try:
        from app.database import get_db as _gdb
        _db_co = _gdb()
        await asyncio.to_thread(_record_host_cooccurrence, _db_co, hosts)
        _host_clusters = await asyncio.to_thread(_apply_network_pairs, _db_co, _host_clusters)
    except Exception as e:
        logger.debug("[Independence] co-occurrence layer skipped: %s", e)
    evidence = _annotated_evidence(findings, host_clusters=_host_clusters)
    from app.config import config as _cfg
    syn_model = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None  # Lever C

    try:
        draft = await _invoke_bg([{"role": "user", "content":
            f"Today is {today}. You READ {len(findings)} credible sources on '{subject}'. Findings below.\n\n"
            "Write a sharp intelligence briefing: the situation, key facts/numbers, what's genuinely new, "
            "and what it means. Cite outlets inline. Use ONLY information present in the findings — never "
            "invent a company, number, person, or event not present.\n\n" + evidence}],
            max_tokens=700, temperature=0.25, model=syn_model, num_ctx=8192)
    except Exception as e:
        logger.warning("[DeepResearch] draft failed: %s", e)
        draft = ""
    draft = (draft or "").strip()
    if not draft:
        return f"## 🔬 {label} — researched briefing\n_subject: {subject}_\n\n(synthesis unavailable)"

    # VERIFICATION PASS — strip any claim the findings don't support (RARR mode:
    # revise partial-support claims to their supported core instead of deleting).
    from app.config import config as _cfg_v
    _rarr = bool(getattr(_cfg_v, "ENABLE_RARR", False))
    try:
        final = await _invoke_bg([{"role": "user", "content":
            _verify_prompt(evidence, draft, overview=False, rarr=_rarr)}],
            max_tokens=750, temperature=0.1, model=syn_model, num_ctx=8192)
        final = (final or "").strip() or draft
    except Exception:
        final = draft
    logger.info("[DeepResearch] verify pass (%s): %d→%d content sentences%s",
                label, _count_content_sentences(draft), _count_content_sentences(final),
                " [RARR]" if _rarr else "")

    final = _strip_prompt_leak(final)   # drop any synthesis-prompt instructions the model echoed
    final = _KNOWN_VS_NEW_RE.sub("", final).strip()   # belt-and-braces: knowing marker must never post
    final = _strip_fake_citations(final, hosts)   # drop fabricated-attribution lines
    final, _ = await _ensure_citations(final, findings, model=syn_model)   # backstop: re-cite an uncited misfire
    final, _ = await _ground_numbers(final, [b for _, _, b in articles], model=syn_model)  # every figure traces to a source
    final, _ = await _ground_claims(final, [f"{t}\n{b}" for t, _u, b in articles], model=syn_model)   # distinctive terms trace too (titles incl.)
    final, _ = await _check_contamination(final, articles, model=syn_model)   # details trace to their CITED source, not a grafted one
    final, _ = await _check_numeric_attribution(final, articles, model=syn_model)   # + figures trace to their CITED source's value
    final, _ = _strip_pseudo_citations(final)   # the digest citing its own analysis is not a source
    final, _ = _decite_analysis(final)   # the digest's own reasoning carries no citation (2026-09-01)
    final, _ = _cite_uncited_sentences(final, articles)   # attribute uncited sibling sentences to their source (deterministic)
    final, _ = await _entailment_gate(final, articles, label=label)   # MiniCheck: cited source must ENTAIL the claim (gated, fail-open)
    final = _correct_currency_mislabels(final, articles)   # re-stamp $-figures the sources state in CNY/HKD/EUR/...
    final, _corr = await _corroborate_numbers(final, articles)   # set of figures ≥2 sources confirm (for Lever A)
    final = _tidy_citations(final)   # strip $-wrapped citations + leaked title fragments
    final = _strip_novel_bottomline_figures(final)   # drop a figure introduced only in the bottom line
    final = _drop_orphan_headers(final)   # remove any section header left empty by the strip passes
    final = _repair_dangling_fragments(final)  # recover sentences broken by an excised entity (prep/possessive)
    final = _repair_broken_sentences(final)   # drop sentences left grammatically broken by the strip passes
    final, _ = _drop_artifact_sentences(final)   # never ship "the launched a …" (2026-09-01)
    learned = await _learn_facts(subject, final, findings, kg, articles=articles, model=syn_model)
    logger.info("[DeepResearch] %s: read %d sources (%s), learned %d facts",
                label, len(findings), ", ".join(hosts[:6]), learned)

    # Lever A: fresh independent verification of the lead's top claims (gated, bounded).
    try:
        from app.config import config as _cfg
        if getattr(_cfg, "ENABLE_CLAIM_VERIFICATION", True):
            final, _ = await _verify_lead_claims(final, label, corroborated=_corr)
    except Exception as e:
        logger.warning("[DeepResearch] claim verification failed: %s", e)

    # Corroboration-gated lead: caveat a headline resting on a single low-credibility
    # source (robust to unknown farms the blocklist misses). Deterministic, always on.
    final, _gated = _gate_lead_credibility(final, host_clusters=_host_clusters, articles=articles)
    if _gated:
        logger.info("[DeepResearch] %s: lead flagged — single low-credibility source", label)

    final = _bound_and_clean(final, max_chars=10800)
    header = (f"## 🔬 {label} — researched briefing\n"
              f"_subject: {subject}\nread {len(findings)} sources: {', '.join(hosts[:6])}"
              f"{' +more' if len(hosts) > 6 else ''} · {learned} facts learned · {today}_\n\n")
    return header + final


async def _deep_analyze(findings: list, label: str, today: str, *,
                        bodies: dict | None = None, model: str | None = None) -> str:
    """REAL deep research: cluster the findings into their distinct stories, then
    analyze EACH story in depth with its own call (concurrent), so the final
    synthesis reasons over ANALYZED material — not raw snippets. Crucially the
    synthesis still receives the full findings too, so this ADDS a layer of analysis
    without the lossy compression that sank the earlier decomposition attempt
    (2026-06-21): more calls to dig into everything, one call to synthesize.

    Depth comes from the SOURCE TEXT, not a pre-summary: when `bodies` (a url→full
    article-body map) is given, each story is analyzed from the FULL bodies of its
    top sources. The 240-token findings are a lossy compression of what was read —
    and the per-story analysis is exactly the step where the model must see the
    actual article text, or the upgraded synthesis model is reasoning over stubs.
    Findings still drive the cheap clustering pass and the final synthesis breadth.
    Per-story input is bounded (top sources by authority, each capped) to stay within
    the 8192 ctx. Returns a '### story\\n<analysis>' block (or '' to fall back)."""
    if len(findings) < 3:
        return ""
    bodies = bodies or {}
    numbered = "\n".join(f"[{i}] [{t}] ({_host(u)})\n{(f or '')[:600]}"
                         for i, (t, u, f) in enumerate(findings))
    # 1) one cheap pass to group findings into distinct stories (findings suffice)
    try:
        raw = await _invoke_bg([{"role": "user", "content":
            f"Group these {label} findings into the distinct ongoing STORIES they cover. Merge findings "
            "about the SAME event into one story; aim for 5-9 stories. Return JSON only: "
            '[{"title": "...", "items": [0, 3, 7]}].\n\n' + numbered}],
            json_mode=True, json_schema=_STORY_GROUPS_SCHEMA, max_tokens=600, temperature=0.1, model=model, num_ctx=8192)
        groups = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(groups, dict):
            groups = groups.get("stories") or groups.get("items") or []
    except Exception as e:
        logger.warning("[DeepResearch] deep-analyze story clustering failed (%s) — "
                       "degrading to single-story analysis for %r", e, label)
        groups = []
    stories = []
    for g in (groups or [])[:10]:
        if not isinstance(g, dict):
            continue
        idxs = [i for i in (g.get("items") or []) if isinstance(i, int) and 0 <= i < len(findings)]
        title = str(g.get("title", "")).strip()
        if title and idxs:
            stories.append((title, idxs))
    if not stories:   # clustering failed — analyze the whole pool as one story
        logger.info("[DeepResearch] deep-analyze fell back to single-story pool for %r "
                    "(%d findings, no clusters)", label, len(findings))
        stories = [(label, list(range(len(findings))))]

    # 2) analyze EACH story in depth, concurrently — the "enough calls" layer.
    # Feed the FULL article bodies of the story's most authoritative sources (capped
    # for the context budget); fall back to the finding text when a body is missing.
    async def _an(title, idxs):
        ranked = sorted(idxs, key=lambda i: _source_quality(findings[i][1]), reverse=True)
        parts = []
        for i in ranked[:8]:
            t, u, f = findings[i]
            txt = bodies.get(u) or f or ""
            parts.append(f"[{t}] ({_host(u)})\n{txt[:4500]}")
        ev = "\n\n".join(parts)[:28000]
        try:
            a = await _invoke_bg([{"role": "user", "content":
                f"Today is {today}. Analyze this {label} story IN DEPTH from its sources — be a sharp "
                "intelligence analyst, not a summarizer:\n"
                "- the key facts, EXACT numbers, named players, and what is genuinely NEW;\n"
                "- WHY it matters and the second-order implications;\n"
                "- any tension, disagreement, or uncertainty across the sources;\n"
                "- what to watch next.\n"
                "Write a COMPLETE analysis of ~500-650 words covering ALL four points — "
                "finish cleanly with a complete final sentence; never stop mid-thought.\n"
                f"STORY: {title}\n\nSOURCES:\n{ev}"}],
                # 2200 (was 1400) + num_ctx 16384 (was 12288), 2026-08-11: Lever C
                # routed this to the 27B, which writes ~6k chars — the 1400 cap cut
                # it mid-generation on ~90% of stories (689 [truncation] warns in
                # 2.5 days), always amputating the tail sections (tension / what to
                # watch). The word-target above makes the model self-bound; the cap
                # is now a backstop with headroom, not the editor.
                max_tokens=2200, temperature=0.3, model=model, num_ctx=16384)
            return (title, (a or "").strip())
        except Exception as e:
            logger.warning("[DeepResearch] per-story analysis failed (story=%r): %s", title, e)
            return (title, "")
    analyses = await asyncio.gather(*[_an(t, ix) for t, ix in stories])
    blocks = [f"### {t}\n{a}" for t, a in analyses if a]
    if blocks:
        logger.info("[DeepResearch] %s deep-analysis: %d stories analyzed (%d calls)",
                    label, len(blocks), len(stories) + 1)
    return "\n\n".join(blocks)


async def _enrich_overview(draft: str, analysis_block: str, common: str, label: str,
                           *, model: str | None = None) -> str:
    """Completeness + depth pass (owner directive: output-at-its-best, more calls fine).
    Re-reads the DEEP ANALYSES + FINDINGS against the draft and (a) ADDS any consequential
    development the synthesis omitted, (b) DEEPENS thin ones — strictly from the sources.
    The verification pass + grounding guards downstream re-check every added claim, so this
    can only add GROUNDED richness; invented additions are stripped there. `common` is the
    shared evidence prefix from `_common_context` (prefix-cache #34) — it carries the
    analyses + findings; instructions and draft come after it."""
    if not draft or not analysis_block:
        return draft
    try:
        out = await _invoke_bg([{"role": "user", "content":
            common +
            f"Below is a DRAFT overview. Return an IMPROVED overview that is more COMPLETE "
            f"and DEEPER, changing nothing already correct:\n"
            f"1. ADD any genuinely consequential {label} development present in the DEEP ANALYSES "
            f"or SOURCE FINDINGS above but MISSING from the draft — as a new Secondary bullet with "
            f"its inline (outlet.com) citation.\n"
            f"2. DEEPEN any thin development using specifics FROM THE SOURCES: exact numbers, the "
            f"mechanism, named players, dates, and the second-order implication.\n"
            f"3. KEEP every existing development and the structure (Lead / Secondary Developments / "
            f"Connections & bottom line). Longer is fine — comprehensiveness is the goal.\n"
            f"Add ONLY what the analyses/findings above support; invent NOTHING; cite every new "
            f"sentence inline.\n\n"
            f"DRAFT:\n{draft}"}],
            # 16384→20480 (2026-08-11): common grew to ~10k tokens with the
            # ANALYSIS_CAP raise; + draft (~2.2k) + 5000 gen ≈ 18k.
            max_tokens=5000, temperature=0.35, num_ctx=20480, model=model)
        out = (out or "").strip()
        return out if _accept_correction(draft, out) else draft   # keep only a structure-preserving, non-shrinking result
    except Exception as e:
        logger.debug("[DeepResearch] enrich pass failed: %s", e)
        return draft


async def _gather_evidence(label: str, n_stories: int, feed_key: str | None):
    """Story selection + pooled deep read + iterative gap loop -> (subjects, findings,
    articles). Split out of domain_overview (2026-07-10) so the ceiling A/B harness can
    capture evidence ONCE and replay synthesis on it. Behavior-preserving move."""
    subjects = await _focus_subjects(label, feed_key=feed_key or label, n=n_stories)
    logger.info("[DeepResearch] %s → %d stories: %s", label, len(subjects), subjects)

    # Read DEEP: broad/meta domains (World Awareness, Current Events) spread their
    # reads across many disparate stories, so a low target leaves each story SINGLE-
    # sourced — which reads vague and risks single-source errors (the 2026-06-29 audit
    # found World Awareness at 8 sources across 7 stories). Higher read_target +
    # browser_budget gets multiple sources PER story (most quality world feeds — BBC,
    # Guardian, France24, DW — are JS-rendered, so depth is gated by browser_budget).
    # Latency-tolerant (background); richer > faster per the owner.
    articles = await _gather_overview(subjects, feed_key or label, read_target=28, browser_budget=20)
    # Relevance anchor = the DOMAIN (broad), NOT the seed subjects. The feeds
    # surface many on-domain stories beyond the seed subjects; gating findings on
    # the specific subjects made the 9B reject them as "irrelevant" (the bug that
    # forced the overview to keep falling back). Broad anchor keeps every
    # on-domain finding → real breadth. Subjects still seed the search + synthesis.
    findings = await _findings(articles, label) if articles else []

    # ITERATIVE gap loop (Phase 3): reflect on the first-pass findings → targeted follow-up
    # searches → read → merge. Later queries conditioned on what the first read revealed is
    # what single-pass gather cannot do; measured to lift coverage/corroboration (Search-o1,
    # FRAMES 0.40→0.66). Bounded: ≤2 rounds, stop when a round adds <2 new sources. The
    # reflection runs on the fast default model; only the extra reads add cost (network, not GPU).
    from app.config import config as _cfg_iter
    if getattr(_cfg_iter, "ENABLE_ITERATIVE_GATHER", True) and len(findings) >= 3:
        from app.monitors.report_grader import _anchors as _anchor_fn
        _seen = {u for _, u, _ in findings}
        _seen_anchors = _anchor_fn(" ".join(f for _, _, f in findings))
        # Loop-until-dry (#67): keep gathering while a round surfaces NEW
        # distinct ENTITIES (not just new URLs — a round can add sources that
        # merely re-cover known stories). Bounded to 3 rounds so a hot domain
        # can't run away; stops early when a round is entity-dry OR source-thin.
        # The yield latch (#54) defers this whole loop while the owner chats.
        for _round in range(3):
            try:
                gaps = await _gap_followup(findings, label)
            except Exception:
                gaps = []
            if not gaps:
                break
            more = await _gather_gap(gaps, read_target=10, browser_budget=8)
            new_arts = [a for a in more if a[1] not in _seen and not _stale_body(a[2])]
            if len(new_arts) < 2:
                break
            _seen.update(a[1] for a in new_arts)
            new_findings = await _findings(new_arts, label)
            round_anchors = _anchor_fn(" ".join(f for _, _, f in new_findings))
            fresh_anchors = round_anchors - _seen_anchors
            articles = articles + new_arts
            findings = findings + new_findings
            logger.info("[DeepResearch] gap-round %d (%s): +%d sources, +%d new entities",
                        _round + 1, label, len(new_arts), len(fresh_anchors))
            if len(fresh_anchors) < 2:   # entity-dry → stop even if URLs were new
                break
            _seen_anchors |= round_anchors
            logger.info("[DeepResearch] %s gap-round %d: +%d sources (%d findings total)",
                        label, _round + 1, len(new_arts), len(findings))
    return subjects, findings, articles


_KVN_SCHEMA = {
    "type": "object",
    "properties": {
        "new": {"type": "integer"},
        "updates": {"type": "integer"},
        "contradictions": {"type": "integer"},
    },
    "required": ["new", "updates", "contradictions"],
}


async def _count_known_vs_new(prior_text: str, final: str, *, model: str | None, label: str) -> dict | None:
    """Knowing instrumentation v2 (2026-08-12): ONE tiny gated call counting the
    finished digest against the prior dossier. 2026-09-01: max_tokens 80 -> 200
    and a JSON schema — the 80-token budget truncated the reply on the 27B and
    logged 'None new | None updates' (the only two primed runs in the window).

    2026-09-03: this is now the ONLY consumer of the dossier in the digest path.
    It runs on the FINISHED briefing, so it measures novelty without being able
    to change what was written — which is why retiring the prompt injection did
    not cost the knowing tier its instrumentation. It carries its own header."""
    try:
        raw = await _invoke_bg([{"role": "user", "content":
            "PRIOR UNDERSTANDING — what Nova knew about this domain before today:\n"
            + prior_text + "\n\n" +
            "TODAY'S BRIEFING:\n" + final[:9000] +
            "\n\nCount today's developments in the briefing against the PRIOR "
            "UNDERSTANDING above: how many are genuinely NEW (absent from prior), "
            "how many UPDATE something already known, and how many CONTRADICT it. "
            'Return JSON only: {"new": <int>, "updates": <int>, "contradictions": <int>}'}],
            json_mode=True, json_schema=_KVN_SCHEMA, max_tokens=200, temperature=0.0,
            model=model, num_ctx=8192)
        data = llm.extract_json_object(raw) if isinstance(raw, str) else (raw or {})
        if not isinstance(data, dict):
            return None
        out = {}
        for k in ("new", "updates", "contradictions"):
            try:
                out[k] = int(data.get(k))
            except (TypeError, ValueError):
                out[k] = None
        return out
    except Exception as e:
        logger.debug("[Knowing] KNOWN-VS-NEW measurement failed (%s): %s", label, e)
        return None


async def _synthesize_from_evidence(label: str, findings: list, articles: list, today: str, *, kg=None, syn_model: str | None = None, dossier_key: str | None = None) -> str:
    """Deep analysis -> best-of-N synthesis -> grounding stack -> fact-banking -> bound.
    Split out of domain_overview (2026-07-10) so the ceiling harness replays the EXACT
    production synthesis on captured evidence with any (model, config). syn_model=None
    -> config.MONITOR_SYNTHESIS_MODEL."""
    hosts = sorted({_host(u) for _, u, _ in findings})
    # Independence map (2026-08-12): near-duplicate bodies collapse for every
    # corroboration count in this chain — mirrors/syndication count once.
    _host_clusters = _independence_clusters(articles)[1] if articles else {}
    # Temporal layer: record this digest's host set (always-on, cheap) and merge
    # any flagged paraphrase-network pairs into the cluster map. Fail-open.
    try:
        from app.database import get_db as _gdb
        _db_co = _gdb()
        await asyncio.to_thread(_record_host_cooccurrence, _db_co, hosts)
        _host_clusters = await asyncio.to_thread(_apply_network_pairs, _db_co, _host_clusters)
    except Exception as e:
        logger.debug("[Independence] co-occurrence layer skipped: %s", e)
    evidence = _annotated_evidence(findings, host_clusters=_host_clusters)

    # REAL deep research: analyze each story in depth FIRST (many calls), then let the
    # final synthesis (this call) reason over those analyses + the full findings.
    # Lever C: route the analysis/synthesis reasoning to a bigger model when set.
    from app.config import config as _cfg
    if syn_model is None:
        syn_model = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None
    if syn_model:
        logger.info("[DeepResearch] %s synthesis on %s (Lever C)", label, syn_model)
    deep = ""
    try:
        if getattr(_cfg, "ENABLE_DEEP_ANALYSIS", True):
            body_map = {u: b for _t, u, b in articles if b}
            deep = await _deep_analyze(findings, label, today, bodies=body_map, model=syn_model)
    except Exception as e:
        logger.warning("[DeepResearch] deep analysis failed: %s", e)
    analysis_block = (
        "DEEP ANALYSES — each story already reasoned over by an analyst; draw on these for the "
        f"insight, the 'so what', and the connections:\n{deep}\n\n" if deep else "")

    # Knowing tier (2026-08-12), injection RETIRED 2026-09-03. The dossier used
    # to be injected here as PRIOR UNDERSTANDING so the digest could lead with
    # what was new. A paired A/B over 16 frozen topics — same evidence, arms run
    # concurrently, priming provably on in one and off in the other — measured
    # the injection as no gain and a real cost:
    #     overall   0.838 primed vs 0.844 unprimed   (primed won 7/16, p=0.61)
    #     support   0.642 primed vs 0.690 unprimed   (primed won 4/16, p=0.099)
    #     judge     4.61  primed vs 4.55  unprimed   (p=0.34)
    #     coverage  0.715 primed vs 0.684 unprimed   (p=0.49)
    #     fabricated 0.000 in both arms
    # Support is the DETERMINISTIC grounding score, and it fell on 12 of 16
    # topics: prior context was being restated as if today's sources carried it.
    # Nothing measured improved. The dossier is still LOADED — the out-of-band
    # KNOWN-VS-NEW count needs it and cannot alter the digest — but it no longer
    # reaches any prompt the briefing is written from. (The 2026-08-13 result
    # that said priming wins is void: priming was silently off for most of its
    # topics.) Revisit only with a harness that scores novelty framing, which
    # this one does not measure.
    prior_text = ""
    if getattr(_cfg, "ENABLE_DOSSIERS", True):
        try:
            from app.core.dossiers import get_domain_dossier, priming_excerpt
            from app.database import get_db
            # Off the event loop — a sync DB read here was the one steady-state
            # [Sync DB call on event-loop] violation (audit 2026-08-23).
            # Keyed by MONITOR NAME (2026-09-01): the short profile label only
            # matched the dossier key for 13/39 domains — the rest ran unprimed.
            _d = await asyncio.to_thread(get_domain_dossier, get_db(), label, dossier_key)
            if _d and _d["body"]:
                prior_text = priming_excerpt(_d["body"])
                logger.info("[Knowing] %s measured against dossier %r "
                            "(prompt injection retired 2026-09-03)", label, _d["dkey"])
            else:
                logger.info("[Knowing] %s has no domain dossier yet (key=%r) — "
                            "no novelty measurement", label, dossier_key or label)
        except Exception as e:
            logger.debug("[Knowing] dossier priming unavailable: %s", e)

    # Prefix-cache (#34): ONE byte-identical evidence pack leads every chain stage
    # (synthesis → judge → enrich → verify) so its KV cache is computed once and
    # reused; only each stage's instructions + draft are new tokens. The synthesis
    # call appends any evidence overflow past the shared cap AFTER the pack — the
    # cache reuses up to the divergence point and synthesis still sees everything.
    common = _common_context(today, label, analysis_block, evidence)
    _extra = evidence[_COMMON_EVIDENCE_CAP:]
    # Coverage checklist (#67b): the distinct entity anchors surfaced in the
    # findings. Synthesis-recall was dropping found stories (the DeepResearch
    # Bench II failure: agents "miss key sources"); listing the surfaced players
    # as a to-cover checklist keeps them in. Fabrication-safe — it only names
    # entities already IN the findings and still requires source support.
    try:
        from app.monitors.report_grader import _anchors as _anchor_fn
        _cov_anchors = sorted(_anchor_fn(" ".join(f for _, _, f in findings)),
                              key=len, reverse=True)[:18]
    except Exception:
        _cov_anchors = []
    _checklist = (
        "COVERAGE CHECKLIST — these named players/entities surfaced in the findings; "
        "make sure every consequential one is addressed somewhere in the brief IF the "
        "findings support it (do NOT invent detail for one that is only named in "
        f"passing): {', '.join(_cov_anchors)}.\n\n" if _cov_anchors else "")
    _syn_prompt = (
        common
        + (f"MORE SOURCE FINDINGS (continued):\n{_extra}\n\n" if _extra else "")
        + _checklist
        + f"You READ {len(findings)} credible {label} sources — the deep analyses and findings "
        f"above are everything you read. Write the overview from THESE ONLY "
        f"(do not add stories you remember but did not read here).\n\n"
        f"You are a sharp {label} intelligence analyst writing for an expert who will ACT on this. "
        f"Be substantive, analytical, and ruthlessly ON-MISSION.\n"
        f"CALIBRATE by evidence strength: each finding is tagged '(outlet · reliability · "
        f"corroboration)'. Lead with developments that are CONFIRMED (multiple sources, or a "
        f"wire/primary/quality outlet); a claim resting on a SINGLE 'single/unverified' or "
        f"'LOW-CREDIBILITY' source must be caveated in-line ('per a single unverified report,') or "
        f"dropped — NEVER lead with one. Cite only the outlet name, not the reliability tag.\n"
        f"**Lead Development** — THE single most consequential {label} development: a full paragraph "
        f"(5-8 sentences) with the key facts, EXACT numbers, named players, what is genuinely new, "
        f"and — critically — WHY it matters and the second-order implications.\n"
        f"**Secondary Developments** — each OTHER genuinely consequential {label} development as its "
        f"own bullet (3-5 sentences of concrete specifics — numbers, names, dates, the mechanism, and "
        f"the 'so what' / second-order implication). Rank by importance and lead with the biggest. Cover "
        f"the FIVE TO SEVEN MOST CONSEQUENTIAL developments — prioritize ruthlessly; a focused brief that "
        f"leads with what matters beats an exhaustive list. Keep the FULL briefing UNDER ~8500 characters "
        f"(a 12k-char digest overruns both the generation budget and Discord readability, 2026-07-08). "
        f"EXCLUDE anything that "
        f"is not real {label} intelligence: sports results, celebrity/personal gossip, individual "
        f"salary/appointment trivia, routine scheduling, promotional/press-release fluff, and off-topic "
        f"tangents — even if a source mentioned them. Prefer developments confirmed by MULTIPLE sources; "
        f"never pad with filler, but richness from REAL sourced detail is the goal.\n"
        "**Connections & bottom line** — 2-3 sentences: the throughline across the stories and the "
        "single MOST CONSEQUENTIAL thing to watch next — a real open question with stakes, NOT a "
        "routine scheduling detail (e.g. a minor earnings date). Draw ONLY on facts already stated "
        "above; introduce NO new number, figure, or named entity in this section.\n"
        "Every FACTUAL sentence — an event, figure, quote, or named action — MUST cite its outlet "
        "inline like (cnbc.com). Your own ANALYSIS (implications, connections, what to watch) "
        "carries NO citation: never attach an outlet to your own reasoning, and never point at "
        "the analyses themselves as if they were a source — they are your own thinking, not "
        "evidence. If you cannot cite a factual claim from the findings, OMIT it. "
        "Use ONLY information present in the analyses/findings "
        "above — never invent a company, number, person, or event.")
    try:
        # best-of-N: FOUR diverse framings, external grounded judge picks the sharpest.
        # (owner directive: maximize the best output; extra generations are fine.)
        draft = await _best_synthesis(_syn_prompt, evidence, n=4,
                                      temps=(0.15, 0.35, 0.55, 0.8), model=syn_model,
                                      context_block=common)
    except Exception as e:
        logger.warning("[DeepResearch] overview draft failed: %s", e)
        draft = ""
    draft = (draft or "").strip()
    if not draft:
        return f"## 🌐 {label} — domain overview\n_{today}_\n\n(synthesis unavailable)"

    # Knowing instrumentation: log how today's reading scored against prior
    # understanding, then strip the marker line from the digest body.
    _kvn = _KNOWN_VS_NEW_RE.search(draft)
    if _kvn:
        logger.info("[Knowing] %s KNOWN-VS-NEW: %s", label, _kvn.group(1).strip())
        draft = _KNOWN_VS_NEW_RE.sub("", draft).strip()

    # ENRICHMENT: add consequential developments the synthesis omitted + deepen thin ones,
    # strictly from the sources (verification + grounding guards below re-check every claim).
    draft = await _enrich_overview(draft, analysis_block, common, label, model=syn_model)

    # Verify pass retired 2026-09-01: the 27B "delete mode" pass changed 0
    # sentences in 17 of 20 live runs while costing a 5,000-token generation;
    # the MiniCheck gate below now covers every cited fact sentence instead.
    final = draft
    logger.info("[DeepResearch] verify pass retired — entail gate covers %d content sentences (%s)",
                _count_content_sentences(final), label)

    final = _strip_prompt_leak(final)   # drop any synthesis-prompt instructions the model echoed
    final = _KNOWN_VS_NEW_RE.sub("", final).strip()   # belt-and-braces: knowing marker must never post
    final = _strip_fake_citations(final, hosts)   # drop fabricated-attribution lines
    final, _ = await _ensure_citations(final, findings, model=syn_model)   # backstop: re-cite an uncited misfire
    final, _ = await _ground_numbers(final, [b for _, _, b in articles], model=syn_model)  # every figure traces to a source
    final, _ = await _ground_claims(final, [f"{t}\n{b}" for t, _u, b in articles], model=syn_model)   # distinctive terms trace too (titles incl.)
    final, _ = await _check_contamination(final, articles, model=syn_model)   # details trace to their CITED source, not a grafted one
    final, _ = await _check_numeric_attribution(final, articles, model=syn_model)   # + figures trace to their CITED source's value
    final, _ = _strip_pseudo_citations(final)   # the digest citing its own analysis is not a source
    final, _ = _decite_analysis(final)   # the digest's own reasoning carries no citation (2026-09-01)
    final, _ = _cite_uncited_sentences(final, articles)   # attribute uncited sibling sentences to their source (deterministic)
    final, _ = await _entailment_gate(final, articles, label=label)   # MiniCheck: cited source must ENTAIL the claim (gated, fail-open)
    final = _correct_currency_mislabels(final, articles)   # re-stamp $-figures the sources state in CNY/HKD/EUR/...
    final, _corr = await _corroborate_numbers(final, articles)   # set of figures ≥2 sources confirm (for Lever A)
    final = _tidy_citations(final)   # strip $-wrapped citations + leaked title fragments
    final = _strip_novel_bottomline_figures(final)   # drop a figure introduced only in the bottom line
    final = _drop_orphan_headers(final)   # remove any section header left empty by the strip passes
    final = _repair_dangling_fragments(final)  # recover sentences broken by an excised entity (prep/possessive)
    final = _repair_broken_sentences(final)   # drop sentences left grammatically broken by the strip passes
    final, _ = _drop_artifact_sentences(final)   # never ship "the launched a …" (2026-09-01)
    # Knowing instrumentation v2 (2026-08-12): measured OUT-OF-BAND — one tiny
    # gated call counting the finished digest against the prior dossier. The
    # v1 in-band marker asked the synthesis to append a count line, but the
    # aggregation merge rewrites candidates and dropped it (probe-verified).
    if prior_text:
        _kvn = await _count_known_vs_new(prior_text, final, model=syn_model, label=label)
        if _kvn:
            logger.info("[Knowing] %s KNOWN-VS-NEW: %s new | %s updates | %s contradictions",
                        label, _kvn.get("new"), _kvn.get("updates"), _kvn.get("contradictions"))

    learned = await _learn_facts(label, final, findings, kg, articles=articles, model=syn_model)
    logger.info("[DeepResearch] %s overview: read %d sources (%s), learned %d facts",
                label, len(findings), ", ".join(hosts[:6]), learned)
    # Recall/coverage signal (task #65): of the distinct entity anchors surfaced
    # during gather, what fraction reached the digest? pool_anchors = gather
    # breadth; coverage = synthesis recall. The before/after metric for the
    # multi-angle sweep (#66) and coverage-driven gap loop (#67).
    try:
        from app.monitors.report_grader import coverage_score
        _cov = coverage_score(final, findings)
        logger.info("[DeepResearch] coverage (%s): CORE %d/%d (%.0f%%) · all %d/%d (%.0f%%) · pool=%d%s",
                    label, _cov["core_covered"], _cov["core_anchors"], _cov["core_coverage"] * 100,
                    _cov["covered"], _cov["pool_anchors"], _cov["coverage"] * 100, _cov["pool_anchors"],
                    (" · missed-core: " + "; ".join(_cov["missed"][:5])) if _cov["missed"] else "")
    except Exception as e:
        logger.warning("[DeepResearch] coverage score failed: %s", e)

    # Lever A: fresh independent verification of the lead's top claims (gated, bounded).
    try:
        from app.config import config as _cfg
        if getattr(_cfg, "ENABLE_CLAIM_VERIFICATION", True):
            final, _ = await _verify_lead_claims(final, label, corroborated=_corr)
    except Exception as e:
        logger.warning("[DeepResearch] claim verification failed: %s", e)

    # Corroboration-gated lead: caveat a headline resting on a single low-credibility
    # source (robust to unknown farms the blocklist misses). Deterministic, always on.
    final, _gated = _gate_lead_credibility(final, host_clusters=_host_clusters, articles=articles)
    if _gated:
        logger.info("[DeepResearch] %s: lead flagged — single low-credibility source", label)

    # Think-leak tripwire (2026-07-08): a `<think>` reaching here means a model
    # pass leaked raw reasoning past the invoke_nothink + strip + accept guards.
    # Never post it — strip to the last close (or drop an unterminated tail) and
    # log ERROR so the failure is loud, not a broken digest on Discord.
    if "<think>" in final.lower() or "</think>" in final.lower():
        logger.error("[DeepResearch] THINK-LEAK in %s digest — stripping before post", label)
        from app.core.llm import _strip_think_tags
        final = _strip_think_tags(final).strip()
        if len(final) < 200:   # strip left nothing usable → this run is unpublishable
            logger.error("[DeepResearch] %s digest empty after think-strip — suppressing", label)
            return f"## 🌐 {label} — domain overview\n_{today}_\n\n(synthesis unavailable this cycle)"

    # Deterministic final bound: guarantee a complete-sentence ending + length
    # under the storage/generation caps (both the Discord post and the stored
    # copy use this return value). Ends the truncation/overrun arms race.
    final = _bound_and_clean(final, max_chars=10800)

    header = (f"## 🌐 {label} — domain overview\n"
              f"_read {len(findings)} sources: {', '.join(hosts[:7])}"
              f"{' +more' if len(hosts) > 7 else ''} · {learned} facts learned · {today}_\n\n")
    return header + final


async def domain_overview(label: str, kg=None, n_stories: int = 5, feed_key: str | None = None) -> str:
    """Broad overlook over a deep base: find the top current stories, then ONE
    pooled deep read across all of them (http-fast-path first, headless browser
    only for a few high-value misses — reliable + bounded), then ONE grounded
    synthesis into a cohesive overview: lead development + secondary developments
    + bottom line. The single strong synthesis pass beat decomposed/atomized
    re-synthesis head-to-head (2026-06-21), so we keep depth AND breadth without
    the lossy collapse. Verified, cited, fact-banking. Falls back to a single deep
    briefing when the pool is thin; honest when there are no readable sources."""
    today = _NOW().strftime("%B %d, %Y")
    subjects, findings, articles = await _gather_evidence(label, n_stories, feed_key)
    if len(findings) < 2:
        # Breadth pool came up thin — fall back to a single strong deep-dive on
        # the top story (proven good), rather than a hollow overview.
        if subjects:
            logger.info("[DeepResearch] %s overview thin (%d findings) — single deep-dive fallback",
                        label, len(findings))
            return await research_and_brief(label, topic=subjects[0], kg=kg)
        return (f"## 🌐 {label} — domain overview\n_no readable credible sources today · {today}_\n\n"
                f"No readable credible sources were found across the top {label} stories today; "
                f"nothing reported rather than speculation.")
    return await _synthesize_from_evidence(label, findings, articles, today, kg=kg, dossier_key=feed_key)

