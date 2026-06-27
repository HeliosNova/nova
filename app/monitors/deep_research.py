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

logger = logging.getLogger(__name__)


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


# --- Source-quality tiers -------------------------------------------------
_TIER1_SUFFIX = (".gov", ".edu", ".int", ".mil")
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
}


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


def _source_quality(url: str) -> float:
    if _blocked(url):
        return 0.0
    h = _host(url)
    if any(h.endswith(s) for s in _TIER1_SUFFIX) or h in _TIER1_HOSTS:
        return 3.0
    if h in _TIER2_HOSTS:
        return 2.0
    # Primary-artifact URL on an otherwise-unranked host: the actual filing/release/
    # paper beats secondary reporting. Lifts above the 1.0 generic floor, below the
    # curated tier-1 wires, and never rescues a junk/blocked host (handled above).
    if _PRIMARY_URL_RE.search(url):
        return 2.5
    if _JUNK_HOSTS_RE.search(url):
        return 0.5
    return 1.0


_CSS_TOKENS = re.compile(
    r"(\{[^}]{0,40}:[^}]{0,40}\}|@media|font-family|margin:|padding:|rgba?\(|px;|<style)",
    re.IGNORECASE)


def _junk_body(b: str) -> bool:
    if not b or len(b) < 400:
        return True
    head = b[:240].lower()
    if ("minimal readable" in head or "%pdf" in head
            or "endstream" in b[:600].lower() or b.count(" obj") > 5):
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
    r"\btrends?\s+(for|to\s+watch|in\s+20|reshaping)\b|\bthings?\s+(you|to)\b|\bways?\s+to\b)")


def _is_seo_headline(title: str) -> bool:
    return bool(_SEO_HEADLINE_RE.search(title or ""))


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


async def _feed_headlines(feed_key: str) -> list[tuple[str, str]]:
    """Headlines from the domain's CURATED feeds — real news (not SEO), so coverage
    counted here is trustworthy. (title, host) pairs."""
    try:
        from app.monitors.rss_feeds import fetch_recent_items
        items = await fetch_recent_items(feed_key, hours=72, max_total=30)
    except Exception:
        return []
    return [(it.title.strip(), it.source_host) for it in items
            if it.title and it.url and not _blocked(it.url)]


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
    clusters.sort(key=lambda c: (len(c["hosts"]), len(c["titles"])), reverse=True)
    top = clusters[:n]
    if not top:
        return [f"{label} latest developments {year}"]
    reps = [max(c["titles"], key=len) for c in top]
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(reps))
    try:
        raw = await llm.invoke_nothink([{"role": "user", "content":
            f"Today is {today}. Rewrite each {label} headline below (already ranked by importance) as "
            f"ONE concrete {year} subject phrase — key entities + what happened, no outlet name, no "
            f"clickbait, no question mark. KEEP THE SAME ORDER AND COUNT.\n" + listing +
            f"\n\nReturn a JSON array of exactly {len(reps)} strings, same order."}],
            json_mode=True, json_prefix='["', max_tokens=320, temperature=0.2)
        subs = [str(s).strip().rstrip("?") for s in _json_array(raw)
                if isinstance(s, str) and len(str(s).strip()) > 10]
    except Exception:
        subs = []
    if len(subs) < len(reps):  # LLM thin/failed → clean the representative headlines deterministically
        subs = [re.sub(r"\s*[-–|:]\s*[A-Z][\w. '&]{2,30}$", "", t).strip() for t in reps]
    return subs[:n] or [f"{label} latest developments {year}"]


async def _focus_subject(label: str) -> str:
    """The single most-COVERED current story (coverage-ranked top-1)."""
    subs = await _focus_subjects(label, n=1)
    return subs[0] if subs else f"{label} latest developments {_NOW().strftime('%Y')}"


# Limit concurrent headless-browser renders (chromium contexts are heavy).
_BROWSER_SEM = asyncio.Semaphore(3)


async def _fetch_body(url: str, *, browser_budget: list[int] | None = None) -> str | None:
    """Read an article body. Fast path: http_fetch (hard 15s cap). Fallback:
    headless browser (hard 22s cap) — most quality news sites (BBC, CNBC,
    Economist, Reuters) are JS-rendered and return a CSS shell to a plain GET, so
    without this the good sources are unreadable. The browser is the latency
    bottleneck (a dead URL can sit on a 60s nav timeout), so callers pass a
    `browser_budget` (mutable [n]) to cap total renders — most reads stay on the
    fast path and only a few escalate to the browser."""
    from app.tools.http_fetch import HttpFetchTool
    try:
        res = await asyncio.wait_for(HttpFetchTool().execute(url=url), timeout=15)
        body = (res.output or "") if getattr(res, "success", False) else ""
        if not _junk_body(body):
            return body[:5000]
    except Exception:
        pass
    # JS-rendered → render with the browser, but only within budget.
    if browser_budget is not None:
        if browser_budget[0] <= 0:
            return None
        browser_budget[0] -= 1
    from app.tools.browser import BrowserTool
    async with _BROWSER_SEM:
        try:
            r = await asyncio.wait_for(
                BrowserTool().execute(action="navigate", url=url), timeout=22)
            body = (r.output or "") if getattr(r, "success", False) else ""
            return body[:5000] if (body and not _junk_body(body)) else None
        except Exception:
            return None


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


_PAYWALL_HOSTS = {
    "bloomberg.com", "ft.com", "wsj.com", "economist.com", "nytimes.com",
    "washingtonpost.com", "newyorker.com", "theatlantic.com", "barrons.com",
    "businessinsider.com", "seekingalpha.com",
}


async def _read_bodies(picks: list, *, read_target: int, browser_budget: int) -> list:
    """Read article bodies cheaply and reliably. Phase 1: http_fetch ALL picks in
    parallel (fast, no browser). Phase 2: escalate to the headless browser for
    only a few HIGH-VALUE (tier ≥ 2) misses — the browser resists asyncio
    cancellation and a single dead URL can burn 60s, so we minimize and bound it.
    Stale bodies dropped throughout."""
    async def _http(r):
        body = await _fetch_body(r.url, browser_budget=[0])  # [0] → http only
        return (r, body)

    good, misses = [], []
    for r, body in await asyncio.gather(*[_http(r) for r in picks]):
        if body and not _stale_body(body):
            good.append((r.title, r.url, body))
        else:
            misses.append(r)

    if len(good) < read_target and browser_budget > 0:
        # Escalate only HIGH-VALUE misses, and skip hard-paywalled hosts: rendering
        # them just burns a ~60s nav timeout on a paywall/subscribe stub that the
        # body gate then drops anyway.
        hv = [r for r in misses
              if _source_quality(r.url) >= 2.0 and _host(r.url) not in _PAYWALL_HOSTS][:browser_budget]
        budget = [browser_budget]

        async def _br(r):
            body = await _fetch_body(r.url, browser_budget=budget)
            return (r.title, r.url, body) if (body and not _stale_body(body)) else None

        good += [a for a in await asyncio.gather(*[_br(r) for r in hv]) if a]
    return good[:read_target]


async def _gather_sources(subject: str, *, read_target: int, browser_budget: int = 6) -> list:
    """Deep single-story gather: facet-expand the subject, search, read."""
    year = _NOW().strftime("%Y")
    raw = await llm.invoke_nothink([{"role": "user", "content":
        f"3 web-search queries digging into this {year} story from different facets "
        f"(what happened, numbers/who, reactions/analysis): '{subject}'. JSON array of 3."}],
        json_mode=True, json_prefix='["', max_tokens=160)
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


async def _overview_angles(subjects: list[str]) -> list[str]:
    """Facet-expand the stories into article-finding queries. The bare subject
    phrase tends to surface landing/SEO pages; a 'what happened / numbers' facet
    surfaces actual articles. ONE LLM call, and the angle count is capped — we get
    'more sources' by reading deeper into each query's results (max_results=16),
    NOT by firing many queries, which is what rate-limits the search engines."""
    year = _NOW().strftime("%Y")
    angles = list(subjects)
    try:
        raw = await llm.invoke_nothink([{"role": "user", "content":
            f"For each of these {year} news stories, write ONE focused web-search query that would "
            f"surface actual news ARTICLES (what happened, key numbers/names) — not homepages.\n" +
            "\n".join(f"- {s}" for s in subjects) +
            "\nReturn one flat JSON array of the query strings."}],
            json_mode=True, json_prefix='["', max_tokens=300, temperature=0.2)
        angles += [a for a in _json_array(raw) if isinstance(a, str) and len(a) > 8]
    except Exception:
        pass
    return list(dict.fromkeys(angles))[:12]   # cap angles → bounded search load


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
    # Feeds first (curated), then aux news (GDELT/Bing — search-independent), then
    # SearXNG search; dedup; ≤3 per host.
    seen, host_ct, picks = set(), {}, []
    for r in feed_picks + aux_picks + search_picks:
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


_NO_CONTENT_RE = re.compile(
    r"\b(irrelevant|no (?:concrete |specific |relevant |substantive )?(?:information|findings|"
    r"content|facts|details|data)|contains no|no (?:actual )?article|boilerplate|"
    r"navigation (?:menu|bar|chrome|links)|cannot extract|nothing (?:to extract|reportable))",
    re.IGNORECASE)


async def _findings(articles: list, subject: str) -> list[tuple[str, str, str]]:
    async def _one(title, url, body):
        try:
            f = await llm.invoke_nothink([{"role": "user", "content":
                f"Extract 2-3 concrete findings (facts, numbers, named events) from this article "
                f"relevant to '{subject}'. Reply IRRELEVANT only if the article is about a COMPLETELY "
                f"different field. Only what is stated.\n\nTITLE: {title}\n\n{body}"}],
                max_tokens=240, temperature=0.2)
            f = (f or "").strip()
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


def _tidy_citations(text: str) -> str:
    """Cosmetic cleanup of synthesis citation artifacts (2026-06-24 verification
    pass): the 9B occasionally wraps a citation in stray '$' ('($quiverquant.com$)')
    or drops a truncated source TITLE into a citation slot ('(… starts making
    noise…)'). Deterministic + scoped to parentheticals, so real figures ('$9.2m')
    and ordinary prose are untouched."""
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
    return text


async def _ground_numbers(text: str, bodies: list[str]) -> tuple[str, int]:
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
        fixed = await llm.invoke_nothink([{"role": "user", "content":
            "SOURCE TEXTS and a DRAFT briefing are below. These figures in the draft do NOT appear "
            "in the sources and were likely misread: " + "; ".join(unver[:12]) + ".\n"
            "Rewrite the draft so EVERY number matches the sources exactly: replace each listed "
            "figure with the value actually stated in the sources, or delete that specific claim if "
            "the sources give no figure. Change nothing else — keep all wording, structure, and "
            "citations.\n"
            "OUTPUT RULES: return ONLY the corrected briefing itself, starting at its first heading "
            "or bold line. Do NOT add any preamble, sign-posting, or notes explaining what you "
            "changed (no 'Based on the sources…', no 'here is the rewritten draft', no '*Note:*').\n\n"
            "SOURCES:\n" + src + "\n\nDRAFT:\n" + text}],
            max_tokens=1400, temperature=0.0, num_ctx=8192)
        fixed = _strip_correction_scaffold((fixed or "").strip())
    except Exception as e:
        logger.warning("[DeepResearch] numeric grounding pass failed: %s", e)
        fixed = ""
    # Use the correction only if it kept the briefing's structure (didn't collapse
    # to commentary) — else fall back to the original draft.
    out = fixed if (fixed and ("**" in fixed or "##" in fixed)) else text
    still = set(_unverified_numbers(out, corpus, corpus_nc))
    if still:
        out = _drop_sentences_with(out, still)
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


def _orphan_terms(text: str, corpus: str) -> list[str]:
    """Distinctive CHECKABLE terms in `text` that appear NOWHERE in the source corpus
    — the qualitative analogue of `_unverified_numbers`. Catches fabricated specifics
    that numeric grounding can't see (e.g. an invented 'LSD therapy' rationale grafted
    onto a real acquisition). Conservative by design: only ACRONYMS and multi-word
    PROPER-NOUN phrases (high distinctiveness, hard to paraphrase). Single names are
    skipped — a body-extraction miss on one name would falsely strip real content."""
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
    return list(dict.fromkeys(orphans))


async def _ground_claims(text: str, bodies: list[str]) -> tuple[str, int]:
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
        fixed = await llm.invoke_nothink([{"role": "user", "content":
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
            max_tokens=1400, temperature=0.0, num_ctx=8192)
        fixed = _strip_correction_scaffold((fixed or "").strip())
    except Exception as e:
        logger.warning("[DeepResearch] claim grounding pass failed: %s", e)
        fixed = ""
    out = fixed if (fixed and ("**" in fixed or "##" in fixed)) else text
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


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def _cited_regs(sent: str) -> set:
    """Registrable domains cited inline in a sentence, e.g. '(thehackernews.com)'."""
    regs = set()
    for cm in _PAREN_CITE_RE.finditer(sent):
        for tok in _DOMAIN_TOKEN_RE.findall(cm.group(0)):
            regs.add(_reg_domain(tok.lower()))
    return regs


async def _check_contamination(text: str, articles: list) -> tuple[str, int]:
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
        fixed = await llm.invoke_nothink([{"role": "user", "content":
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
            max_tokens=1400, temperature=0.0, num_ctx=8192)
        fixed = _strip_correction_scaffold((fixed or "").strip())
    except Exception as e:
        logger.warning("[DeepResearch] contamination guard failed: %s", e)
        fixed = ""
    out = fixed if (fixed and ("**" in fixed or "##" in fixed)) else text
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


def _figure_support(text: str, articles: list) -> dict[str, tuple[int, set]]:
    """Deterministic cross-source support: for each magnitude figure in `text`,
    how many DISTINCT source hosts state that exact value (matched with comma/
    expanded/word variants). Pure value-agreement — no semantic judgment, so unlike
    the 9B's contradiction-guessing it can't be wrong about 'same quantity'."""
    host_corpus: dict[str, str] = {}
    for _t, url, body in articles:
        h = _host(url)
        host_corpus[h] = host_corpus.get(h, "") + " " + (body or "").lower()
    hc = {h: (c, c.replace(",", "")) for h, c in host_corpus.items()}
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
        hosts = {h for h, (c, cnc) in hc.items() if any(v in c or v in cnc for v in variants)}
        raw = m.group(0).strip()
        prev = out.get(raw, (0, set()))[1]
        out[raw] = (len(hosts | prev), hosts | prev)
    return out


async def _corroborate_numbers(text: str, articles: list) -> tuple[str, int]:
    """Cross-source numeric corroboration — RELIABLE version. The LLM-judgment
    approach (have the 9B decide which figures 'contradict for the same quantity')
    was built and tested and it does the judgment BACKWARDS — it missed a genuine
    50-vs-200 contradiction and falsely flagged the FortiBleed targeted-vs-compromised
    figures (different quantities). So we don't ask the model to judge: we
    DETERMINISTICALLY count how many independent sources state each figure's value
    and badge the ones ≥2 sources confirm with a ✓. Reliable, only adds a positive
    mark to genuinely-corroborated numbers (never a false accusation), and gives the
    reader real confidence calibration."""
    if not text or len(articles) < 2:
        return text, 0
    support = _figure_support(text, articles)
    confirmed = [f for f, (c, _h) in support.items() if c >= 2]
    if not confirmed:
        return text, 0
    badged = 0
    for fig in sorted(confirmed, key=len, reverse=True):  # longest first → no substring collisions
        idx = text.find(fig)
        if idx < 0:
            continue
        after = idx + len(fig)
        if text[after:after + 2] == " ✓":          # already badged
            continue
        text = text[:after] + " ✓" + text[after:]
        badged += 1
    if badged:
        logger.info("[DeepResearch] numeric corroboration: %d figure(s) confirmed by ≥2 sources", badged)
    return text, badged


async def _best_synthesis(prompt: str, evidence: str, *, n: int = 2,
                          temps: tuple = (0.2, 0.55), max_tokens: int = 1900) -> str:
    """Parallel best-of-N synthesis: generate N full candidate briefings (varied
    temperature → different framings), then an external grounded JUDGE picks the
    sharpest. This is the research-backed way to get more analytical depth from a
    small model — N independent shots + external selection, NOT self-refine (which
    degrades the 9B) and NOT decomposition (which proved lossy here). One judge call."""
    temps = list(temps)[:max(1, n)]
    raw = await asyncio.gather(*[
        llm.invoke_nothink([{"role": "user", "content": prompt}],
                           max_tokens=max_tokens, temperature=t, num_ctx=8192)
        for t in temps], return_exceptions=True)
    cands = [(c or "").strip() for c in raw if isinstance(c, str) and (c or "").strip()]
    if len(cands) <= 1:
        return cands[0] if cands else ""
    listing = "\n\n".join(f"=== CANDIDATE {i + 1} ===\n{c[:2600]}" for i, c in enumerate(cands))
    try:
        pick = await llm.invoke_nothink([{"role": "user", "content":
            "SOURCE FINDINGS and candidate briefings are below. Pick the SINGLE best candidate — the "
            "one that is (a) best grounded in the findings (nothing beyond them), (b) most specific "
            "(concrete numbers, names, dates), (c) sharpest on analysis and the 'so what', (d) covers "
            "the most distinct stories, (e) cleanest structure. Reply with ONLY the candidate number.\n\n"
            f"SOURCE FINDINGS:\n{evidence[:9000]}\n\n{listing}"}],
            max_tokens=6, temperature=0.0)
        m = re.search(r"\d+", pick or "")
        idx = (int(m.group()) - 1) if m else 0
    except Exception:
        idx = 0
    return cands[idx] if 0 <= idx < len(cands) else cands[0]


async def _learn_facts(topic: str, brief: str, findings: list, kg) -> int:
    """Bank grounded facts from the verified briefing — kept only if they trace to
    what was actually read (anti-hallucination), garbage-gated."""
    if kg is None or not brief or len(findings) < 1:
        return 0
    from app.core.kg import is_garbage_triple
    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": _FACT_PROMPT.format(topic=topic, evidence=brief[:4500])}],
            json_mode=True, json_prefix='[{', max_tokens=600)
        cands = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(cands, dict):
            cands = cands.get("facts") or cands.get("triples") or []
    except Exception:
        return 0
    if not isinstance(cands, list):
        return 0
    combined = " ".join(f.lower() for _, _, f in findings)
    src_texts = [f.lower() for _, _, f in findings]
    stored = 0
    for c in cands[:10]:
        if not isinstance(c, dict):
            continue
        s, p, o = (str(c.get("subject", "")).strip(), str(c.get("predicate", "")).strip(),
                   str(c.get("object", "")).strip())
        if not (s and p and o) or is_garbage_triple(s, p, o):
            continue
        st, ot = _key_terms(s), _key_terms(o)
        if not st or not ot:
            continue
        if not (any(t in combined for t in st) and any(t in combined for t in ot)):
            continue
        support = sum(1 for txt in src_texts
                      if any(t in txt for t in st) and any(t in txt for t in ot))
        try:
            if await kg.add_fact(s, p, o, confidence=min(0.9, 0.6 + 0.1 * max(1, support)),
                                 source="researched", provenance=f"deep_research:{topic[:60]}"):
                stored += 1
        except Exception:
            pass
    return stored


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
    evidence = "\n\n".join(f"[{t}] ({_host(u)})\n{f}" for t, u, f in findings)

    try:
        draft = await llm.invoke_nothink([{"role": "user", "content":
            f"Today is {today}. You READ {len(findings)} credible sources on '{subject}'. Findings below.\n\n"
            "Write a sharp intelligence briefing: the situation, key facts/numbers, what's genuinely new, "
            "and what it means. Cite outlets inline. Use ONLY information present in the findings — never "
            "invent a company, number, person, or event not present.\n\n" + evidence}],
            max_tokens=700, temperature=0.25)
    except Exception as e:
        logger.warning("[DeepResearch] draft failed: %s", e)
        draft = ""
    draft = (draft or "").strip()
    if not draft:
        return f"## 🔬 {label} — researched briefing\n_subject: {subject}_\n\n(synthesis unavailable)"

    # VERIFICATION PASS — strip any claim the findings don't support.
    try:
        final = await llm.invoke_nothink([{"role": "user", "content":
            "Below are SOURCE FINDINGS and a DRAFT briefing. Remove or correct EVERY statement in the "
            "draft that is not directly supported by the findings (invented numbers, names, events). "
            "Keep only supported claims; preserve structure. Output the corrected briefing only.\n\n"
            f"SOURCE FINDINGS:\n{evidence}\n\nDRAFT:\n{draft}"}],
            max_tokens=750, temperature=0.1)
        final = (final or "").strip() or draft
    except Exception:
        final = draft

    final = _strip_fake_citations(final, hosts)   # drop fabricated-attribution lines
    final, _ = await _ground_numbers(final, [b for _, _, b in articles])  # every figure traces to a source
    final, _ = await _ground_claims(final, [f"{t}\n{b}" for t, _u, b in articles])   # distinctive terms trace too (titles incl.)
    final, _ = await _check_contamination(final, articles)   # details trace to their CITED source, not a grafted one
    final, _ = await _corroborate_numbers(final, articles)   # ✓ badge figures ≥2 sources confirm
    final = _tidy_citations(final)   # strip $-wrapped citations + leaked title fragments
    learned = await _learn_facts(subject, final, findings, kg)
    logger.info("[DeepResearch] %s: read %d sources (%s), learned %d facts",
                label, len(findings), ", ".join(hosts[:6]), learned)

    header = (f"## 🔬 {label} — researched briefing\n"
              f"_subject: {subject}\nread {len(findings)} sources: {', '.join(hosts[:6])}"
              f"{' +more' if len(hosts) > 6 else ''} · {learned} facts learned · {today}_\n\n")
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
    subjects = await _focus_subjects(label, feed_key=feed_key or label, n=n_stories)
    logger.info("[DeepResearch] %s → %d stories: %s", label, len(subjects), subjects)

    articles = await _gather_overview(subjects, feed_key or label, read_target=18, browser_budget=10)
    # Relevance anchor = the DOMAIN (broad), NOT the seed subjects. The feeds
    # surface many on-domain stories beyond the seed subjects; gating findings on
    # the specific subjects made the 9B reject them as "irrelevant" (the bug that
    # forced the overview to keep falling back). Broad anchor keeps every
    # on-domain finding → real breadth. Subjects still seed the search + synthesis.
    findings = await _findings(articles, label) if articles else []

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

    hosts = sorted({_host(u) for _, u, _ in findings})
    evidence = "\n\n".join(f"[{t}] ({_host(u)})\n{f}" for t, u, f in findings)

    _syn_prompt = (
        f"Today is {today}. You READ {len(findings)} credible {label} sources. Their findings are "
        f"below — write the overview from THESE FINDINGS ONLY (do not add stories you remember but "
        f"did not read here).\n\n"
        "Write a thorough DOMAIN INTELLIGENCE OVERVIEW — this is the full picture, be substantive:\n"
        "**Lead Development** — the single most significant story: a full paragraph (5-8 sentences) "
        "with all the key facts, numbers, named players, what is genuinely new, and why it matters.\n"
        "**Secondary Developments** — EVERY other distinct story the findings support, each as its "
        "own bullet with 2-4 sentences of concrete specifics (numbers, names, dates). Cover as "
        "many distinct stories as the findings support — do NOT stop at two or three.\n"
        "**Connections & bottom line** — 2-3 sentences: the throughline across the stories and the "
        "single thing to watch next.\n"
        "EVERY sentence and bullet MUST cite its outlet inline like (cnbc.com). If you cannot cite "
        "a claim from the findings, OMIT it. Group findings into stories yourself. Use ONLY "
        "information present in the findings — never invent a company, number, person, or event.\n\n"
        + evidence)
    try:
        # best-of-N: two framings, judge picks the sharpest grounded one
        draft = await _best_synthesis(_syn_prompt, evidence, n=2, temps=(0.2, 0.55))
    except Exception as e:
        logger.warning("[DeepResearch] overview draft failed: %s", e)
        draft = ""
    draft = (draft or "").strip()
    if not draft:
        return f"## 🌐 {label} — domain overview\n_{today}_\n\n(synthesis unavailable)"

    # VERIFICATION PASS — ground every claim AND require a citation on each.
    try:
        final = await llm.invoke_nothink([{"role": "user", "content":
            "Below are SOURCE FINDINGS and a DRAFT domain overview. Do all of:\n"
            "1. Remove or correct EVERY statement not directly supported by the findings (invented "
            "numbers, names, events).\n"
            "2. DELETE any sentence or bullet that lacks an inline outlet citation like (cnbc.com) — "
            "an uncited claim is where fabrication hides; cut it.\n"
            "3. Keep all supported, cited claims, their specifics, and the structure.\n"
            "Output the corrected overview only.\n\n"
            f"SOURCE FINDINGS:\n{evidence}\n\nDRAFT:\n{draft}"}],
            max_tokens=1900, temperature=0.1, num_ctx=8192)
        final = (final or "").strip() or draft
    except Exception:
        final = draft

    final = _strip_fake_citations(final, hosts)   # drop fabricated-attribution lines
    final, _ = await _ground_numbers(final, [b for _, _, b in articles])  # every figure traces to a source
    final, _ = await _ground_claims(final, [f"{t}\n{b}" for t, _u, b in articles])   # distinctive terms trace too (titles incl.)
    final, _ = await _check_contamination(final, articles)   # details trace to their CITED source, not a grafted one
    final, _ = await _corroborate_numbers(final, articles)   # ✓ badge figures ≥2 sources confirm
    final = _tidy_citations(final)   # strip $-wrapped citations + leaked title fragments
    learned = await _learn_facts(label, final, findings, kg)
    logger.info("[DeepResearch] %s overview: read %d sources (%s), learned %d facts",
                label, len(findings), ", ".join(hosts[:6]), learned)

    header = (f"## 🌐 {label} — domain overview\n"
              f"_read {len(findings)} sources: {', '.join(hosts[:7])}"
              f"{' +more' if len(hosts) > 7 else ''} · {learned} facts learned · {today}_\n\n")
    return header + final
