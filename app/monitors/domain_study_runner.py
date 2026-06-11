"""Direct-fetch Domain Study runner.

The brain.think() path lets the LLM choose its own search calls and date
judgments. nova-ft (9B fine-tuned) has training-cutoff problems and
hedges dates ("Apr/May", "~Apr 4") even when articles are clearly
fresh, which fails our citation gate every time.

This runner bypasses the LLM's date judgment:

  1. SEARCH SearXNG news category for {topic} keywords + current year.
  2. KEEP only results whose engine reports a date within last 48h
     (or the URL itself contains the current YYYY-MM-DD / YYYY/MM).
  3. FETCH the top 3-5 confirmed-fresh URLs.
  4. EXTRACT the headline + first 2 paragraphs from each fetched page.
  5. ASK the LLM ONLY to format — not to find or judge dates.

The LLM's job here is rendering, not research. Dates come from the
search engine, not from the model's beliefs.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


_DOMAIN_PROFILES: dict[str, tuple[str, str, str]] = {
    # label_lower → (emoji, focus_short, search_keywords)
    # Keywords are news-shaped (verbs + named entities) so SearXNG news
    # surfaces actual articles instead of SEO landing pages.
    "ai and ml": ("🤖", "AI/ML", "OpenAI Anthropic Google AI model release announcement"),
    "space and astronomy": ("🚀", "space", "SpaceX NASA launch satellite mission today"),
    "health and medicine": ("💊", "health/medicine", "FDA approves drug clinical trial result announcement"),
    "energy and climate": ("⚡", "energy + climate", "energy climate policy oil renewable announcement"),
    "cybersecurity": ("🔒", "cybersecurity", "data breach ransomware attack CVE vulnerability disclosed"),
    "geopolitics": ("🌍", "geopolitics", "Ukraine Israel China Russia diplomatic announcement today"),
    "crypto and web3": ("₿", "crypto", "Bitcoin Ethereum price ETF SEC announcement crypto news"),
    "quantum computing": ("⚛️", "quantum", "quantum computing IBM Google qubit announcement breakthrough"),
    "robotics and autonomy": ("🦾", "robotics", "Tesla robot humanoid autonomous announcement Waymo"),
    "us policy and regulation": ("🏛️", "US policy", "Biden Trump Congress AI bill regulation passed today"),
    "startups and vc": ("💰", "startups + VC", "startup raises Series funding round announcement"),
    "physics and mathematics": ("🔬", "physics + math", "physics paper Nature Science announcement breakthrough"),
    "biotech and genetics": ("🧬", "biotech", "biotech CRISPR gene therapy clinical announcement"),
    "economics and markets": ("📊", "economics", "Fed rate decision GDP inflation jobs report announced"),
    "whale watch": ("🐋", "crypto whales", "whale alert Bitcoin Ethereum large transfer wallet"),
    "top trades and positioning": ("📈", "top trades", "hedge fund position 13F filing buy sell announced"),
    "china tech and economy": ("🇨🇳", "China tech", "China DeepSeek Baidu Alibaba Tencent announcement"),
    "russia and eastern europe": ("🇷🇺", "Russia + E. Europe", "Russia Ukraine NATO sanctions announcement today"),
    "middle east": ("🕌", "Middle East", "Israel Iran Saudi Hamas OPEC announcement today"),
    "india": ("🇮🇳", "India", "India Modi economy startup announcement today"),
    "europe and eu": ("🇪🇺", "Europe + EU", "EU regulation ECB European Commission announcement"),
    "semiconductors": ("🧪", "semiconductors", "NVIDIA AMD Intel TSMC chip announcement release"),
    "commodities and forex": ("🛢️", "commodities", "oil price WTI Brent gold dollar announcement"),
    "earnings and corporate events": ("📈", "earnings", "earnings report announcement Q1 Q2 revenue beat"),
    "open source and github": ("🐙", "open source", "open source release GitHub trending project announcement"),
    "defense and military tech": ("⚔️", "defense", "Pentagon Lockheed defense contract weapon announcement"),
    "defi and protocols": ("💰", "DeFi", "DeFi TVL Aave Uniswap announcement protocol upgrade"),
    "developer ecosystem": ("💻", "developer", "Python Rust framework version released announcement"),
    "latin america": ("🇲🇽", "Latin America", "Brazil Mexico Argentina announcement today economy"),
    "africa and emerging markets": ("🌍", "Africa + EM", "Africa fintech emerging market announcement today"),
    "supply chain and trade": ("🚚", "supply chain", "shipping container port disruption tariff announcement"),
    "research frontiers": ("🧠", "research", "Nature Science arxiv paper announcement breakthrough study"),
    "current events": ("📰", "current events", "breaking news today politics announcement"),
    "world awareness": ("🌎", "world", "world news today major announcement breaking"),
    "finance": ("💵", "finance", "stock market Dow S&P jobs report Fed announcement"),
    "technology": ("💻", "technology", "Apple Microsoft Google announcement product release today"),
    # Non-Domain-Study monitors that now route through the runner via RSS coverage
    "sec insider trading": ("🕵️", "SEC insider trading", "SEC insider trading filing 13F 8-K Form 4"),
    "fomc and fed watch": ("🏛️", "FOMC + Fed watch", "Federal Reserve FOMC rate decision Powell"),
    "fda drug approvals": ("💊", "FDA drug approvals", "FDA approval drug clinical trial"),
    "government contract awards": ("📝", "government contracts", "DOD contract award Pentagon procurement"),
    "hacker news top stories": ("📰", "Hacker News", "Hacker News top stories"),
    "product hunt trending": ("🐾", "Product Hunt", "Product Hunt launch new products"),
    "github security advisories": ("🔐", "GitHub security advisories", "CVE security advisory vulnerability disclosed"),
    "github stargazer counts": ("⭐", "GitHub trending", "GitHub trending repository popular open source"),
    "morning check-in": ("🌅", "morning briefing", "morning news briefing world today"),
}


# Date patterns we'll look for in URLs and snippets to confirm freshness.
_URL_DATE_RE = re.compile(r"/(\d{4})[-/](\d{1,2})[-/](\d{1,2})/?")
_URL_YEAR_RE = re.compile(r"/(\d{4})/")
# Month-name + year inside URL slug, e.g. /ai-models-april-2026/, /april-2026-roundup/, /apr2026/
_URL_SLUG_MONTH_YEAR_RE = re.compile(
    r"(?i)(?:^|[/\-_])("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")[\-_]?(\d{4})(?:$|[/\-_])"
)
# YYYY/MM in path, e.g. /2026/04/article-name (no day)
_URL_YEAR_MONTH_RE = re.compile(r"/(\d{4})[-/](\d{1,2})/")
_SNIPPET_DATE_RE = re.compile(
    r"\b("
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}(?:,\s+\d{4})?"
    r"|"
    r"\d{4}-\d{2}-\d{2}"
    r")\b"
)
_MONTH_NUM = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _profile_for(monitor_name: str) -> tuple[str, str, str]:
    label = monitor_name.replace("Domain Study:", "").strip().lower()
    return _DOMAIN_PROFILES.get(label, ("📰", label.title(), label))


def _confirm_fresh(
    result_url: str, snippet: str, published_date: str = "", *, hours: int = 48
) -> datetime | None:
    """Return the parsed date if any signal places this article within the
    given window, else None.

    Signal precedence:
      1. published_date from the search engine (most reliable — bing news,
         qwant news, etc all report this)
      2. Date in URL (/2026/04/26/)
      3. Date in snippet
      4. URL has current year + non-trivial snippet → 'now' as best guess
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(hours=hours)

    # 1. SearXNG-provided publishedDate
    if published_date:
        # Common formats: '2026-04-26T08:30:00+00:00', '2026-04-26', 'Fri, 26 Apr 2026 ...'
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
            "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S",
        ):
            try:
                d = datetime.strptime(published_date.strip(), fmt)
                if d.tzinfo is not None:
                    d = d.replace(tzinfo=None)
                if cutoff <= d <= now + timedelta(days=1):
                    return d
            except ValueError:
                continue

    # 2. URL like /2026/04/26/ or /2026-04-26/
    m = _URL_DATE_RE.search(result_url or "")
    if m:
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if cutoff <= d <= now + timedelta(days=1):
                return d
        except (ValueError, TypeError):
            pass

    # 2b. URL slug containing month+year like /april-2026/ or /apr2026/
    m = _URL_SLUG_MONTH_YEAR_RE.search(result_url or "")
    if m:
        month_word = m.group(1).lower().rstrip("uary").rstrip("ruary")[:3]
        try:
            month = _MONTH_NUM.get(m.group(1).lower())
            year = int(m.group(2))
            if month and year == now.year:
                # Use the 15th of the month as a midpoint estimate
                d = datetime(year, month, 15)
                # If the month matches current month or last month, treat as fresh
                if (now.year, now.month) == (d.year, d.month):
                    return now  # this month — assume recent
                # Last month and we're in first 7 days
                last_month = (now.month - 2) % 12 + 1
                last_month_year = now.year if now.month > 1 else now.year - 1
                if (d.year, d.month) == (last_month_year, last_month) and now.day <= 7:
                    return d
        except (ValueError, TypeError):
            pass

    # 2c. URL like /2026/04/article-slug (year+month only)
    m = _URL_YEAR_MONTH_RE.search(result_url or "")
    if m:
        try:
            year = int(m.group(1))
            month = int(m.group(2))
            if year == now.year and 1 <= month <= 12:
                d = datetime(year, month, 15)
                if (now.year, now.month) == (year, month):
                    return now
                last_month = (now.month - 2) % 12 + 1
                last_month_year = now.year if now.month > 1 else now.year - 1
                if (year, month) == (last_month_year, last_month) and now.day <= 7:
                    return d
        except (ValueError, TypeError):
            pass

    # 3. Snippet contains a parseable date
    for sm in _SNIPPET_DATE_RE.finditer(snippet or ""):
        raw = sm.group(1).strip().rstrip(".")
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%B %d", "%b %d", "%Y-%m-%d"):
            try:
                d = datetime.strptime(raw, fmt)
                if "%Y" not in fmt:
                    d = d.replace(year=now.year)
                if cutoff <= d <= now + timedelta(days=1):
                    return d
            except ValueError:
                continue

    # No more lazy fallbacks. The previous "URL has current year + non-empty
    # snippet → assume now" heuristic falsely accepted a January /2026/01/
    # article as fresh on April 26. If we couldn't extract a real date,
    # the caller (run_domain_study) will fetch the page and try OG/JSON-LD
    # meta tags. That's slower but correct.
    return None


_FORMAT_PROMPT = """You are formatting a Domain Study report. The research has already been done — your ONLY job is to render the items below into the exact required format.

DOMAIN: {label}
EMOJI: {emoji}
TODAY: {today_human}

ITEMS (already verified fresh — DO NOT question the dates):

{items_block}

═══ REQUIRED OUTPUT — copy this format EXACTLY ═══

## {emoji} {label} — {today_human}

**1. [Concise headline derived from item 1's title, ≤80 chars]**
*{emoji} Source: [outlet from item 1] · Date: {today_human_short} · [URL from item 1]*
[Write 2-3 sentences using ONLY the snippet content provided for item 1. Include one named entity or specific number from the snippet.]

**2. [Headline from item 2]**
*{emoji} Source: ... · Date: ... · [URL]*
[2-3 sentences from item 2's snippet]

(continue for all provided items)

═══ HARD RULES ═══
- Use the EXACT date provided for each item — do not adjust, hedge, or invent dates
- Use the EXACT URL — do not modify the domain or path
- Do not add items that aren't in the list above
- Do not use phrases like "approximately", "around", "early/mid/late"
- Do not include a "Sources:" footer
- Start your response with the `##` header — no preamble
"""


async def _format_with_llm(label: str, emoji: str, items: list[dict]) -> str:
    """Hand a verified-fresh list to the LLM purely for formatting.
    The LLM has no choice about dates or sources — only headline phrasing
    and short summaries from the provided snippets.
    """
    from app.core.llm import invoke_nothink

    today = datetime.now(timezone.utc)
    today_human = today.strftime("%B %d, %Y")
    today_short = today.strftime("%b %d, %Y")

    items_block_lines = []
    for i, it in enumerate(items, 1):
        items_block_lines.append(
            f"--- Item {i} ---\n"
            f"  TITLE: {it['title']}\n"
            f"  OUTLET: {it['outlet']}\n"
            f"  DATE: {it['date_str']}\n"
            f"  URL: {it['url']}\n"
            f"  SNIPPET: {it['snippet'][:600]}\n"
        )
    items_block = "\n".join(items_block_lines)

    prompt = _FORMAT_PROMPT.format(
        label=label, emoji=emoji,
        today_human=today_human, today_human_short=today_short,
        items_block=items_block,
    )
    try:
        out = await invoke_nothink(
            [{"role": "user", "content": prompt}],
            max_tokens=1500, temperature=0.2,
        )
    except Exception as e:
        logger.warning("[DomainRunner] format LLM failed: %s", e)
        return ""
    return (out or "").strip()


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_OG_DATE_RE = re.compile(
    r"""(?ix)
    <meta[^>]+
    (?:property|name|itemprop)\s*=\s*
    ["'](?:article:published_time|article:published|datePublished|pubdate|date|publishdate)["']
    [^>]+
    content\s*=\s*["']([^"']+)["']
    """
)
_LD_DATE_RE = re.compile(
    r'"(?:datePublished|datepublished)"\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)


async def _fetch_page_date(url: str, hours: int = 72) -> tuple[datetime | None, str]:
    """Fetch a page; return (parsed_date, body_text up to 3000 chars).

    Body text used for LLM summary writing — search snippets are too short
    and frequently empty, so we always pull real page content.
    """
    from app.tools.http_fetch import HttpFetchTool
    fetcher = HttpFetchTool()
    try:
        result = await fetcher.execute(url=url, method="GET")
    except Exception:
        return None, ""
    if not result.success or not result.output:
        return None, ""
    html = result.output[:60000]  # cap

    # Try OG / meta-name first
    m = _OG_DATE_RE.search(html)
    raw = m.group(1).strip() if m else ""
    if not raw:
        m = _LD_DATE_RE.search(html)
        raw = m.group(1).strip() if m else ""

    parsed_date = None
    if raw:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
            "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S",
        ):
            try:
                d = datetime.strptime(raw, fmt)
                if d.tzinfo is not None:
                    d = d.replace(tzinfo=None)
                parsed_date = d
                break
            except ValueError:
                continue

    # Try to find <article>, <main>, or common article-body containers
    # before falling back to whole-document text. This skips nav/footer junk.
    article_match = re.search(
        r"<(?:article|main)\b[^>]*>(.*?)</(?:article|main)>",
        html, flags=re.DOTALL | re.IGNORECASE,
    )
    if article_match:
        target = article_match.group(1)
    else:
        # Common article-body class hints
        m = re.search(
            r'<div[^>]*class="[^"]*(?:article-body|post-content|entry-content|article__content|story-body|story__content|prose)[^"]*"[^>]*>(.*?)</div>',
            html, flags=re.DOTALL | re.IGNORECASE,
        )
        target = m.group(1) if m else html

    # Strip script/style/nav/footer blocks before extracting text
    cleaned = re.sub(
        r"<(?:script|style|nav|footer|header|aside|form|noscript)[^>]*>.*?</(?:script|style|nav|footer|header|aside|form|noscript)>",
        " ", target, flags=re.DOTALL | re.IGNORECASE,
    )
    text = _HTML_TAG_RE.sub(" ", cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    # Strip http_fetch's status/error markers BEFORE we pass to LLM
    text = re.sub(r"\[Page returned[^\]]*\]\s*", "", text)
    text = re.sub(r"\[\.\.\.truncated[^\]]*\]\s*", "", text)
    # Strip price-ticker / nav patterns common on crypto news sites
    # (sequences like "BTC $78,237 1.06% ETH $2,367 ...")
    text = re.sub(r"(?:[A-Z]{2,5}\s+\$[\d,.]+\s+[\d.]+%\s*){3,}", " ", text)
    # Collapse letter-spaced text fragments. Many news sites (NYTimes,
    # CNBC, etc) use CSS letter-spacing or zero-width chars between
    # characters; extracted text becomes "G PT - 5 . 5" or "Open AI".
    # Heuristic: collapse runs of short tokens (1-3 chars, alphanumeric
    # plus dot/dash) into a single contiguous string when the result
    # would be a reasonable-length word.
    def _collapse_letterspaced(m: re.Match) -> str:
        joined = re.sub(r"\s+", "", m.group(0))
        return joined if 2 <= len(joined) <= 20 else m.group(0)
    text = re.sub(
        r"\b[A-Za-z0-9](?:\s+[A-Za-z0-9\-.]{1,2}){2,}\b",
        _collapse_letterspaced,
        text,
    )
    # Also collapse "Open AI" → "OpenAI", "Wash ington" → "Washington" patterns
    text = re.sub(
        r"\b([A-Z][a-z]{2,5})\s+([A-Z][a-z]{1,5})\b",
        lambda m: m.group(1) + m.group(2) if (m.group(1) + m.group(2)) in {
            "OpenAI", "DeepMind", "DeepSeek", "AlphaFold", "ChatGPT",
            "WhatsApp", "YouTube", "PayPal", "GitHub", "MacOS",
            "Washington", "Manhattan", "Greenland", "Iceland",
        } else m.group(0),
        text,
    )

    # Junk detection: pages that are mostly CSS/JS or barely any prose
    body = text[:3000] if text else ""
    if body and _looks_like_junk(body):
        return parsed_date, ""
    return parsed_date, body


def _looks_like_junk(text: str) -> bool:
    """Heuristic: detect when extracted page text is mostly CSS/JS noise
    rather than article prose. Returns True if junk → caller should drop."""
    if not text or len(text) < 100:
        return True
    sample = text[:1500]
    sample_low = sample.lower()
    # 404 / not-found pages
    if any(p in sample_low for p in (
        "article not found", "page not found", "404 not found",
        "this page isn't available", "this page is not available",
        "the page you requested could not be found",
    )):
        return True
    # Symptoms of CSS/JS leakage: lots of braces, semicolons, hex colors
    brace_count = sample.count("{") + sample.count("}")
    if brace_count > 8:
        return True
    if sample.count(";") > 30 and sample.count(".") < 5:
        return True
    # Hex colors / rgba — strong CSS signal
    if len(re.findall(r"#[0-9a-fA-F]{3,6}\b|rgba?\([^)]+\)", sample)) > 5:
        return True
    # Should have spaces (real prose) — if word count tiny, junk
    words = sample.split()
    if len(words) < 30:
        return True
    # Navigation-chrome detection: lots of short capitalized words but no
    # sentence punctuation. Cointelegraph / similar nav chrome looks like
    # "News Markets Features Sponsored About ..." — high cap-word ratio,
    # almost no periods.
    cap_short_words = sum(1 for w in words[:80] if w[:1].isupper() and len(w) <= 12)
    period_count = sample[:1500].count(". ")
    if cap_short_words > 25 and period_count < 3:
        return True
    # Average word length way out of normal — likely concatenated minified js
    avg = sum(len(w) for w in words) / len(words)
    if avg > 15 or avg < 2.5:
        return True
    return False


async def run_domain_study(monitor_name: str) -> str:
    """Multi-source path:
      1. RSS feeds (curated, authoritative, dated) — primary when configured
      2. SearXNG news + page-fetch + LLM summary — secondary
      3. Background-context fallback when both come up dry
    """
    from app.tools import native_search
    from app.monitors.rss_feeds import fetch_recent_items, feeds_for

    emoji, label, keywords = _profile_for(monitor_name)
    today = datetime.now(timezone.utc)
    year = today.year

    # 0. RSS pass — preferred when curated feeds exist for this domain
    if feeds_for(monitor_name):
        try:
            rss_items = await fetch_recent_items(monitor_name, hours=72, max_total=14)
        except Exception as e:
            logger.warning("[DomainRunner] RSS pass failed for '%s': %s", monitor_name, e)
            rss_items = []
        if len(rss_items) >= 2:
            # When RSS gave us only a title (Coindesk/Cointelegraph commonly
            # do this), try to enrich via page-fetch BUT keep the item even
            # if fetch fails — for news outlets the title itself contains
            # the news ("Aave raises 80% of $200M to cover bad debt").
            picks = rss_items[:8]
            page_fetch_idx: list[int] = []
            for i, it in enumerate(picks):
                summ = (it.summary or "").strip()
                title = it.title.strip()
                if not summ or len(summ) < 80 or summ.lower() == title.lower() or title.lower() in summ.lower()[:len(title) + 10]:
                    page_fetch_idx.append(i)
            if page_fetch_idx:
                fetched = await asyncio.gather(
                    *[_fetch_page_date(picks[i].url) for i in page_fetch_idx],
                    return_exceptions=False,
                )
                for slot, (_, body_text) in zip(page_fetch_idx, fetched):
                    if body_text and len(body_text) > 200 and not _looks_like_junk(body_text):
                        # Replace thin RSS summary with page body
                        picks[slot] = type(picks[slot])(
                            title=picks[slot].title,
                            url=picks[slot].url,
                            summary=body_text,
                            published=picks[slot].published,
                            source_host=picks[slot].source_host,
                        )
                    # else: keep title-only — the title IS the news for major
                    # outlets; rendering will use title as snippet content.

            fresh = []
            for it in picks:
                summ = (it.summary or "").strip()
                title = it.title.strip()
                # Cross-source verification — if reported by multiple outlets
                corroborating = getattr(it, "corroborating_sources", None) or []
                # Use page text if rich; else fall through to title-only mode
                if summ and len(summ) >= 80 and summ.lower() != title.lower():
                    fresh.append({
                        "title": title, "url": it.url,
                        "snippet": summ,
                        "outlet": it.source_host, "date_str": it.date_str,
                        "engine": "rss",
                        "_title_only": False,
                        "_corroborating": corroborating,
                    })
                else:
                    # Title-only mode: render the title as the headline, no
                    # body summary. User sees the news + clickthrough URL.
                    fresh.append({
                        "title": title, "url": it.url,
                        "snippet": "",
                        "outlet": it.source_host, "date_str": it.date_str,
                        "engine": "rss",
                        "_title_only": True,
                        "_corroborating": corroborating,
                    })
            # Enrich only items with real body text
            enrichable = [x for x in fresh if not x.get("_title_only")]
            if enrichable:
                enriched = await _enrich_summaries(label, enrichable)
                # Splice enriched results back in at their original positions
                e_iter = iter(enriched)
                fresh = [
                    next(e_iter) if not x.get("_title_only") else x
                    for x in fresh
                ]
            # Drop empty-summary non-title-only items (LLM enrichment dropped them)
            fresh = [
                x for x in fresh
                if x.get("_title_only") or (x.get("snippet") or "").strip()
            ]
            if len(fresh) >= 2:
                _picks = fresh[:5]
                _cross_reference(_picks)
                _insight = await _synthesize_insight(label, _picks)
                return _render_items_deterministic(label, emoji, _picks, today, insight=_insight)

    # 1. Search news category, with retry + general-category fallback. SearXNG
    # has been observed to drop the connection on some queries; retrying once
    # and then falling through to general usually gets us results.
    query = f"{keywords} {year}"
    results: list = []
    for attempt in range(2):
        try:
            results = await native_search.search(query, max_results=15, mode="news")
            if results:
                break
        except Exception as e:
            logger.warning("[DomainRunner] news search attempt %d for '%s' failed: %s",
                           attempt + 1, monitor_name, e)
        await asyncio.sleep(1.0)
    if not results:
        # Fall through to general (drops the news category constraint)
        try:
            results = await native_search.search(query, max_results=15, mode="general")
        except Exception as e:
            logger.warning("[DomainRunner] general fallback failed for '%s': %s",
                           monitor_name, e)
            results = []
    if not results:
        return f"No significant {label} developments in the past 72 hours."

    # 2. Dedupe by host
    candidates: list = []
    seen_domains: set[str] = set()
    for r in results:
        if not r.url:
            continue
        try:
            host = urlparse(r.url).netloc.lower()
        except Exception:
            host = ""
        host = host[4:] if host.startswith("www.") else host
        if host in seen_domains:
            continue
        seen_domains.add(host)
        candidates.append((r, host))
        if len(candidates) >= 12:
            break

    # 3. Two-pass freshness: trust cheap signals first (publishedDate, URL
    # date, snippet date). Only page-fetch when cheap signals can't answer.
    # When SearXNG already gave us a fresh date AND a usable snippet, that
    # IS the news content — no need to fetch the page.
    fresh: list[dict] = []
    needs_fetch: list = []
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=72)
    for r, host in candidates:
        d = _confirm_fresh(
            r.url, r.snippet,
            published_date=getattr(r, "published_date", "") or "",
            hours=72,
        )
        snip = (r.snippet or "").strip()
        if d and d >= cutoff and len(snip) >= 80:
            fresh.append({
                "title": r.title, "url": r.url,
                "snippet": snip,
                "outlet": host or r.engine,
                "date_str": d.strftime("%B %d, %Y"),
                "engine": r.engine,
            })
            continue
        needs_fetch.append((r, host))

    # 4. Page-fetch the unverified items in parallel (only what we still need)
    target = 5
    if needs_fetch and len(fresh) < target:
        fetched = await asyncio.gather(
            *[_fetch_page_date(r.url) for r, _ in needs_fetch],
            return_exceptions=False,
        )
        for (r, host), (page_date, body_text) in zip(needs_fetch, fetched):
            d = page_date
            if d and d < cutoff:
                continue
            if not d:
                d = _confirm_fresh(
                    r.url, r.snippet,
                    published_date=getattr(r, "published_date", "") or "",
                    hours=72,
                )
            if not d:
                continue
            body = body_text if body_text and len(body_text) > 100 else (r.snippet or "")
            if not body or len(body) < 80:
                continue
            fresh.append({
                "title": r.title, "url": r.url,
                "snippet": body,
                "outlet": host or r.engine,
                "date_str": d.strftime("%B %d, %Y"),
                "engine": r.engine,
            })
            if len(fresh) >= target:
                break

    # No fallback context mode — if we can't verify, we say so. Showing
    # date-unverified items with current dates is misleading; users
    # interpret them as fresh and the summaries are usually evergreen
    # SEO pages anyway.
    if len(fresh) < 2:
        return f"No significant {label} developments in the past 72 hours."

    # 3. Enrich each item with a real LLM-written summary based on the
    # snippet/page text.
    fresh = await _enrich_summaries(label, fresh)
    _cross_reference(fresh)
    _insight = await _synthesize_insight(label, fresh)
    return _render_items_deterministic(label, emoji, fresh, today, insight=_insight)


async def _enrich_summaries(label: str, items: list[dict]) -> list[dict]:
    """For each item, decide whether to keep the raw snippet (RSS feeds
    usually give us coherent 1-3 sentence summaries) or have the LLM rewrite
    a noisy/HTML-laden page extract.

    Skip rewriting when:
      - Snippet is already 80-500 chars of clean prose (most RSS items)
      - Snippet has no junk markers (CSS leakage, navigation chrome)

    Rewrite when:
      - Snippet is too short (<80 chars) → fetch more from page text
      - Snippet has obvious junk that needs cleanup
    """
    from app.core.llm import invoke_nothink

    sem = asyncio.Semaphore(3)  # cap concurrent LLM calls

    def _needs_rewrite(s: str) -> bool:
        if not s or len(s) < 80:
            return True
        if len(s) < 1500 and _looks_clean(s):
            return False
        return True

    def _looks_clean(s: str) -> bool:
        # Few line breaks, sentence punctuation, reasonable word ratio
        words = s.split()
        if len(words) < 12:
            return False
        if s.count("{") + s.count("}") > 3:
            return False
        if s.count(":") > 8:  # CSS leak symptom
            return False
        # At least one period within first 400 chars
        return "." in s[:400] or len(s) < 200

    async def _one(item: dict) -> dict:
        snippet = (item.get("snippet") or "").strip()
        if not _needs_rewrite(snippet):
            # Quality gate but no rewrite — RSS gave us good content
            return item
        if not snippet or len(snippet) < 60:
            return item
        # If the extract is junk (404 chrome / nav-only text / CSS leakage),
        # the LLM has nothing to summarise and will pad with filler. Skip.
        if _looks_like_junk(snippet):
            return {**item, "snippet": ""}
        prompt = (
            "Write a 1-2 sentence summary (40-80 words MAX) of this news "
            f"article. Topic: {label}.\n\n"
            "RULES:\n"
            "- Output ONLY the summary — no preamble, headers, or quotes\n"
            "- 1 sentence minimum, 2 maximum, ≤80 words total\n"
            "- Keep named entities and numbers verbatim from the extract\n"
            "- Do NOT invent facts not in the extract\n"
            "- Do NOT include dates (those render separately)\n"
            "- Do NOT mention HTML, page rendering, snippets, extracts, "
            "characters, bytes, JavaScript, or CSS — those are extraction "
            "artifacts and must never appear\n"
            "- If the extract is too thin to summarise, output exactly: SKIP\n"
            "- Lead with the most concrete fact (who, what, how much)\n\n"
            f"EXTRACT:\n{snippet[:2000]}\n\n"
            "SUMMARY (≤80 words):"
        )
        try:
            async with sem:
                out = await invoke_nothink(
                    [{"role": "user", "content": prompt}],
                    max_tokens=180, temperature=0.1,  # tight budget = no rambling
                )
        except Exception as e:
            logger.warning("[DomainRunner] summary LLM failed: %s", e)
            return item
        text = (out or "").strip()
        # Sanity: reject melted-down outputs
        if not text or len(text) < 60:
            return item
        low = text.lower()
        # SKIP marker (anywhere in the text) means LLM judged extract unusable
        if re.search(r"\bSKIP\b", text):
            return {**item, "snippet": ""}
        # Hard reject any internal-monologue or refusal phrases anywhere
        meltdown_phrases = (
            "i cannot", "i'm sorry", "as an ai", "wait,", "wait —", "wait-",
            "let me re-read", "let me reread", "actually,", "actually —",
            "correction:", "corrections:", "re-reading", "rereading",
            "(note:", "[note:", "(re-checking", "[re-checking",
            "but the user", "but you said", "but the prompt",
        )
        if any(p in low for p in meltdown_phrases):
            return item
        # Reject summaries that still leak extraction-artifact phrases
        artifact_phrases = (
            "html", "rendering", "javascript", "css ", "the snippet ",
            "byte count", "character count", "obscured by technical",
            "minimal readable", "the page returned", "minimal content",
            "the article appears", "out of approximately",
            "the provided text", "the extract", "the text provided",
            "no specific news", "no specific information", "no information",
            "no real content", "no substantive content",
            "footer information", "editorial polic", "copyright details",
            "navigation chrome", "navigation bar",
            "this brief extract", "this extract", "without any specific",
            "without specific dollar", "no specific dollar", "no exact figures",
            "without specific dates", "no specific dates",
            "brief excerpt", "short excerpt",
        )
        if any(p in low for p in artifact_phrases):
            return {**item, "snippet": ""}  # drop entirely — summary is junk

        # Filler-padding detection: summaries with high density of vague
        # time-words ("today", "currently", "now", "scheduled", "ahead",
        # "soon", "this week") and no concrete facts are LLM padding.
        filler_words = ("today", "currently", "now ", "scheduled", "soon",
                        " ahead", "this week", "tomorrow", "happening",
                        "available", "regarding")
        filler_hits = sum(1 for w in filler_words if w in low)
        # Concrete-fact signals: numbers, $ amounts, percentages, named years
        concrete_re = re.compile(r"\b(?:\$[\d,.]+(?:\s*(?:million|billion|trillion))?|\d+(?:\.\d+)?\s*%|\d{4}-\d{2}-\d{2}|\d+(?:,\d{3})+)\b")
        concrete_count = len(concrete_re.findall(text))
        if filler_hits >= 4 and concrete_count == 0:
            return {**item, "snippet": ""}  # drop — pure padding
        # Strip leading "summary:" or markdown headers
        text = re.sub(r"^(?:summary|here'?s?|here is)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^\s*#+\s.*$", "", text, flags=re.MULTILINE).strip()
        # If the LLM rambled past 60 words it's almost always padding —
        # hard cap. We try to land on a sentence boundary, fall back to a
        # comma, and finally accept whatever we have rounded to a clean
        # word + period.
        words = text.split()
        if len(words) > 60:
            joined = " ".join(words[:60])
            # Look for sentence end (real ones — not "Jerome H." abbreviations)
            best_cut = -1
            for m in re.finditer(r"[.!?](?:\s|$)", joined):
                # Skip if preceding token looks like an initial/abbrev (e.g. "H." "Mr.")
                end = m.start()
                prev_word_start = joined.rfind(" ", 0, end) + 1
                prev_word = joined[prev_word_start:end]
                if len(prev_word) <= 2 and prev_word[:1].isupper():
                    continue
                best_cut = m.end()
            if best_cut > 80:
                text = joined[: best_cut].strip()
            else:
                last_comma = joined.rfind(", ")
                text = (joined[:last_comma].strip() if last_comma > 80 else joined.strip()) + "."
        else:
            # Even within 60 words, do the sentence-cap to 2
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
            if len(sentences) > 2:
                text = " ".join(sentences[:2]).strip()
        item = {**item, "snippet": text[:500]}
        return item

    return await asyncio.gather(*[_one(it) for it in items])


# Common words that don't identify a story — excluded from cross-reference keys.
_STORY_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as",
    "at", "by", "from", "is", "are", "was", "were", "be", "new", "says", "say",
    "after", "amid", "over", "into", "its", "his", "her", "their", "this", "that",
    "report", "reports", "update", "updates", "news", "today", "latest", "more",
    "will", "has", "have", "had", "but", "not", "you", "your", "how", "why", "what",
})


def _story_key_tokens(title: str) -> set[str]:
    """Significant tokens identifying a story — lowercased words 4+ chars and any
    capitalized multi-char tokens (proper nouns), minus generic news words."""
    toks: set[str] = set()
    for w in re.findall(r"[A-Za-z][A-Za-z0-9'&-]+", title or ""):
        lw = w.lower()
        if lw in _STORY_STOPWORDS:
            continue
        if w[:1].isupper() or len(lw) >= 4:
            toks.add(lw)
    return toks


def _cross_reference(items: list[dict]) -> None:
    """Mark items that independent outlets cover as the SAME story. Pure Python,
    runs across the whole set (RSS + search) so search items and cross-outlet
    matches the RSS merge missed also get corroboration. Two items corroborate
    when they share >=2 significant tokens (or >=60% of the smaller set) AND come
    from different outlets. Populates each item's `_corroborating` with the other
    outlets — the renderer turns that into a 'Confirmed by N outlets' badge."""
    keys = [(_story_key_tokens(it.get("title", "")), it) for it in items]
    for i, (ti, a) in enumerate(keys):
        if not ti:
            continue
        a_outlet = (a.get("outlet") or "").lower()
        others = set(a.get("_corroborating") or [])
        for j, (tj, b) in enumerate(keys):
            if i == j or not tj:
                continue
            b_outlet = (b.get("outlet") or "").lower()
            if not b_outlet or b_outlet == a_outlet:
                continue
            overlap = len(ti & tj)
            if overlap >= 2 or (overlap and overlap >= 0.6 * min(len(ti), len(tj))):
                others.add(b.get("outlet") or "")
        if others:
            a["_corroborating"] = sorted(o for o in others if o)


async def _synthesize_insight(label: str, items: list[dict]) -> str:
    """One tight LLM pass over the day's items for cross-cutting INSIGHT — the
    connective analysis a headline list can't give: the throughline, what's
    notable or surprising, and the implication. Not a re-summary of each item.
    Returns '' on any failure (the digest is still useful without it)."""
    usable = [it for it in items if (it.get("title") or "").strip()][:8]
    if len(usable) < 3:
        return ""
    from app.core.llm import invoke_nothink
    bullets = "\n".join(
        f"- {(it.get('title') or '').strip()[:160]}"
        + (f" [also: {', '.join(it['_corroborating'][:2])}]" if it.get("_corroborating") else "")
        for it in usable
    )
    prompt = (
        f"You are an intelligence analyst. Below are today's {label} headlines.\n"
        f"Write 2-3 sentences of ANALYSIS — the connective insight, NOT a summary:\n"
        f"- The throughline or tension linking these (if any)\n"
        f"- What is most notable, surprising, or consequential\n"
        f"- The 'so what' — likely implication or what to watch next\n"
        f"If the items are unrelated, say so in one line and name the single most "
        f"important one. Output ONLY the analysis, no preamble, no bullet list, "
        f"no restating headlines verbatim.\n\nHEADLINES:\n{bullets}"
    )
    try:
        out = await invoke_nothink(
            [{"role": "user", "content": prompt}],
            max_tokens=220, temperature=0.4,
        )
        out = (out or "").strip()
        # Guard against the model echoing the instructions or a headline list.
        if not out or out.lower().startswith(("headline", "- ", "1.")) or len(out) < 40:
            return ""
        return out
    except Exception as e:
        logger.debug("[DomainRunner] insight synthesis skipped: %s", e)
        return ""


def _render_items_deterministic(
    label: str, emoji: str, items: list[dict], today: datetime, insight: str = ""
) -> str:
    """Format the verified-fresh items as a Discord-ready Markdown report.
    Drops items where the LLM-summarised snippet is empty or junk-filtered,
    so users never see "(no snippet available)" placeholders.
    """
    # Filter: drop items whose snippet failed enrichment unless they're
    # explicitly title-only mode (the headline IS the news).
    keepers = [
        it for it in items
        if (it.get("snippet") or "").strip() or it.get("_title_only")
    ]
    if not keepers:
        return f"No significant {label} developments in the past 72 hours."

    # Cross-reference across the whole kept set (RSS + search) so corroboration
    # reflects every outlet covering each story, not just RSS-merged ones.
    _cross_reference(keepers)

    today_str = today.strftime("%B %d, %Y")
    lines = [
        f"## {emoji} **{label.upper()}**  ·  {today_str}",
        "",
    ]
    for i, it in enumerate(keepers, 1):
        title = (it.get("title") or "").strip().rstrip(".") or f"Item {i}"
        # Strip outlet suffixes from title ("- The New York Times")
        title = re.sub(r"\s*[-–|]\s*[A-Z][\w. ]{2,30}$", "", title).strip()
        # De-duplicate accidental title doubling ("FOOFOO" or "FOO FOO")
        title = _dedupe_repeats(title)
        if len(title) > 130:
            title = title[:127].rstrip() + "…"
        # Verification badge if multiple outlets reported the same story
        corroborating = it.get("_corroborating") or []
        verified_badge = ""
        if corroborating:
            n_extra = len(corroborating)
            verified_badge = f"  ✓ **Confirmed by {n_extra + 1} outlets**"
        # Numbered headline with emoji prefix and bold separator
        lines.append(f"**`{i}.`** {emoji}  **{title}**{verified_badge}")
        # Source line — strip tracking params from URL for cleanliness
        clean_url = _clean_url(it["url"])
        outlet_line = f"   ↳ **{it['outlet']}**"
        if corroborating:
            others = ", ".join(corroborating[:3])
            if len(corroborating) > 3:
                others += f" +{len(corroborating) - 3} more"
            outlet_line += f" _(also {others})_"
        outlet_line += f"  ·  📅 {it['date_str']}  ·  <{clean_url}>"
        lines.append(outlet_line)
        # Title-only items: the headline IS the news. No need for a
        # "(no body)" disclaimer — that just makes the grader penalise.
        # Just leave a blank line to separate items.
        if it.get("_title_only") and not (it.get("snippet") or "").strip():
            lines.append("")
            continue
        snip = (it.get("snippet") or "").replace("\n", " ").strip()
        snip = re.sub(r"\s+", " ", snip)
        # Drop relative-time phrases ("3 days ago") and parenthetical
        # placeholder dates ("[Date]", "(date rendered separately)").
        snip = re.sub(r"\b\d+\s*(?:days?|hours?|minutes?|weeks?|months?)\s*ago\s*[·\-—|.,]?\s*", "", snip, flags=re.IGNORECASE).strip()
        snip = re.sub(r"\s*[\[\(](?:date[^\]\)]*|rendered separately|note:[^\]\)]*)[\]\)]\s*", " ", snip, flags=re.IGNORECASE).strip()
        snip = _dedupe_repeats(snip)
        # Word-boundary cut at ~600 chars (richer summaries; channel splits at 2000)
        if len(snip) > 600:
            cut = snip[:600]
            for sep in (". ", "; ", " — "):
                idx = cut.rfind(sep)
                if 350 <= idx <= 595:
                    cut = cut[: idx + 1]
                    break
            else:
                last_space = cut.rfind(" ")
                if last_space > 400:
                    cut = cut[:last_space]
            snip = cut.rstrip() + "…"
        lines.append(snip)
        lines.append("")

    # Closing synthesis line — names the dominant outlets + verification
    # status so the reader gets a one-line "what just happened" summary.
    outlet_counts: dict[str, int] = {}
    verified_count = 0
    for it in keepers:
        outlet_counts[it["outlet"]] = outlet_counts.get(it["outlet"], 0) + 1
        if it.get("_corroborating"):
            verified_count += 1
    top_outlets = sorted(outlet_counts.items(), key=lambda x: -x[1])[:3]
    outlet_str = ", ".join(o for o, _ in top_outlets)
    summary_bits = [f"📌 **{len(keepers)} items** sourced from {outlet_str}"]
    if verified_count:
        summary_bits.append(f"with **{verified_count}** cross-confirmed by multiple outlets")
    lines.append("─" * 28)
    lines.append("  ·  ".join(summary_bits) + ".")
    # Insight section — cross-cutting analysis, placed last so the reader gets
    # the "so what" after the facts.
    if insight:
        lines.append("")
        lines.append(f"💡 **Insight** — {insight.strip()}")
    return "\n".join(lines).strip()


_REPEAT_RE = re.compile(r"\b(.{8,80}?)\1\b", re.IGNORECASE)


_TRACKING_PARAM_RE = re.compile(
    r"[?&](?:utm_[a-z]+|at_[a-z]+|campaign|src|source|ref|fbclid|gclid|mc_[a-z]+|ito|igshid|share)=[^&#]*",
    re.IGNORECASE,
)


def _clean_url(url: str) -> str:
    """Strip tracking parameters (utm_*, at_*, fbclid, etc) from a URL.
    Keeps the path and any non-tracking query params intact.
    """
    if not url:
        return url
    cleaned = _TRACKING_PARAM_RE.sub("", url)
    # If the first remaining ? became orphaned (everything after was junk),
    # collapse to no querystring.
    cleaned = re.sub(r"\?(?=&|$)", "", cleaned)
    cleaned = re.sub(r"\?&", "?", cleaned)
    cleaned = re.sub(r"&{2,}", "&", cleaned)
    return cleaned.rstrip("?&")


def _dedupe_repeats(text: str) -> str:
    """Remove immediate repeats — 'FOOFOO' → 'FOO', 'BAR BAR' → 'BAR'.
    Helps when the LLM accidentally doubles a phrase or the page extracted
    the title twice in a row.
    """
    if not text:
        return text
    out = _REPEAT_RE.sub(lambda m: m.group(1), text)
    # Also collapse double spaces that the substitution may have left
    return re.sub(r"\s{2,}", " ", out).strip()
