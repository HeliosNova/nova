"""Native web search — direct HTML scraping of public search engines.

Removes the SearXNG dependency. SearXNG was returning garbage rankings (AT&T
forums and Albuquerque census results for "current population of Japan"),
which made tool-using agents fabricate around bad data.

This module hits public search frontends directly and parses the result HTML.
No API keys, no third-party services, no SearXNG container.

Engines (in priority order, with quick failover):
  1. DuckDuckGo HTML (https://html.duckduckgo.com/html/) — no API, no JS, fast
  2. Bing HTML (https://www.bing.com/search) — fallback
  3. Brave Search HTML (https://search.brave.com/search) — second fallback

Each engine is parsed with stdlib `html.parser` (no BeautifulSoup dep). If one
returns 0 results or errors, we fall through to the next.

Output is a normalized list of {title, url, snippet}.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

logger = logging.getLogger(__name__)

# 12s was too tight for SearXNG: it talks to multiple upstream engines
# (Bing, Startpage, Yandex, etc.) sequentially, and a single slow upstream
# made the whole call return "Server disconnected" 5-8% of queries on the
# eval suite. User mandate: we're optimizing for best, not fastest. 30s
# lets every upstream engine have time to respond, and the tool round-trip
# is still well under the GENERATION_TIMEOUT ceiling.
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RESULTS = 10

# Rotating User-Agents to avoid being fingerprinted as a bot from a fixed UA.
import random
USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
]


def _headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        # Brotli decoding requires the `brotli` Python package; gzip+deflate are
        # built into httpx. Stick to those.
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    engine: str
    published_date: str = ""  # ISO-ish date string from the search engine if available

    def __str__(self) -> str:
        return f"{self.title}\n  {self.url}\n  {self.snippet}"


# ---------------------------------------------------------------------------
# DuckDuckGo HTML parser
# ---------------------------------------------------------------------------

class _DDGParser(HTMLParser):
    """Extract results from the html.duckduckgo.com/html/ response."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._current: dict[str, str] = {}
        self._mode: str | None = None  # "title" | "url" | "snippet" | None
        self._url_buf: list[str] = []
        self._title_buf: list[str] = []
        self._snippet_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr = dict(attrs)
        cls = attr.get("class", "") or ""
        if tag == "a" and "result__a" in cls:
            self._mode = "title"
            self._title_buf = []
            href = attr.get("href", "")
            self._current["raw_href"] = href
        elif tag == "a" and "result__url" in cls:
            self._mode = "url"
            self._url_buf = []
        elif tag == "a" and "result__snippet" in cls:
            self._mode = "snippet"
            self._snippet_buf = []
        elif tag == "div" and ("result__snippet" in cls or "result-snippet" in cls):
            self._mode = "snippet"
            self._snippet_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._mode == "title":
            self._current["title"] = "".join(self._title_buf).strip()
            self._mode = None
        elif tag == "a" and self._mode == "url":
            url_text = "".join(self._url_buf).strip()
            # The visible URL is partial (no scheme); use raw_href if it's a real URL
            raw = self._current.get("raw_href", "")
            self._current["url"] = self._extract_real_url(raw, url_text)
            self._mode = None
            # Once we have URL, this result is "ready" — flush if title also there
            self._maybe_flush()
        elif tag in ("a", "div") and self._mode == "snippet":
            self._current["snippet"] = "".join(self._snippet_buf).strip()
            self._mode = None
            self._maybe_flush()

    def handle_data(self, data: str) -> None:
        if self._mode == "title":
            self._title_buf.append(data)
        elif self._mode == "url":
            self._url_buf.append(data)
        elif self._mode == "snippet":
            self._snippet_buf.append(data)

    def _extract_real_url(self, raw_href: str, fallback: str) -> str:
        """DDG wraps URLs in /l/?uddg=... Decode if so."""
        if not raw_href:
            return "https://" + fallback if fallback else ""
        if raw_href.startswith("//duckduckgo.com/l/") or raw_href.startswith("/l/"):
            qs = parse_qs(urlparse(raw_href).query)
            if "uddg" in qs:
                return unquote(qs["uddg"][0])
        if raw_href.startswith("//"):
            return "https:" + raw_href
        if raw_href.startswith("http"):
            return raw_href
        if fallback:
            return "https://" + fallback
        return raw_href

    def _maybe_flush(self) -> None:
        c = self._current
        if c.get("title") and c.get("url"):
            self.results.append(SearchResult(
                title=html.unescape(c["title"]),
                url=c["url"],
                snippet=html.unescape(c.get("snippet", "")),
                engine="duckduckgo",
            ))
            self._current = {}


# ---------------------------------------------------------------------------
# Bing HTML parser
# ---------------------------------------------------------------------------

_BING_RESULT_RE = re.compile(
    r'<li class="b_algo">.*?<h2><a href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
    r'(?:<p[^>]*>(?P<snippet>.*?)</p>|<div class="b_caption"[^>]*><p[^>]*>(?P<snippet2>.*?)</p>)',
    re.DOTALL,
)


def _parse_bing(text: str) -> list[SearchResult]:
    results = []
    for m in _BING_RESULT_RE.finditer(text):
        title = re.sub(r"<[^>]+>", "", m.group("title")).strip()
        url = m.group("url").strip()
        snippet = re.sub(r"<[^>]+>", "", m.group("snippet") or m.group("snippet2") or "").strip()
        if title and url:
            results.append(SearchResult(
                title=html.unescape(title),
                url=url,
                snippet=html.unescape(snippet),
                engine="bing",
            ))
    return results


# ---------------------------------------------------------------------------
# Brave HTML parser (very lightweight, regex)
# ---------------------------------------------------------------------------

_BRAVE_RESULT_RE = re.compile(
    r'<a class="[^"]*result-header[^"]*"\s+href="(?P<url>[^"]+)"[^>]*>'
    r'.*?<span class="[^"]*title[^"]*"[^>]*>(?P<title>.*?)</span>'
    r'.*?<div class="[^"]*snippet[^"]*"[^>]*>(?P<snippet>.*?)</div>',
    re.DOTALL,
)


def _parse_brave(text: str) -> list[SearchResult]:
    results = []
    for m in _BRAVE_RESULT_RE.finditer(text):
        title = re.sub(r"<[^>]+>", "", m.group("title")).strip()
        url = m.group("url").strip()
        snippet = re.sub(r"<[^>]+>", "", m.group("snippet")).strip()
        if title and url:
            results.append(SearchResult(
                title=html.unescape(title),
                url=url,
                snippet=html.unescape(snippet),
                engine="brave",
            ))
    return results


# ---------------------------------------------------------------------------
# Engine drivers
# ---------------------------------------------------------------------------

async def _fetch(url: str, *, params: dict | None = None, data: dict | None = None, timeout: float = DEFAULT_TIMEOUT, method: str = "GET") -> str | None:
    try:
        async with httpx.AsyncClient(
            headers=_headers(),
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            if method == "POST":
                resp = await client.post(url, params=params, data=data)
            else:
                resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        logger.warning("native_search fetch failed for %s: %s", url, e)
        return None


async def _search_wikipedia(query: str, max_results: int) -> list[SearchResult]:
    """Search Wikipedia via its REST API. No rate limit, no anti-bot, no API key.

    Best for factual lookups: populations, dates, definitions, biographies,
    geography. Returns the top N article matches with extract snippets.

    Uses srsearch with srqiprofile=engine_autoselect for smarter ranking that
    favors broad/general articles over narrow date-stamped ones (so "French
    Revolution" returns the 1789 main article, not the "of 1848" sub-article).
    """
    import urllib.parse, json as _json
    try:
        # Wikipedia requires a descriptive UA per their API policy.
        # Including a contact URL avoids being flagged as anonymous bot traffic.
        async with httpx.AsyncClient(
            headers={
                "User-Agent": "NovaBot/1.0 (https://github.com/anthropics/nova; sovereign personal assistant) python-httpx",
                "Accept": "application/json",
            },
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
        ) as client:
            # Step 1: search for matching pages. Default profile (classic
            # textual relevance) — popularity-weighted profiles bias toward
            # current-events articles like "2026 Hungarian elections" which
            # are wrong for most factual queries.
            r = await client.get(WIKIPEDIA_API, params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            })
            r.raise_for_status()
            data = r.json()
            search_hits = data.get("query", {}).get("search", [])
            if not search_hits:
                return []

            # Step 2: get extracts for those pages in one batch call
            titles = "|".join(h["title"] for h in search_hits[:max_results])
            r2 = await client.get(WIKIPEDIA_API, params={
                "action": "query",
                "prop": "extracts|info",
                "exintro": "1",
                "explaintext": "1",
                "exchars": "1000",
                "inprop": "url",
                "titles": titles,
                "format": "json",
                "redirects": "1",
            })
            r2.raise_for_status()
            pages = r2.json().get("query", {}).get("pages", {})

            results = []
            for hit in search_hits[:max_results]:
                title = hit["title"]
                # Find the corresponding page entry (titles after redirect resolution)
                page = next((p for p in pages.values() if p.get("title") == title), None)
                if page is None:
                    page = next((p for p in pages.values()), None)
                if page is None:
                    continue
                url = page.get("fullurl") or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                snippet = page.get("extract", "") or hit.get("snippet", "")
                # Strip Wikipedia's HTML tags from the search snippet
                snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    engine="wikipedia",
                ))
            return results
    except Exception as e:
        logger.warning("wikipedia search failed for %r: %s", query[:60], e)
        return []


async def _search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    # DDG's html endpoint requires POST with form-encoded body for actual
    # results — GET returns the homepage with no results.
    text = await _fetch(
        "https://html.duckduckgo.com/html/",
        data={"q": query, "kl": "us-en", "df": ""},
        method="POST",
    )
    if not text:
        return []
    parser = _DDGParser()
    try:
        parser.feed(text)
    except Exception as e:
        logger.warning("DDG parse error: %s", e)
        return []
    return parser.results[:max_results]


async def _search_bing(query: str, max_results: int) -> list[SearchResult]:
    text = await _fetch(
        "https://www.bing.com/search",
        params={"q": query, "form": "QBLH", "setlang": "en-US"},
    )
    if not text:
        return []
    return _parse_bing(text)[:max_results]


async def _search_brave(query: str, max_results: int) -> list[SearchResult]:
    text = await _fetch(
        "https://search.brave.com/search",
        params={"q": query, "source": "web"},
    )
    if not text:
        return []
    return _parse_brave(text)[:max_results]


async def _search_searxng(
    query: str,
    max_results: int,
    *,
    categories: str = "general",
) -> list[SearchResult]:
    """Hit the local SearXNG instance — gives us 5+ engines in one call
    (bing, startpage, ecosia, yandex, yahoo, wikipedia, etc) with category
    filtering. SearXNG is the preferred primary because it aggregates
    multiple engines, dedupes, and supports the `news` category.
    """
    from app.config import config as _cfg
    base = getattr(_cfg, "SEARXNG_URL", "http://searxng:8080")
    if not base:
        return []
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            params: dict = {
                "q": query,
                "format": "json",
                "language": "en",
                "categories": categories,
            }
            # Allow operator override of engines from .env
            engines_csv = getattr(_cfg, "WEB_SEARCH_ENGINES", "")
            if engines_csv and categories == "general":
                params["engines"] = engines_csv
            resp = await client.get(f"{base}/search", params=params)
            if resp.status_code >= 400:
                return []
            data = resp.json()
    except Exception as e:
        logger.warning("searxng (%s) failed for %r: %s", categories, query[:60], e)
        return []

    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for r in data.get("results", []):
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or "").strip()
        if not url or not title:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        engine = r.get("engine", "searxng")
        # SearXNG news results carry publishedDate (ISO-ish). Surface it
        # so the Domain Study runner can verify freshness without trusting
        # the LLM's date judgment.
        pub = (r.get("publishedDate") or r.get("publishedDate", "") or "").strip()
        results.append(SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            engine=f"searxng:{engine}",
            published_date=pub,
        ))
        if len(results) >= max_results:
            break
    return results


# ---------------------------------------------------------------------------
# Intent detection — route queries to specialized search categories
# ---------------------------------------------------------------------------

_NEWS_SIGNAL_WORDS = (
    "today", "tonight", "yesterday", "this week", "past 24", "past 48",
    "past week", "past month", "latest", "recent", "current", "currently",
    "now", "breaking", "news", "headline", "update", "developments",
    "what's happening", "whats happening", "happened", "announced",
    "announcement", "launched", "released",
)

_SCIENCE_SIGNAL_WORDS = (
    "research paper", "preprint", "arxiv", "doi:", "doi.org", "study finds",
    "peer-reviewed", "peer reviewed", "abstract:", "citation", "pubmed",
    "clinical trial", "meta-analysis", "systematic review",
)

_CODE_SIGNAL_WORDS = (
    "github.com", "stack overflow", "stackoverflow", "code example",
    "implementation of", "library for", "package for", "pip install",
    "npm install", "cargo install", "how do i implement", "how to implement",
    "docstring", "traceback", "stacktrace", "compile error",
    " api ", " sdk ",
)

_DISCUSSION_SIGNAL_WORDS = (
    "reddit", "hackernews", "hacker news", "discussion of", "people think",
    "opinions on", "what do people", "subreddit",
)


def _detect_category(query: str) -> str:
    """Return the SearXNG category that best fits a query.

    Priority: code > science > discussion > news > general. A query that says
    "latest research paper on X" should go to science, not news. A query
    asking for a code library should go to it, not general.
    """
    if not query:
        return "general"
    q = query.lower()
    if any(w in q for w in _CODE_SIGNAL_WORDS):
        return "it"
    if any(w in q for w in _SCIENCE_SIGNAL_WORDS):
        return "science"
    if any(w in q for w in _DISCUSSION_SIGNAL_WORDS):
        return "social"
    if any(w in q for w in _NEWS_SIGNAL_WORDS):
        return "news"
    return "general"


def _looks_like_news(query: str) -> bool:
    return _detect_category(query) == "news"


def _merge_unique(
    results_lists: list[list[SearchResult]],
    *,
    max_results: int,
) -> list[SearchResult]:
    """Round-robin merge so each engine gets representation; dedupe by URL."""
    out: list[SearchResult] = []
    seen: set[str] = set()
    pointers = [0] * len(results_lists)
    while len(out) < max_results:
        progressed = False
        for i, lst in enumerate(results_lists):
            if pointers[i] >= len(lst):
                continue
            r = lst[pointers[i]]
            pointers[i] += 1
            progressed = True
            url_key = r.url.split("#")[0].rstrip("/")
            if url_key in seen:
                continue
            seen.add(url_key)
            out.append(r)
            if len(out) >= max_results:
                break
        if not progressed:
            break
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def search(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    mode: str = "auto",  # "auto" | "news" | "factual" | "general" | "science" | "code" | "social"
) -> list[SearchResult]:
    """Search across engines, MERGING results so no single source dominates.

    Modes:
      - auto: detect intent (news/science/code/social/general)
      - news: SearXNG news category (bing news, yahoo news, qwant news, ...)
      - factual: Wikipedia first then SearXNG general
      - general: SearXNG general (bing, startpage, duckduckgo, yandex, brave, ...)
      - science: SearXNG science (arxiv, semantic scholar, pubmed, crossref)
      - code: SearXNG IT (github, stackoverflow, gitlab, askubuntu, hackernews)
      - social: SearXNG social (reddit, hackernews)

    SearXNG already aggregates 200+ engines per category. This wrapper picks
    the right category based on intent and adds Wikipedia as a factual
    augmentation for non-news queries. Engines run concurrently and results
    are round-robin merged so no single source dominates the output.
    """
    effective_mode = mode
    if mode == "auto":
        effective_mode = _detect_category(query)
        # Map detected SearXNG category to mode. The detector returns the
        # SearXNG category name directly; only "news" needs special handling
        # for the legacy mode string.
        if effective_mode == "general":
            effective_mode = "general"

    # Build engine ladder. Each tuple = (async callable taking (q,n), label).
    if effective_mode == "news":
        ladder = [
            (lambda q, n: _search_searxng(q, n, categories="news"), "searxng:news"),
            (lambda q, n: _search_searxng(q, n, categories="general"), "searxng:general"),
            (_search_duckduckgo, "duckduckgo"),
        ]
    elif effective_mode == "factual":
        ladder = [
            (_search_wikipedia, "wikipedia"),
            (lambda q, n: _search_searxng(q, n, categories="general"), "searxng:general"),
        ]
    elif effective_mode == "science":
        ladder = [
            (lambda q, n: _search_searxng(q, n, categories="science"), "searxng:science"),
            (_search_wikipedia, "wikipedia"),
            (lambda q, n: _search_searxng(q, n, categories="general"), "searxng:general"),
        ]
    elif effective_mode in ("code", "it"):
        ladder = [
            (lambda q, n: _search_searxng(q, n, categories="it"), "searxng:it"),
            (lambda q, n: _search_searxng(q, n, categories="general"), "searxng:general"),
        ]
    elif effective_mode == "social":
        ladder = [
            (lambda q, n: _search_searxng(q, n, categories="social"), "searxng:social"),
            (lambda q, n: _search_searxng(q, n, categories="general"), "searxng:general"),
        ]
    else:  # general
        ladder = [
            (lambda q, n: _search_searxng(q, n, categories="general"), "searxng:general"),
            (_search_wikipedia, "wikipedia"),
            (_search_duckduckgo, "duckduckgo"),
        ]

    # Run the first 3 engines concurrently — gives a rich merged result set
    # without waiting on every engine. Engines beyond 3 only get hit if we
    # come up short.
    primary = ladder[:3]
    fan_results: list[list[SearchResult]] = []
    coros = [fn(query, max_results) for fn, _ in primary]
    settled = await asyncio.gather(*coros, return_exceptions=True)
    for (fn, name), out in zip(primary, settled):
        if isinstance(out, Exception):
            logger.warning("native_search: %s raised: %s", name, out)
            continue
        if out:
            logger.info("native_search: %s returned %d for %r", name, len(out), query[:60])
            fan_results.append(out)
        else:
            logger.info("native_search: %s returned 0 for %r", name, query[:60])

    merged = _merge_unique(fan_results, max_results=max_results)

    # If we still don't have enough, try the rest sequentially as fallback
    if len(merged) < max_results:
        for fn, name in ladder[3:]:
            try:
                extra = await fn(query, max_results)
                if extra:
                    fan_results.append(extra)
                    merged = _merge_unique(fan_results, max_results=max_results)
                    if len(merged) >= max_results:
                        break
            except Exception as e:
                logger.warning("native_search: %s failed: %s", name, e)

    return _demote_low_credibility(merged)


def _result_host(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _demote_low_credibility(results: list[SearchResult]) -> list[SearchResult]:
    """Stable-partition: results from low-credibility domains (dataset score
    < 0.3 — content farms, known misinformation hosts) move to the BOTTOM.
    Visible-but-last, never silently dropped: the model and user can still see
    them, but trustworthy sources lead. Relevance order is preserved within
    each partition. Same source-authority backbone as the monitor digests
    (Lin et al. PNAS Nexus 2023 consensus ratings)."""
    try:
        from app.core.source_authority import authority
    except Exception:
        return results
    trusted = [r for r in results if authority(_result_host(r.url)) >= 0.3]
    lowcred = [r for r in results if authority(_result_host(r.url)) < 0.3]
    if lowcred:
        logger.info("native_search: demoted %d low-credibility result(s): %s",
                    len(lowcred), [_result_host(r.url) for r in lowcred[:3]])
    return trusted + lowcred


def format_results(results: list[SearchResult]) -> str:
    """Format results as a numbered list with engine, host, and a RELIABILITY
    tier per result (primary/wire, reputable, general, mixed, low-credibility —
    from the same 11,520-domain consensus dataset the monitor digests use).

    Annotating reliability lets the model weigh conflicting results and cite
    by credibility instead of treating a content farm and Reuters as equals.
    Low-credibility results carry an explicit warning tag.
    """
    if not results:
        return "No results found."
    try:
        from app.core.source_authority import authority, tier
    except Exception:
        def authority(h):  # degrade gracefully if module unavailable
            return 0.5

        def tier(h):
            return "general"
    lines = []
    any_lowcred = False
    for i, r in enumerate(results, 1):
        snippet = r.snippet[:800] if r.snippet else ""
        engine = r.engine or "?"
        host = _result_host(r.url)
        t = tier(host) if host else "general"
        warn = ""
        if host and authority(host) < 0.3:
            warn = " ⚠ treat as unverified"
            any_lowcred = True
        lines.append(f"[{i}] ({engine} · {host or '?'} — {t}{warn}) {r.title}\n    {r.url}\n    {snippet}")
    out = "\n\n".join(lines)
    if any_lowcred:
        out += ("\n\nNOTE: results tagged low-credibility come from sources with poor "
                "factual-reliability ratings. Do not state their claims as fact; prefer "
                "the primary/wire and reputable sources above, or flag the claim as unverified.")
    return out
