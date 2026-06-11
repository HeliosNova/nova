"""Iterative search agent — multi-step research with full-page reads.

The plain `web_search` tool returns 10 snippets and the LLM synthesises
an answer from them. Snippets are lossy: Wikipedia's first 200 chars
rarely contain the specific number you need, news headlines drop the
key date, etc.

This tool wraps `web_search` + `http_fetch` + a tiny LLM critique loop:

  1. SEARCH the query (auto category routing).
  2. RANK results — pick the top 2-3 most authoritative URLs (HN, Reuters,
     Wikipedia, .gov, .edu beat random blog spam).
  3. FETCH the top URLs in parallel (cap concurrency to 3).
  4. EXTRACT the relevant section from each page using a fast LLM call.
  5. CRITIQUE: does the extracted content directly answer the question?
     - YES → SYNTHESISE final answer with citations.
     - NO  → REFINE the query (keyword shift) and recurse, max 2 rounds.

Output format includes [Source: URL] citations after every claim so the
caller can verify.

Used directly by the agent loop and exposed as a tool the LLM can pick
when a single web_search isn't enough.
"""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlparse

from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)


# Domain trust scores (very rough — drives ranking, not gate)
_DOMAIN_PRIORITY = {
    # Tier 1 — primary sources / encyclopedic / regulator
    "wikipedia.org": 100, "en.wikipedia.org": 100,
    "reuters.com": 95, "apnews.com": 95, "bbc.co.uk": 95, "bbc.com": 95,
    "nature.com": 95, "science.org": 95, "arxiv.org": 95,
    "sec.gov": 95, "fda.gov": 95, "nih.gov": 95, "europa.eu": 95,
    "irs.gov": 90, "noaa.gov": 90, "nist.gov": 90,
    # Tier 2 — quality outlets
    "nytimes.com": 85, "washingtonpost.com": 85, "wsj.com": 85, "economist.com": 85,
    "ft.com": 85, "bloomberg.com": 85, "theguardian.com": 80,
    "techcrunch.com": 80, "theverge.com": 80, "arstechnica.com": 80,
    "news.ycombinator.com": 75, "github.com": 80, "stackoverflow.com": 75,
    # Tier 3 — aggregators
    "msn.com": 50, "yahoo.com": 50, "google.com": 50,
}


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _score_result(title: str, url: str, snippet: str) -> int:
    """Return a heuristic relevance/authority score, higher = better."""
    score = 50
    d = _domain(url)
    score += _DOMAIN_PRIORITY.get(d, 0)
    # .gov / .edu / .org gets a small bump
    if d.endswith(".gov") or d.endswith(".edu"):
        score += 15
    elif d.endswith(".org"):
        score += 5
    # Snippet length hint
    if snippet and len(snippet) > 120:
        score += 5
    # Penalise pure aggregators / forum spam
    if any(bad in d for bad in ("forums.", "answers.", "quora", "pinterest")):
        score -= 25
    return score


_RESULT_RE = re.compile(
    r"\[\d+\]\s*\(([^)]+)\)\s*(.+?)\n\s*(https?://\S+)\n\s*(.+?)(?=\n\[|$)",
    re.DOTALL,
)


def _parse_search_output(text: str) -> list[dict]:
    """Parse the format produced by native_search.format_results into dicts."""
    out: list[dict] = []
    for m in _RESULT_RE.finditer(text or ""):
        engine, title, url, snippet = m.group(1), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
        out.append({"engine": engine, "title": title, "url": url, "snippet": snippet})
    return out


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    """Crude HTML→text. Good enough for extracting paragraphs."""
    if not html:
        return ""
    # Remove script/style blocks first
    html = re.sub(r"<(?:script|style)[^>]*>.*?</(?:script|style)>",
                  " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = _HTML_TAG_RE.sub(" ", html)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


_EXTRACT_PROMPT = (
    "You will be given a question and the body text of a web page. Extract "
    "ONLY the sentences that directly answer the question. Quote exactly. "
    "If the page does not answer the question, reply 'NO ANSWER'.\n\n"
    "QUESTION: {question}\n\n"
    "PAGE TEXT (first 6000 chars):\n{body}\n\n"
    "RELEVANT SENTENCES:"
)


_REFINE_PROMPT = (
    "You searched for: '{query}'\nThe results don't directly answer the original "
    "question: '{question}'.\n\nWrite ONE refined search query (different keywords, "
    "more specific, or a quoted phrase) that would find the answer. Output ONLY "
    "the query — no preamble, no quotes around the whole thing."
)


_SYNTHESIS_PROMPT = (
    "Synthesise an answer to the question using ONLY the extracted snippets "
    "below. Every claim must be followed by a [Source: <url>] citation. If "
    "the snippets don't fully answer the question, say what you can answer "
    "and what's still unknown.\n\n"
    "QUESTION: {question}\n\n"
    "EXTRACTED EVIDENCE:\n{evidence}\n\n"
    "ANSWER:"
)


async def _extract_relevant(question: str, page_text: str) -> str:
    """Ask LLM to pull the answer-relevant sentences out of a fetched page."""
    from app.core.llm import invoke_nothink
    body = page_text[:6000]
    try:
        out = await invoke_nothink(
            [{"role": "user", "content": _EXTRACT_PROMPT.format(question=question, body=body)}],
            max_tokens=400, temperature=0.0,
        )
    except Exception as e:
        logger.warning("[SearchAgent] extract failed: %s", e)
        return ""
    return (out or "").strip()


async def _refine_query(question: str, prev_query: str) -> str:
    from app.core.llm import invoke_nothink
    try:
        out = await invoke_nothink(
            [{"role": "user", "content": _REFINE_PROMPT.format(question=question, query=prev_query)}],
            max_tokens=80, temperature=0.3,
        )
    except Exception as e:
        logger.warning("[SearchAgent] refine failed: %s", e)
        return prev_query
    return (out or "").strip().splitlines()[0][:200]


async def _synthesise(question: str, evidence: list[dict]) -> str:
    """Build the final answer from extracted evidence with citations."""
    from app.core.llm import invoke_nothink
    blocks = []
    for ev in evidence:
        if not ev.get("excerpt"):
            continue
        blocks.append(f"[Source: {ev['url']}]\n{ev['excerpt']}\n")
    if not blocks:
        return "Could not extract a direct answer from the searched pages."
    try:
        out = await invoke_nothink(
            [{"role": "user", "content": _SYNTHESIS_PROMPT.format(
                question=question, evidence="\n".join(blocks))}],
            max_tokens=600, temperature=0.2,
        )
    except Exception as e:
        logger.warning("[SearchAgent] synth failed: %s", e)
        return ""
    return (out or "").strip()


async def deep_research(question: str, *, max_rounds: int = 2, max_pages: int = 3) -> str:
    """The actual agent: search → rank → fetch → extract → synth, with one
    refinement round if the first pass doesn't answer.
    """
    from app.tools import native_search
    from app.tools.http_fetch import HttpFetchTool

    fetcher = HttpFetchTool()
    query = question
    accumulated: list[dict] = []
    seen_urls: set[str] = set()

    for round_idx in range(max_rounds):
        # 1. Search
        try:
            results = await native_search.search(query, max_results=8, mode="auto")
        except Exception as e:
            logger.warning("[SearchAgent] round %d search failed: %s", round_idx, e)
            results = []

        if not results:
            if round_idx == 0:
                return f"No results found for '{query}'."
            break

        # 2. Rank — score and dedupe by domain
        ranked = []
        seen_domains: set[str] = set()
        for r in results:
            score = _score_result(r.title, r.url, r.snippet)
            ranked.append((score, r))
        ranked.sort(key=lambda x: x[0], reverse=True)

        # Pick top max_pages whose URLs we haven't fetched yet
        picks: list = []
        for _, r in ranked:
            if r.url in seen_urls:
                continue
            d = _domain(r.url)
            if d in seen_domains:
                continue
            seen_domains.add(d)
            seen_urls.add(r.url)
            picks.append(r)
            if len(picks) >= max_pages:
                break

        if not picks:
            break

        # 3. Fetch in parallel
        async def _fetch_one(r):
            try:
                fetched = await fetcher.execute(url=r.url, method="GET")
            except Exception as e:
                return r, "", f"fetch raised: {e}"
            if not fetched.success:
                return r, "", fetched.error or "fetch failed"
            text = _strip_html(fetched.output or "")
            return r, text, ""

        fetched = await asyncio.gather(*[_fetch_one(r) for r in picks], return_exceptions=False)

        # 4. Extract
        for r, page_text, err in fetched:
            if err or not page_text:
                logger.info("[SearchAgent] fetch %s failed: %s", r.url, err)
                continue
            excerpt = await _extract_relevant(question, page_text)
            if excerpt and "no answer" not in excerpt.lower()[:25]:
                accumulated.append({
                    "url": r.url,
                    "title": r.title,
                    "engine": r.engine,
                    "excerpt": excerpt[:1500],
                })

        # If we have enough evidence, stop
        if len(accumulated) >= 2:
            break

        # 5. Refine for the next round (only if we still have room)
        if round_idx < max_rounds - 1:
            new_query = await _refine_query(question, query)
            if new_query and new_query.lower() != query.lower():
                logger.info("[SearchAgent] round %d: refined '%s' → '%s'", round_idx, query, new_query)
                query = new_query

    # 6. Synthesise
    if not accumulated:
        return (
            f"Searched {max_rounds} rounds, fetched several pages, but could "
            f"not extract a direct answer to: '{question}'."
        )
    return await _synthesise(question, accumulated)


class SearchAgentTool(BaseTool):
    name = "deep_research"
    description = (
        "Multi-step research tool. Use when a single web_search isn't enough — "
        "the question needs reading actual source pages, comparing multiple "
        "sources, or refining the query based on early results. Returns a "
        "synthesized answer with [Source: URL] citations under every claim."
    )
    parameters = "question: str, max_rounds: int = 2, max_pages: int = 3"
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to research."},
            "max_rounds": {"type": "integer", "description": "Search rounds (default 2)."},
            "max_pages": {"type": "integer", "description": "Pages to fetch per round (default 3)."},
        },
        "required": ["question"],
    }

    async def execute(self, *, question: str = "", max_rounds: int = 2, max_pages: int = 3, **kw) -> ToolResult:
        if not question:
            return ToolResult(output="", success=False, error="No question provided",
                              error_category=ErrorCategory.VALIDATION)
        try:
            answer = await deep_research(question, max_rounds=max_rounds, max_pages=max_pages)
        except Exception as e:
            logger.exception("deep_research raised")
            return ToolResult(output="", success=False, error=str(e),
                              error_category=ErrorCategory.TRANSIENT, retriable=True)
        if not answer:
            return ToolResult(output="", success=False,
                              error="empty result", error_category=ErrorCategory.NOT_FOUND)
        return ToolResult(output=answer, success=True)
