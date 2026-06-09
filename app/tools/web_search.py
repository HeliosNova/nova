"""Web search tool — multi-engine merged via native_search.

native_search internally fans out to SearXNG (news/general categories) +
DuckDuckGo + Bing + Wikipedia and round-robin merges results, so callers
get diverse sources in a single call. This wrapper exists for the
ToolRegistry contract; all the routing logic lives in native_search.
"""

from __future__ import annotations

import logging

from app.config import config
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web for current events, facts, prices, news, and real-time "
        "information. Returns results merged across multiple engines (SearXNG "
        "with bing/startpage/yandex/yahoo + Wikipedia for factual queries; "
        "SearXNG news category for time-sensitive queries). Each result names "
        "its source engine. Use when you need information not in your training "
        "data or ingested documents."
    )
    parameters = "query: str"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string. Be specific for better results.",
            },
            "mode": {
                "type": "string",
                "enum": ["auto", "news", "factual", "general", "science", "code", "social"],
                "description": (
                    "Search mode. 'auto' (default) auto-detects intent. "
                    "'news' forces news engines (bing news, qwant news, ...). "
                    "'factual' prioritises Wikipedia. 'science' targets "
                    "arxiv/semanticscholar/pubmed/crossref. 'code' targets "
                    "github/stackoverflow/gitlab. 'social' targets "
                    "reddit/hackernews. 'general' uses general web engines."
                ),
            },
        },
        "required": ["query"],
    }

    async def execute(self, *, query: str = "", mode: str = "auto", **kwargs) -> ToolResult:
        if not query:
            return ToolResult(
                output="",
                success=False,
                error="No query provided",
                error_category=ErrorCategory.VALIDATION,
            )

        try:
            from app.tools import native_search
            results = await native_search.search(
                query,
                max_results=config.WEB_SEARCH_MAX_RESULTS,
                mode=mode,
            )
        except Exception as e:
            logger.exception("native_search raised")
            return ToolResult(
                output="",
                success=False,
                error=f"search failed: {e}",
                error_category=ErrorCategory.TRANSIENT,
                retriable=True,
            )

        if not results:
            # Auto-curiosity: a search that finds nothing is a topic Nova
            # doesn't have ambient awareness on yet. Queue it for the
            # Curiosity Research monitor to pursue more thoroughly.
            try:
                from app.database import get_db
                from app.core.curiosity import CuriosityQueue
                cq = CuriosityQueue(get_db())
                cq.add(topic=query[:280], source="search_zero_result", urgency=0.4)
            except Exception as e:
                logger.debug("Auto-curiosity queue from empty search failed: %s", e)
            return ToolResult(
                output="No results found.",
                success=False,
                error="all engines returned 0 results",
                error_category=ErrorCategory.NOT_FOUND,
                retriable=False,
            )

        output = native_search.format_results(results)
        if config.ENABLE_INJECTION_DETECTION:
            from app.core.injection import sanitize_content
            output = sanitize_content(output, context="search result")
        return ToolResult(output=output, success=True)
