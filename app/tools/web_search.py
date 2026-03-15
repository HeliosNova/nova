"""Web search tool — SearXNG integration."""

from __future__ import annotations

import logging

import httpx

from app.config import config
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web. Use for current events, facts you don't know, prices, news."
    parameters = "query: str"

    async def execute(self, *, query: str = "", **kwargs) -> ToolResult:
        if not query:
            return ToolResult(output="", success=False, error="No query provided")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{config.SEARXNG_URL}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "engines": "google,duckduckgo,bing",
                        "language": "en",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])[:5]
            if not results:
                return ToolResult(output="No results found.", success=True)

            lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                snippet = r.get("content", "")[:300]
                lines.append(f"[{i}] {title}\n    {url}\n    {snippet}")

            output = "\n\n".join(lines)
            if config.ENABLE_INJECTION_DETECTION:
                from app.core.injection import sanitize_content
                output = sanitize_content(output, context="search result")
            return ToolResult(output=output, success=True)

        except Exception as e:
            logger.warning("Web search failed: %s", e)
            return ToolResult(output="", success=False, error=f"Search failed: {e}", retriable=True)
