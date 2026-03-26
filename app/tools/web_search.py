"""Web search tool — SearXNG integration."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.config import config
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_RETRY_DELAY = 1.0


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web via SearXNG for current events, facts, prices, news, and real-time information. "
        "Returns up to 5 results, each with title, URL, and snippet. "
        "Use when you need information not in your training data or ingested documents. "
        "Do NOT use for questions answerable from memory, knowledge_search, or conversation history."
    )
    parameters = "query: str"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string. Be specific for better results.",
            },
        },
        "required": ["query"],
    }

    async def execute(self, *, query: str = "", **kwargs) -> ToolResult:
        if not query:
            return ToolResult(output="", success=False, error="No query provided", error_category=ErrorCategory.VALIDATION)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=config.WEB_SEARCH_TIMEOUT) as client:
                    resp = await client.get(
                        f"{config.SEARXNG_URL}/search",
                        params={
                            "q": query,
                            "format": "json",
                            "engines": config.WEB_SEARCH_ENGINES,
                            "language": "en",
                        },
                    )
                    if resp.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Server error {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )
                    resp.raise_for_status()
                    data = resp.json()

                results = data.get("results", [])[:config.WEB_SEARCH_MAX_RESULTS]
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

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    logger.warning("Web search attempt %d failed (%s), retrying in %.0fs", attempt + 1, e, _RETRY_DELAY)
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_error = e
                    if attempt < _MAX_RETRIES - 1:
                        logger.warning("Web search attempt %d got HTTP %d, retrying in %.0fs", attempt + 1, e.response.status_code, _RETRY_DELAY)
                        await asyncio.sleep(_RETRY_DELAY)
                        continue
                else:
                    last_error = e
                    break
            except Exception as e:
                last_error = e
                break

        logger.warning("Web search failed after %d attempt(s): %s", _MAX_RETRIES, last_error)
        return ToolResult(output="", success=False, error=f"Search failed: {last_error}", retriable=True, error_category=ErrorCategory.TRANSIENT)
