"""Knowledge search tool — query ingested documents via the retriever."""

from __future__ import annotations

import logging

from app.config import config
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = (
        "Search ingested documents using hybrid retrieval (vector similarity + BM25 keyword matching with RRF fusion). "
        "Returns ranked chunks with source, relevance tier (HIGH/MODERATE/LOW), and content. "
        "Use for questions about uploaded files and documents. "
        "Do NOT use for web searches (use web_search) or past conversations (use memory_search)."
    )
    parameters = "query: str"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant document chunks.",
            },
        },
        "required": ["query"],
    }

    def trim_output(self, output: str) -> str:
        """Keep top 3 search result chunks, each trimmed."""
        if len(output) <= 2000:
            return output
        # Split on result boundaries (numbered [1], [2], etc.)
        import re
        chunks = re.split(r'\n\n(?=\[\d+\])', output)
        trimmed_chunks = []
        for chunk in chunks[:3]:
            if len(chunk) > 500:
                chunk = chunk[:500] + '...'
            trimmed_chunks.append(chunk)
        result = '\n\n'.join(trimmed_chunks)
        if len(chunks) > 3:
            result += f'\n\n[...{len(chunks) - 3} more results truncated]'
        return result

    def __init__(self, retriever=None):
        self._retriever = retriever

    async def execute(self, *, query: str = "", **kwargs) -> ToolResult:
        if not query:
            return ToolResult(output="", success=False, error="No query provided", error_category=ErrorCategory.VALIDATION)

        if not self._retriever:
            return ToolResult(
                output="No documents have been ingested yet.",
                success=True,
            )

        try:
            chunks = await self._retriever.search(query)
            if not chunks:
                return ToolResult(
                    output="No relevant documents found for this query.",
                    success=True,
                )

            lines = []
            for i, chunk in enumerate(chunks, 1):
                source = chunk.title or chunk.source or chunk.document_id
                # Add relevance tier label
                if chunk.score > 0.7:
                    relevance = "HIGH"
                elif chunk.score > 0.4:
                    relevance = "MODERATE"
                else:
                    relevance = "LOW"
                lines.append(
                    f"[{i}] Source: {source} (relevance: {relevance}, score: {chunk.score:.3f})\n{chunk.content}"
                )

            output = "\n\n".join(lines)
            if config.ENABLE_INJECTION_DETECTION:
                from app.core.injection import sanitize_content
                output = sanitize_content(output, context="knowledge base")
            return ToolResult(output=output, success=True)

        except Exception as e:
            logger.warning("Knowledge search failed: %s", e)
            return ToolResult(output="", success=False, error=f"Search failed: {e}", error_category=ErrorCategory.TRANSIENT)
