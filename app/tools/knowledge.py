"""Knowledge search tool — query ingested documents via the retriever."""

from __future__ import annotations

import logging

from app.config import config
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class KnowledgeSearchTool(BaseTool):
    name = "knowledge_search"
    description = "Search your owner's ingested documents. Use for questions about uploaded content."
    parameters = "query: str"

    def __init__(self, retriever=None):
        self._retriever = retriever

    async def execute(self, *, query: str = "", **kwargs) -> ToolResult:
        if not query:
            return ToolResult(output="", success=False, error="No query provided")

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
            return ToolResult(output="", success=False, error=f"Search failed: {e}")
