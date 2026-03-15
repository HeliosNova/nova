"""Nova_ MCP Server — expose Nova's intelligence as MCP tools.

Allows external agents (Claude Code, Cursor, etc.) to query Nova's
long-term memory, knowledge graph, lessons, and document store.

This module defines the MCP server and its tools. It does NOT start the
server — that's done by scripts/mcp_server_runner.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from app.config import config
from app.core.kg import KnowledgeGraph
from app.core.learning import LearningEngine
from app.core.memory import ConversationStore, UserFactStore
from app.core.retriever import Retriever
from app.database import SafeDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema for each tool's inputSchema)
# ---------------------------------------------------------------------------

_TOOLS = [
    Tool(
        name="nova_memory_query",
        description=(
            "Search Nova's long-term memory for information about the user "
            "or past conversations"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for memory lookup",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="nova_knowledge_graph",
        description=(
            "Query Nova's knowledge graph for structured facts about entities"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Entity name to look up in the knowledge graph",
                },
                "hops": {
                    "type": "integer",
                    "description": "Number of hops for graph traversal (1=direct, 2=neighbors of neighbors)",
                    "default": 1,
                },
            },
            "required": ["entity"],
        },
    ),
    Tool(
        name="nova_lessons",
        description=(
            "Retrieve lessons Nova has learned from corrections, relevant to a topic"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or question to find relevant lessons for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lessons to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="nova_document_search",
        description=(
            "Search Nova's ingested documents using hybrid retrieval (vector + BM25)"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for document retrieval",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of document chunks to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="nova_facts_about",
        description="Get all stored facts about the user/owner",
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category: fact, preference, capability, or constraint. Omit for all.",
                    "enum": ["fact", "preference", "capability", "constraint"],
                },
            },
            "required": [],
        },
    ),
]


# ---------------------------------------------------------------------------
# MCP Server factory
# ---------------------------------------------------------------------------

def create_mcp_server(
    db: SafeDB,
    *,
    user_facts: UserFactStore | None = None,
    conversations: ConversationStore | None = None,
    learning: LearningEngine | None = None,
    kg: KnowledgeGraph | None = None,
    retriever: Retriever | None = None,
) -> Server:
    """Create and configure a Nova_ MCP server with all tool handlers.

    Callers must provide a SafeDB instance (already init_schema'd).
    Service instances are created lazily if not provided.
    """
    server = Server(config.MCP_SERVER_NAME)

    # Lazily build services from the db if not injected
    _user_facts = user_facts or UserFactStore(db)
    _conversations = conversations or ConversationStore(db)
    _learning = learning or LearningEngine(db)
    _kg = kg or KnowledgeGraph(db)

    # Retriever may fail (ChromaDB not available) — that's OK
    _retriever = retriever
    if _retriever is None:
        try:
            _retriever = Retriever(db)
        except Exception as e:
            logger.warning("Retriever unavailable (document search disabled): %s", e)

    # ------------------------------------------------------------------
    # list_tools handler
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _TOOLS

    # ------------------------------------------------------------------
    # call_tool handler
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            if name == "nova_memory_query":
                return await _handle_memory_query(arguments)
            elif name == "nova_knowledge_graph":
                return await _handle_knowledge_graph(arguments)
            elif name == "nova_lessons":
                return await _handle_lessons(arguments)
            elif name == "nova_document_search":
                return await _handle_document_search(arguments)
            elif name == "nova_facts_about":
                return await _handle_facts_about(arguments)
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )]
        except Exception as e:
            logger.exception("MCP tool '%s' failed", name)
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}),
            )]

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _handle_memory_query(args: dict) -> list[TextContent]:
        query = args.get("query", "")
        limit = int(args.get("limit", 5))

        if not query:
            return [TextContent(type="text", text=json.dumps({"error": "query is required"}))]

        # Gather user facts
        all_facts = _user_facts.get_all()
        query_lower = query.lower()
        matching_facts = [
            {"key": f.key, "value": f.value, "category": f.category, "confidence": f.confidence}
            for f in all_facts
            if query_lower in f.key.lower() or query_lower in f.value.lower()
        ]

        # Search conversation messages
        message_results = _conversations.search_messages(query, limit=limit)

        result = {
            "matching_facts": matching_facts[:limit],
            "conversation_excerpts": message_results[:limit],
            "total_facts_checked": len(all_facts),
        }
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _handle_knowledge_graph(args: dict) -> list[TextContent]:
        entity = args.get("entity", "")
        hops = int(args.get("hops", 1))

        if not entity:
            return [TextContent(type="text", text=json.dumps({"error": "entity is required"}))]

        facts = _kg.query(entity, hops=hops)
        result = {
            "entity": entity,
            "hops": hops,
            "facts": [
                {
                    "subject": f["subject"],
                    "predicate": f["predicate"],
                    "object": f["object"],
                    "confidence": f["confidence"],
                    "source": f.get("source", ""),
                    "depth": f.get("depth", 0),
                }
                for f in facts
            ],
            "total": len(facts),
        }
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _handle_lessons(args: dict) -> list[TextContent]:
        query = args.get("query", "")
        limit = int(args.get("limit", 5))

        if not query:
            return [TextContent(type="text", text=json.dumps({"error": "query is required"}))]

        lessons = _learning.get_relevant_lessons(query, limit=limit)
        result = {
            "query": query,
            "lessons": [
                {
                    "id": lesson.id,
                    "topic": lesson.topic,
                    "correct_answer": lesson.correct_answer,
                    "lesson_text": lesson.lesson_text or "",
                    "confidence": lesson.confidence,
                    "times_retrieved": lesson.times_retrieved,
                    "times_helpful": lesson.times_helpful,
                }
                for lesson in lessons
            ],
            "total": len(lessons),
        }
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _handle_document_search(args: dict) -> list[TextContent]:
        query = args.get("query", "")
        top_k = int(args.get("top_k", 5))

        if not query:
            return [TextContent(type="text", text=json.dumps({"error": "query is required"}))]

        if _retriever is None:
            return [TextContent(type="text", text=json.dumps({
                "error": "Document search is unavailable (ChromaDB not initialized)"
            }))]

        chunks = await _retriever.search(query, top_k=top_k)
        result = {
            "query": query,
            "results": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "content": c.content,
                    "score": round(c.score, 4),
                    "source": c.source,
                    "title": c.title,
                }
                for c in chunks
            ],
            "total": len(chunks),
        }
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _handle_facts_about(args: dict) -> list[TextContent]:
        category = args.get("category")

        all_facts = _user_facts.get_all()

        if category:
            facts = [f for f in all_facts if f.category == category]
        else:
            facts = all_facts

        result = {
            "facts": [
                {
                    "key": f.key,
                    "value": f.value,
                    "category": f.category,
                    "confidence": f.confidence,
                }
                for f in facts
            ],
            "total": len(facts),
        }
        if category:
            result["filter"] = category

        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server
