"""Nova MCP Server — expose Nova's intelligence as MCP tools.

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
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import Resource, TextContent, Tool, CallToolResult

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
            "Search Nova's long-term memory for user facts and past conversation excerpts. "
            "Returns matching facts (key, value, category, confidence) and conversation snippets "
            "ranked by keyword relevance. Use for: recalling user preferences, finding past "
            "discussions, checking what Nova knows about the user. Prefer nova_document_search "
            "for ingested document content. Prefer nova_knowledge_graph for structured entity "
            "relationships. Limit default: 5, max: 20."
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
            "Query Nova's temporal knowledge graph for structured facts about entities and "
            "their relationships. Returns subject-predicate-object triples with confidence "
            "scores, source provenance, and traversal depth. Use for: looking up entity facts, "
            "exploring connections between concepts, checking what Nova has learned from research. "
            "Set hops=2 to include neighbors of neighbors. Prefer nova_memory_query for "
            "user-specific facts. Prefer nova_document_search for full-text document content."
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
            "Retrieve lessons Nova has learned from user corrections, relevant to a query topic. "
            "Returns lesson details including topic, correct/wrong answers, lesson text, "
            "confidence, and retrieval/helpfulness counts. Use for: understanding how Nova was "
            "corrected on similar topics, checking if Nova has learned from past mistakes. "
            "Limit default: 5."
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
            "Search Nova's ingested documents using hybrid retrieval (vector similarity + BM25 "
            "keyword matching with Reciprocal Rank Fusion). Returns ranked document chunks with "
            "content, relevance score, source, and title. Use for: finding information in "
            "uploaded files and documents. Prefer nova_memory_query for user facts and "
            "conversation history. Top_k default: 5."
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
        description=(
            "Get all stored facts about the user/owner, optionally filtered by category. "
            "Returns facts with key, value, category (fact/preference/capability/constraint), "
            "and confidence score. Use for: getting a complete picture of known user attributes. "
            "Omit category parameter to retrieve all facts across all categories."
        ),
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
# Structured error helper
# ---------------------------------------------------------------------------


def _mcp_error(message: str, category: str, is_retryable: bool = False) -> CallToolResult:
    """Return a structured MCP error response with isError=True."""
    return CallToolResult(
        content=[TextContent(
            type="text",
            text=json.dumps({
                "error": message,
                "error_category": category,
                "is_retryable": is_retryable,
            }),
        )],
        isError=True,
    )


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
    """Create and configure a Nova MCP server with all tool handlers.

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
    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        try:
            if name == "nova_memory_query":
                content = await _handle_memory_query(arguments)
            elif name == "nova_knowledge_graph":
                content = await _handle_knowledge_graph(arguments)
            elif name == "nova_lessons":
                content = await _handle_lessons(arguments)
            elif name == "nova_document_search":
                content = await _handle_document_search(arguments)
            elif name == "nova_facts_about":
                content = await _handle_facts_about(arguments)
            else:
                return _mcp_error(f"Unknown tool: {name}", "not_found", False)
            # Wrap list[TextContent] in CallToolResult for consistent return type
            if isinstance(content, list):
                return CallToolResult(content=content, isError=False)
            return content  # Already a CallToolResult (from _mcp_error)
        except Exception as e:
            logger.exception("MCP tool '%s' failed", name)
            return _mcp_error(str(e), "internal", True)

    # ------------------------------------------------------------------
    # Resource handlers
    # ------------------------------------------------------------------

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="nova://facts",
                name="User Facts",
                description="User fact categories and counts",
                mimeType="application/json",
            ),
            Resource(
                uri="nova://lessons",
                name="Learned Lessons",
                description="Lesson topic index and counts",
                mimeType="application/json",
            ),
            Resource(
                uri="nova://documents",
                name="Ingested Documents",
                description="Document metadata catalog",
                mimeType="application/json",
            ),
            Resource(
                uri="nova://knowledge-graph/entities",
                name="Knowledge Graph Entities",
                description="Top entities and their fact counts",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri) -> list[ReadResourceContents]:
        uri_str = str(uri)

        if uri_str == "nova://facts":
            all_facts = _user_facts.get_all()
            categories: dict[str, int] = {}
            for f in all_facts:
                categories[f.category] = categories.get(f.category, 0) + 1
            return [ReadResourceContents(
                content=json.dumps({"total": len(all_facts), "by_category": categories}),
                mime_type="application/json",
            )]

        elif uri_str == "nova://lessons":
            lessons = _learning.get_all_lessons(limit=100)
            topics = [lesson.topic for lesson in lessons]
            return [ReadResourceContents(
                content=json.dumps({"total": len(topics), "topics": topics[:50]}),
                mime_type="application/json",
            )]

        elif uri_str == "nova://documents":
            if _retriever is None:
                return [ReadResourceContents(
                    content=json.dumps({"error": "Document store unavailable", "total": 0}),
                    mime_type="application/json",
                )]
            try:
                docs = _retriever.list_documents(limit=50)
                return [ReadResourceContents(
                    content=json.dumps({
                        "total": len(docs),
                        "documents": [
                            {"id": d.get("id", ""), "title": d.get("title", ""), "source": d.get("source", "")}
                            for d in docs
                        ],
                        "note": "Use nova_document_search to query content",
                    }),
                    mime_type="application/json",
                )]
            except Exception:
                return [ReadResourceContents(
                    content=json.dumps({"total": 0, "note": "Document metadata unavailable"}),
                    mime_type="application/json",
                )]

        elif uri_str == "nova://knowledge-graph/entities":
            try:
                top_entities = _kg.get_top_entities(limit=50)
                entity_counts = {r["subject"]: r["cnt"] for r in top_entities}
                stats = _kg.get_stats()
                return [ReadResourceContents(
                    content=json.dumps({
                        "total": stats.get("unique_entities", 0),
                        "top_entities": entity_counts,
                    }),
                    mime_type="application/json",
                )]
            except Exception:
                return [ReadResourceContents(
                    content=json.dumps({"total": 0, "entities": {}}),
                    mime_type="application/json",
                )]

        else:
            return [ReadResourceContents(
                content=json.dumps({"error": f"Unknown resource: {uri_str}"}),
                mime_type="application/json",
            )]

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _handle_memory_query(args: dict) -> list[TextContent]:
        query = args.get("query", "")
        limit = min(max(1, int(args.get("limit", 5))), 20)

        if not query:
            return _mcp_error("query is required", "validation", False)

        # Gather user facts — word-overlap scoring instead of naive substring
        from app.core.text_utils import normalize_words
        all_facts = _user_facts.get_all()
        query_words = normalize_words(query, min_length=2)
        scored_facts = []
        for f in all_facts:
            fact_words = normalize_words(f.key, min_length=2) | normalize_words(f.value, min_length=2)
            overlap = len(query_words & fact_words)
            if overlap >= 1:
                scored_facts.append((overlap, f))
        scored_facts.sort(key=lambda x: -x[0])
        matching_facts = [
            {"key": f.key, "value": f.value, "category": f.category, "confidence": f.confidence}
            for _, f in scored_facts
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
        hops = min(max(1, int(args.get("hops", 1))), 5)

        if not entity:
            return _mcp_error("entity is required", "validation", False)

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
        limit = min(max(1, int(args.get("limit", 5))), 20)

        if not query:
            return _mcp_error("query is required", "validation", False)

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
        top_k = min(max(1, int(args.get("top_k", 5))), 50)

        if not query:
            return _mcp_error("query is required", "validation", False)

        if _retriever is None:
            return _mcp_error(
                "Document search is unavailable (ChromaDB not initialized)",
                "transient",
                True,
            )

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
