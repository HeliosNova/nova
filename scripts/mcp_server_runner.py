"""Nova MCP Server runner — entry point for stdio-based MCP transport.

This script is spawned as a subprocess by MCP clients (Claude Code, Cursor, etc.).
It initializes the database and required services, then runs the MCP server
over stdin/stdout using the standard MCP stdio transport.

Usage (by MCP client config):
    python scripts/mcp_server_runner.py

Environment variables:
    DB_PATH       — path to Nova's SQLite database (default: /data/nova.db)
    CHROMADB_PATH — path to ChromaDB storage (default: /data/chromadb)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app.*` imports work
# when running as `python scripts/mcp_server_runner.py` from any cwd.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Configure logging to stderr (stdout is reserved for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("nova_mcp")


async def main() -> None:
    """Initialize services and run the MCP server over stdio."""

    # Late imports — after sys.path is set up
    from mcp.server.stdio import stdio_server

    from app.config import config
    from app.core.kg import KnowledgeGraph
    from app.core.learning import LearningEngine
    from app.core.memory import ConversationStore, UserFactStore
    from app.core.retriever import Retriever
    from app.database import get_db
    from app.mcp_server import create_mcp_server

    # --- Initialize database ---
    logger.info("Initializing database at %s", config.DB_PATH)
    db = get_db()
    db.init_schema()

    # --- Create service instances ---
    user_facts = UserFactStore(db)
    conversations = ConversationStore(db)
    learning = LearningEngine(db)
    kg = KnowledgeGraph(db)

    retriever = None
    try:
        retriever = Retriever(db)
        logger.info("Retriever initialized (ChromaDB at %s)", config.CHROMADB_PATH)
    except Exception as e:
        logger.warning("Retriever unavailable (document search disabled): %s", e)

    # --- Build the MCP server ---
    server = create_mcp_server(
        db,
        user_facts=user_facts,
        conversations=conversations,
        learning=learning,
        kg=kg,
        retriever=retriever,
    )

    logger.info("Nova MCP server starting (name=%s)...", config.MCP_SERVER_NAME)

    # --- Run over stdio transport ---
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Nova MCP server stopped.")
    except Exception:
        logger.exception("Nova MCP server crashed")
        sys.exit(1)
