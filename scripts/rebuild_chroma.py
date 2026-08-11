"""Rebuild every ChromaDB collection from SQLite — the source of truth.

Disaster-recovery doctrine (audit 2026-07-08): the vector store is DERIVED
data. SQLite (nova.db) holds every fact, lesson, and document chunk in raw
text; this script re-embeds all of it into a fresh Chroma persistence dir.
So Chroma needs no backup of its own — a verified nova.db snapshot plus this
script IS the vector-store backup.

Run inside the container (embedder must be reachable):

    docker exec nova-app python scripts/rebuild_chroma.py

IMPORTANT (from the 2026-06-09 postmortem): if you run this against a LIVE
app, the running process holds stale collection handles afterwards — restart
with `docker compose up -d nova --force-recreate` when done.
"""

from __future__ import annotations

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("rebuild_chroma")


async def main() -> int:
    from app.database import get_db
    from app.core.kg import KnowledgeGraph
    from app.core.learning import LearningEngine

    db = get_db()
    failures = 0

    kg = KnowledgeGraph(db)
    try:
        n = await asyncio.to_thread(kg.reindex_kg_facts)
        logger.info("kg_facts reindexed: %s vectors", n)
    except Exception as e:
        logger.error("kg_facts reindex FAILED: %s", e)
        failures += 1

    learning = LearningEngine(db=db)
    try:
        n = await asyncio.to_thread(learning.reindex_lessons)
        logger.info("lessons reindexed: %s vectors", n)
    except Exception as e:
        logger.error("lessons reindex FAILED: %s", e)
        failures += 1

    # Documents: Retriever.ingest is idempotent by doc_id; re-ingest raw text
    # stored in SQLite. If the documents table doesn't exist or is empty this
    # is a no-op.
    try:
        from app.core.retriever import Retriever
        retriever = Retriever()
        rows = db.fetchall(
            "SELECT DISTINCT doc_id, title, source FROM document_chunks"
        ) if db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='document_chunks'"
        ) else []
        redone = 0
        for row in rows:
            chunks = db.fetchall(
                "SELECT content FROM document_chunks WHERE doc_id = ? ORDER BY chunk_index",
                (row["doc_id"],),
            )
            text = "\n\n".join(c["content"] for c in chunks)
            if text.strip():
                await retriever.ingest(text=text, title=row["title"] or "restored",
                                       source=row["source"] or "rebuild")
                redone += 1
        logger.info("documents re-ingested: %s", redone)
    except Exception as e:
        logger.error("document re-ingest FAILED (non-fatal if unused): %s", e)

    if failures:
        logger.error("rebuild finished with %d failure(s)", failures)
        return 1
    logger.info("rebuild complete — restart the app to drop stale collection handles")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
