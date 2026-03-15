"""Hybrid retrieval — ChromaDB vector search + SQLite FTS5 BM25 + RRF fusion.

Simple and effective: two signals, reciprocal rank fusion, done.
No reranker until we have evidence retrieval quality is bad.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import uuid
from dataclasses import dataclass

from app.config import config
from app.database import get_db

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A retrieved document chunk."""
    chunk_id: str
    document_id: str
    content: str
    score: float = 0.0
    source: str = ""
    title: str = ""


class Retriever:
    """Hybrid search: ChromaDB vectors + SQLite FTS5 BM25 + RRF fusion."""

    def __init__(self, db=None, chroma_collection=None):
        self._db = db or get_db()
        self._collection = chroma_collection
        self._chroma_client = None
        self._collection_lock = threading.Lock()

    def _get_collection(self):
        """Lazy-init ChromaDB collection."""
        if self._collection is not None:
            return self._collection
        with self._collection_lock:
            if self._collection is None:
                try:
                    import chromadb
                    self._chroma_client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
                    self._collection = self._chroma_client.get_or_create_collection(
                        name="documents",
                        metadata={"hnsw:space": "cosine"},
                    )
                except Exception as e:
                    logger.error("Failed to init ChromaDB: %s", e)
                    raise
        return self._collection

    def close(self):
        """Release ChromaDB client resources."""
        if self._chroma_client is not None:
            del self._chroma_client
            self._chroma_client = None
            self._collection = None

    async def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Hybrid search: vector + BM25 + RRF fusion."""
        top_k = top_k or config.RETRIEVAL_TOP_K

        # Run both searches in parallel
        vector_task = asyncio.to_thread(self._vector_search, query, top_k * 2)
        fts_task = asyncio.to_thread(self._fts5_search, query, top_k * 2)

        vector_results, fts_results = await asyncio.gather(
            vector_task, fts_task, return_exceptions=True
        )

        # Handle failures gracefully
        if isinstance(vector_results, Exception):
            logger.warning("Vector search failed: %s", vector_results)
            vector_results = []
        if isinstance(fts_results, Exception):
            logger.warning("FTS5 search failed: %s", fts_results)
            fts_results = []

        if not vector_results and not fts_results:
            return []

        # Reciprocal Rank Fusion
        fused = _reciprocal_rank_fusion(vector_results, fts_results, k=config.RRF_K)

        # Entity relevance guard — drop chunks where query entities don't appear
        fused = _entity_relevance_filter(query, fused[:top_k])

        return fused[:top_k]

    def _vector_search(self, query: str, top_k: int) -> list[Chunk]:
        """Search ChromaDB using embeddings."""
        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return []

            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            chunks = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    # ChromaDB cosine distance: 0 = identical, 2 = opposite
                    # Convert to similarity score: 1 - (distance / 2)
                    score = 1.0 - (distance / 2.0)
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        document_id=metadata.get("document_id", ""),
                        content=results["documents"][0][i],
                        score=score,
                        source=metadata.get("source", ""),
                        title=metadata.get("title", ""),
                    ))
            return chunks

        except Exception as e:
            logger.warning("Vector search error: %s", e)
            return []

    def _fts5_search(self, query: str, top_k: int) -> list[Chunk]:
        """Search SQLite FTS5 using BM25."""
        try:
            # Escape FTS5 special characters
            safe_query = _escape_fts5(query)
            if not safe_query.strip():
                return []

            rows = self._db.fetchall(
                """SELECT chunk_id, document_id, content, rank
                   FROM chunks_fts
                   WHERE chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, top_k),
            )

            chunks = []
            for row in rows:
                # FTS5 rank is negative (lower = better match)
                # Normalize to 0-1 score
                raw_rank = abs(float(row["rank"])) if row["rank"] else 0
                score = raw_rank / (1.0 + raw_rank)  # Sigmoid-like normalization
                chunks.append(Chunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    score=score,
                ))
            return chunks

        except Exception as e:
            logger.warning("FTS5 search error: %s", e)
            return []

    async def ingest(
        self,
        text: str,
        *,
        source: str = "",
        title: str = "",
        doc_id: str | None = None,
    ) -> tuple[str, int]:
        """Ingest text: chunk → store in ChromaDB + FTS5.

        Returns (document_id, chunk_count).
        """
        doc_id = doc_id or str(uuid.uuid4())
        chunks = _recursive_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        if not chunks:
            return doc_id, 0

        # Remove old chunks from ChromaDB if re-ingesting
        try:
            collection = self._get_collection()
            old_results = await asyncio.to_thread(collection.get, where={"document_id": doc_id}, include=[])
            if old_results["ids"]:
                await asyncio.to_thread(collection.delete, ids=old_results["ids"])
        except Exception as e:
            logger.warning("Failed to clean old chunks for doc %s: %s", doc_id, e)

        # Prepare chunk data
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        # Store document metadata + FTS5 chunks atomically
        with self._db.transaction() as tx:
            tx.execute(
                "INSERT OR REPLACE INTO documents (id, title, source, chunk_count) VALUES (?, ?, ?, ?)",
                (doc_id, title or source or "Untitled", source, len(chunks)),
            )
            # Remove old chunks if re-ingesting (prevents duplicates)
            tx.execute("DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,))
            # Insert into FTS5
            tx.executemany(
                "INSERT INTO chunks_fts (chunk_id, document_id, content) VALUES (?, ?, ?)",
                [(cid, doc_id, text) for cid, text in zip(chunk_ids, chunks)],
            )

        # Insert into ChromaDB
        try:
            collection = self._get_collection()
            await asyncio.to_thread(
                collection.add,
                ids=chunk_ids,
                documents=chunks,
                metadatas=[
                    {"document_id": doc_id, "source": source, "title": title, "chunk_index": i}
                    for i in range(len(chunks))
                ],
            )
        except Exception as e:
            logger.error("ChromaDB ingest failed: %s", e)

        return doc_id, len(chunks)

    def get_document(self, doc_id: str) -> dict | None:
        """Get document metadata."""
        row = self._db.fetchone("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return dict(row) if row else None

    def list_documents(self, limit: int = 50) -> list[dict]:
        """List all documents."""
        rows = self._db.fetchall(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks from both stores."""
        doc = self.get_document(doc_id)
        if not doc:
            return False

        # Delete from FTS5
        self._db.execute("DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,))
        self._db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

        # Delete from ChromaDB
        try:
            collection = self._get_collection()
            # Get all chunk IDs for this document
            results = collection.get(
                where={"document_id": doc_id},
                include=[],
            )
            if results["ids"]:
                collection.delete(ids=results["ids"])
        except Exception as e:
            logger.warning("ChromaDB delete failed: %s", e)

        return True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    *result_lists: list[Chunk],
    k: int = 60,
) -> list[Chunk]:
    """Reciprocal Rank Fusion across multiple result lists.

    RRF score = sum(1 / (k + rank_i)) for each list the doc appears in.
    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank + 1)
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

    # Sort by fused score descending
    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    result = []
    for cid in sorted_ids:
        chunk = chunk_map[cid]
        chunk.score = scores[cid]
        result.append(chunk)

    return result


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursively split text into chunks by paragraphs, then sentences, then characters.

    2026 benchmarks show simple recursive splitting (69%) outperforms semantic chunking (54%).
    """
    if not text or not text.strip():
        return []

    # Approximate tokens: ~4 chars per token
    max_chars = chunk_size * 4
    overlap_chars = overlap * 4

    if len(text) <= max_chars:
        return [text.strip()]

    # Try splitting by double newlines (paragraphs)
    chunks = _split_by_separator(text, "\n\n", max_chars, overlap_chars)
    if chunks:
        return chunks

    # Fall back to single newlines
    chunks = _split_by_separator(text, "\n", max_chars, overlap_chars)
    if chunks:
        return chunks

    # Fall back to sentences
    chunks = _split_by_separator(text, ". ", max_chars, overlap_chars)
    if chunks:
        return chunks

    # Last resort: character-level splitting
    result = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        result.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap_chars
    return [c for c in result if c]


def _split_by_separator(text: str, sep: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split text by separator, keeping chunks under max_chars."""
    parts = text.split(sep)
    if len(parts) <= 1:
        return []

    chunks = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            # Overlap: keep the last part of the previous chunk
            if overlap_chars > 0 and len(current) > overlap_chars:
                current = current[-overlap_chars:] + sep + part
            else:
                current = part
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks if len(chunks) > 0 else []


from app.core.text_utils import STOP_WORDS as _STOP_WORDS, content_words as _content_words  # noqa: E402


def _entity_relevance_filter(
    query: str, chunks: list[Chunk], threshold: float = 0.3
) -> list[Chunk]:
    """Drop retrieved chunks where query content words don't appear enough.

    Prevents the embedding-collapse bug where "capital of France" retrieves
    "capital of Australia" because embedding models produce near-identical vectors.
    Uses lower threshold for short queries. Always returns at least min(3, available) results.
    """
    query_words = _content_words(query)
    if len(query_words) < 2:
        return chunks  # Too few words to filter meaningfully

    # Lower threshold for short queries (≤3 content words)
    effective_threshold = 0.2 if len(query_words) <= 3 else threshold

    filtered = []
    for chunk in chunks:
        chunk_words = _content_words(chunk.content)
        overlap = len(query_words & chunk_words)
        ratio = overlap / len(query_words)
        if ratio >= effective_threshold:
            filtered.append(chunk)

    # Always return at least min(3, available) results
    min_results = min(3, len(chunks))
    return filtered if len(filtered) >= min_results else chunks[:min_results]


def _escape_fts5(query: str) -> str:
    """Escape special FTS5 characters for safe matching."""
    # Remove FTS5 operators that could cause syntax errors
    safe = query.replace('"', "").replace("'", "")
    # Remove other FTS5 special chars
    for char in ["*", "(", ")", "{", "}", ":", "^", "~", "+", "-", "?", "!", "@", "#", "$", "%", "&", ".", ",", ";", "/", "=", "|", "\\", "<", ">"]:
        safe = safe.replace(char, " ")
    # Collapse whitespace
    return " ".join(safe.split())
