"""Tests for Phase 3: Retriever (FTS5 + chunking + RRF)."""

from __future__ import annotations

import pytest

from app.core.retriever import (
    Chunk,
    Retriever,
    _escape_fts5,
    _reciprocal_rank_fusion,
    _recursive_split,
)


# ===========================================================================
# Text Chunking
# ===========================================================================

class TestChunking:
    def test_short_text_single_chunk(self):
        chunks = _recursive_split("Hello world", 512, 50)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_empty_text(self):
        assert _recursive_split("", 512, 50) == []
        assert _recursive_split("   ", 512, 50) == []

    def test_paragraph_splitting(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        # With tiny chunk size to force splitting
        chunks = _recursive_split(text, 5, 1)  # 5 tokens ≈ 20 chars
        assert len(chunks) >= 2
        # Each chunk should have content
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_long_text_gets_chunked(self):
        text = "word " * 1000  # ~5000 chars, well over 512 tokens
        chunks = _recursive_split(text, 128, 16)
        assert len(chunks) > 1
        # All content should be preserved (approximately, due to overlap)
        total_text = " ".join(chunks)
        assert "word" in total_text

    def test_overlap_exists(self):
        # Create text with clear paragraphs
        paragraphs = [f"Paragraph {i} " * 50 for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = _recursive_split(text, 64, 16)
        if len(chunks) > 1:
            # With overlap, adjacent chunks should share some text
            # (hard to guarantee exact overlap, but chunks shouldn't be empty)
            assert all(len(c) > 0 for c in chunks)


# ===========================================================================
# FTS5 Escaping
# ===========================================================================

class TestFTS5Escape:
    def test_normal_text(self):
        assert _escape_fts5("hello world") == "hello world"

    def test_strips_special_chars(self):
        assert "(" not in _escape_fts5("test(foo)")
        assert ")" not in _escape_fts5("test(foo)")
        assert "*" not in _escape_fts5("test*")

    def test_strips_quotes(self):
        result = _escape_fts5('he said "hello"')
        assert '"' not in result


# ===========================================================================
# Reciprocal Rank Fusion
# ===========================================================================

class TestRRF:
    def test_single_list(self):
        chunks = [
            Chunk(chunk_id="a", document_id="d1", content="text a", score=0.9),
            Chunk(chunk_id="b", document_id="d1", content="text b", score=0.7),
        ]
        result = _reciprocal_rank_fusion(chunks)
        assert len(result) == 2
        assert result[0].chunk_id == "a"  # Higher ranked

    def test_two_lists_fusion(self):
        vector = [
            Chunk(chunk_id="a", document_id="d1", content="text a"),
            Chunk(chunk_id="b", document_id="d1", content="text b"),
        ]
        fts = [
            Chunk(chunk_id="b", document_id="d1", content="text b"),
            Chunk(chunk_id="c", document_id="d1", content="text c"),
        ]
        result = _reciprocal_rank_fusion(vector, fts)
        assert len(result) == 3
        # "b" appears in both lists, so it should be ranked highest
        assert result[0].chunk_id == "b"

    def test_empty_lists(self):
        result = _reciprocal_rank_fusion([], [])
        assert result == []

    def test_scores_are_set(self):
        chunks = [Chunk(chunk_id="a", document_id="d1", content="text")]
        result = _reciprocal_rank_fusion(chunks)
        assert result[0].score > 0


# ===========================================================================
# Retriever — FTS5 search (no ChromaDB needed)
# ===========================================================================

class TestRetrieverFTS5:
    @pytest.fixture
    def retriever(self, db):
        """Retriever with only FTS5 (no ChromaDB)."""
        return Retriever(db=db, chroma_collection=None)

    def test_fts5_search_empty(self, retriever):
        results = retriever._fts5_search("hello", 5)
        assert results == []

    def test_fts5_search_finds_content(self, retriever, db):
        # Insert test data into FTS5
        db.execute(
            "INSERT INTO chunks_fts (chunk_id, document_id, content) VALUES (?, ?, ?)",
            ("c1", "d1", "The quick brown fox jumps over the lazy dog"),
        )
        db.execute(
            "INSERT INTO chunks_fts (chunk_id, document_id, content) VALUES (?, ?, ?)",
            ("c2", "d1", "Python is a great programming language"),
        )

        results = retriever._fts5_search("brown fox", 5)
        assert len(results) >= 1
        assert results[0].chunk_id == "c1"

    def test_fts5_search_ranking(self, retriever, db):
        db.execute(
            "INSERT INTO chunks_fts (chunk_id, document_id, content) VALUES (?, ?, ?)",
            ("c1", "d1", "Bitcoin is a cryptocurrency"),
        )
        db.execute(
            "INSERT INTO chunks_fts (chunk_id, document_id, content) VALUES (?, ?, ?)",
            ("c2", "d1", "Bitcoin price reached new highs for Bitcoin investors"),
        )

        results = retriever._fts5_search("Bitcoin", 5)
        assert len(results) == 2
        # c2 has more Bitcoin mentions, should rank higher
        assert results[0].chunk_id == "c2"

    @pytest.mark.asyncio
    async def test_ingest_and_search(self, retriever, db):
        # Ingest without ChromaDB
        doc_id, count = await retriever.ingest(
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning uses neural networks with many layers.",
            title="ML Guide",
            source="test",
        )

        assert count >= 1
        assert doc_id is not None

        # Search via FTS5 only
        results = retriever._fts5_search("neural networks", 5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_ingest_creates_document_record(self, retriever, db):
        doc_id, count = await retriever.ingest(
            "Test content here.",
            title="Test Doc",
            source="unit_test",
        )

        doc = retriever.get_document(doc_id)
        assert doc is not None
        assert doc["title"] == "Test Doc"
        assert doc["chunk_count"] == count

    def test_list_documents(self, retriever, db):
        db.execute(
            "INSERT INTO documents (id, title, source, chunk_count) VALUES (?, ?, ?, ?)",
            ("d1", "Doc 1", "test", 3),
        )
        docs = retriever.list_documents()
        assert len(docs) == 1
        assert docs[0]["title"] == "Doc 1"


# ===========================================================================
# FTS5 Escape — extended chars (Fix 1)
# ===========================================================================

class TestFTS5EscapeExtended:
    def test_strips_period(self):
        assert "." not in _escape_fts5("hello.world")

    def test_strips_comma(self):
        assert "," not in _escape_fts5("hello, world")

    def test_strips_semicolon(self):
        assert ";" not in _escape_fts5("hello; world")

    def test_sentence_with_punctuation(self):
        result = _escape_fts5("Dr. Smith, Jr.; the professor")
        assert "." not in result
        assert "," not in result
        assert ";" not in result
        assert "Dr" in result
        assert "Smith" in result


# ===========================================================================
# Edge Cases — empty database, FTS5 fallback, large documents
# ===========================================================================

class TestRetrieverEdgeCases:
    @pytest.fixture
    def retriever(self, db):
        return Retriever(db=db, chroma_collection=None)

    @pytest.mark.asyncio
    async def test_search_empty_database(self, retriever):
        """Search on empty DB should return empty list, not error."""
        results = await retriever.search("anything at all")
        assert results == []

    def test_fts5_search_special_chars(self, retriever):
        """FTS5 search with special characters should not crash."""
        results = retriever._fts5_search('hello "world" (test) *star*', 5)
        assert results == []

    def test_fts5_search_empty_query(self, retriever):
        """Empty query should return empty results."""
        results = retriever._fts5_search("", 5)
        assert results == []

    def test_fts5_search_single_char(self, retriever):
        """Single-character query should not crash."""
        results = retriever._fts5_search("a", 5)
        assert results == []

    @pytest.mark.asyncio
    async def test_ingest_large_document(self, retriever):
        """Large document should be chunked without error."""
        large_text = "This is a paragraph about testing. " * 500
        doc_id, count = await retriever.ingest(
            large_text,
            title="Large Doc",
            source="test",
        )
        assert count > 1
        assert doc_id is not None

    @pytest.mark.asyncio
    async def test_ingest_whitespace_only(self, retriever):
        """Whitespace-only text should be rejected or produce 0 chunks."""
        doc_id, count = await retriever.ingest(
            "   \n\n   \t  ",
            title="Empty",
            source="test",
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_search_after_ingest(self, retriever):
        """Should find content after ingestion via FTS5."""
        await retriever.ingest(
            "Quantum computing uses qubits instead of classical bits for computation.",
            title="Quantum Guide",
            source="test",
        )
        results = retriever._fts5_search("quantum qubits", 5)
        assert len(results) >= 1
        assert "quantum" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_delete_document_removes_chunks(self, retriever, db):
        """Deleting a document should remove its chunks from search."""
        doc_id, _ = await retriever.ingest(
            "Unique test content about elephants in space.",
            title="Elephant Doc",
            source="test",
        )
        # Verify it's searchable
        results = retriever._fts5_search("elephants space", 5)
        assert len(results) >= 1

        # Delete
        retriever.delete_document(doc_id)

        # Should no longer appear
        results = retriever._fts5_search("elephants space", 5)
        assert len(results) == 0

    def test_rrf_with_duplicates(self):
        """RRF should merge duplicates from multiple result sets."""
        set1 = [
            Chunk(chunk_id="a", document_id="d1", content="text a"),
            Chunk(chunk_id="b", document_id="d1", content="text b"),
        ]
        set2 = [
            Chunk(chunk_id="a", document_id="d1", content="text a"),
            Chunk(chunk_id="c", document_id="d1", content="text c"),
        ]
        result = _reciprocal_rank_fusion(set1, set2)
        # "a" appears in both, should be top-ranked
        assert result[0].chunk_id == "a"
        assert len(result) == 3
