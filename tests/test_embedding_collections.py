"""open_collection() must give every Chroma collection ONE embedder and heal a
stale-dimension collection left by a pre-upgrade default.

Background: lessons/kg/retriever were wired to the bge-m3 embedding function on
2026-06-09, but reflexions/skills/active_memory were NOT — they silently used
ChromaDB's 384-dim MiniLM default (production was caught downloading the MiniLM
ONNX mid-chat). Mixed embedders + a stale 384-dim collection meeting a 1024-dim
function = InvalidDimension at query time. open_collection centralizes both.
"""

from unittest.mock import MagicMock, patch

from app.core import embedding


def _client_with(collection):
    c = MagicMock()
    c.get_or_create_collection.return_value = collection
    return c


class TestOpenCollection:
    def test_no_embedder_returns_plain_collection(self):
        coll = MagicMock()
        client = _client_with(coll)
        with patch.object(embedding, "get_embedding_function", return_value=None):
            out = embedding.open_collection(client, "lessons")
        assert out is coll
        # No embedding_function kwarg when none is configured
        _, kw = client.get_or_create_collection.call_args
        assert "embedding_function" not in kw

    def test_embedder_attached_when_available(self):
        coll = MagicMock()
        coll.count.return_value = 0  # empty: no dim probe
        client = _client_with(coll)
        ef = MagicMock()
        with patch.object(embedding, "get_embedding_function", return_value=ef):
            embedding.open_collection(client, "reflexions")
        _, kw = client.get_or_create_collection.call_args
        assert kw["embedding_function"] is ef

    def test_stale_dimension_triggers_rebuild_and_reindex(self):
        stale = MagicMock()
        stale.count.return_value = 5
        stale.query.side_effect = Exception("Collection expecting embedding with dimension of 384, got 1024")
        fresh = MagicMock()
        fresh.count.return_value = 0
        client = MagicMock()
        client.get_or_create_collection.side_effect = [stale, fresh]
        ef = MagicMock()
        reindex = MagicMock(return_value=42)
        with patch.object(embedding, "get_embedding_function", return_value=ef):
            out = embedding.open_collection(client, "reflexions", reindex=reindex)
        client.delete_collection.assert_called_once_with("reflexions")
        assert out is fresh
        reindex.assert_called_once_with(fresh)

    def test_non_dimension_query_error_propagates(self):
        coll = MagicMock()
        coll.count.return_value = 3
        coll.query.side_effect = RuntimeError("disk I/O error")
        client = _client_with(coll)
        ef = MagicMock()
        with patch.object(embedding, "get_embedding_function", return_value=ef):
            try:
                embedding.open_collection(client, "kg_facts")
                assert False, "should have re-raised the non-dimension error"
            except RuntimeError as e:
                assert "disk I/O" in str(e)
        client.delete_collection.assert_not_called()

    def test_healthy_collection_not_rebuilt(self):
        coll = MagicMock()
        coll.count.return_value = 100
        coll.query.return_value = {"ids": [[]]}  # probe succeeds
        client = _client_with(coll)
        ef = MagicMock()
        with patch.object(embedding, "get_embedding_function", return_value=ef):
            out = embedding.open_collection(client, "lessons", reindex=MagicMock())
        assert out is coll
        client.delete_collection.assert_not_called()
