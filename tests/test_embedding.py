"""Tests for the pluggable embedding function (app/core/embedding.py)."""
from __future__ import annotations

import app.core.embedding as emb


def test_prefixes_for_known_and_unknown():
    assert emb._prefixes_for("bge-m3") == ("", "")
    assert emb._prefixes_for("bge-m3:latest") == ("", "")
    q, d = emb._prefixes_for("mxbai-embed-large")
    assert q.startswith("Represent this sentence") and d == ""
    # unknown model -> no prefixes
    assert emb._prefixes_for("some-other-embedder") == ("", "")


def test_default_aliases_return_none(monkeypatch):
    for alias in ("", "default", "minilm", "all-MiniLM-L6-v2"):
        monkeypatch.setattr(emb.config, "EMBEDDING_MODEL", alias, raising=False)
        assert emb.get_embedding_function(force=True) is None


def test_ollama_ef_applies_doc_prefix_and_batches(monkeypatch):
    captured = []

    def fake_embed(self, chunk):
        captured.append(list(chunk))
        return [[0.1, 0.2, 0.3] for _ in chunk]

    monkeypatch.setattr(emb.OllamaEmbeddingFunction, "_embed", fake_embed)
    ef = emb.OllamaEmbeddingFunction("bge-m3", "http://x:11434",
                                     doc_prefix="search_document: ", batch=2)
    out = ef(["a", "b", "c"])
    assert len(out) == 3 and len(out[0]) == 3
    # prefix applied to every input
    assert all(t.startswith("search_document: ") for batch in captured for t in batch)
    # batched in groups of 2 -> [2, 1]
    assert [len(b) for b in captured] == [2, 1]


def test_ollama_ef_rejects_count_mismatch(monkeypatch):
    import json as _json

    class FakeResp:
        def __init__(self, payload):
            self._p = payload
        def read(self):
            return _json.dumps(self._p).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    # returns only 1 vector for 2 inputs -> must raise
    monkeypatch.setattr(emb.urllib.request, "urlopen",
                        lambda *a, **k: FakeResp({"embeddings": [[0.1, 0.2]]}))
    ef = emb.OllamaEmbeddingFunction("bge-m3", "http://x:11434")
    try:
        ef(["a", "b"])
        assert False, "expected RuntimeError on vector-count mismatch"
    except RuntimeError:
        pass


def test_get_embedding_function_probe_failure_falls_back(monkeypatch):
    # A model that errors on probe must resolve to None (MiniLM fallback).
    monkeypatch.setattr(emb.config, "EMBEDDING_MODEL", "broken-embedder", raising=False)

    def boom(self, input):
        raise RuntimeError("probe 400")

    monkeypatch.setattr(emb.OllamaEmbeddingFunction, "__call__", boom)
    assert emb.get_embedding_function(force=True) is None
