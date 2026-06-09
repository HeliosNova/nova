"""Pluggable embedding function for ChromaDB collections.

Historically every collection used ChromaDB's bundled default
(all-MiniLM-L6-v2, 384-dim, 2020-era). A 2026 offline bake-off over 80
paraphrase->fact retrieval queries (entity kept, relation paraphrased — the
case keyword search misses) measured MiniLM *last* of five candidates:

    embedder                 r@1   r@3   r@5   r@10   MRR
    snowflake-arctic-embed2  0.93  1.00  1.00  1.00   0.960
    bge-m3                   0.89  1.00  1.00  1.00   0.938
    mxbai-embed-large        0.86  0.95  1.00  1.00   0.917
    qwen3-embedding:0.6b     0.85  0.97  0.99  0.99   0.911
    all-MiniLM-L6 (old)      0.84  0.95  0.96  0.97   0.899

bge-m3 is chosen: it recovers every target by rank 3 (== arctic at the depth
that matters for top-K injection), is *symmetric* (no query/doc prefix, so it
drops cleanly into ChromaDB's embedding-function abstraction), multilingual,
and a production standard.

This module exposes an Ollama-backed EmbeddingFunction plus a cached factory
`get_embedding_function()` that returns it ONLY when EMBEDDING_MODEL names a
reachable Ollama embedder that returns a valid vector on a probe. Otherwise it
returns None, so the collection falls back to ChromaDB's default — a sovereign
install on modest hardware that never pulled the embedder still works.
"""
from __future__ import annotations

import json
import logging
import urllib.request

from ..config import config

logger = logging.getLogger(__name__)

# Models that mean "use ChromaDB's bundled default (MiniLM), no Ollama call".
_DEFAULT_ALIASES = {"", "default", "minilm", "all-minilm-l6-v2", "all-minilm"}

# Per-model asymmetric prefixes. bge-m3 is symmetric (no prefix). Kept here so a
# future swap to an asymmetric model (arctic 'query: ', nomic 'search_*: ') only
# needs a table entry — though asymmetric models also need query-time handling.
_PREFIXES = {
    # model-name-prefix: (query_prefix, doc_prefix)
    "bge-m3": ("", ""),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}


def _prefixes_for(model: str) -> tuple[str, str]:
    base = model.split(":")[0]
    for key, val in _PREFIXES.items():
        if base == key or base.startswith(key):
            return val
    return ("", "")


class OllamaEmbeddingFunction:
    """ChromaDB EmbeddingFunction backed by Ollama's /api/embed.

    Synchronous (ChromaDB calls embedders synchronously). Uses stdlib urllib so
    it never touches the asyncio event loop. Applies a document prefix to every
    input by default; query-time code that needs the *query* prefix for an
    asymmetric model must pass embeddings explicitly (bge-m3 needs neither).
    """

    def __init__(self, model: str, base_url: str, doc_prefix: str = "",
                 timeout: int = 120, batch: int = 64):
        self._model = model
        self._url = base_url.rstrip("/") + "/api/embed"
        self._doc_prefix = doc_prefix
        self._timeout = timeout
        self._batch = batch

    # ChromaDB requires the parameter to be named `input`.
    def __call__(self, input):  # noqa: A002 - name mandated by ChromaDB
        texts = list(input)
        if self._doc_prefix:
            texts = [self._doc_prefix + t for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            out.extend(self._embed(texts[i:i + self._batch]))
        return out

    def _embed(self, chunk: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self._model, "input": chunk}).encode()
        req = urllib.request.Request(
            self._url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            data = json.loads(r.read().decode())
        embs = data.get("embeddings")
        if not embs or len(embs) != len(chunk):
            raise RuntimeError(
                f"Ollama embed returned {len(embs) if embs else 0} vectors for "
                f"{len(chunk)} inputs (model={self._model})")
        return embs

    # Some ChromaDB versions want a stable name for (de)serialization.
    @staticmethod
    def name() -> str:
        return "ollama"


_CACHED: object = False  # sentinel: not yet resolved


def get_embedding_function(force: bool = False):
    """Return an OllamaEmbeddingFunction for the configured model, or None to
    fall back to ChromaDB's default. Result is cached after the first probe."""
    global _CACHED
    if _CACHED is not False and not force:
        return _CACHED

    model = (getattr(config, "EMBEDDING_MODEL", "") or "").strip()
    if model.lower() in _DEFAULT_ALIASES:
        _CACHED = None
        return None

    base_url = getattr(config, "OLLAMA_URL", "http://ollama:11434")
    _, doc_prefix = _prefixes_for(model)
    ef = OllamaEmbeddingFunction(model, base_url, doc_prefix=doc_prefix)
    # Probe: a model can be "present" yet fail on /api/embed (nomic-v2-moe 400s).
    try:
        vec = ef(["probe"])
        if not vec or not isinstance(vec[0], (list, tuple)) or len(vec[0]) < 8:
            raise RuntimeError("probe returned no usable vector")
        logger.info("Embedding model active: %s (dim=%d)", model, len(vec[0]))
        _CACHED = ef
    except Exception as e:
        logger.warning(
            "Embedding model %r not usable via Ollama (%s) — falling back to "
            "ChromaDB default (all-MiniLM-L6-v2)", model, e)
        _CACHED = None
    return _CACHED
