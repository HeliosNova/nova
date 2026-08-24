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
import threading
import time
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
        # Single-query coalescing cache. Per-turn the brain fires several
        # retrievals (lessons, KG, reflexions, success-patterns) that each embed
        # the SAME query; running them concurrently used to flood the one GPU
        # embedder so calls queued and timed out. Cache (short TTL) + in-flight
        # dedup so identical concurrent/recent query embeds compute ONCE.
        self._cache_lock = threading.Lock()
        self._cache: dict[str, tuple[float, list[float]]] = {}   # key -> (expiry, vec)
        self._inflight: dict[str, threading.Event] = {}
        self._cache_ttl = 30.0
        self._cache_max = 256

    # ChromaDB requires the parameter to be named `input`.
    def __call__(self, input):  # noqa: A002 - name mandated by ChromaDB
        texts = list(input)
        # Coalesce the single-text QUERY case (the retrieval hot path). Batch
        # (indexing) embeds are distinct texts — no coalescing benefit.
        if len(texts) == 1:
            return [self._embed_one_cached(texts[0])]
        if self._doc_prefix:
            texts = [self._doc_prefix + t for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            out.extend(self._embed(texts[i:i + self._batch]))
        return out

    def _embed_one_cached(self, text: str) -> list[float]:
        key = (self._doc_prefix + text) if self._doc_prefix else text
        now = time.monotonic()
        leader = False
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit and hit[0] > now:
                return hit[1]
            ev = self._inflight.get(key)
            if ev is None:
                ev = threading.Event()
                self._inflight[key] = ev
                leader = True
        if not leader:
            # Another thread is already embedding this exact query — wait for it.
            ev.wait(timeout=self._timeout + 5)
            with self._cache_lock:
                hit = self._cache.get(key)
            if hit:
                return hit[1]
            # Leader failed/evicted (rare) — fall through and compute directly.
            return self._embed([key])[0]
        try:
            vec = self._embed([key])[0]
            with self._cache_lock:
                self._cache[key] = (now + self._cache_ttl, vec)
                if len(self._cache) > self._cache_max:
                    self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
            return vec
        finally:
            with self._cache_lock:
                done = self._inflight.pop(key, None)
            if done is not None:
                done.set()

    def _embed(self, chunk: list[str]) -> list[list[float]]:
        # num_gpu=0: run the embedder on CPU. With the 27B + the owner's VRAM
        # reserve resident, the 1.2GB embedder became UNLOADABLE on GPU — every
        # embed call waited on VRAM that never freed, all four knowledge
        # retrievals (kg/lessons/reflexions/patterns) hit their 5s context
        # timeout, and chat answered with ZERO injected knowledge (found
        # 2026-07-07: kg-retrieval causal-fix 0.83→0.0). A 1024-dim embed costs
        # ~100-300ms on CPU and never competes with generation.
        payload = json.dumps({"model": self._model, "input": chunk,
                              "options": {"num_gpu": 0}}).encode()
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
# If we fell back to the default embedder because a probe FAILED (transient), the
# monotonic deadline after which get_embedding_function re-probes — so a startup
# hiccup on the embed instance (still pulling bge-m3, compose ordering) doesn't pin
# the whole process on 2020-era MiniLM for its lifetime (audit 2026-08-17). A
# DEFINITIVE default (the model IS an alias for the built-in) leaves this at 0.0.
_FALLBACK_UNTIL: float = 0.0
_REPROBE_COOLDOWN_S: float = 300.0


def get_embedding_function(force: bool = False):
    """Return an OllamaEmbeddingFunction for the configured model, or None to
    fall back to ChromaDB's default. Result is cached after the first probe."""
    global _CACHED, _FALLBACK_UNTIL
    if _CACHED is not False and not force:
        import time
        should_reprobe = (_CACHED is None and _FALLBACK_UNTIL
                          and time.monotonic() >= _FALLBACK_UNTIL)
        if not should_reprobe:
            return _CACHED
        logger.info("[embedding] re-probing embedder after prior fallback (cooldown elapsed)")

    model = (getattr(config, "EMBEDDING_MODEL", "") or "").strip()
    if model.lower() in _DEFAULT_ALIASES:
        _CACHED = None
        _FALLBACK_UNTIL = 0.0  # definitive: caller wants the built-in default
        return None

    # Prefer the dedicated CPU embed instance (isolated request queue → never
    # waits behind a 27B generation); fall back to the main Ollama if it isn't
    # reachable yet (e.g. still pulling bge-m3, or single-instance dev).
    _main = getattr(config, "OLLAMA_URL", "http://ollama:11434")
    _embed = (getattr(config, "EMBED_OLLAMA_URL", "") or "").strip() or _main
    _, doc_prefix = _prefixes_for(model)

    def _try(url):
        e = OllamaEmbeddingFunction(model, url, doc_prefix=doc_prefix)
        e(["probe"])   # a model can be "present" yet 400 on /api/embed
        return e

    base_url = _embed
    ef = OllamaEmbeddingFunction(model, base_url, doc_prefix=doc_prefix)
    # Probe: a model can be "present" yet fail on /api/embed (nomic-v2-moe 400s).
    try:
        if _embed != _main:
            try:
                ef = _try(_embed)
                logger.info("[embedding] using dedicated embed instance %s", _embed)
            except Exception as _e:
                logger.warning("[embedding] embed instance %s unreachable (%s) — falling back to %s",
                               _embed, _e, _main)
                base_url = _main
                ef = OllamaEmbeddingFunction(model, base_url, doc_prefix=doc_prefix)
        vec = ef(["probe"])
        if not vec or not isinstance(vec[0], (list, tuple)) or len(vec[0]) < 8:
            raise RuntimeError("probe returned no usable vector")
        logger.info("Embedding model active: %s (dim=%d)", model, len(vec[0]))
        _CACHED = ef
        _FALLBACK_UNTIL = 0.0
    except Exception as e:
        import time
        _FALLBACK_UNTIL = time.monotonic() + _REPROBE_COOLDOWN_S
        logger.warning(
            "Embedding model %r not usable via Ollama (%s) — falling back to "
            "ChromaDB default (all-MiniLM-L6-v2); will re-probe in %.0fs",
            model, e, _REPROBE_COOLDOWN_S)
        _CACHED = None
    return _CACHED


def open_collection(client, name: str, *, reindex=None, metadata=None):
    """get_or_create a Chroma collection wired to the configured embedder,
    reconciling a dimension mismatch from a previously-defaulted collection.

    Every collection MUST go through here so the whole store uses ONE embedder.
    Collections created before the embedder upgrade persist as 384-dim MiniLM;
    attaching the 1024-dim bge-m3 function then throws InvalidDimension on the
    first query. When that happens we drop and recreate the collection under
    the new embedder and (if a `reindex` callable is supplied) repopulate it
    from the SQLite source of truth — the vectors are always re-derivable.

    `reindex(collection)` should backfill rows; it's only called after a
    rebuild, so it must NOT early-return on `count()>0`.
    """
    md = metadata or {"hnsw:space": "cosine"}
    ef = get_embedding_function()
    kw = {"name": name, "metadata": md}
    if ef is not None:
        kw["embedding_function"] = ef
    coll = client.get_or_create_collection(**kw)

    if ef is None:
        return coll

    # Detect a stale-dimension collection by probing a query. A fresh/empty
    # collection can't mismatch, so skip the probe when it's empty.
    try:
        if coll.count() > 0:
            coll.query(query_texts=["__dim_probe__"], n_results=1)
    except Exception as e:
        if "dimension" not in str(e).lower() and "InvalidDimension" not in type(e).__name__:
            raise
        logger.warning(
            "Collection %r has a stale embedding dimension — rebuilding under %s",
            name, getattr(config, "EMBEDDING_MODEL", "?"))
        client.delete_collection(name)
        coll = client.get_or_create_collection(**kw)
        if reindex is not None:
            try:
                n = reindex(coll)
                logger.info("Rebuilt %r: reindexed %s items under the new embedder", name, n)
            except Exception as re:
                logger.error("Reindex of rebuilt collection %r failed: %s", name, re)
    return coll
