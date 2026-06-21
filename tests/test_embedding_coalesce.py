"""Single-query embed coalescing (2026-06-14).

Per-turn the brain fires several retrievals that each embed the SAME query;
running them concurrently flooded the one GPU embedder (calls queued/timed out).
The embedder now coalesces identical concurrent/recent single-text embeds into
one actual call.
"""
import threading
import time

from app.core.embedding import OllamaEmbeddingFunction


def _make(counter):
    ef = OllamaEmbeddingFunction("test-model", "http://x")
    calls = {"n": 0}

    def fake_embed(chunk):
        calls["n"] += 1
        time.sleep(0.15)  # simulate a slow GPU embed so concurrency overlaps
        return [[0.1, 0.2, 0.3] for _ in chunk]

    ef._embed = fake_embed
    return ef, calls


def test_concurrent_identical_query_embeds_once():
    ef, calls = _make(None)
    results = []
    barrier = threading.Barrier(6)

    def worker():
        barrier.wait()  # release all at once
        results.append(ef(["what is the price of gold"])[0])

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert calls["n"] == 1, f"expected 1 embed call (coalesced), got {calls['n']}"
    assert len(results) == 6 and all(r == [0.1, 0.2, 0.3] for r in results)


def test_recent_query_served_from_cache():
    ef, calls = _make(None)
    ef(["repeat query"])
    ef(["repeat query"])  # within TTL -> cached
    assert calls["n"] == 1


def test_distinct_queries_not_coalesced():
    ef, calls = _make(None)
    ef(["query a"])
    ef(["query b"])
    assert calls["n"] == 2


def test_batch_indexing_bypasses_cache():
    ef, calls = _make(None)
    ef(["doc one", "doc two", "doc three"])  # batch path
    assert calls["n"] == 1  # one batched call
    # A single-text query after is independent (not served from a batch).
    ef(["doc one"])
    assert calls["n"] == 2
