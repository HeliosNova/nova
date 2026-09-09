"""The search fan-out is bounded, and a rate-limited fallback rests (2026-09-09).

Measured 2026-09-08 14:22-14:23 UTC in the post-outage wave: SearXNG answered
67 queries in one minute, then three digests fanned out their angle searches
at once (every angle on both news and general, roughly 16 requests each) and
56 of them timed out upstream while the ladder fell through to Brave 26 times
in the same second and got 429. A direct SearXNG query answered in 1.5 s the
whole time; the instance was healthy. Baseline failure rate 4%; in the burst
minute over 50%, and the digests fell back to Bing-direct and Wikipedia -
thinner evidence exactly when the wave made them most numerous.

Two rules: at most `_SEARXNG_MAX_CONCURRENT` requests in flight to the local
SearXNG at once (the rest queue for a second or two instead of timing out),
and after Brave answers 429 it is left alone for `_BRAVE_COOLDOWN_S` so a
burst of failures cannot turn into a burst of rate-limit strikes.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.tools import native_search as ns


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload) if isinstance(payload, dict) else str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(f"{self.status_code}", request=None, response=self)


def _client_factory(handler, stats):
    class _C:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, **k):
            stats["inflight"] += 1
            stats["peak"] = max(stats["peak"], stats["inflight"])
            stats["calls"] += 1
            try:
                await asyncio.sleep(0.03)
                return handler(url, params)
            finally:
                stats["inflight"] -= 1

        async def post(self, url, **k):
            return await self.get(url, **k)
    return _C


@pytest.mark.asyncio
async def test_searxng_requests_are_bounded(monkeypatch):
    stats = {"inflight": 0, "peak": 0, "calls": 0}
    payload = {"results": [{"url": "https://example.com/a", "title": "A", "content": "c", "engine": "bing"}]}
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(lambda u, p: _Resp(200, payload), stats))
    outs = await asyncio.gather(*[ns._search_searxng(f"query {i}", 5) for i in range(24)])
    assert stats["calls"] == 24
    assert all(len(o) == 1 for o in outs), "bounding must not drop requests, only queue them"
    assert stats["peak"] <= ns._SEARXNG_MAX_CONCURRENT, stats
    assert stats["peak"] >= 2, "the cap must still allow real concurrency"


@pytest.mark.asyncio
async def test_brave_rests_after_a_429(monkeypatch):
    stats = {"inflight": 0, "peak": 0, "calls": 0}
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(lambda u, p: _Resp(429, "Too Many Requests"), stats))
    ns._brave_blocked_until = 0.0
    assert await ns._search_brave("first", 5) == []
    assert stats["calls"] == 1
    for i in range(5):
        assert await ns._search_brave(f"again {i}", 5) == []
    assert stats["calls"] == 1, "while resting, Brave must not be asked at all"
    ns._brave_blocked_until = 0.0
    await ns._search_brave("after the rest", 5)
    assert stats["calls"] == 2


@pytest.mark.asyncio
async def test_brave_success_does_not_rest(monkeypatch):
    stats = {"inflight": 0, "peak": 0, "calls": 0}
    html = '<div class="snippet"></div>'
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _client_factory(lambda u, p: _Resp(200, html), stats))
    ns._brave_blocked_until = 0.0
    await ns._search_brave("a", 5)
    await ns._search_brave("b", 5)
    assert stats["calls"] == 2
