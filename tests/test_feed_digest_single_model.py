"""Feed-backed digests run on the synthesis model too (2026-09-08).

`_monitor_class` puts feed-backed query monitors (Hacker News, Product Hunt,
SEC, FDA, GitHub advisories, government contracts, Research Frontiers) in the
same "digest" residency class as the Domain Studies, on the stated premise that
they drive the 27B. They did not: the feed chain's five LLM calls in
domain_study_runner - the insight line, the extractive retry, thin-item
enrichment, snippet rewriting and the cross-cutting analysis - were all
model-less, i.e. the default 9B. So the scheduler ran a 9B chain three-wide
beside 27B chains, and Ollama evicted one for the other on every call.

Measured 2026-09-08 15:20-15:27 UTC, Hacker News beside two finishing Domain
Studies: "predicted 10.7 GiB" evicting "predicted 20.7 GiB" and back every
10-20 seconds, six loads in five minutes, the 27B resident for twenty seconds
at a stretch.

The 2026-09-01 one-model change covered deep_research.py; this completes it
for the feed chain. Same shape as tests/test_digest_single_model.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import config as _cfg
from app.core import llm as llm_mod
from app.monitors import domain_study_runner as dsr


@pytest.fixture
def syn_model():
    old = getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "")
    _cfg.update(MONITOR_SYNTHESIS_MODEL="qwen3.8:27b")
    yield "qwen3.8:27b"
    _cfg.update(MONITOR_SYNTHESIS_MODEL=old)


def _record(monkeypatch, reply="A grounded one-line insight about the items."):
    seen: list[dict] = []

    async def _fake(messages, **kwargs):
        seen.append(kwargs)
        if kwargs.get("json_schema"):
            n = kwargs["json_schema"]["properties"]["summaries"].get("minItems", 1)
            return '{"summaries": [' + ", ".join(['"A verifiable sentence from the item."'] * n) + "]}"
        return reply

    monkeypatch.setattr(llm_mod, "invoke_nothink", _fake)
    return seen


def _items(n=4):
    return [SimpleNamespace(title=f"Headline number {i} about a concrete thing", summary="",
                            link=f"https://example.com/{i}") for i in range(n)]


@pytest.mark.asyncio
async def test_the_insight_line_uses_the_synthesis_model(monkeypatch, syn_model):
    seen = _record(monkeypatch)
    await dsr._native_insight("Hacker News", _items())
    assert seen, "no LLM call recorded"
    assert [k.get("model") for k in seen] == [syn_model] * len(seen)


@pytest.mark.asyncio
async def test_the_cross_cutting_analysis_uses_the_synthesis_model(monkeypatch, syn_model):
    seen = _record(monkeypatch, reply="The throughline is a concrete tension between the items, "
                                      "and the implication is worth watching next week.")
    await dsr._synthesize_insight("Hacker News", [{"title": f"Headline {i} about something"} for i in range(4)])
    assert seen, "no LLM call recorded"
    assert [k.get("model") for k in seen] == [syn_model] * len(seen)


@pytest.mark.asyncio
async def test_the_extractive_retry_uses_the_synthesis_model(monkeypatch, syn_model):
    seen = _record(monkeypatch)
    redo = [(SimpleNamespace(title="A page about a thing", link="https://example.com/a"),
             "The page says the thing happened on September 8, 2026 and cost $4 million. " * 5)]
    await dsr._extractive_retry("Product Hunt Trending", "Product Hunt", redo)
    assert seen, "no LLM call recorded"
    assert [k.get("model") for k in seen] == [syn_model] * len(seen)


@pytest.mark.asyncio
async def test_without_a_synthesis_model_the_default_is_used(monkeypatch):
    old = getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "")
    _cfg.update(MONITOR_SYNTHESIS_MODEL="")
    seen = _record(monkeypatch)
    try:
        await dsr._native_insight("Hacker News", _items())
    finally:
        _cfg.update(MONITOR_SYNTHESIS_MODEL=old)
    assert seen and seen[0].get("model") is None


@pytest.mark.asyncio
async def test_snippet_rewriting_uses_the_synthesis_model(monkeypatch, syn_model):
    seen = _record(monkeypatch, reply="The article reports a concrete development with named parties and figures.")
    # >= 200 chars with no sentence break: long enough that the junk detector
    # keeps it, unpunctuated enough that _looks_clean() says no, so the rewrite
    # path runs.
    noisy = ("Widget maker announces new version with faster sync and a lower price for "
             "teams across Europe while analysts expect adoption to rise sharply this "
             "quarter as rivals delay their own launches and customers wait for details "
             "on pricing tiers and regional availability")
    assert len(noisy) >= 200 and "." not in noisy
    await dsr._enrich_summaries("Product Hunt", [{"title": "A launch", "snippet": noisy,
                                                   "url": "https://example.com/launch"}])
    assert seen, "no LLM call recorded"
    assert [k.get("model") for k in seen] == [syn_model] * len(seen)


@pytest.mark.asyncio
async def test_thin_item_enrichment_uses_the_synthesis_model(monkeypatch, syn_model):
    seen = _record(monkeypatch)

    async def _fake_fetch(url, **kwargs):
        return "2026-09-08", ("The advisory describes a remote code execution flaw in the widget "
                             "library affecting versions before 2.4.1, fixed upstream on September 8, 2026. ") * 6

    monkeypatch.setattr(dsr, "_fetch_page_date", _fake_fetch)
    items = [SimpleNamespace(title=f"GHSA-2026-000{i}: widget library flaw", summary="",
                             url=f"https://github.com/advisories/GHSA-2026-000{i}",
                             link=f"https://github.com/advisories/GHSA-2026-000{i}") for i in range(3)]
    await dsr._enrich_thin_native_items("GitHub Security Advisories", "GitHub Security Advisories", items)
    assert seen, "no LLM call recorded (the thin items were not enriched)"
    assert [k.get("model") for k in seen] == [syn_model] * len(seen)
