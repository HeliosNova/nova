"""The KG contradiction judge runs on the model its caller is already holding (2026-09-08).

_extract_kg_triples takes a `model` and the post-digest extraction passes the
synthesis model, so the extraction itself runs on the resident 27B. But when a
freshly extracted triple conflicts with a live fact, add_fact's guard
`check_and_resolve_contradictions` asked the DEFAULT model which to keep -
the 9B - which evicted the 27B for one 16-token verdict and then reloaded it
for the next digest step. Live 2026-09-08 16:49:51 UTC: "KG: extracted 1
triple(s) (source='Domain Study: Top Trades and Positioning')" ten seconds
after a 9B load, the 27B back at 16:50:20.

The judge now takes the caller's model. Chat and curiosity pass nothing and
keep the default, so an interactive turn never drags the 27B in for it.
"""
from __future__ import annotations

import json

import pytest

from app.core import llm as llm_mod
from app.core.kg import KnowledgeGraph

SYN = "qwen3.8:27b"


@pytest.fixture
def kg(tmp_path):
    from app.database import SafeDB
    d = SafeDB(str(tmp_path / "kg.db"))
    d.init_schema()
    graph = KnowledgeGraph(d)
    yield graph
    d.close()


def _record(monkeypatch, replies):
    seen: list[dict] = []
    queue = list(replies)

    async def _fake(messages, **kwargs):
        seen.append(kwargs)
        return queue.pop(0) if queue else '{"keep": "both"}'

    monkeypatch.setattr(llm_mod, "invoke_nothink", _fake)
    return seen


@pytest.mark.asyncio
async def test_the_judge_uses_the_model_it_is_given(kg, monkeypatch):
    assert await kg.add_fact("Nvidia", "headquartered_in", "Santa Clara", confidence=0.9, source="extracted")
    seen = _record(monkeypatch, ['{"keep": "A"}'])
    safe = await kg.check_and_resolve_contradictions("Nvidia", "headquartered_in", "Austin", 0.8, model=SYN)
    assert safe is False
    assert seen, "the judge never ran"
    assert seen[0].get("model") == SYN


@pytest.mark.asyncio
async def test_without_a_model_the_judge_keeps_the_default(kg, monkeypatch):
    assert await kg.add_fact("Nvidia", "headquartered_in", "Santa Clara", confidence=0.9, source="extracted")
    seen = _record(monkeypatch, ['{"keep": "A"}'])
    await kg.check_and_resolve_contradictions("Nvidia", "headquartered_in", "Austin", 0.8)
    assert seen and seen[0].get("model") is None


@pytest.mark.asyncio
async def test_a_digest_extraction_carries_its_model_into_the_judge(kg, monkeypatch):
    from app.core.brain_kg import _extract_kg_triples
    assert await kg.add_fact("Nvidia", "headquartered_in", "Santa Clara", confidence=0.9, source="extracted")
    triples = json.dumps([{"subject": "Nvidia", "predicate": "headquartered_in", "object": "Austin",
                           "confidence": 0.85}])
    seen = _record(monkeypatch, [triples, '{"keep": "A"}'])
    await _extract_kg_triples(kg, "Domain Study: Semiconductors",
                              "The report says Nvidia moved its headquarters to Austin this week.",
                              source_name="Domain Study: Semiconductors", model=SYN, trust=0.7)
    assert len(seen) == 2, [k.get("model") for k in seen]
    assert seen[0].get("model") == SYN, "extraction"
    assert seen[1].get("model") == SYN, "the judge must not fall back to the default model"
