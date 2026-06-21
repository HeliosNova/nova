"""Cross-domain causal synthesis (Monitor Intelligence v2, Phase B).

The synthesis probe extracts a CAUSAL CHAIN across domains and stores it as real
`caused_by` entity triples (durable, queryable) instead of the old
`cross_pattern:X recurs_across <paragraph>` meta-noise the KG gate now rejects.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app.core.cross_monitor as cm
from app.core.cross_monitor import ThemeCluster
from app.core.kg import is_garbage_triple, CANONICAL_PREDICATES


def _cluster():
    return ThemeCluster(
        key="export controls",
        monitors={"Domain Study: Geopolitics", "Domain Study: Finance", "Domain Study: AI and ML"},
        snippets=[
            ("Domain Study: Geopolitics", "US tightens semiconductor export controls on China."),
            ("Domain Study: Finance", "NVIDIA cuts revenue guidance citing export limits."),
            ("Domain Study: AI and ML", "AI labs warn of slower compute buildout."),
        ],
    )


@pytest.mark.asyncio
async def test_causal_probe_extracts_chain():
    payload = ('{"chain": [{"cause": "export controls", "effect": "chip revenue"}, '
               '{"cause": "chip revenue", "effect": "ai buildout"}], "confidence": 0.7}')
    with patch("app.core.llm.invoke_nothink", AsyncMock(return_value=payload)):
        chain = await cm._causal_probe(_cluster(), hours=36)
    assert len(chain) == 2
    assert chain[0] == {"cause": "export controls", "effect": "chip revenue", "confidence": 0.7}


@pytest.mark.asyncio
async def test_causal_probe_drops_sentence_fragments():
    # Sentences masquerading as entities must be dropped (would pollute the KG).
    payload = ('{"chain": [{"cause": "the fed raised interest rates again this week amid", '
               '"effect": "markets fell"}], "confidence": 0.6}')
    with patch("app.core.llm.invoke_nothink", AsyncMock(return_value=payload)):
        chain = await cm._causal_probe(_cluster(), hours=36)
    assert chain == []  # cause is a >5-word fragment → rejected


@pytest.mark.asyncio
async def test_coincidental_returns_empty():
    with patch("app.core.llm.invoke_nothink", AsyncMock(return_value='{"chain": []}')):
        chain = await cm._causal_probe(_cluster(), hours=36)
    assert chain == []


def test_digest_scaffolding_not_extracted_as_themes():
    # Live-caught 2026-06-21: "insight"/"cross-confirmed" digest scaffolding was
    # surfacing as false cross-monitor themes. They must be stopworded.
    from app.core.cross_monitor import _extract_signals
    sig = _extract_signals(
        "💡 Insight — markets react. 5 items sourced, with 2 cross-confirmed by multiple outlets."
    )
    for junk in ("insight", "cross-confirmed", "outlets", "sourced"):
        assert junk not in sig, f"{junk!r} should be stopworded, got {sig}"


def test_causal_triples_are_valid_kg_facts():
    # The triples the probe produces must pass the KG garbage gate and use a
    # canonical predicate (so they're real, queryable knowledge).
    assert "caused_by" in CANONICAL_PREDICATES
    assert is_garbage_triple("chip revenue", "caused_by", "export controls") is False
    assert is_garbage_triple("ai buildout", "caused_by", "chip revenue") is False
