"""Cross-Monitor Synthesis runs on the synthesis model its residency class names (2026-09-08).

`_SYNTHESIS_MODEL_TYPES` files check_type "synthesis" in the "digest" residency
class - the 27B - so the scheduler batches it beside Domain Studies. Its three
LLM calls (cluster-key validation, per-cluster synthesis, the big-picture meta
synthesis) were model-less, i.e. the default 9B: the same classification gap
that made the feed digests bounce the two models on 2026-09-08, found by an AST
scan for background LLM calls without a `model=` keyword after two live cases
in one afternoon. The causal probe in this module was routed to the synthesis
model on 2026-08-14 for quality reasons; this brings the other three with it.
"""
from __future__ import annotations

import pytest

from app.config import config as _cfg
from app.core import cross_monitor as cm
from app.core import llm as llm_mod


@pytest.fixture
def syn_model():
    old = getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "")
    _cfg.update(MONITOR_SYNTHESIS_MODEL="qwen3.8:27b")
    yield "qwen3.8:27b"
    _cfg.update(MONITOR_SYNTHESIS_MODEL=old)


def _record(monkeypatch, reply):
    seen: list[dict] = []

    async def _fake(messages, **kwargs):
        seen.append(kwargs)
        return reply

    monkeypatch.setattr(llm_mod, "invoke_nothink", _fake)
    return seen


LEAD = ("Brent crude rose above 95 dollars a barrel on September 8, 2026 after the Strait "
        "of Hormuz blockade entered its sixth month, and refiners in Europe reported product "
        "shortages that pushed diesel margins to a two-year high (reuters.com).")


@pytest.mark.asyncio
async def test_cluster_key_validation_uses_the_synthesis_model(monkeypatch, syn_model):
    seen = _record(monkeypatch, '{"valid": ["tariffs", "oil prices"]}')
    await cm._validate_cluster_keys(["tariffs", "oil prices", "the"])
    assert seen, "no LLM call recorded"
    assert [k.get("model") for k in seen] == [syn_model] * len(seen)


@pytest.mark.asyncio
async def test_cluster_synthesis_uses_the_synthesis_model(monkeypatch, syn_model):
    seen = _record(monkeypatch, "Oil prices are moving energy, markets and trade at once because "
                                "the blockade has exhausted inventory buffers.")
    cluster = cm.ThemeCluster(key="oil prices", monitors={"Energy", "Markets", "Trade"},
                              snippets=[("Energy", LEAD), ("Markets", LEAD), ("Trade", LEAD)])
    await cm._synthesize_cluster(cluster, hours=36)
    assert seen, "no LLM call recorded"
    assert [k.get("model") for k in seen] == [syn_model] * len(seen)


@pytest.mark.asyncio
async def test_meta_synthesis_uses_the_synthesis_model(monkeypatch, syn_model):
    seen = _record(monkeypatch, "THREAD ONE: the blockade [Energy] [Markets] [Trade] is the single "
                                "force behind this week's moves, because inventories are exhausted "
                                "and refiners cannot substitute. " * 3)
    grouped = {f"Domain Study: Area {i}": [LEAD] for i in range(5)}
    monkeypatch.setattr(cm, "_gather_recent_outputs", lambda db, **kw: grouped)
    await cm.meta_synthesis(object(), hours=36)
    assert seen, "no LLM call recorded (fewer than four leads reached the model?)"
    assert [k.get("model") for k in seen] == [syn_model] * len(seen)
