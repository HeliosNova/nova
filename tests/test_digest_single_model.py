"""One-model digest chain (audit 2026-09-01).

Each Domain Study digest swapped weights 3-4 times: subjects on the 27B,
angles / per-article findings / gap follow-up / fresh-check on the default 9B,
then synthesis on the 27B again. With 17 GB + 10 GB models that cannot
co-reside on the 24 GB card, every swap was a reload, and 5 of 6 truncation
warnings came from the 512-token findings step on the 9B. Every stage of the
chain now runs on the synthesis model when one is configured.
"""
from __future__ import annotations

import pytest

from app.config import config as _cfg
from app.monitors import deep_research


@pytest.fixture
def syn_model():
    old = getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "")
    _cfg.update(MONITOR_SYNTHESIS_MODEL="qwen3.8:27b")
    yield "qwen3.8:27b"
    _cfg.update(MONITOR_SYNTHESIS_MODEL=old)


@pytest.mark.asyncio
async def test_angles_findings_and_gap_followup_use_the_synthesis_model(monkeypatch, syn_model):
    seen: list[dict] = []

    async def _fake(messages, **kwargs):
        seen.append(kwargs)
        if kwargs.get("json_schema"):
            return '["what happened with the widget", "widget makers involved"]'
        return "- The widget shipped on August 30, 2026 (reuters.com)."

    monkeypatch.setattr(deep_research, "_invoke_bg", _fake)
    await deep_research._overview_angles(["Widget ships"])
    await deep_research._findings([("Widget ships", "https://reuters.com/x", "body " * 50)], "widgets")
    await deep_research._gap_followup([("Widget ships", "https://reuters.com/x", "finding")], "widgets")
    assert seen, "no LLM calls recorded"
    assert all(k.get("model") == syn_model for k in seen), [k.get("model") for k in seen]


@pytest.mark.asyncio
async def test_findings_budget_is_no_longer_512(monkeypatch, syn_model):
    seen: list[dict] = []

    async def _fake(messages, **kwargs):
        seen.append(kwargs)
        return "- finding"

    monkeypatch.setattr(deep_research, "_invoke_bg", _fake)
    await deep_research._findings([("t", "https://a.com/x", "body")], "x")
    assert seen and seen[0].get("max_tokens", 0) >= 800


@pytest.mark.asyncio
async def test_without_a_synthesis_model_the_default_model_is_used(monkeypatch):
    old = getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "")
    _cfg.update(MONITOR_SYNTHESIS_MODEL="")
    seen: list[dict] = []

    async def _fake(messages, **kwargs):
        seen.append(kwargs)
        return '["q"]'

    monkeypatch.setattr(deep_research, "_invoke_bg", _fake)
    try:
        await deep_research._overview_angles(["s"])
    finally:
        _cfg.update(MONITOR_SYNTHESIS_MODEL=old)
    assert seen and seen[0].get("model") is None
