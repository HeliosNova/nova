"""Integration tests for the grounding-honesty caveat through the REAL
_refine_response (not just the _maybe_unverified_caveat helper). Stubs only the
stochastic LLM/critique calls so the deterministic fire/suppress wiring is
exercised: critique flags -> rewrite (rejected|accepted) -> caveat (fires|suppressed).

conftest sets ENABLE_CRITIQUE=false; the frozen config is overridden here via
object.__setattr__ (same pattern as test_config) so the critique block runs.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.core.brain as brain

CAVEAT_MARK = "couldn't confirm some specifics"

FLAGGED = (
    "The Aurora Accord was signed on June 9, 2026 in Geneva by 14 nations, "
    "establishing the first binding international framework on autonomous "
    "systems, with a compliance deadline of March 2027 and a $4.2 billion "
    "enforcement fund administered from Zurich.")


class _Resp:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []


@pytest.fixture
def critique_on():
    """Force the (frozen) config to run critique + skip best-of-N for the test."""
    cfg = brain.config
    saved = {k: getattr(cfg, k) for k in ("ENABLE_CRITIQUE", "ENABLE_BEST_OF_N", "ENABLE_PLANNING")}
    object.__setattr__(cfg, "ENABLE_CRITIQUE", True)
    object.__setattr__(cfg, "ENABLE_BEST_OF_N", False)
    object.__setattr__(cfg, "ENABLE_PLANNING", False)
    try:
        yield
    finally:
        for k, v in saved.items():
            object.__setattr__(cfg, k, v)


def _common_stubs(monkeypatch):
    async def fake_critique(query, answer, **kw):
        return {"pass": False, "issues": ["Date/amount unsupported by any source."]}

    async def fake_adv(query, answer, **kw):
        return {"verdict": "pass", "flaws": []}

    monkeypatch.setattr("app.core.critique.critique_answer", fake_critique)
    monkeypatch.setattr("app.core.critique.adversarial_critique", fake_adv)
    monkeypatch.setattr("app.core.reflexion.should_use_llm_critique", lambda *a, **k: False)
    monkeypatch.setattr("app.core.reflexion.assess_quality", lambda *a, **k: (0.85, "stub"))
    monkeypatch.setattr("app.core.brain.get_services", lambda: SimpleNamespace(reflexions=None))


async def _refine():
    return await brain._refine_response(
        messages=[{"role": "user", "content": "Tell me about the Aurora Accord."}],
        tools=[], final_content=FLAGGED, query="Tell me about the Aurora Accord.",
        intent="general", tool_results=[], was_planned=False, plan=None,
    )


@pytest.mark.asyncio
async def test_caveat_fires_when_rewrite_rejected(monkeypatch, critique_on):
    _common_stubs(monkeypatch)
    # Meta-commentary rewrite -> trips _has_meta gate -> REJECTED -> original kept.
    async def fake_rewrite(messages, tools, **kw):
        return _Resp("I made a mistake in my previous answer; let me correct the "
                     "earlier response — I cannot confirm those details.")
    monkeypatch.setattr("app.core.llm.generate_with_tools", fake_rewrite)

    out, _q, _r = await _refine()
    assert CAVEAT_MARK in out, "caveat must fire when flagged text survives a rejected rewrite"
    assert out.startswith(FLAGGED)


@pytest.mark.asyncio
async def test_caveat_suppressed_when_rewrite_accepted(monkeypatch, critique_on):
    _common_stubs(monkeypatch)
    # Clean substantive rewrite (no meta, reasonable length) -> ACCEPTED ->
    # final_content becomes the rewrite -> caveat must NOT fire.
    clean = ("The Aurora Accord is an international agreement on autonomous systems; "
             "I don't have verified specifics on its signing date or funding, so treat "
             "any precise figures as unconfirmed pending a source check.")
    async def fake_rewrite(messages, tools, **kw):
        return _Resp(clean)
    monkeypatch.setattr("app.core.llm.generate_with_tools", fake_rewrite)

    out, _q, _r = await _refine()
    assert CAVEAT_MARK not in out, "caveat must be suppressed once an accepted rewrite replaces the flagged text"
