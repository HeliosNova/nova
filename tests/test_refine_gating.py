"""_refine_response latency fixes.

Stage 1 (2026-06-14): the two pre-rewrite judges (coverage + adversarial) are
merged into ONE `critique_unified` LLM call, and the whole judge chain is gated
on grounding — when the answer is already grounded (all tools clean, or a general
answer backed by KG/owner facts) the judges are skipped and the quality score
comes from the free heuristic, because `validate_claims` re-checks grounding
right after refine. The adversarial hunter is still gated to answers that make
checkable claims.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.config import config as _cfg
from app.core.brain import _refine_response


@contextlib.contextmanager
def _critique_enabled():
    # conftest pins ENABLE_CRITIQUE=false; flip the (frozen) singleton for these
    # tests so the critique/adversarial path actually executes.
    old = _cfg.ENABLE_CRITIQUE
    object.__setattr__(_cfg, "ENABLE_CRITIQUE", True)
    try:
        yield
    finally:
        object.__setattr__(_cfg, "ENABLE_CRITIQUE", old)


CONVERSATIONAL = (
    "I think learning to cook is a genuinely rewarding hobby. It brings people "
    "together around the table, teaches patience, and rewards you a little more "
    "each time you practice and slowly refine your instincts over many meals."
)
FACTUAL = (
    "The Mona Lisa was painted by Leonardo da Vinci during the Italian Renaissance "
    "and now hangs in the Louvre in Paris. It measures 77 by 53 centimeters, was "
    "acquired by King Francis the First, and today draws millions of visitors a "
    "year who line up for a brief glimpse of the famous portrait behind glass."
)

CLEAN_TOOL = [{"tool": "web_search", "output": "the capital is Paris"}]
FAILED_TOOL = [{"tool": "web_search", "output": "", "error": "request failed"}]


def _uni_mock(pass_=True, verdict="pass", issues=None, flaws=None):
    """Mock the single merged judge call."""
    blocking = [f for f in (flaws or []) if f.get("blocking")]
    return AsyncMock(return_value={
        "pass": pass_,
        "issues": issues or [],
        "flaws": flaws or [],
        "blocking_flaws": blocking,
        "verdict": verdict,
    })


async def _run(content, *, tool_results=None, kg_facts="", user_facts="",
               uni=None, llm_critique=False, crit_resp=None):
    uni = uni or _uni_mock()
    crit_resp = crit_resp or AsyncMock(return_value=(0.9, "ok"))
    with _critique_enabled(), \
         patch("app.core.critique.critique_unified", uni), \
         patch("app.core.reflexion.should_use_llm_critique", return_value=llm_critique), \
         patch("app.core.reflexion.critique_response", crit_resp):
        out, q, r = await _refine_response(
            messages=[{"role": "user", "content": "q"}],
            tools=[],
            final_content=content,
            query="tell me about it",
            intent="general",
            tool_results=tool_results or [],
            was_planned=False,
            plan=None,
            retrieved_context="",
            user_facts_text=user_facts,
            kg_facts_text=kg_facts,
        )
    return out, uni, crit_resp


# --- 1B: two judges merged into one call -----------------------------------

@pytest.mark.asyncio
async def test_ungrounded_conversational_runs_one_unified_judge():
    out, uni, _ = await _run(CONVERSATIONAL)
    assert uni.await_count == 1, "exactly one merged judge call (not two)"
    assert out  # answer preserved


@pytest.mark.asyncio
async def test_ungrounded_factual_runs_one_unified_judge():
    out, uni, _ = await _run(FACTUAL)
    assert uni.await_count == 1, "factual answer judged by the single merged call"
    # the merged call carries the same content the judges used to get separately
    assert uni.await_args.args[1] == FACTUAL


@pytest.mark.asyncio
async def test_unified_fail_triggers_adversarial_rewrite():
    # verdict=fail with a blocking flaw must still drive the adversarial rewrite,
    # proving the unified result is correctly split into the adv verdict shape.
    uni = _uni_mock(
        pass_=True,  # self-critique passes (no coverage rewrite)
        verdict="fail",
        flaws=[{"type": "factual", "description": "wrong date", "blocking": True}],
    )
    rewritten = "A corrected, fully accurate answer about the painting. " * 4
    fake = SimpleNamespace(content=rewritten, tool_calls=None)
    with patch("app.core.brain.llm.generate_with_tools",
               AsyncMock(return_value=fake)) as gen:
        out, uni_used, _ = await _run(FACTUAL, uni=uni)
    assert uni_used.await_count == 1
    assert gen.await_count >= 1, "blocking flaw must trigger a corrective regeneration"
    assert rewritten in out, "the regenerated answer must replace the flagged one"


# --- 1A: grounding gate -----------------------------------------------------

@pytest.mark.asyncio
async def test_grounded_by_clean_tools_skips_judge():
    out, uni, _ = await _run(FACTUAL, tool_results=CLEAN_TOOL)
    assert uni.await_count == 0, "clean tools => grounded => judge skipped"
    assert out


@pytest.mark.asyncio
async def test_grounded_by_kg_facts_skips_judge():
    out, uni, _ = await _run(FACTUAL, kg_facts="The Mona Lisa is by Leonardo da Vinci.")
    assert uni.await_count == 0, "KG-grounded general answer => judge skipped"


@pytest.mark.asyncio
async def test_failed_tools_still_judged():
    # A failed tool is NOT grounding — the judge must still run.
    out, uni, _ = await _run(FACTUAL, tool_results=FAILED_TOOL)
    assert uni.await_count == 1, "failed tools are not grounding => still judged"


@pytest.mark.asyncio
async def test_grounded_routes_scorer_to_heuristic():
    # Even when should_use_llm_critique would say yes, a grounded answer must use
    # the free heuristic scorer — the calibrated LLM scorer is not invoked.
    crit_resp = AsyncMock(return_value=(0.1, "bad"))
    out, uni, crit_used = await _run(
        FACTUAL, kg_facts="The Mona Lisa is by Leonardo da Vinci.",
        llm_critique=True, crit_resp=crit_resp,
    )
    assert uni.await_count == 0
    assert crit_used.await_count == 0, "grounded answer scored by heuristic, not the LLM judge"


@pytest.mark.asyncio
async def test_ungrounded_uses_llm_scorer_when_enabled():
    # Sanity: when NOT grounded and should_use_llm_critique=True, the calibrated
    # scorer IS used (we only route around it for grounded answers).
    crit_resp = AsyncMock(return_value=(0.8, "fine"))
    out, uni, crit_used = await _run(FACTUAL, llm_critique=True, crit_resp=crit_resp)
    assert crit_used.await_count == 1
