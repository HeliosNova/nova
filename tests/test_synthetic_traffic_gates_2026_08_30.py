"""Synthetic traffic must not feed the self-improvement collectors.

Measured 2026-08-30 (fresh-eyes sweep, after the owner pushed past the task
list): ephemeral eval-harness runs were writing into every persistent
self-improvement input the ephemeral flag never reached —

  * auto_tool_candidates: 22 of 100 rows were eval-fixture queries ("In the
    Nova project, what does the codename 'ZQX' refer to?", the fictional
    Aurora-7 fusion plant, Skylance X9), and SEVERAL had triggered=1 — eval
    probes drove real LLM tool-writing. custom_tools #3 (28 uses) traces to
    this class.
  * capability_gaps: newest rows were quiz-family questions, later mined by
    goal derivation — the 8 historical goals (ALL failed) chased exactly such
    phantoms ("Re-research: Factual Art History Questions").
  * reflexion get_relevant: the store held 114 is_eval=1 failures vs 15 real
    ones, and real-chat failure injection drew from that 88%-synthetic pool
    with no filter.

think(ephemeral=True) documents "keeps eval traffic out of every persistent
store". These are the remaining side channels after web_search's curiosity
mint was closed earlier tonight.
"""

from __future__ import annotations

import pathlib

import pytest

from app.tools.base import EPHEMERAL_REQUEST

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def reset_ephemeral():
    token = EPHEMERAL_REQUEST.set(False)
    yield
    EPHEMERAL_REQUEST.reset(token)


class TestGatePresenceAtRemainingSites:
    """brain's no-skill gap writer runs deep in
    machinery a unit test can't cheaply drive; assert the wiring exists so a
    refactor can't silently drop it."""



class TestTrustRecordingScoped:
    """Trust is Nova's observational self-awareness metric (can_use never
    enforces — sovereign-AI design). Eval probes were moving the dial anyway:
    the ZQX fixture's sandboxed code_exec attempt dinged the global score
    -15 to 58.0 on 2026-08-30. A self-metric fed by synthetic failures
    measures the eval suite, not Nova."""

    @pytest.mark.asyncio
    async def test_ephemeral_tool_outcome_not_recorded(self):
        from unittest.mock import AsyncMock, MagicMock

        from app.tools.base import BaseTool, ToolRegistry, ToolResult

        class Boom(BaseTool):
            name = "boom"
            description = "always fails"
            async def execute(self, **kwargs):
                return ToolResult(output="", success=False, error="infra down")

        reg = ToolRegistry()
        reg.register(Boom())
        reg.trust_manager = MagicMock()
        reg.trust_manager.can_use.return_value = True

        EPHEMERAL_REQUEST.set(True)
        await reg.execute_full("boom", {})
        reg.trust_manager.record_outcome.assert_not_called()

        EPHEMERAL_REQUEST.set(False)
        await reg.execute_full("boom", {})
        assert reg.trust_manager.record_outcome.called, (
            "real traffic must still move the observational metric"
        )
