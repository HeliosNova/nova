"""Ephemeral requests must not write to persistent stores via tool side-channels.

brain.think(ephemeral=True) documents: "keeps eval traffic out of every
persistent store". The eval harness calls think(ephemeral=True, channel="eval")
with queries about FICTIONAL entities (Vorenza, Skylance X9, mem_scheduler).

Measured leak (2026-08-30): web_search's zero-result auto-curiosity mint never
saw the flag, so the curiosity queue collected
    [167] "Vorenza location where is it based headquartered"
    [238] "Skylance X9 aircraft maximum altitude"
    [239] "Where is Vorenza based?"
as REAL research topics. Defense in depth held — the curiosity judge dismissed
all three and 0 fictional facts reached the KG — but the vector was open, and
monitor 77 already proved probe-shaped traffic can spawn real work.

The fix: EPHEMERAL_REQUEST ContextVar in tools/base.py, set by think() at
entry, consulted by web_search before minting.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tools.base import EPHEMERAL_REQUEST
from app.tools.web_search import WebSearchTool


@pytest.fixture
def zero_result_search():
    """native_search.search returns [] — the mint trigger."""
    async def _empty(*a, **k):
        return []
    with patch("app.tools.native_search.search", _empty):
        yield


@pytest.fixture(autouse=True)
def reset_ephemeral():
    token = EPHEMERAL_REQUEST.set(False)
    yield
    EPHEMERAL_REQUEST.reset(token)


class TestEphemeralSuppressesCuriosityMint:
    @pytest.mark.asyncio
    async def test_ephemeral_zero_result_does_not_mint(self, zero_result_search):
        EPHEMERAL_REQUEST.set(True)
        with patch("app.core.curiosity.CuriosityQueue") as cq_cls:
            tool = WebSearchTool()
            result = await tool.execute(query="Where is Vorenza based?")
            assert not result.success  # zero results is still a failed search
            cq_cls.assert_not_called(), (
                "eval traffic minted a curiosity item — the exact leak that "
                "queued fictional entities as real research topics"
            )

    @pytest.mark.asyncio
    async def test_real_zero_result_still_mints(self, zero_result_search):
        """The mint is a FEATURE for real traffic — ambient-awareness gaps
        should still be queued. Suppressing it everywhere would quietly kill
        the search_zero_result curiosity source (7 live rows use it)."""
        EPHEMERAL_REQUEST.set(False)
        added = MagicMock()
        with patch("app.core.curiosity.CuriosityQueue") as cq_cls:
            cq_cls.return_value.add = added
            tool = WebSearchTool()
            result = await tool.execute(query="obscure real topic with no hits")
            assert not result.success
            added.assert_called_once()
            kw = added.call_args.kwargs
            assert kw.get("source") == "search_zero_result"

    def test_default_is_not_ephemeral(self):
        """Fail-open in the right direction: code that never touches the var
        behaves exactly as before this fix (mints allowed)."""
        assert EPHEMERAL_REQUEST.get() is False


class TestThinkPublishesTheFlag:
    def test_think_sets_contextvar_from_ephemeral_param(self):
        """Source-level check that think() wires the param to the ContextVar
        before any tool can run — behavioral verification would need the full
        service stack, but the wiring is the load-bearing line."""
        import inspect

        from app.core import brain

        src = inspect.getsource(brain.think)
        assert "EPHEMERAL_REQUEST.set(ephemeral)" in src, (
            "think() must publish its ephemeral flag to tools; without it the "
            "'stays out of every persistent store' promise has a side channel"
        )
