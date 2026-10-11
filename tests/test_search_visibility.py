"""R1 (plan 2026-10-09): engine suspensions are visible, refusals are directive.

Measured 2026-10-09: SearXNG answers every call with an `unresponsive_engines`
list that nothing read; a suspended engine looked exactly like one with no
opinion. And a zero-result search told the model "No results found." — an
invitation to infer. Now every suspension is counted per engine, `search()`
names them in its log line, the engineering report carries the per-engine
non-empty rate, and the tool output starts with SEARCH REFUSED and an
instruction not to infer.
"""

from __future__ import annotations

import inspect
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools import native_search as ns


@pytest.fixture(autouse=True)
def _reset():
    ns._ENGINE_STATS.clear()
    ns._LAST_UNRESPONSIVE.clear()
    yield
    ns._ENGINE_STATS.clear()
    ns._LAST_UNRESPONSIVE.clear()


def _resp(payload: dict, status: int = 200):
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=payload)
    return r


@pytest.mark.asyncio
async def test_unresponsive_engines_are_logged_and_counted(caplog):
    payload = {
        "results": [{"url": "https://a.example/x", "title": "A", "content": "c", "engine": "bing"}],
        "unresponsive_engines": [["brave", "Suspended: too many requests"], ["yandex", "timeout"]],
    }
    client = MagicMock()
    client.get = AsyncMock(return_value=_resp(payload))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch.object(ns.httpx, "AsyncClient", return_value=client), \
         caplog.at_level(logging.INFO, logger="app.tools.native_search"):
        out = await ns._search_searxng("copper tariffs", 5)
    assert len(out) == 1
    assert any("unresponsive" in r.message and "brave" in r.message for r in caplog.records)
    health = ns.engine_health()
    assert health["brave"]["unresponsive"] == 1
    assert health["yandex"]["unresponsive"] == 1
    assert health["bing"]["nonempty"] == 1
    assert health["bing"]["calls"] == 1 and health["brave"]["calls"] == 1
    assert ns._LAST_UNRESPONSIVE == ["brave", "yandex"]


def test_engine_health_shape_is_per_engine_counters():
    ns._note_engine("wikipedia", nonempty=True)
    ns._note_engine("wikipedia", nonempty=False)
    ns._note_engine("brave", unresponsive=True)
    assert ns.engine_health() == {
        "wikipedia": {"calls": 2, "nonempty": 1, "unresponsive": 0},
        "brave": {"calls": 1, "nonempty": 0, "unresponsive": 1},
    }


def test_engine_health_summary_names_rates_and_suspensions():
    for _ in range(4):
        ns._note_engine("bing", nonempty=True)
    ns._note_engine("bing", nonempty=False)
    for _ in range(3):
        ns._note_engine("brave", unresponsive=True)
    text = ns.engine_health_summary()
    assert "bing 80%" in text
    assert "brave" in text and "suspended x3" in text


def test_search_log_line_names_the_suspended_engines():
    src = inspect.getsource(ns.search)
    assert "unresponsive=" in src


@pytest.mark.asyncio
async def test_web_search_zero_results_is_a_directive_refusal():
    from app.tools.web_search import WebSearchTool
    tool = WebSearchTool()
    with patch.object(ns, "search", AsyncMock(return_value=[])):
        res = await tool.execute(query="copper tariffs")
    assert res.success is False
    assert res.output.startswith("SEARCH REFUSED: every engine returned nothing for 'copper tariffs'")
    assert "Do not infer an answer" in res.output


def test_engineering_report_carries_the_engine_field():
    from app.monitors import engineering_report as er
    src = inspect.getsource(er.build_report)
    assert 'fields["search_engines"]' in src
    assert "engine_health_summary" in src
