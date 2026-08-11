"""#5: the chat-facing deep_research tool must read via the monitor engine's
_fetch_body (http fast-path → headless-browser fallback), not http-only — so it
can read JS-rendered quality news (BBC/CNBC/Reuters/Economist) instead of getting
a CSS shell and reporting NO ANSWER on the best sources.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_interactive_deep_research_uses_browser_capable_reader(monkeypatch):
    import app.tools.search_agent as sa
    import app.tools.native_search as ns
    import app.monitors.deep_research as dr
    from app.core import llm

    async def fake_search(query, **k):
        return [SimpleNamespace(title="Story", url="https://bbc.com/news/x",
                                snippet="snippet", engine="bing")]
    monkeypatch.setattr(ns, "search", fake_search)

    seen = {"fetched": [], "budget_passed": False}

    async def fake_fetch_body(url, **k):
        seen["fetched"].append(url)
        seen["budget_passed"] = "browser_budget" in k    # the bounded-render budget is threaded
        return "The full article body states the answer is 42, per officials."
    monkeypatch.setattr(dr, "_fetch_body", fake_fetch_body)

    async def fake_llm(msgs, **k):
        content = msgs[0]["content"]
        if "RELEVANT SENTENCES" in content:
            return "The answer is 42."
        return "The answer is 42 [Source: https://bbc.com/news/x]."
    monkeypatch.setattr(llm, "invoke_nothink", fake_llm)

    out = await sa.deep_research("what is the answer?", max_rounds=1, max_pages=2)

    assert seen["fetched"] == ["https://bbc.com/news/x"]   # routed through the browser-capable reader
    assert seen["budget_passed"]                            # with a bounded browser budget
    assert "42" in out


@pytest.mark.asyncio
async def test_interactive_deep_research_browser_budget_is_bounded(monkeypatch):
    # The browser budget is a single shared mutable cap for the whole call, so a
    # research request can't fan out into unbounded headless renders on the chat path.
    import app.tools.search_agent as sa
    import app.tools.native_search as ns
    import app.monitors.deep_research as dr
    from app.core import llm

    async def fake_search(query, **k):
        return [SimpleNamespace(title=f"S{i}", url=f"https://site{i}.com/a",
                                snippet="s", engine="bing") for i in range(3)]
    monkeypatch.setattr(ns, "search", fake_search)

    budgets_seen = []

    async def fake_fetch_body(url, *, browser_budget=None, **k):
        budgets_seen.append(id(browser_budget))   # same list object across all fetches
        return "body text answering the question fully"
    monkeypatch.setattr(dr, "_fetch_body", fake_fetch_body)

    async def fake_llm(msgs, **k):
        return "ans [Source: x]" if "RELEVANT" not in msgs[0]["content"] else "ans"
    monkeypatch.setattr(llm, "invoke_nothink", fake_llm)

    await sa.deep_research("q?", max_rounds=1, max_pages=3)
    assert len(set(budgets_seen)) == 1   # one shared budget object, not a fresh cap per fetch
