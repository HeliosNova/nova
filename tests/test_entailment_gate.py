"""MiniCheck entailment gate (#48) — deterministic coverage with a mocked
sidecar. The gate must: skip when disabled; leave entailed sentences alone;
re-cite a claim its cited source doesn't entail but another read source does;
drop a claim no source entails; and fail OPEN when the sidecar is down."""
from __future__ import annotations

import pytest

import app.monitors.deep_research as dr
from app.config import config

ARTS = [
    ("Fed report", "https://reuters.com/fed", "The Federal Reserve held rates steady at 4 percent."),
    ("Tech story", "https://cnbc.com/apple", "Apple launched a new laptop with the M5 chip today."),
]

TEXT = ("* The Federal Reserve held rates steady at 4 percent this week (reuters.com).\n"
        "* Apple launched a new laptop with the M5 chip (reuters.com).\n"
        "* A secret merger between two unnamed giants was finalized quietly (cnbc.com).\n")


class _FakeResponse:
    def __init__(self, results):
        self._results = results

    def raise_for_status(self):
        pass

    def json(self):
        return {"results": self._results}


def _fake_client_factory(script):
    """script: list of results-lists, one per successive POST call."""
    calls = {"n": 0, "payloads": []}

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            calls["payloads"].append(json)
            res = script[min(calls["n"], len(script) - 1)]
            calls["n"] += 1
            if isinstance(res, Exception):
                raise res
            return _FakeResponse(res)

    return _FakeClient, calls


@pytest.fixture
def minicheck_on():
    config.update(ENABLE_MINICHECK=True)
    yield
    config.update(ENABLE_MINICHECK=False)


@pytest.mark.asyncio
async def test_gate_disabled_is_noop():
    out, n = await dr._entailment_gate(TEXT, ARTS)
    assert out == TEXT and n == 0


@pytest.mark.asyncio
async def test_recite_and_drop(minicheck_on, monkeypatch):
    # first batch: sentence1 supported, sentence2 unsupported (wrong host),
    # sentence3 unsupported. second batch (alternates): apple claim entailed by
    # cnbc.com; merger claim entailed by nobody.
    first = [{"supported": True, "prob": 0.97},
             {"supported": False, "prob": 0.04},
             {"supported": False, "prob": 0.02}]
    # alt pairs: (apple↔cnbc), (merger↔reuters)  — order follows per_host iteration
    second = [{"supported": True, "prob": 0.93},
              {"supported": False, "prob": 0.03}]
    fake, calls = _fake_client_factory([first, second])
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    out, n = await dr._entailment_gate(TEXT, ARTS)
    assert n == 2
    # the Fed sentence survives untouched
    assert "held rates steady at 4 percent this week (reuters.com)" in out
    # the Apple claim got RE-CITED to the source that entails it
    assert "M5 chip (cnbc.com)" in out and "M5 chip (reuters.com)" not in out
    # the fabricated merger claim is gone
    assert "secret merger" not in out


@pytest.mark.asyncio
async def test_fail_open_when_sidecar_down(minicheck_on, monkeypatch):
    fake, _ = _fake_client_factory([ConnectionError("sidecar down")])
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    out, n = await dr._entailment_gate(TEXT, ARTS)
    assert out == TEXT and n == 0
