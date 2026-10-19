"""E1 — union-of-sources grounding (2026-10-18, flag-gated).

The gate re-cites a dropped sentence only when a SINGLE other read host entails
the whole claim. A cross-source conclusion supported by sources A+B together but
by neither alone still drops. ENABLE_UNION_GROUNDING adds a rescue that scores
the claim against a document built from the best windows across ALL sources.
Default OFF — gate behaviour is unchanged until the A/B reads the rule.

The test drives the real _entailment_gate with a scripted MiniCheck sidecar
(monkeypatched httpx) where NO single source supports the spread claim but the
concatenation does, and asserts: OFF -> sentence dropped; ON -> sentence kept.
"""

from __future__ import annotations

import httpx
import pytest

from app.monitors import deep_research as dr


# Two sources, each carrying ONE half of the claim. No single source entails the
# whole sentence; the union does.
ARTS = [
    ("Fed holds rates", "https://reuters.com/a",
     "The Federal Reserve held its benchmark rate steady on Wednesday, citing elevated inflation. " * 6),
    ("Jobs report soft", "https://apnews.com/b",
     "Nonfarm payrolls rose far less than forecast in the latest jobs report, signalling a cooling labor market. " * 6),
]
# A synthesized sentence whose support is split across the two sources.
DRAFT = ("The Federal Reserve held rates steady even as the jobs report showed a cooling "
         "labor market, a divergence that tightens its policy path (reuters.com).")


class _FakeResp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


def _install_fake_minicheck(monkeypatch, *, union_supports: bool):
    """A MiniCheck that supports a claim only when the doc contains BOTH halves
    ('federal reserve' AND 'jobs report'/'payrolls') — i.e. only the union doc."""
    def _verdict(doc, claim):
        d = (doc or "").lower()
        has_fed = "federal reserve" in d
        has_jobs = "payrolls" in d or "jobs report" in d
        return {"supported": bool(has_fed and has_jobs), "prob": 0.9 if (has_fed and has_jobs) else 0.05}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None):
            pairs = json["pairs"]
            return _FakeResp({"results": [_verdict(p["doc"], p["claim"]) for p in pairs]})
    monkeypatch.setattr(httpx, "AsyncClient", _Client)


async def _run(monkeypatch, flag):
    from app.config import config
    monkeypatch.setattr(config, "ENABLE_MINICHECK", True)
    monkeypatch.setattr(config, "MINICHECK_URL", "http://fake:8080")
    monkeypatch.setattr(config, "ENABLE_UNION_GROUNDING", flag)
    _install_fake_minicheck(monkeypatch, union_supports=True)
    out, _ = await dr._entailment_gate(DRAFT, ARTS, label="test")
    return out


@pytest.mark.asyncio
async def test_spread_claim_drops_without_the_flag(monkeypatch):
    out = await _run(monkeypatch, False)
    assert "Federal Reserve held rates steady" not in out      # no single source entailed it -> dropped


@pytest.mark.asyncio
async def test_spread_claim_is_kept_by_the_union_with_the_flag(monkeypatch):
    out = await _run(monkeypatch, True)
    assert "Federal Reserve held rates steady" in out          # union of the two sources entailed it -> kept


def test_flag_wired_and_default_off():
    from app.config import config
    import inspect
    assert "ENABLE_UNION_GROUNDING" in inspect.getsource(dr._entailment_gate)
    import os
    if "ENABLE_UNION_GROUNDING" not in os.environ:
        assert config.ENABLE_UNION_GROUNDING is False
