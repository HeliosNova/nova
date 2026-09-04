"""The enrichment entailment gate: verdicts, and what they cost (2026-09-04).

This gate had no test at all, which is how it came to publish unverified
summaries on a timeout and, once that was fixed, to become the digest chain's
dominant expense without anyone costing it. Both properties are pinned here:
the verdict a summary gets, and the number of times its page is read.

The cascade reads the head of a page first and the whole page only when the
head came up short. It is exact by construction — a claim entailed by part of a
document is entailed by the document — so these tests assert the verdicts are
the ones full-width scoring would give, and that the cheap path stays cheap.
"""
from __future__ import annotations

import pytest

import app.monitors.domain_study_runner as dsr
from app.config import config

HEAD = "The central bank raised its policy rate by 50 basis points on Tuesday. "
FILLER = "Unrelated background about the building and its architecture. " * 90
TAIL = "Officials also confirmed the balance sheet runoff will continue through June. "

EARLY_BODY = HEAD + FILLER                 # support is in the first 2,600 chars
LATE_BODY = HEAD + FILLER + TAIL           # support sits past the narrow cut

EARLY_CLAIM = "The central bank raised its policy rate by 50 basis points."
LATE_CLAIM = "Balance sheet runoff continues through June."


class _Resp:
    def __init__(self, results):
        self._r = results

    def raise_for_status(self):
        pass

    def json(self):
        return {"results": self._r}


def _client(*, fail_first: bool = False):
    """A sidecar that entails a claim when the document actually contains its
    supporting sentence, and records every document it was shown."""
    seen = {"docs": [], "posts": 0}

    class _C:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            seen["posts"] += 1
            if fail_first and seen["posts"] == 1:
                raise RuntimeError("sidecar busy")
            out = []
            for pair in json["pairs"]:
                doc = pair["doc"]
                seen["docs"].append(doc)
                if pair["claim"] == EARLY_CLAIM:
                    ok = "50 basis points" in doc
                elif pair["claim"] == LATE_CLAIM:
                    ok = "balance sheet runoff" in doc.lower()
                else:
                    ok = False
                out.append({"supported": ok, "prob": 0.94 if ok else 0.02})
            return _Resp(out)

    return _C, seen


@pytest.fixture
def minicheck_on():
    config.update(ENABLE_MINICHECK=True, MINICHECK_URL="http://minicheck:9000")
    yield
    config.update(ENABLE_MINICHECK=False)


@pytest.mark.asyncio
async def test_support_in_the_first_page_costs_one_read(minicheck_on, monkeypatch):
    fake, seen = _client()
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    cand = [({"title": "rate"}, EARLY_BODY, EARLY_CLAIM)]
    kept = await dsr._entail_gate_enrich_summaries("Finance", cand)

    assert len(kept) == 1
    assert seen["posts"] == 1, "an entailed summary must not pay for the full page"
    assert len(seen["docs"][0]) <= 2600


@pytest.mark.asyncio
async def test_support_past_the_cut_is_found_at_full_width(minicheck_on, monkeypatch):
    """The needle-past-the-cut case that motivated the wide window: the summary
    must survive, and the second read must be the one that saves it."""
    fake, seen = _client()
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    cand = [({"title": "runoff"}, LATE_BODY, LATE_CLAIM)]
    kept = await dsr._entail_gate_enrich_summaries("Finance", cand)

    assert len(kept) == 1, "a grounded summary was dropped by the narrow pass"
    assert seen["posts"] == 2
    assert "balance sheet runoff" not in seen["docs"][0].lower()
    assert "balance sheet runoff" in seen["docs"][1].lower()


@pytest.mark.asyncio
async def test_a_summary_no_page_supports_still_drops(minicheck_on, monkeypatch):
    fake, seen = _client()
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    cand = [({"title": "invented"}, EARLY_BODY,
             "The bank announced a merger with a regional lender.")]
    kept = await dsr._entail_gate_enrich_summaries("Finance", cand)

    assert kept == [], "an unsupported summary must not survive the cascade"
    assert seen["posts"] == 2, "it must be refused at full width, not on the head alone"


@pytest.mark.asyncio
async def test_a_degraded_sidecar_publishes_unverified_without_a_second_pass(
        minicheck_on, monkeypatch):
    """Fail-open is the documented posture, and a failed chunk must not buy a
    second timeout on the way out."""
    fake, seen = _client(fail_first=True)
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    cand = [({"title": "rate"}, EARLY_BODY, EARLY_CLAIM)]
    kept = await dsr._entail_gate_enrich_summaries("Finance", cand)

    assert len(kept) == 1
    assert seen["posts"] == 1, "a degraded chunk must not be re-checked at full width"


@pytest.mark.asyncio
async def test_the_gate_is_a_noop_when_minicheck_is_off(monkeypatch):
    config.update(ENABLE_MINICHECK=False)
    cand = [({"title": "rate"}, EARLY_BODY, EARLY_CLAIM)]
    assert await dsr._entail_gate_enrich_summaries("Finance", cand) is cand
