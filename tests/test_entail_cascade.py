"""What the entailment cascade costs, and when it refuses to run (2026-09-04).

Entailment is 64% of a digest's wall clock (19.8 of 30.8 minutes measured over
25 live digests) and runs on the CPU sidecar while the GPU idles. Cost scales
steeply with document length: on 60 real claim/document pairs, 5,508 characters
took 8.79 seconds a pair and 2,754 took 2.42. Scoring the narrow document first
and re-checking only its failures reproduces the full-width verdict set,
because nothing the narrow document supported was rejected at full width.

The saving, though, is conditional, and the first version of this file asserted
it unconditionally. A narrow pass costs 2.42 s and a full one 8.79, so a cascade
only pays while more than about 27% of pairs clear the narrow document. The
60-pair sample sat at 70% supported; a live crypto digest the same morning ran
20 of 23 UNSUPPORTED, and there the cascade would have been ~10% SLOWER. So the
rate is probed on a spread sample, and the cascade steps aside when the probe
says it would lose.

These are behavioural tests. The ones they replace asserted on the source text
of the gate, which passes whether or not the gate works and fails whenever it
is edited - the same defect as scripting a stub by call order.
"""
from __future__ import annotations

import re

import pytest

import app.monitors.deep_research as dr
from app.config import config

MARKER = "SECONDARYSOURCEMARKER"

# One host, two articles. The first carries every supported fact, so it wins the
# ranking and IS the narrow document; the second is only ever read at full
# width, which is what makes the two widths distinguishable from outside.
LEAD = ("Northwind Energy commissioned the Barrow tidal array in March. "
        "The array delivers 240 megawatts to the regional grid. "
        "Regulators approved the second phase in June. ")
ARTS = [("Main report", "https://example.com/a1", LEAD),
        ("Sidebar", "https://example.com/a2", MARKER + " brief follow-up note.")]

SUPPORTED = [
    "Northwind Energy commissioned the Barrow tidal array in March (example.com).",
    "The Barrow tidal array delivers 240 megawatts to the regional grid (example.com).",
    "Regulators approved the second phase of Barrow in June (example.com).",
]
UNSUPPORTED = [
    "Northwind Energy opened a hydrogen electrolyser plant in Aberdeen (example.com).",
    "The company signed a supply agreement with a Norwegian shipping group (example.com).",
    "A consortium acquired the remaining stake in the offshore division (example.com).",
    "Executives confirmed a listing on the Frankfurt exchange next quarter (example.com).",
    "The regulator opened an inquiry into transmission pricing in Wales (example.com).",
    "A pilot storage facility began operating on the eastern seaboard (example.com).",
    "Analysts revised their capacity forecast for the coming decade upward (example.com).",
    "The board appointed a former minister as its incoming chair (example.com).",
    "Construction crews completed the substation ahead of the winter (example.com).",
]

_SUPPORT_PHRASES = ("tidal array in March", "240 megawatts", "second phase in June")
_SUPPORT_KEYS = ("Barrow", "240 megawatts", "second phase")


def _text(sentences) -> str:
    return "".join("* " + s + "\n" for s in sentences)


class _Resp:
    def __init__(self, results):
        self._r = results

    def raise_for_status(self):
        pass

    def json(self):
        return {"results": self._r}


def _client():
    """A sidecar that entails a claim when the document actually says it, and
    records the width of every document it was shown."""
    seen = {"pairs": [], "posts": 0}

    class _C:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            seen["posts"] += 1
            out = []
            for p in json["pairs"]:
                doc, claim = p["doc"], p["claim"]
                seen["pairs"].append((claim, MARKER in doc))
                ok = (any(w in doc for w in _SUPPORT_PHRASES)
                      and any(t in claim for t in _SUPPORT_KEYS))
                out.append({"supported": ok, "prob": 0.93 if ok else 0.02})
            return _Resp(out)

    return _C, seen


@pytest.fixture
def minicheck_on():
    config.update(ENABLE_MINICHECK=True)
    yield
    config.update(ENABLE_MINICHECK=False)


@pytest.fixture(autouse=True)
def _forget_narrow_rates():
    """The cascade remembers each call site's narrow-support rate in process,
    which is the point of it — and which leaks between tests if left alone."""
    dr._NARROW_RATE.clear()
    yield
    dr._NARROW_RATE.clear()


def _key(claim: str) -> str:
    """Compare claims by their letters: the gate strips the citation and leaves
    the bullet and stray spacing, none of which is the claim."""
    return re.sub(r"[^a-z0-9]+", "", claim.split(" (")[0].lower())


def _widths(seen, sentences):
    """(narrow, full) counts for the whole-sentence claims only - the clause
    rescue judges sub-claims, which are a different question."""
    whole = {_key(s) for s in sentences}
    narrow = full = 0
    for claim, was_full in seen["pairs"]:
        if _key(claim) not in whole:
            continue
        full += was_full
        narrow += not was_full
    return narrow, full


@pytest.mark.asyncio
async def test_a_well_supported_digest_is_scored_narrow_throughout(
        minicheck_on, monkeypatch):
    """When the probe clears the bar, every claim gets the cheap document."""
    sentences = SUPPORTED * 4                       # 12 claims, all entailed
    fake, seen = _client()
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    out, n = await dr._entailment_gate(_text(sentences), ARTS)
    assert n == 0, "entailed sentences must survive untouched"
    narrow, full = _widths(seen, sentences)
    assert narrow == len(sentences), "every claim should have been scored narrow"
    assert full == 0, "nothing failed, so nothing should have been re-read"


@pytest.mark.asyncio
async def test_a_mostly_unsupported_digest_stops_narrowing_after_the_probe(
        minicheck_on, monkeypatch):
    """The case a cascade loses on: the narrow pass is a tax on work that gets
    redone anyway, so only the probe pays it."""
    sentences = UNSUPPORTED + SUPPORTED[:1]         # 10 claims, 1 entailed
    fake, seen = _client()
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    await dr._entailment_gate(_text(sentences), ARTS)
    narrow, full = _widths(seen, sentences)
    assert narrow <= 8, "the probe must be bounded; " + str(narrow) + " went narrow"
    assert narrow < len(sentences), "the cascade should have stepped aside"
    assert full >= len(sentences) - narrow, "the rest must still get full width"


@pytest.mark.asyncio
async def test_the_probe_samples_across_the_briefing_not_its_lead(
        minicheck_on, monkeypatch):
    """A briefing's opening sentences are its lead, and their support rate is
    not the document's, so a head-only probe would misjudge a long digest."""
    sentences = UNSUPPORTED[:6] + SUPPORTED         # unsupported first, then not
    fake, seen = _client()
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    await dr._entailment_gate(_text(sentences), ARTS)
    probed = [c for c, was_full in seen["pairs"] if not was_full]
    assert any(t in c for c in probed for t in _SUPPORT_KEYS), \
        "the probe never looked past the lead"


def test_a_site_judged_not_worth_narrowing_is_re_measured_eventually():
    """The rate memory was a one-way latch: once a call site measured below the
    bar it never measured again for the life of the process. Restarts reset it,
    so it would never have been loud - which is the shape of thing that ends up
    silently off for weeks here."""
    dr._NARROW_RATE.clear()
    dr._NARROW_SKIPS.clear()
    key = "some monitor:gate"

    assert dr._narrow_worth_it(key), "an unknown site must be probed"
    dr._record_narrow_rate(key, 0.10)                 # measured: not worth it

    seq = [dr._narrow_worth_it(key) for _ in range(dr._NARROW_REPROBE_EVERY + 5)]
    assert seq[0] is False, "it should stop narrowing immediately"
    assert sum(seq) == 1, "exactly one re-probe per cycle, not every call"
    assert seq.index(True) == dr._NARROW_REPROBE_EVERY - 1

    dr._record_narrow_rate(key, 0.90)                 # sources improved
    assert dr._narrow_worth_it(key), "a recovered site must narrow again at once"
    dr._NARROW_RATE.clear()
    dr._NARROW_SKIPS.clear()


@pytest.mark.asyncio
async def test_a_dead_sidecar_still_leaves_the_text_alone(minicheck_on, monkeypatch):
    class _Dead:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            raise ConnectionError("sidecar down")

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Dead)
    text = _text(SUPPORTED)
    out, n = await dr._entailment_gate(text, ARTS)
    assert out == text and n == 0
@pytest.mark.asyncio
async def test_a_miss_reports_how_many_articles_the_host_had(
        minicheck_on, monkeypatch, caplog):
    """83% of cited sentences fail this gate. Whether that is the model writing
    past its sources or the gate reading the wrong two of a host's articles is
    the whole question, and the drop log cannot tell them apart without this."""
    arts = ARTS + [("Extra one", "https://example.com/a3", "Unrelated filler about weather."),
                   ("Extra two", "https://example.com/a4", "More filler about traffic.")]
    fake, _seen = _client()
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    import logging
    with caplog.at_level(logging.INFO, logger="app.monitors.deep_research"):
        await dr._entailment_gate(_text(UNSUPPORTED[:1]), arts)

    miss = [r.getMessage() for r in caplog.records if "[entail-miss]" in r.getMessage()]
    assert miss, "an unsupported claim must still be inspectable"
    assert "arts=2/4" in miss[0], miss[0]
    assert "unread_best=" in miss[0] and "read_worst=" in miss[0]
