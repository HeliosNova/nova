"""MiniCheck entailment gate (#48) — deterministic coverage with a mocked
sidecar. The gate must: skip when disabled; leave entailed sentences alone;
re-cite a claim its cited source doesn't entail but another read source does;
drop a claim no source entails; and fail OPEN when the sidecar is down.

v3 (2026-08-12): anchored-in-source rescue — a synthesis sentence whose cited
source entails at least one informative sub-claim keeps its citation instead
of dropping; evidence is built per-ARTICLE (a busy host's later articles used
to be amputated by the per-host [:24000] concatenation cap)."""
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


# ---------------------------------------------------------------------------
# v3: anchored-in-source clause rescue + per-article evidence
# ---------------------------------------------------------------------------

FRAMED = ("* The most structurally transformative element is the election commission "
          "announcement expanding parliament seats in April (reuters.com).\n")
FRAMED_ARTS = [
    ("Election report", "https://reuters.com/vote",
     "The election commission announcement expanding parliament seats came in April."),
]


@pytest.mark.asyncio
async def test_clause_rescue_keeps_anchored_sentence(minicheck_on, monkeypatch):
    # Whole-sentence entailment fails on the analytic lead-in ("the most
    # structurally transformative element is …") but the copula tail — the
    # factual core — IS entailed by the cited source: the sentence must be
    # KEPT with its citation, not dropped.
    first = [{"supported": False, "prob": 0.02}]     # primary: whole sentence fails
    second = [{"supported": True, "prob": 0.94}]     # clause pass: tail entailed
    fake, calls = _fake_client_factory([first, second])
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    out, n = await dr._entailment_gate(FRAMED, FRAMED_ARTS)
    assert out == FRAMED and n == 0                  # kept verbatim, nothing changed
    # the clause pass judged a SUB-claim, not the whole sentence again
    clause_claim = calls["payloads"][1]["pairs"][0]["claim"]
    assert "most structurally transformative" not in clause_claim
    assert "election commission announcement" in clause_claim


@pytest.mark.asyncio
async def test_fabrication_drops_even_with_clause_pass(minicheck_on, monkeypatch):
    # Every sub-claim ALSO fails and no other source can re-cite it — the
    # fabricated-attribution case must still drop.
    text = ("* A secret merger between two unnamed giants was finalized quietly, "
            "which analysts called unprecedented in scale (cnbc.com).\n")
    arts = [
        ("Tech story", "https://cnbc.com/apple", "Apple launched a new laptop with the M5 chip today."),
        ("Fed report", "https://reuters.com/fed", "The Federal Reserve held rates steady at 4 percent."),
    ]
    first = [{"supported": False, "prob": 0.02}]
    clause = [{"supported": False, "prob": 0.03}] * 8   # replayed for however many sub-claims
    fake, _ = _fake_client_factory([first, clause])
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    out, n = await dr._entailment_gate(text, arts)
    assert "secret merger" not in out and n == 1


@pytest.mark.asyncio
async def test_multi_article_host_evidence_not_amputated(minicheck_on, monkeypatch):
    # A busy host's LATER article must still be usable evidence. The old
    # per-host concatenation capped at 24000 chars, so a 30k first article
    # amputated the second entirely — its claims had no evidence at all.
    junk = ("Nothing relevant here. " * 1500)[:30000]   # 30k chars, article 1
    fact = ("Perovskite solar cells crossed the industrial viability threshold "
            "with 26 percent efficiency modules this quarter.")
    arts = [
        ("Junk story", "https://nature.com/a1", junk),
        ("Solar story", "https://nature.com/a2", fact),
    ]
    text = ("* Perovskite solar cells crossed the industrial viability threshold "
            "at 26 percent efficiency (nature.com).\n")
    fake, calls = _fake_client_factory([[{"supported": True, "prob": 0.96}]])
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    out, n = await dr._entailment_gate(text, arts)
    assert out == text and n == 0
    doc = calls["payloads"][0]["pairs"][0]["doc"]
    assert "Perovskite" in doc                      # article 2 present in evidence


@pytest.mark.asyncio
async def test_analytical_sentence_decited_not_dropped(minicheck_on, monkeypatch):
    # gate v4: an analysis-shaped sentence (implication verbs, no hard numbers)
    # that fails entailment keeps its prose but loses the citation — the 48h
    # drop corpus showed ~half of all final drops were the digest's own
    # reasoning wearing a citation no source could entail.
    text = ("* This consolidation mirrors similar moves by rivals, suggesting "
            "sector-wide fatigue with fragmented offerings (cnbc.com).\n")
    arts = [("Tech story", "https://cnbc.com/apple",
             "Apple launched a new laptop with the M5 chip today.")]
    first = [{"supported": False, "prob": 0.03}]
    rest = [{"supported": False, "prob": 0.03}] * 8
    fake, _ = _fake_client_factory([first, rest])
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    out, n = await dr._entailment_gate(text, arts)
    assert "mirrors similar moves" in out          # sentence survives
    assert "cnbc.com" not in out and n == 1        # citation stripped


def test_is_analytical_never_matches_numbers():
    assert dr._is_analytical("This shift mirrors similar moves across the sector, "
                             "suggesting a broader retreat.")
    # a hard figure disqualifies — unsupported numbers must still drop
    assert not dr._is_analytical("This shift mirrors 2021 patterns with revenue up 40% "
                                 "suggesting a broader retreat.")
    assert not dr._is_analytical("Apple launched a new laptop with the M5 chip.")


@pytest.mark.asyncio
async def test_failed_lead_kept_when_section_survives(minicheck_on, monkeypatch):
    # v4.1: a bold-headline lead that fails entailment is a SUMMARY of its
    # section — when a sibling sentence on the line survives, the lead stays
    # (de-cited) instead of decapitating a living section.
    text = ("* **Fed Holds Amid Dissent:** The central bank navigated a fractured "
            "committee landscape this quarter (reuters.com). The Federal Reserve "
            "held rates steady at 4 percent (reuters.com).\n")
    arts = [("Fed report", "https://reuters.com/fed",
             "The Federal Reserve held rates steady at 4 percent.")]
    # lead fails, factual sibling passes
    first = [{"supported": False, "prob": 0.04}, {"supported": True, "prob": 0.96}]
    rest = [{"supported": False, "prob": 0.03}] * 8
    fake, _ = _fake_client_factory([first, rest])
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    out, n = await dr._entailment_gate(text, arts)
    assert "**Fed Holds Amid Dissent:**" in out          # lead survives
    assert "held rates steady at 4 percent (reuters.com)" in out
    assert "landscape this quarter (reuters.com)" not in out   # lead de-cited


@pytest.mark.asyncio
async def test_failed_lead_dies_with_dead_section(minicheck_on, monkeypatch):
    # A lead whose ENTIRE section failed must still die — a headline with no
    # surviving support beneath it is exactly the fabrication case.
    text = ("* **Secret Merger Shakes Markets:** Two unnamed giants finalized "
            "a quiet merger through undisclosed intermediaries (cnbc.com).\n")
    arts = [("Tech story", "https://cnbc.com/apple",
             "Apple launched a new laptop with the M5 chip today.")]
    first = [{"supported": False, "prob": 0.02}]
    rest = [{"supported": False, "prob": 0.02}] * 8
    fake, _ = _fake_client_factory([first, rest])
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", fake)

    out, n = await dr._entailment_gate(text, arts)
    assert "Secret Merger" not in out and n == 1


def test_sentence_split_protects_abbreviations():
    # Live digest led with "death of Army Staff Sgt (defenseone.com). Benjamin
    # Pennington at…" — the splitter broke at the rank period and the gate
    # processed the halves separately (consumer read, 2026-08-14).
    parts = dr._SENT_SPLIT_RE.split(
        "the death of Army Staff Sgt. Benjamin Pennington at Prince Sultan Air "
        "Base (defenseone.com). Economically, the closure triggered ruptures.")
    assert len(parts) == 2
    assert "Sgt. Benjamin Pennington" in parts[0]
    # normal sentences still split; titles mid-sentence stay glued
    parts = dr._SENT_SPLIT_RE.split(
        "Rates held steady. Gen. Smith disagreed with Dr. Jones. Final one here.")
    assert len(parts) == 3


def test_sub_claims_decomposition():
    framed = ("The most structurally transformative element is the election commission "
              "announcement expanding parliament seats in April.")
    subs = dr._sub_claims(framed)
    assert any("election commission announcement" in s and "transformative" not in s
               for s in subs)
    # short atomic sentences yield no useful decomposition — no rescue pass
    assert dr._sub_claims("Apple launched a new laptop with the M5 chip.") == []


def test_sub_claims_bare_but_and_participial():
    # the live [entail-miss] shape that v3.0 could not decompose: compounds
    # joined by bare "but" and a participial trailer.
    dense = ("Core CPI also softened slightly to 2.5 percent annually but rose "
             "0.2 percent month-over-month, remaining structurally elevated above target.")
    subs = dr._sub_claims(dense)
    assert any(s.startswith("Core CPI also softened") and "rose" not in s for s in subs)
    assert any("rose 0.2 percent month-over-month" in s and "softened" not in s for s in subs)

    trailer = ("Shelter costs remain stubborn, accounting for roughly two-thirds "
               "of the monthly headline increase.")
    subs = dr._sub_claims(trailer)
    assert any(s.startswith("accounting for roughly two-thirds") for s in subs)


def test_scrub_chrome_removes_player_junk():
    body = ("Consumer prices rose 0.2 percent in July according to the report.\n"
            "00:00\n00:00\n1x\n"
            "This video file cannot be played.\n"
            "(Error\xa0Code:\xa0102630)\n"
            "ADD US ON GOOGLE\n"
            "Advertisement\n"
            "3 min read\n"
            "Shelter costs accounted for two-thirds of the monthly increase.")
    out = dr._scrub_chrome(body)
    assert "Consumer prices rose" in out and "Shelter costs accounted" in out
    assert "video file" not in out and "Error" not in out
    assert "ADD US ON GOOGLE" not in out and "Advertisement" not in out
    # prose that merely mentions a chrome word survives
    prose = "The advertisement industry spent 40 billion dollars on streaming platforms this year."
    assert dr._scrub_chrome(prose) == prose
