"""Whole-system quality read 2026-10-17: the four reader-facing defects the
instruments did not show, pinned.

1. 5% of delivered digests carried an orphan curly possessive after an excised
   entity ("the Department of ’s system").
2. 38% of digest sections opened on a dangling referent after the gate dropped
   the section's first sentence.
3. A "what changed … this week" chat went tool-less (a dossier matched) and
   narrated a search it never ran.
4. 34% of researched KG facts were banked as related_to; relational predicates
   carried objects that cannot fill them.
"""

from __future__ import annotations

import pytest

from app.monitors import deep_research as dr


# --- 1 -----------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("cross-reference this with the Department of ’s system to identify", "cross-reference this with the Department of system to identify"),
    ("The Netherlands, the ’s host nation, has", "The Netherlands, the host nation, has"),
    ("**Lead Development: ’s Record Q3 Resets Demand**", "**Lead Development: Record Q3 Resets Demand**"),
    ("growth is accelerating away from ’s 2% target (cnbc.com).", "growth is accelerating away from 2% target (cnbc.com)."),
    ("'s small-molecule agonist won approval.", "small-molecule agonist won approval."),
])
def test_orphan_curly_possessive_is_removed(text, expected):
    assert dr._repair_dangling_fragments(text) == expected


@pytest.mark.parametrize("text", [
    "Micron’s margin ceiling warning (bloomberg.com).",
    "the Fed's September meeting marked a break.",
    "*The Social Reckoning*’s flop pushed studios elsewhere.",
    "NVIDIA’s CUDA ecosystem dominates (reuters.com).",
])
def test_attached_possessives_are_untouched(text):
    assert dr._repair_dangling_fragments(text) == text


# --- 2 -----------------------------------------------------------------------
def test_dangling_opener_regex_shapes():
    assert dr._DANGLING_OPENER_RE.match("This fiscal burden is compounded by a critical admission")
    assert dr._DANGLING_OPENER_RE.match("Such a shift implies")
    assert dr._DANGLING_OPENER_RE.match("Consequently, investors")
    assert not dr._DANGLING_OPENER_RE.match("Meanwhile, the RBA lifted rates")
    assert not dr._DANGLING_OPENER_RE.match("Iranian Foreign Minister Abbas Araghchi presented")


def test_line_head_strips_bullet_and_bold_header():
    body = dr._LINE_HEAD_RE.sub("", "* **US-Russia Diesel Accord:** The U.S. Treasury issued a waiver. This deal fractured unity.", count=1)
    assert body.startswith("The U.S. Treasury issued a waiver.")


def test_gate_phase_three_is_wired():
    import inspect
    src = inspect.getsource(dr._entailment_gate)
    assert "dangling follower" in src and "_DANGLING_OPENER_RE.match" in src


# --- 3 -----------------------------------------------------------------------
@pytest.mark.parametrize("q", [
    "What changed in the Strait of Hormuz situation this week and what does it mean for oil?",
    "what's new with the Fed",
    "latest on TSMC",
    "give me an update on the Ukraine ceasefire",
    "what happened today in semiconductors",
])
def test_recency_queries_keep_their_tools(q):
    from app.core.brain import _RECENCY_QUERY_RE
    assert _RECENCY_QUERY_RE.search(q)


@pytest.mark.parametrize("q", [
    "Who is the current CEO of Nvidia?",
    "What do you believe about US inflation and where did that belief come from?",
    "Explain what the Strait of Hormuz is.",
])
def test_knowledge_queries_may_go_tool_less(q):
    from app.core.brain import _RECENCY_QUERY_RE
    assert not _RECENCY_QUERY_RE.search(q)


def test_tool_less_note_is_appended_to_the_system_turn():
    from app.core.brain import _with_tool_less_note, _TOOL_LESS_NOTE
    msgs = [{"role": "system", "content": "You are Nova."}, {"role": "user", "content": "hi"}]
    out = _with_tool_less_note(msgs)
    assert out[0]["content"] == "You are Nova." + _TOOL_LESS_NOTE
    assert msgs[0]["content"] == "You are Nova."          # original untouched
    assert "Do not say you searched" in out[0]["content"]
    assert out[1] == msgs[1]


def test_gate_wiring_in_think():
    import inspect
    from app.core import brain
    src = inspect.getsource(brain.think)
    assert "recency query — tools kept" in src and "_with_tool_less_note(messages)" in src


# --- 4 -----------------------------------------------------------------------
@pytest.mark.parametrize("s,p,o", [
    ("ASML", "owns", "23.0% market share"),
    ("MARA Holdings", "acquired", "$3.6 billion in total borrowings"),
    ("Ghostscript", "owns", "CVE-2026-39919"),
    ("Blazars", "leads", "ultra-high-energy neutrinos"),
    ("Arch Lending", "partnered_with", "tokenized stocks as loan collateral"),
    ("Gaza olive oil presses", "related_to", "7 units"),
    ("Strait of Hormuz", "connected with", "one-fifth of global oil supply"),   # degrades to related_to
    ("Modal", "acquired", "$16 billion Series D valuation"),                    # k8-era eyeball 2026-10-17
    ("Narendra Modi", "owns", "Prime Minister position in India"),
])
def test_vague_or_misfit_triples_are_refused(s, p, o):
    from app.core.kg import is_vague_or_misfit
    assert is_vague_or_misfit(s, p, o)


@pytest.mark.parametrize("s,p,o", [
    ("Saudi Arabia", "acquired", "730 Patriot missiles"),
    ("Microsoft", "owns", "Xbox Labs"),
    ("Eyal Zamir", "leads", "IDF"),
    ("U.S. Army", "partnered_with", "Hanwha Defense USA"),
    ("China", "sanctioned", "Paraguay officials"),
    ("Brent Crude", "price_of", "$105.56 per barrel"),
    ("Quantedge Global", "located_in", "Singapore"),
    ("MARA Holdings", "owns", "35,577 BTC treasury balance"),
    ("BlackRock", "owns", "5% position in Nvidia"),
])
def test_well_formed_triples_pass(s, p, o):
    from app.core.kg import is_vague_or_misfit
    assert not is_vague_or_misfit(s, p, o)


def test_both_banking_sites_check_fit():
    import inspect
    from app.core import brain_kg
    assert "is_vague_or_misfit" in inspect.getsource(brain_kg._extract_kg_triples)
    assert "is_vague_or_misfit" in inspect.getsource(dr._learn_facts)
