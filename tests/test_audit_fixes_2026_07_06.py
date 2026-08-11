"""Regression coverage for the 2026-07-06 whole-plan audit fixes:
1. KG fact banking grounds against FULL article bodies (abstract domains banked
   0 facts when the gate only saw findings text);
2. retire_stale_snapshots: point-in-time research facts expire, timeless don't;
3. claim validator tolerates honorifics/possessives (it was stripping the very
   answers the memory loop taught: "Dr. Sarah Chen" vs lesson "Sarah Chen")."""
from __future__ import annotations

import json

import pytest

import app.monitors.deep_research as dr
from app.core.claim_validator import _claim_parts, build_evidence, validate_claims
from app.core.kg import KnowledgeGraph


# ---------------------------------------------------------------------------
# 1. fact banking grounds against article bodies
# ---------------------------------------------------------------------------

class _FakeKG:
    def __init__(self):
        self.stored = []

    async def add_fact(self, s, p, o, **kw):
        self.stored.append((s, p, o))
        return True


@pytest.mark.asyncio
async def test_learn_facts_grounds_against_article_bodies(monkeypatch):
    async def fake_invoke(messages, **kw):
        return json.dumps([
            # terms appear ONLY in the article body, not in the findings text —
            # the old findings-only gate rejected exactly this shape
            {"subject": "European Union", "predicate": "coordinates", "object": "sanctions framework"},
        ])

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake_invoke)
    kg = _FakeKG()
    findings = [("t", "https://apnews.com/x", "leaders met to discuss measures")]
    articles = [("t", "https://apnews.com/x",
                 "The European Union is coordinating a new sanctions framework against the regime.")]
    n = await dr._learn_facts("geopolitics", "briefing text long enough", findings, kg,
                              articles=articles, model=None)
    assert n == 1 and kg.stored[0][0] == "European Union"


@pytest.mark.asyncio
async def test_learn_facts_still_rejects_ungrounded(monkeypatch):
    async def fake_invoke(messages, **kw):
        return json.dumps([
            {"subject": "Zorblax Corp", "predicate": "acquired", "object": "Quuxian Assets"},
        ])

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake_invoke)
    kg = _FakeKG()
    findings = [("t", "https://apnews.com/x", "leaders met to discuss measures")]
    articles = [("t", "https://apnews.com/x", "The summit produced no agreements.")]
    n = await dr._learn_facts("geopolitics", "briefing text long enough", findings, kg,
                              articles=articles, model=None)
    assert n == 0 and not kg.stored


# ---------------------------------------------------------------------------
# 2. temporal decay for point-in-time facts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retire_stale_snapshots_expires_volatile_keeps_timeless(db):
    kg = KnowledgeGraph(db)
    await kg.add_fact("bitcoin", "price_of", "$97,500", source="researched",
                      provenance="deep_research:crypto")
    await kg.add_fact("inflation", "related_to", "3.2% annual rate", source="researched",
                      provenance="deep_research:economics")
    await kg.add_fact("bitcoin", "is_a", "cryptocurrency", source="researched",
                      provenance="deep_research:crypto")
    # different subject: price_of is FUNCTIONAL — same-subject would supersede
    # the researched fact before retirement runs, invalidating the assertion
    await kg.add_fact("solana", "price_of", "$45,000", source="user")  # not research provenance
    # age all rows past the horizon
    db.execute("UPDATE kg_facts SET created_at = datetime('now','-10 days')")

    retired = await kg.retire_stale_snapshots(days=7)
    assert retired == 2  # the $97,500 price and the 3.2% figure

    current = {(r["subject"], r["predicate"], r["object"])
               for r in db.fetchall("SELECT subject, predicate, object FROM kg_facts WHERE valid_to IS NULL")}
    assert ("bitcoin", "is_a", "cryptocurrency") in current           # timeless survives
    assert ("solana", "price_of", "$45,000") in current               # non-research survives
    assert ("bitcoin", "price_of", "$97,500") not in current          # snapshot retired


@pytest.mark.asyncio
async def test_retire_stale_snapshots_spares_fresh_facts(db):
    kg = KnowledgeGraph(db)
    await kg.add_fact("ethereum", "price_of", "$5,100", source="researched",
                      provenance="deep_research:crypto")
    retired = await kg.retire_stale_snapshots(days=7)
    assert retired == 0  # created just now — inside the horizon


# ---------------------------------------------------------------------------
# 3. claim validator honorific/possessive tolerance
# ---------------------------------------------------------------------------

def test_claim_parts_drops_decorations_keeps_entities():
    assert _claim_parts("Dr. Sarah Chen") == ["Sarah", "Chen"]
    assert _claim_parts("OpenAI's") == ["OpenAI"]
    assert _claim_parts("Prof. Miller") == ["Miller"]
    # entity tokens themselves are untouched — "Mars" keeps its trailing s
    assert _claim_parts("Mars Explorer") == ["Mars", "Explorer"]


def test_validator_keeps_lesson_taught_honorific_answer():
    evidence = build_evidence(lessons_text="- [HIGH] people: Sarah Chen is the founder of Helios Data.")
    answer = "Dr. Sarah Chen, founder of Helios Data, presented the roadmap."
    out, reasons = validate_claims(answer, evidence)
    assert "Sarah Chen" in out and not reasons, f"stripped a lesson-supported claim: {reasons}"


def test_validator_still_strips_fabricated_person():
    evidence = build_evidence(lessons_text="- [HIGH] people: Sarah Chen is the founder of Helios Data.")
    answer = "Dr. Marcus Vole, founder of Helios Data, presented the roadmap."
    out, reasons = validate_claims(answer, evidence)
    assert "Marcus Vole" not in out and reasons


# ---------------------------------------------------------------------------
# 4. garbage gate: related_to + date-fragment pairings are junk
# ---------------------------------------------------------------------------

def test_related_to_date_fragments_are_garbage():
    from app.core.kg import is_garbage_triple
    assert is_garbage_triple("FIFA Disciplinary Committee", "related_to", "July 5")
    assert is_garbage_triple("Khaled al-Halabi", "related_to", "eight years")
    assert is_garbage_triple("Cuba state institutions", "related_to", "July")
    # specific predicates keep legitimate date/duration objects
    assert not is_garbage_triple("Khaled al-Halabi", "sentenced_to", "eight years")
    assert not is_garbage_triple("Google", "founded_in", "September 1998")
    # related_to with a REAL entity object stays valid
    assert not is_garbage_triple("CLARITY Act", "related_to", "Senate Banking Committee")
