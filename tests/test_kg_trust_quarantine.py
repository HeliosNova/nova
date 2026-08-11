"""Memory-poisoning defense: trust weights + quarantine-before-promote (2026-07-08).

Facts banked from untrusted web content (source in {extracted, researched},
pipeline provenance set, trust < 0.7) are stored but QUARANTINED — excluded
from get_relevant_facts (the prompt-injection surface) until an independent
pipeline re-observes the same triple (corroboration promotes: quarantined=0,
trust +0.2 capped at 0.9). Owner/chat/internal facts are never quarantined.
Empty provenance = manual/local add — never quarantined (keeps tests + API
seeding working).

Counters the AgentPoison/MINJA class: >=80% attack success at <0.1% poison
rate on unguarded auto-banking stores (OWASP ASI06).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def kg(db):
    from app.core.kg import KnowledgeGraph
    return KnowledgeGraph(db)


def _row(kg, s, p, o):
    # Match on subject+object only: add_fact normalizes predicates on storage
    # ("develops" may canonicalize or degrade to related_to), so the raw
    # predicate string is not a reliable key for verification.
    return kg._db.fetchone(
        "SELECT COALESCE(trust,0.5) AS trust, COALESCE(quarantined,0) AS quarantined "
        "FROM kg_facts WHERE LOWER(subject)=LOWER(?) AND LOWER(object)=LOWER(?) "
        "AND valid_to IS NULL",
        (s, o),
    )


class TestQuarantineGate:
    @pytest.mark.asyncio
    async def test_low_trust_web_fact_is_quarantined(self, kg):
        ok = await kg.add_fact("Vexcorp", "develops", "quantum reactors",
                               source="researched", provenance="deep_research:tech",
                               trust=0.45)
        assert ok
        row = _row(kg, "Vexcorp", "develops", "quantum reactors")
        assert row["quarantined"] == 1
        assert row["trust"] == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_quarantined_fact_excluded_from_retrieval(self, kg):
        await kg.add_fact("Vexcorp", "develops", "quantum reactors",
                          source="researched", provenance="deep_research:tech",
                          trust=0.45)
        facts = kg.get_relevant_facts("What does Vexcorp develop?")
        assert not any("vexcorp" in (f.subject or "").lower() for f in facts)

    @pytest.mark.asyncio
    async def test_high_support_web_fact_not_quarantined(self, kg):
        await kg.add_fact("Vexcorp", "develops", "quantum reactors",
                          source="researched", provenance="deep_research:tech",
                          trust=0.75)
        row = _row(kg, "Vexcorp", "develops", "quantum reactors")
        assert row["quarantined"] == 0

    @pytest.mark.asyncio
    async def test_chat_fact_not_quarantined(self, kg):
        await kg.add_fact("Rogelio", "works_on", "Nova",
                          source="extracted", provenance="chat", trust=0.85)
        row = _row(kg, "Rogelio", "works_on", "Nova")
        assert row["quarantined"] == 0

    @pytest.mark.asyncio
    async def test_manual_add_without_provenance_not_quarantined(self, kg):
        """Empty provenance = local/manual add (tests, API) — never gated."""
        await kg.add_fact("Halcyon", "made_by", "Vertex Labs")
        row = _row(kg, "Halcyon", "made_by", "Vertex Labs")
        assert row["quarantined"] == 0
        facts = kg.get_relevant_facts("Who makes Halcyon?")
        assert any("halcyon" in (f.subject or "").lower() for f in facts)

    @pytest.mark.asyncio
    async def test_source_default_trust_applied(self, kg):
        """No explicit trust: web sources default 0.5 → quarantined when
        pipeline-attributed."""
        await kg.add_fact("Umbra Group", "acquired", "Nimbus Ltd",
                          source="extracted", provenance="monitor:Finance")
        row = _row(kg, "Umbra Group", "acquired", "Nimbus Ltd")
        assert row["quarantined"] == 1


class TestCorroborationPromotion:
    @pytest.mark.asyncio
    async def test_independent_observation_promotes(self, kg):
        await kg.add_fact("Vexcorp", "develops", "quantum reactors",
                          source="researched", provenance="deep_research:tech",
                          trust=0.45)
        # Same triple re-observed by a DIFFERENT pipeline
        await kg.add_fact("Vexcorp", "develops", "quantum reactors",
                          source="extracted", provenance="monitor:Technology",
                          trust=0.6)
        row = _row(kg, "Vexcorp", "develops", "quantum reactors")
        assert row["quarantined"] == 0
        assert row["trust"] == pytest.approx(0.65)  # 0.45 + 0.2
        facts = kg.get_relevant_facts("What does Vexcorp develop?")
        assert any("vexcorp" in (f.subject or "").lower() for f in facts)

    @pytest.mark.asyncio
    async def test_same_pipeline_reobservation_does_not_promote(self, kg):
        """The SAME provenance re-banking the same fact is not independent
        evidence — a poisoned page re-fetched daily must not self-promote."""
        for _ in range(3):
            await kg.add_fact("Vexcorp", "develops", "quantum reactors",
                              source="researched", provenance="deep_research:tech",
                              trust=0.45)
        row = _row(kg, "Vexcorp", "develops", "quantum reactors")
        assert row["quarantined"] == 1

    @pytest.mark.asyncio
    async def test_trust_promotion_capped(self, kg):
        await kg.add_fact("A Corp", "leads", "B Market",
                          source="researched", provenance="deep_research:x", trust=0.85)
        await kg.add_fact("A Corp", "leads", "B Market",
                          source="extracted", provenance="monitor:y", trust=0.6)
        row = _row(kg, "A Corp", "leads", "B Market")
        assert row["trust"] <= 0.9


class TestAgedPromotion:
    @pytest.mark.asyncio
    async def test_aged_quarantined_fact_promotes(self, kg):
        # Bank a low-trust web fact (quarantined), then age it past the window.
        await kg.add_fact("Zephyr Corp", "leads", "the widget market",
                          source="researched", provenance="deep_research:x", trust=0.5)
        row = _row(kg, "Zephyr Corp", "leads", "the widget market")
        assert row["quarantined"] == 1
        # Backdate created_at beyond the promotion window.
        kg._db.execute(
            "UPDATE kg_facts SET created_at = datetime('now','-10 days') "
            "WHERE LOWER(subject)=LOWER(?)", ("Zephyr Corp",))
        n = await kg.promote_aged_quarantine(days=7)
        assert n >= 1
        row2 = _row(kg, "Zephyr Corp", "leads", "the widget market")
        assert row2["quarantined"] == 0
        # HARDENED 2026-07-09 (full-system exploration): age-release SURFACES the
        # fact (quarantined=0) but must NOT grant it authority. Everything in
        # quarantine is a low-credibility single-source claim, so releasing it to
        # trust 0.7 handed a patient poisoner an authoritative injected fact.
        # Release now caps at 0.6 (MIN), so it renders sub-authoritative and is
        # never stated as established fact. Default window is also 21d (not 7).
        assert row2["trust"] <= 0.6

    @pytest.mark.asyncio
    async def test_fresh_quarantined_fact_not_promoted(self, kg):
        await kg.add_fact("Nimbus Ltd", "based_in", "Etherford",
                          source="researched", provenance="deep_research:y", trust=0.5)
        n = await kg.promote_aged_quarantine(days=7)  # created just now → too fresh
        assert n == 0
        assert _row(kg, "Nimbus Ltd", "based_in", "Etherford")["quarantined"] == 1
