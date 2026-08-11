"""Natural-language fact rendering ↔ facts-first gate contract (task #63).

format_for_prompt renders facts as natural sentences (paraphrased evidence
measurably increases small-model receptiveness — ACL 2025, arXiv 2409.10955),
but brain._kg_answers_query PARSES those lines to fire tool-less generation.
The contract: every rendered line is "SUBJECT <verb phrase> OBJECT" with the
subject immediately before a verb the gate's alternation matches. These tests
pin both sides so neither can drift alone.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.brain import _kg_answers_query
from app.core.kg import KnowledgeGraph, Fact, _PRED_PHRASES


def _fact(subject, predicate, obj, conf=0.9):
    now = datetime.now(timezone.utc).isoformat()
    return Fact(id=1, subject=subject, predicate=predicate, object=obj,
                confidence=conf, source="researched", created_at=now,
                valid_from=now, valid_to=None, provenance="deep_research:x",
                superseded_by=None)


class TestRenderedLinesFireTheGate:
    @pytest.mark.parametrize("subject,pred,obj,query", [
        ("Vertex Dynamics Labs", "developed_by", "Halcyon", "Who is behind Vertex Dynamics Labs?"),
        ("Nimbus", "leads", "cloud analytics", "Who runs Nimbus?"),
        ("Vorenza", "based_in", "Brindlemark", "Where is Vorenza located?"),
        ("Bitcoin", "price_of", "$97,500", "What do you know about Bitcoin pricing?"),
    ])
    def test_rendered_fact_fires_gate(self, subject, pred, obj, query):
        text = KnowledgeGraph.format_for_prompt([_fact(subject, pred, obj)])
        assert subject in text
        assert _kg_answers_query(query, text) is True, f"gate missed: {text}"

    def test_action_query_still_vetoed(self):
        text = KnowledgeGraph.format_for_prompt([_fact("Nimbus", "leads", "cloud analytics")])
        assert _kg_answers_query("Search the web for Nimbus news", text) is False

    def test_every_phrase_keeps_subject_first_contract(self):
        """No phrase may start with anything that would absorb into the
        gate's subject capture (leading adverbs break token matching)."""
        gate_verbs = ("related to", "is", "was", "has", "makes", "leads",
                      "develops", "regulates", "produces", "created", "owns",
                      "acquired", "developed", "works", "lives", "currently")
        for pred, phrase in _PRED_PHRASES.items():
            first_word = phrase.split()[0]
            assert any(
                phrase.startswith(v) or first_word == v.split()[0]
                for v in gate_verbs
            ), f"phrase for {pred!r} ({phrase!r}) not in the gate's verb alternation"


class TestDomainQuerySubgraph:
    def test_domain_regex_captures_entity(self):
        from app.core.brain import _DOMAIN_QUERY_RE
        m = _DOMAIN_QUERY_RE.search("What do you know about Vertex Dynamics Labs?")
        assert m and m.group(1).strip(" ?") == "Vertex Dynamics Labs"
        m2 = _DOMAIN_QUERY_RE.search("tell me everything about the semiconductor market")
        assert m2 and "semiconductor market" in m2.group(1)
        assert _DOMAIN_QUERY_RE.search("What is 2+2?") is None

    @pytest.mark.asyncio
    async def test_entity_subgraph_pulls_hops_and_excludes_quarantined(self, db):
        kg = KnowledgeGraph(db)
        await kg.add_fact("Vexcorp", "acquired", "Nimbus Ltd", source="user")
        await kg.add_fact("Nimbus Ltd", "based_in", "Brindlemark", source="user")
        await kg.add_fact("Vexcorp", "develops", "quantum reactors",
                          source="researched", provenance="deep_research:x", trust=0.45)
        facts = kg.entity_subgraph("Vexcorp", limit=10)
        objs = {f.object.lower() for f in facts}
        assert "nimbus ltd" in objs                    # direct
        assert any("brindlemark" in o for o in objs)   # 1-hop
        assert not any("quantum" in o for o in objs)   # quarantined excluded
