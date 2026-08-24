"""Principle predicates must survive normalization (audit 2026-08-23).

`principles.distill` writes durable KG facts with predicate `principle_says`
(solo high-helpful lesson) / `principle_consensus` (cluster consensus). Those
were NOT in CANONICAL_PREDICATES, so `normalize_predicate` degraded them to
`related_to` — making distilled principles indistinguishable from generic
associations in retrieval, and defeating the point of promoting lessons to
durable facts that survive lesson decay.
"""

from app.core.kg import normalize_predicate, CANONICAL_PREDICATES


def test_principle_says_is_canonical():
    assert "principle_says" in CANONICAL_PREDICATES
    assert normalize_predicate("principle_says") == "principle_says"


def test_principle_consensus_is_canonical():
    assert "principle_consensus" in CANONICAL_PREDICATES
    assert normalize_predicate("principle_consensus") == "principle_consensus"


def test_principle_predicates_not_degraded_to_related_to():
    # The regression: both silently degraded to related_to.
    assert normalize_predicate("principle_says") != "related_to"
    assert normalize_predicate("principle_consensus") != "related_to"
