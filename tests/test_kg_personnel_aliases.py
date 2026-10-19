"""Personnel / membership predicate aliases (2026-10-19).

A diagnostic over delivered digests showed clean entity->entity facts being
DISCARDED because their verb wasn't in the predicate map and degraded to the
vague 'related_to', which the quality gate refuses: "X replaced Y", "A plays
for B", "official appointed to body", "Pfizer merged with Seagen". These now
map to existing canonical predicates and bank. Transaction/quote verbs stay
OUT by design — their objects are quantities/statements, not entities.
"""

from __future__ import annotations

import pytest

from app.core.kg import normalize_predicate, is_vague_or_misfit


@pytest.mark.parametrize("pred,canon", [
    ("replaced", "successor_of"),
    ("replaces", "successor_of"),
    ("succeeds", "successor_of"),
    ("succeeded", "successor_of"),
    ("took over from", "successor_of"),
    ("plays for", "member_of"),
    ("plays_for", "member_of"),
    ("appointed to", "member_of"),
    ("elected to", "member_of"),
    ("named to", "member_of"),
    ("merged with", "partnered_with"),
])
def test_clean_relational_verbs_resolve_to_canonical(pred, canon):
    assert normalize_predicate(pred) == canon


@pytest.mark.parametrize("s,p,o", [
    ("Oluwatoyin Babalola", "replaced", "Cornelius Ajayi"),
    ("Hyron Andrews", "plays for", "Lions rugby team"),
    ("the minister", "appointed to", "UN Security Council"),
    ("Pfizer", "merged with", "Seagen"),
])
def test_recovered_facts_pass_the_vague_gate(s, p, o):
    assert not is_vague_or_misfit(s, p, o)


@pytest.mark.parametrize("p", ["transferred", "estimated", "reported", "sold", "urges", "targets",
                               "joined", "joins"])
def test_ambiguous_and_transaction_verbs_still_degrade(p):
    # Transaction/attribution verbs stay OUT: their objects are quantities/
    # statements, not entities. Bare "joined"/"joins" stay OUT too: polysemous
    # ("joined the race/forces/call"), so aliasing them would bank noise the
    # gates can't catch.
    assert normalize_predicate(p) == "related_to"
