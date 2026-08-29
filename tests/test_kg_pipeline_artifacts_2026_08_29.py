"""The extractor banks triples about Nova's OWN output instead of the world.

Measured on the live KG 2026-08-29: 16 such triples were 0.3% of live facts but
9.6% of ALL retrievals — a 32x over-representation that crowded real knowledge
out of every prompt, while `acquired` facts sat 83% never-retrieved.

  2834x  semiconductors  is_a        domain overview
  2468x  geopolitics     is_a        researched briefing
  2430x  whale_watch     is_a        domain_intelligence_overview
  2321x  July 08, 2026   has_status  date of facts learned

The repaired LLM curation pass (c4faf8a) retired all of them within hours of
coming back online, so this rule is PREVENTION: block them at write time rather
than pay for a curation round to remove them later. Verified retroactively —
13/13 artifacts blocked, 0 false positives across 4944 live facts.
"""

from __future__ import annotations

import pytest

from app.core.kg import is_garbage_triple


ARTIFACTS = [
    ("semiconductors", "is_a", "domain overview"),
    ("geopolitics", "is_a", "researched briefing"),
    ("whale_watch", "is_a", "domain_intelligence_overview"),
    ("July 08, 2026", "has_status", "date of facts learned"),
    ("supply chain", "is_a", "domain overview"),
    ("quantum", "is_a", "domain overview"),
    ("World", "is_a", "domain overview"),
    ("cybersecurity", "is_a", "domain overview"),
]

# Real facts that were ALSO retired in the same window, for other reasons.
# The rule must not claim these — they are knowledge, not bookkeeping.
REAL_KNOWLEDGE = [
    ("Bank of Japan", "regulates", "1% policy rate"),
    ("Strait of Hormuz", "related_to", "one-fifth global oil shipments"),
    ("Open Secure AI Alliance", "related_to", "Linux Foundation"),
    ("travel ban policies", "caused_by", "Trump Administration"),
    ("CVE-2026-42530", "related_to", "NGINX Open Source"),
    ("Nvidia", "developed_by", "Vera Rubin platform"),
    ("TSMC", "located_in", "Taiwan"),
    ("Samsung", "produces", "LPDDR5X chips"),
    # A titled document that merely CONTAINS a date is a legitimate subject —
    # only a BARE date is rejected.
    ("Contracts for Aug. 3, 2026", "part_of", "Government Contract Awards"),
]


@pytest.mark.parametrize("s,p,o", ARTIFACTS)
def test_pipeline_artifacts_are_rejected(s, p, o):
    assert is_garbage_triple(s, p, o), (
        f"({s}, {p}, {o}) is a triple about Nova's own output, not the world; "
        f"this class was 0.3% of facts but 9.6% of all retrieval"
    )


@pytest.mark.parametrize("s,p,o", REAL_KNOWLEDGE)
def test_real_knowledge_survives(s, p, o):
    assert not is_garbage_triple(s, p, o), (
        f"({s}, {p}, {o}) is real knowledge — a garbage rule that eats facts is "
        f"worse than the pollution it removes"
    )


@pytest.mark.parametrize("subject", [
    "July 08, 2026", "August 3, 2026", "Aug. 14, 2026", "2026-08-29", "3 August 2026",
])
def test_bare_date_cannot_be_a_subject(subject):
    assert is_garbage_triple(subject, "has_status", "something"), (
        f"bare date {subject!r} is not an entity and cannot be a fact's subject"
    )
