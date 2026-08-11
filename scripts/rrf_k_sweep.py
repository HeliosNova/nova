"""Offline RRF-k sweep for KG retrieval (task #63).

2026 measurement (arXiv 2604.01733): RRF k=10 beat k=60 by ~2pp Recall@5 on a
document corpus. This sweeps k over OUR live KG (realistic distractor mass)
using the kg-retrieval suite's seeded fictional facts + paraphrase queries,
measuring hit@8 of the seeded fact. Seeds are cleaned up afterward.

Run: docker exec nova-app python scripts/rrf_k_sweep.py
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/app")

CASES = [
    (("Aetherion Guild", "led_by", "Mara Quill"), "Who leads the Aetherion Guild?"),
    (("Vortex Institute", "located_in", "Talloran City"), "Where is the Vortex Institute located?"),
    (("Vorenza", "based_in", "Brindlemark"), "Where is Vorenza based?"),
    (("Kestrel", "created_by", "Dario Venn"), "Who is the person behind Kestrel?"),
    (("Halcyon", "developed_by", "Iris Quon"), "Which team built Halcyon?"),
    (("Nimbus", "led_by", "Petra Kovacs"), "Who runs Nimbus?"),
]
KS = (10, 30, 60)


async def main() -> None:
    from app.database import get_db
    from app.core.kg import KnowledgeGraph

    kg = KnowledgeGraph(get_db())

    for (s, p, o), _q in CASES:
        await kg.delete_fact(s, p, o)  # defensive pre-clean
        await kg.add_fact(s, p, o, confidence=0.95, source="eval", provenance="rrf-sweep")

    orig_fuse = KnowledgeGraph._rrf_fuse
    results: dict[int, int] = {}
    try:
        for k in KS:
            def fuse_with_k(keyword_ids, vector_ids, ppr_ids=None, k=k):
                return orig_fuse(keyword_ids, vector_ids, ppr_ids, k=k)
            KnowledgeGraph._rrf_fuse = staticmethod(fuse_with_k)
            hits = 0
            for (s, p, o), q in CASES:
                facts = await asyncio.to_thread(kg.get_relevant_facts, q, 8)
                if any(f.subject.lower() == s.lower() and f.object.lower() == o.lower()
                       for f in facts):
                    hits += 1
            results[k] = hits
            print(f"k={k}: hit@8 = {hits}/{len(CASES)}")
    finally:
        KnowledgeGraph._rrf_fuse = orig_fuse
        for (s, p, o), _q in CASES:
            await kg.delete_fact(s, p, o)
    best = max(results, key=results.get)
    print(f"BEST: k={best} ({results[best]}/{len(CASES)}) — "
          f"change kg.py _rrf_fuse default only if a k strictly beats 60.")


if __name__ == "__main__":
    asyncio.run(main())
