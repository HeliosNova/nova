"""#2 verification — KG fact-RATE climb from the monitor extraction change.

#2 changed ONLY the extraction step (heartbeat_loop:434 → `_extract_kg_triples`),
not the slow 27B digest generation. So we measure it precisely + fast by running
the REAL production extraction over the FULL set of 44 real monitor digests,
old-params vs new-params, into two SCRATCH KGs:
    OLD: max_answer_chars=1000, max_triples=5    (the prior hard caps)
    NEW: max_answer_chars=6000, max_triples=15   (the fix)
The OLD/NEW delta isolates #2a (the cap) — both arms use the now-global 43-predicate
vocab. #2b (predicate preservation) is reported separately = facts in the NEW arm
carrying one of the 11 added news verbs (acquired/sued/sanctioned/…), each of which
the pre-#2b 32-predicate code would have FLATTENED to `related_to` (verb discarded).

SAFE: isolates ChromaDB to a temp path + uses the MiniLM embedder (no Ollama swap),
writes only to throwaway SQLite KGs — production /data KG + chromadb are untouched.

Usage: python /data/_kg_rate_ab.py
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="kgab_")
os.environ["CHROMADB_PATH"] = os.path.join(_TMP, "chroma")   # isolate vectors from prod
os.environ["EMBEDDING_MODEL"] = "default"                    # MiniLM (CPU) — no Ollama embedder swap

import asyncio
import logging
import re
import shutil
from collections import Counter

logging.disable(logging.CRITICAL)

from app.database import SafeDB
from app.core.kg import KnowledgeGraph
from app.core.brain import _extract_kg_triples

DIGESTS = "/data/all_digests_live.md"
NEW_PREDS = {"acquired", "owns", "subsidiary_of", "invested_in", "partnered_with",
             "competes_with", "supplies", "sued", "sanctioned", "launched", "regulates"}


def _parse(path):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"={80,}\n", text)
    out = []
    for i in range(1, len(parts) - 1, 2):
        m = re.match(r"##\s+(.+?)\s+[—-]\s+\d+\s+sources", parts[i].strip())
        if not m:
            continue
        name = m.group(1).strip()
        digest = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        if digest and not digest.startswith("(") and len(digest) > 200:
            out.append((name, digest))
    return out


async def _run(digests, max_chars, max_tr, dbpath):
    db = SafeDB(dbpath)
    db.init_schema()
    kg = KnowledgeGraph(db)
    for name, digest in digests:
        try:
            await _extract_kg_triples(kg, name, digest, source_name=name,
                                      max_answer_chars=max_chars, max_triples=max_tr)
        except Exception:
            pass
    facts = kg.get_all_facts()
    db.close()
    return facts


async def main():
    digests = _parse(DIGESTS)
    print(f"=== #2 KG fact-rate A/B over {len(digests)} real monitor digests ===", flush=True)
    if len(digests) < 5:
        print("too few digests parsed — check file format.", flush=True)
        return
    print("running OLD extraction (1000 char / 5-cap)…", flush=True)
    old = await _run(digests, 1000, 5, os.path.join(_TMP, "old.db"))
    print("running NEW extraction (6000 char / 15-cap)…", flush=True)
    new = await _run(digests, 6000, 15, os.path.join(_TMP, "new.db"))

    n_old, n_new = len(old), len(new)
    print("\n" + "=" * 56, flush=True)
    print(f"{'arm':<30}{'facts':>9}{'per digest':>14}", flush=True)
    print(f"{'OLD (1000 char / 5 cap)':<30}{n_old:>9}{n_old / len(digests):>14.1f}", flush=True)
    print(f"{'NEW (6000 char / 15 cap)':<30}{n_new:>9}{n_new / len(digests):>14.1f}", flush=True)
    print(f"\nFACT-RATE CLIMB (#2a, same digests): {(n_new / max(n_old, 1) - 1) * 100:+.0f}%  "
          f"({n_old} → {n_new} facts)", flush=True)

    np_facts = [f for f in new if getattr(f, "predicate", "") in NEW_PREDS]
    by_pred = Counter(getattr(f, "predicate", "") for f in np_facts)
    rt_new = sum(1 for f in new if getattr(f, "predicate", "") == "related_to")
    rt_old = sum(1 for f in old if getattr(f, "predicate", "") == "related_to")
    print(f"\n#2b — NEW-verb facts (pre-#2b code → related_to, verb LOST): {len(np_facts)}", flush=True)
    print(f"   by predicate: {dict(by_pred.most_common())}", flush=True)
    print("   examples:", flush=True)
    for f in np_facts[:12]:
        print(f"     {f.subject}  --{f.predicate}-->  {f.object}", flush=True)
    print(f"\nrelated_to share — OLD {rt_old}/{n_old}={100 * rt_old / max(n_old,1):.0f}%  "
          f"NEW {rt_new}/{n_new}={100 * rt_new / max(n_new,1):.0f}%", flush=True)
    print("=" * 56, flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
