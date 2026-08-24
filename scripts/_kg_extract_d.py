"""Arm D of the extraction-lever A/B: 27B + CURRENT prompt (the precision-safe
candidate). Completes the 2x2 — if D keeps most of C's (27B+relaxed) recall gain
with FEWER misframed event-verbs, the ship config is 27B + current prompt (the
model's win without the relaxed prompt's predicate-coercion). Dumps every grounded
event-verb fact to /data/_d_eventverbs.md so the misframe rate can be eyeballed
against C's.
"""
import os
os.environ["EMBEDDING_MODEL"] = "default"

import asyncio
import json
import logging
import re

logging.disable(logging.CRITICAL)

from app.config import config as _cfg
from app.core import llm
from app.core.kg import CANONICAL_PREDICATES, is_garbage_triple, normalize_predicate
import app.monitors.deep_research as dr

DIGESTS = "/data/all_digests_live.md"
MAX_TR = 15
NEW_PREDS = {"acquired", "owns", "subsidiary_of", "invested_in", "partnered_with",
             "competes_with", "supplies", "sued", "sanctioned", "launched", "regulates"}

_DIRECTION = (
    "DIRECTION RULES — these predicates are NOT symmetric. The subject and object roles are fixed:\n"
    "  acquired: subject=ACQUIRER, object=ACQUIRED ; invested_in: subject=INVESTOR, object=RECIPIENT ;\n"
    "  sued: subject=PLAINTIFF, object=DEFENDANT ; sanctioned: subject=SANCTIONER, object=TARGET ;\n"
    "  leads/works_at: subject=PERSON, object=ORG ; capital_of: subject=CITY, object=COUNTRY.\n"
    "Before emitting a triple, check the roles match. REJECT tautologies, meta-statements about the\n"
    "source, underscored variable names, and question-label entities. If nothing substantive, return [].\n")
_CURRENT = ("Extract factual (subject, predicate, object) triples from this Q&A.\n"
            "Use ONLY these predicates: {predicates}\nReturn a JSON array. Max {max_triples} triples.\n"
            "Only verifiable facts, not opinions.\nRate each triple's confidence 0.3–0.95.\n\n" + _DIRECTION +
            '\nExample: [{{"subject":"python","predicate":"created_by","object":"guido van rossum","confidence":0.9}}]\n\n'
            "Q: {query}\nA: {answer}")


def _parse(path):
    parts = re.split(r"={80,}\n", open(path, encoding="utf-8").read())
    out = []
    for i in range(1, len(parts) - 1, 2):
        m = re.match(r"##\s+(.+?)\s+[—-]\s+\d+\s+sources", parts[i].strip())
        d = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        if m and d and not d.startswith("(") and len(d) > 200:
            out.append((m.group(1).strip(), d))
    return out


def _grounded(s, o, low):
    st, ot = dr._key_terms(s), dr._key_terms(o)
    return bool(st and ot and any(t in low for t in st) and any(t in low for t in ot))


async def _extract(model, name, digest, preds, sem):
    clean = re.sub(r"^Domain Study:\s*", "", name, flags=re.IGNORECASE).strip() or name
    prompt = _CURRENT.format(predicates=preds, max_triples=MAX_TR, query=clean, answer=digest[:6000])
    async with sem:
        try:
            raw = await llm.invoke_nothink([{"role": "user", "content": prompt}],
                                           json_mode=True, json_prefix="[{", model=model)
        except Exception:
            return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            data = data.get("triples") or []
        if not isinstance(data, list):
            return []
    except Exception:
        return []
    low = digest.lower()
    out = []
    for t in data[:MAX_TR]:
        if not isinstance(t, dict):
            continue
        s, p, o = (str(t.get("subject", "")).strip(), str(t.get("predicate", "")).strip(),
                   str(t.get("object", "")).strip())
        if not (s and p and o) or len(s) > 100 or len(o) > 100 or is_garbage_triple(s, p, o):
            continue
        out.append((s, normalize_predicate(p), o, _grounded(s, o, low)))
    return out


async def main():
    digests = _parse(DIGESTS)
    preds = ", ".join(sorted(CANONICAL_PREDICATES))
    model27 = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None
    print(f"=== arm D: 27B + CURRENT prompt over {len(digests)} digests (27B={model27}) ===", flush=True)
    sem = asyncio.Semaphore(4)
    res = await asyncio.gather(*[_extract(model27, n, d, preds, sem) for n, d in digests])
    facts = [f for lst in res for f in lst]
    grounded = [f for f in facts if f[3]]
    ev = [f for f in grounded if f[1] in NEW_PREDS]
    gr = 100 * len(grounded) / max(len(facts), 1)
    print(f"\nraw {len(facts)} | grounded {len(grounded)} ({gr:.0f}%) | "
          f"grounded/digest {len(grounded)/len(digests):.1f} | event-verb facts {len(ev)}", flush=True)
    print("compare: A 9B+current=1.4/dig(18ev) · C 27B+relaxed=5.7/dig(154ev)", flush=True)
    with open("/data/_d_eventverbs.md", "w", encoding="utf-8") as f:
        f.write(f"# arm D (27B + current) grounded event-verb facts: {len(ev)}\n\n")
        for s, p, o, _ in ev:
            f.write(f"{s}  --{p}-->  {o}\n")
    print("\nALL grounded event-verb facts (read for misframe rate):", flush=True)
    for s, p, o, _ in ev:
        print(f"   {s} --{p}--> {o}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
