"""A/B the 27B-extraction lever for `_extract_kg_triples` (KG fact-rate).

The #2 verification showed the EXTRACTION STEP (not the cap) is the binding
constraint on KG fact-rate. This tests the data-pointed lever on the FULL set of
real monitor digests, mirroring production's exact parse + garbage gate (no KG
writes — safe + fast), across three arms:
    A = 9B  + CURRENT prompt   (production control)
    B = 9B  + RELAXED prompt   (isolates the prompt change)
    C = 27B + RELAXED prompt   (the full lever)
A→B = prompt effect, B→C = model effect, A→C = total.

CRUCIAL metric = GROUNDING RATE (precision): a fact counts as grounded only if its
subject AND object tokens actually appear in the digest. A bigger model must not
"win" by hallucinating more facts — more GROUNDED facts at an equal-or-better
grounding rate is the only real win.

Usage: python /data/_kg_extract_ab.py
"""
import os
os.environ["EMBEDDING_MODEL"] = "default"   # no Ollama embedder needed (no KG writes)

import asyncio
import json
import logging
import re
from collections import Counter

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

_BASE = ("Extract factual (subject, predicate, object) triples from this Q&A.\n"
         "Use ONLY these predicates: {predicates}\n"
         "Return a JSON array. Max {max_triples} triples.\n")
_CURRENT = (_BASE + "Only verifiable facts, not opinions.\n"
            "Rate each triple's confidence 0.3–0.95.\n\n" + _DIRECTION +
            '\nExample: [{{"subject":"python","predicate":"created_by","object":"guido van rossum","confidence":0.9}}]\n\n'
            "Q: {query}\nA: {answer}")
_RELAXED = (_BASE + "Capture verifiable facts AND notable ACTIONS/EVENTS as relationships using the action "
            "predicates — acquired, launched, invested_in, partnered_with, sued, sanctioned, supplies, "
            "competes_with, regulates (e.g. \"Google acquired Wiz\", \"Aave launched V4\", \"US sanctioned "
            "Huawei\"). Extract who-did-what-to-whom, not just static facts. Still exclude vague opinions/"
            "commentary.\nRate each triple's confidence 0.3–0.95.\n\n" + _DIRECTION +
            '\nExample: [{{"subject":"google","predicate":"acquired","object":"wiz","confidence":0.9}}]\n\n'
            "Q: {query}\nA: {answer}")


def _parse(path):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"={80,}\n", text)
    out = []
    for i in range(1, len(parts) - 1, 2):
        m = re.match(r"##\s+(.+?)\s+[—-]\s+\d+\s+sources", parts[i].strip())
        if not m:
            continue
        digest = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        if digest and not digest.startswith("(") and len(digest) > 200:
            out.append((m.group(1).strip(), digest))
    return out


def _grounded(s, o, digest_low):
    st, ot = dr._key_terms(s), dr._key_terms(o)
    return bool(st and ot and any(t in digest_low for t in st) and any(t in digest_low for t in ot))


async def _extract(prompt_tpl, model, name, digest, preds, sem):
    clean = re.sub(r"^Domain Study:\s*", "", name, flags=re.IGNORECASE).strip() or name
    prompt = prompt_tpl.format(predicates=preds, max_triples=MAX_TR, query=clean, answer=digest[:6000])
    async with sem:
        try:
            raw = await llm.invoke_nothink([{"role": "user", "content": prompt}],
                                           json_mode=True, json_prefix="[{", model=model)
        except Exception:
            return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw          # mirror production parse
        if isinstance(data, dict):
            data = data.get("triples") or []
        if not isinstance(data, list):
            return []
    except Exception:
        return []
    digest_low = digest.lower()
    out = []
    for t in data[:MAX_TR]:                                              # mirror production gate
        if not isinstance(t, dict):
            continue
        s, p, o = (str(t.get("subject", "")).strip(), str(t.get("predicate", "")).strip(),
                   str(t.get("object", "")).strip())
        if not (s and p and o) or len(s) > 100 or len(o) > 100 or is_garbage_triple(s, p, o):
            continue
        out.append((s, normalize_predicate(p), o, _grounded(s, o, digest_low)))
    return out


async def _arm(prompt_tpl, model, digests, preds):
    sem = asyncio.Semaphore(4)
    res = await asyncio.gather(*[_extract(prompt_tpl, model, n, d, preds, sem) for n, d in digests])
    return [f for lst in res for f in lst]


def _report(tag, facts, ndig):
    grounded = [f for f in facts if f[3]]
    ev = [f for f in grounded if f[1] in NEW_PREDS]
    rt = sum(1 for f in grounded if f[1] == "related_to")
    preds = len({f[1] for f in grounded})
    gr = 100 * len(grounded) / max(len(facts), 1)
    print(f"{tag:<26}{len(facts):>7}{len(grounded):>10}{gr:>9.0f}%{len(grounded)/ndig:>11.1f}"
          f"{len(ev):>9}{preds:>8}{rt:>6}", flush=True)
    return grounded, ev


async def main():
    digests = _parse(DIGESTS)
    preds = ", ".join(sorted(CANONICAL_PREDICATES))
    model27 = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None
    print(f"=== 27B-extraction lever A/B over {len(digests)} real digests "
          f"(27B={model27}) ===", flush=True)
    print("running A (9B + current)…", flush=True)
    A = await _arm(_CURRENT, None, digests, preds)
    print("running B (9B + relaxed)…", flush=True)
    B = await _arm(_RELAXED, None, digests, preds)
    print("running C (27B + relaxed)…", flush=True)
    C = await _arm(_RELAXED, model27, digests, preds)

    nd = len(digests)
    print("\n" + "=" * 80, flush=True)
    print(f"{'arm':<26}{'raw':>7}{'grounded':>10}{'g-rate':>10}{'grnd/dig':>11}{'event-v':>9}{'preds':>8}{'rel_to':>6}", flush=True)
    gA, eA = _report("A 9B + current", A, nd)
    gB, eB = _report("B 9B + relaxed", B, nd)
    gC, eC = _report("C 27B + relaxed", C, nd)
    print("=" * 80, flush=True)

    def climb(g):
        return (len(g) / max(len(gA), 1) - 1) * 100
    print(f"\nGROUNDED fact-rate vs control A:  B(prompt) {climb(gB):+.0f}%   C(full lever) {climb(gC):+.0f}%", flush=True)
    print(f"event-verb facts (grounded):      A {len(eA)}   B {len(eB)}   C {len(eC)}", flush=True)
    print("\nexamples of grounded event-verb facts ONLY the 27B (C) surfaced:", flush=True)
    seen = {(s, p, o) for s, p, o, _ in gB}
    only_c = [(s, p, o) for s, p, o, g in C if g and p in NEW_PREDS and (s, p, o) not in seen]
    for s, p, o in only_c[:12]:
        print(f"   {s} --{p}--> {o}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
