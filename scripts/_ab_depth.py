"""A/B PROOF of fix #1 (deep-analysis body-feed) depth gain.

Holds EVERYTHING constant — same captured findings+bodies, same clustering (done
ONCE, same stories to both arms), same 27B model, same prompt+temp — and varies
ONLY the per-story evidence:
    A (control / OLD): the 240-token finding STUBS         (bodies=None path)
    B (treatment / NEW): the FULL article bodies           (top-6 by authority, capped)

Depth is measured OBJECTIVELY = count of source-GROUNDED specifics (magnitude
figures + distinctive named entities that actually appear in the read article
bodies). No LLM judge: the only strong local model is the 27B itself (self-
preference bias) and gemma3:4b is too weak for depth — grounded-specific density
is deterministic and unbiasable. A specific that traces to a body is real detail
the analyst surfaced; B should surface far more of it than A, since A only ever
saw the lossy stubs.

Usage: python /data/_ab_depth.py "Cybersecurity" [--capture]
"""
import asyncio
import json
import os
import sys
import logging

logging.disable(logging.CRITICAL)

from app.core.brain import Services, set_services
from app.core.kg import KnowledgeGraph
from app.database import get_db
from app.config import config as _cfg
from app.core import llm
import app.monitors.deep_research as dr

FIX = "/data/_ab_fixture.json"
_TODAY = "June 29, 2026"

_ANALYST = (
    "Today is {today}. Analyze this {label} story IN DEPTH from its sources — be a sharp "
    "intelligence analyst, not a summarizer:\n"
    "- the key facts, EXACT numbers, named players, and what is genuinely NEW;\n"
    "- WHY it matters and the second-order implications;\n"
    "- any tension, disagreement, or uncertainty across the sources;\n"
    "- what to watch next.\n"
    "STORY: {title}\n\nSOURCES:\n{ev}")


async def _capture(label):
    subjects = await dr._focus_subjects(label, feed_key=label, n=5)
    articles = await dr._gather_overview(subjects, label, read_target=18, browser_budget=10)
    findings = await dr._findings(articles, label) if articles else []
    data = {"label": label,
            "articles": [[t, u, b] for (t, u, b) in articles],
            "findings": [[t, u, f] for (t, u, f) in findings]}
    json.dump(data, open(FIX, "w"))
    return data


async def _cluster_once(findings, label, model):
    numbered = "\n".join(f"[{i}] [{t}] ({dr._host(u)})\n{(f or '')[:600]}"
                         for i, (t, u, f) in enumerate(findings))
    raw = await llm.invoke_nothink([{"role": "user", "content":
        f"Group these {label} findings into the distinct ongoing STORIES they cover. Merge findings "
        "about the SAME event into one story; aim for 3-7 stories. Return JSON only: "
        '[{"title": "...", "items": [0, 3, 7]}].\n\n' + numbered}],
        json_mode=True, json_prefix="[{", max_tokens=600, temperature=0.1, model=model, num_ctx=8192)
    groups = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(groups, dict):
        groups = groups.get("stories") or groups.get("items") or []
    stories = []
    for g in (groups or [])[:8]:
        if not isinstance(g, dict):
            continue
        idxs = [i for i in (g.get("items") or []) if isinstance(i, int) and 0 <= i < len(findings)]
        title = str(g.get("title", "")).strip()
        if title and idxs:
            stories.append((title, idxs))
    if not stories:
        stories = [(label, list(range(len(findings))))]
    return stories


async def _analyze(stories, findings, body_map, label, model, mode):
    async def _an(title, idxs):
        if mode == "A":   # OLD: all finding stubs
            ev = "\n\n".join(f"[{findings[i][0]}] ({dr._host(findings[i][1])})\n{findings[i][2]}"
                             for i in idxs)
        else:             # NEW: top-6 full bodies by authority, capped
            ranked = sorted(idxs, key=lambda i: dr._source_quality(findings[i][1]), reverse=True)
            parts = []
            for i in ranked[:6]:
                t, u, f = findings[i]
                txt = body_map.get(u) or f or ""
                parts.append(f"[{t}] ({dr._host(u)})\n{txt[:3500]}")
            ev = "\n\n".join(parts)[:20000]
        a = await llm.invoke_nothink(
            [{"role": "user", "content": _ANALYST.format(today=_TODAY, label=label, title=title, ev=ev)}],
            max_tokens=520, temperature=0.3, model=model, num_ctx=8192)
        return (title, (a or "").strip())
    return await asyncio.gather(*[_an(t, ix) for t, ix in stories])


def _grounded(analyses, corpus, corpus_nc):
    """Distinct source-GROUNDED specifics across an arm's analyses."""
    text = "\n".join(a for _, a in analyses)
    g_figs, all_figs = set(), set()
    for m in dr._MAGNITUDE_RE.finditer(text):
        num, unit, cur = m.group("num"), m.group("unit"), m.group("cur")
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        u = (unit or "").lower()
        mag = ("," in num) or (u in dr._MULT) or (cur is not None) \
            or (u in ("percent", "per cent", "%")) or val >= 1000
        if not mag:
            continue
        raw = m.group(0).strip()
        all_figs.add(raw)
        if any(v in corpus or v in corpus_nc for v in dr._num_variants(num, unit, cur)):
            g_figs.add(raw)
    g_named = set()
    for rx in (dr._ACRONYM_RE, dr._COMPOUND_RE, dr._PROPER_PHRASE_RE):
        for m in rx.finditer(text):
            tok = m.group(1)
            if tok.lower() in corpus:
                g_named.add(tok.lower())
    return {"chars": len(text), "all_figs": all_figs, "g_figs": g_figs, "g_named": g_named,
            "grounded": g_figs | {("#" + n) for n in g_named}}


async def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "Cybersecurity"
    force = "--capture" in sys.argv
    set_services(Services(kg=KnowledgeGraph(get_db())))
    model = (getattr(_cfg, "MONITOR_SYNTHESIS_MODEL", "") or "").strip() or None
    print(f"=== A/B depth proof: {label} | model={model or 'default-9B'} ===", flush=True)

    if force or not os.path.exists(FIX):
        print("capturing fresh sources (gather+findings)…", flush=True)
        data = await _capture(label)
    else:
        data = json.load(open(FIX))
        if data.get("label") != label:
            data = await _capture(label)
    articles = [(t, u, b) for t, u, b in data["articles"]]
    findings = [(t, u, f) for t, u, f in data["findings"]]
    body_map = {u: b for (t, u, b) in articles if b}
    print(f"fixture: {len(articles)} articles, {len(findings)} findings, "
          f"{len(body_map)} bodies ({sum(len(b) for b in body_map.values())//1000}k chars)", flush=True)
    if len(findings) < 3:
        print("too few findings to A/B — re-run with --capture once search recovers.", flush=True)
        return

    corpus = " ".join(b for b in body_map.values()).lower()
    corpus_nc = corpus.replace(",", "")

    stories = await _cluster_once(findings, label, model)
    print(f"clustered ONCE → {len(stories)} stories (identical for both arms): "
          f"{[t for t, _ in stories]}", flush=True)

    A = await _analyze(stories, findings, body_map, label, model, "A")   # OLD stubs
    B = await _analyze(stories, findings, body_map, label, model, "B")   # NEW bodies

    ma = _grounded(A, corpus, corpus_nc)
    mb = _grounded(B, corpus, corpus_nc)

    open("/data/_ab_A_stubs.md", "w").write("\n\n".join(f"### {t}\n{a}" for t, a in A))
    open("/data/_ab_B_bodies.md", "w").write("\n\n".join(f"### {t}\n{a}" for t, a in B))

    def dens(m):
        return 1000 * len(m["grounded"]) / max(m["chars"], 1)
    print("\n================ DEPTH METRICS (source-grounded specifics) ================", flush=True)
    print(f"{'metric':<34}{'A: stubs (old)':>16}{'B: bodies (new)':>18}", flush=True)
    for key, lbl in [("g_figs", "grounded magnitude figures"),
                     ("g_named", "grounded named entities")]:
        print(f"{lbl:<34}{len(ma[key]):>16}{len(mb[key]):>18}", flush=True)
    print(f"{'grounded specifics (total)':<34}{len(ma['grounded']):>16}{len(mb['grounded']):>18}", flush=True)
    print(f"{'analysis chars':<34}{ma['chars']:>16}{mb['chars']:>18}", flush=True)
    print(f"{'grounded per 1k chars (density)':<34}{dens(ma):>16.2f}{dens(mb):>18.2f}", flush=True)

    only_b = mb["grounded"] - ma["grounded"]
    only_a = ma["grounded"] - mb["grounded"]
    gain = (len(mb["grounded"]) / max(len(ma["grounded"]), 1) - 1) * 100
    print(f"\nNEW grounded specifics B surfaced that A entirely MISSED: {len(only_b)}", flush=True)
    print(f"grounded specifics A had but B lost:                      {len(only_a)}", flush=True)
    print(f"DEPTH GAIN (grounded specifics, B vs A):                  {gain:+.0f}%", flush=True)
    sample_new = [s for s in only_b if not s.startswith('#')][:12]
    sample_new += [s[1:] for s in only_b if s.startswith('#')][:12]
    print(f"examples of detail ONLY B surfaced: {sorted(set(sample_new))[:16]}", flush=True)

    # side-by-side of the single story with the biggest B-vs-A grounded gap
    gaps = []
    for (t, a), (_, b) in zip(A, B):
        ga = _grounded([(t, a)], corpus, corpus_nc)
        gb = _grounded([(t, b)], corpus, corpus_nc)
        gaps.append((len(gb["grounded"]) - len(ga["grounded"]), t, a, b))
    gaps.sort(reverse=True)
    _, t, a, b = gaps[0]
    print(f"\n================ SIDE-BY-SIDE (biggest-gap story: {t}) ================", flush=True)
    print(f"\n----- A (STUBS, old) -----\n{a}", flush=True)
    print(f"\n----- B (BODIES, new) -----\n{b}", flush=True)
    print("\n(full arms saved: /data/_ab_A_stubs.md, /data/_ab_B_bodies.md)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
