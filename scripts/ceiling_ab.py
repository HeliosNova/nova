"""Same-evidence A/B harness for the quality-ceiling program (2026-07-10).

Kills the gather-variance confound that muddied the 9B-vs-27B test: capture ONE
gather per frozen topic, then replay the IDENTICAL findings+articles through the
EXACT production synthesis (`_synthesize_from_evidence`) with any (model, config),
and grade deterministic FACT + coverage + pinned cross-family RACE (gemma4:e4b).

Run inside nova-app (PYTHONPATH=/app):
  # 1. build the frozen benchmark once (GPU+network heavy, ~40 min for 8 topics):
  python /tmp/ceiling_ab.py --capture
  # 2. baseline (syn_model=None -> config MONITOR_SYNTHESIS_MODEL = 27B):
  python /tmp/ceiling_ab.py --replay --tag 27B
  # 3. a candidate model / config lever:
  python /tmp/ceiling_ab.py --replay --model gemma4:26b-a4b-it-qat --tag moe
  python /tmp/ceiling_ab.py --replay --set ENABLE_MINICHECK=true --tag minicheck
  # subset for a quick smoke: --only ai_and_ml,geopolitics,finance
"""
import argparse, asyncio, json, os, re, statistics

CEIL = "/data/ceiling"
# Frozen benchmark: production MONITOR NAMES. label + feed resolution mirror
# production exactly (via _profile_for + feed_key=monitor_name, n_stories=7).
# Diverse: abstract (geopolitics), numeric (finance/economics), technical
# (AI/cyber/china), scientific (biotech).
TOPICS = [
    "Domain Study: AI and ML", "Domain Study: Geopolitics", "Domain Study: Finance",
    "Domain Study: Cybersecurity", "Domain Study: China Tech and Economy",
    "Domain Study: Economics and Markets", "Domain Study: Energy and Climate",
    "Domain Study: Biotech and Genetics",
    # n=16 expansion (2026-09-01): the 08-13 dossier-priming A/B won every
    # mean at n=8 but p≈0.14 — doubled for proof-grade. Additions keep the
    # diversity axes: +2 regions, +2 hard-tech, medical, crypto-numeric,
    # space-science, defense.
    "Domain Study: Space and Astronomy", "Domain Study: Health and Medicine",
    "Domain Study: Semiconductors", "Domain Study: Middle East",
    "Domain Study: Crypto and Web3", "Domain Study: Supply Chain and Trade",
    "Domain Study: Latin America", "Domain Study: Defense and Military Tech",
]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


async def capture(only=None):
    from app.monitors.deep_research import _gather_evidence
    from app.monitors.domain_study_runner import _profile_for
    os.makedirs(CEIL, exist_ok=True)
    for monitor_name in TOPICS:
        if only and not any(o in slug(monitor_name) for o in only):
            continue
        _emoji, label, _kw = _profile_for(monitor_name)
        try:
            subjects, findings, articles = await _gather_evidence(label, 7, monitor_name)
        except Exception as e:
            print(f"[capture] {label}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        json.dump({"label": label, "monitor": monitor_name, "subjects": subjects,
                   "findings": findings, "articles": articles},
                  open(f"{CEIL}/{slug(monitor_name)}.json", "w"))
        print(f"[capture] {label}: {len(findings)} findings, {len(articles)} articles saved", flush=True)


def _apply(sets):
    from app.config import config
    for kv in sets:
        k, v = kv.split("=", 1)
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        elif re.fullmatch(r"-?\d+", v):
            v = int(v)
        try:
            object.__setattr__(config, k, v)
            print(f"[config] {k} = {v!r}", flush=True)
        except Exception as e:
            print(f"[config] FAILED {k}: {e}", flush=True)


async def replay(model, tag, sets, only, judge):
    from app.monitors.deep_research import _synthesize_from_evidence, _NOW
    from app.monitors.report_grader import grade_report, coverage_score
    if sets:
        _apply(sets)
    today = _NOW().strftime("%B %d, %Y")
    rows = []
    for fn in sorted(os.listdir(CEIL)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(f"{CEIL}/{fn}"))
        if only and not any(o in slug(d.get("monitor", d["label"])) for o in only):
            continue
        label = d["label"]
        findings = [tuple(x) for x in d["findings"]]
        articles = [tuple(x) for x in d["articles"]]
        try:
            digest = await _synthesize_from_evidence(label, findings, articles, today,
                                                     kg=None, syn_model=model,
                                                     dossier_key=d.get("monitor"))
        except Exception as e:
            print(f"[{tag}] {label}: SYNTH FAILED {type(e).__name__}: {e}", flush=True)
            continue
        g = await grade_report(digest, label, model=judge)
        cov = coverage_score(digest, findings)
        rows.append((label, g, cov, len(digest)))
        print(f"[{tag}] {label[:24]:24} overall={g['overall']:.3f} race={g['race_avg']:.2f} "
              f"support={g['fact']['support']:.2f} fab={g['fact']['fabricated_rate']:.2f} "
              f"core_cov={cov['core_coverage']:.2f} chars={len(digest)}", flush=True)
    if rows:
        m = lambda f: statistics.mean(f(r) for r in rows)
        summary = {
            "tag": tag, "model": model or "config-default", "judge": judge, "n": len(rows),
            "overall": round(m(lambda r: r[1]["overall"]), 3),
            "race": round(m(lambda r: r[1]["race_avg"]), 3),
            "support": round(m(lambda r: r[1]["fact"]["support"]), 3),
            "fabricated": round(m(lambda r: r[1]["fact"]["fabricated_rate"]), 3),
            "core_cov": round(m(lambda r: r[2]["core_coverage"]), 3),
            # Per-topic rows (2026-09-01): the 08-13 priming A/B persisted only
            # means, so the paired per-topic comparison the p-value needs had
            # to be scraped from stdout. A/B evidence must outlive the session.
            "rows": [{"label": r[0], "overall": r[1]["overall"],
                      "race": r[1]["race_avg"], "support": r[1]["fact"]["support"],
                      "fabricated": r[1]["fact"]["fabricated_rate"],
                      "core_cov": r[2]["core_coverage"], "chars": r[3]}
                     for r in rows],
        }
        print(f"\n[{tag}] MEANS n={summary['n']}: overall={summary['overall']} race={summary['race']} "
              f"support={summary['support']} fab={summary['fabricated']} core_cov={summary['core_cov']}", flush=True)
        os.makedirs(f"{CEIL}/results", exist_ok=True)
        json.dump(summary, open(f"{CEIL}/results/{tag}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--judge", default="gemma4:e4b")
    ap.add_argument("--only", default="")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x] or None
    if a.capture:
        asyncio.run(capture(only))
    if a.replay:
        # Quiet window (2026-09-02): the replay owns the GPU for its run —
        # the heartbeat skips its LLM lane instead of the operator hand-
        # disabling 39 monitors and a guardian re-enabling them.
        from app.database import get_db
        from app.monitors.quiet import clear_quiet, set_quiet
        set_quiet(get_db(), 6, f"ceiling A/B replay {a.tag}")
        try:
            asyncio.run(replay(a.model, a.tag, a.set, only, a.judge))
        finally:
            clear_quiet(get_db())
