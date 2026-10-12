"""What kind of claim does the primary entailment gate reject? (2026-10-12)

Run inside nova-app: python scripts/gate_miss_classes.py  (reads /data/entail).

First reading, 3,557 primary pairs over 24 h: 80% unsupported at the primary
pass (54% after the rescues). Of the unsupported: 34% low overlap (the claim's
content words are not in the cited source's best windows), 25% analytical
framing (de-cited, not dropped), 15% carry a number the source does not,
7% high-overlap candidates for model false negatives, 3% borderline. Evidence
quality does NOT separate the classes: hub/teaser pages 14% vs 15%, chrome in
the window 22% vs 24%, prose share 0.96 vs 0.97 for rejected vs accepted. The
median rejection probability is 0.03 — decisive, not borderline. The gate is
mostly right; the lever is the synthesis attributing claims to outlets whose
text does not carry them.

Item 4 of the 2026-10-12 program.

Reads the gate record (site == 'gate', the primary pass) and classifies the
unsupported pairs so the 54% drop rate stops being one number:
  no_evidence     doc window under 300 chars (nothing to entail against)
  low_overlap     < 35% of the claim's content words (>= 5 chars) appear in the doc
                  -> the evidence picker chose the wrong window/article
  number_missing  a number in the claim is absent from the doc (figure mismatch)
  analytical      claim opens with analytic framing the v4 gate de-cites anyway
  borderline      prob in [0.35, 0.5): one re-phrase from passing
  high_overlap    >= 70% of content words present AND no number missing AND
                  prob < 0.35 -> candidate model false negative (the costly class)
Prints a summary JSON and 8 samples from the high_overlap class.
"""
import glob, json, re, sys, random
from collections import Counter

WORD = re.compile(r"[a-z][a-z\-']{4,}")
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
ANALYTIC = re.compile(r"(?i)^(this|these|the (most|key|real|broader)|such|overall|in (short|sum)|taken together|crucially|notably|importantly|the (implication|significance|takeaway|upshot))")
STOP = {"which", "their", "there", "these", "those", "about", "would", "could", "should", "after", "before", "while", "where", "being", "other", "between", "through", "during", "against", "under", "since", "within", "across"}

recs = []
for f in sorted(glob.glob("/data/entail/gate-*.jsonl")):
    for line in open(f, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("site") == "gate":
            recs.append(r)
total = len(recs)
uns = [r for r in recs if not r.get("supported")]
cls = Counter()
hi_samples = []
for r in uns:
    claim = (r.get("claim") or ""); doc = (r.get("doc") or ""); low = doc.lower()
    words = [w for w in WORD.findall(claim.lower()) if w not in STOP]
    overlap = sum(1 for w in set(words) if w in low) / max(1, len(set(words)))
    nums = NUM.findall(claim)
    num_missing = any(n.replace(",", "") not in low.replace(",", "") for n in nums) if nums else False
    prob = r.get("prob") if isinstance(r.get("prob"), (int, float)) else -1
    tags = []
    if len(doc) < 300: tags.append("no_evidence")
    if overlap < 0.35: tags.append("low_overlap")
    if num_missing: tags.append("number_missing")
    if ANALYTIC.match(claim.strip()): tags.append("analytical")
    if 0.35 <= prob < 0.5: tags.append("borderline")
    if overlap >= 0.7 and not num_missing and prob < 0.35 and len(doc) >= 300:
        tags.append("high_overlap_false_negative?")
        hi_samples.append((round(prob, 3), claim[:160], doc[:220].replace("\n", " ")))
    if not tags: tags.append("other")
    for t in tags: cls[t] += 1
probs = sorted(r.get("prob", 0) for r in uns if isinstance(r.get("prob"), (int, float)))
summary = {
    "gate_pairs": total, "unsupported": len(uns), "drop_share": round(len(uns) / total, 3) if total else None,
    "narrow_share_of_unsupported": round(sum(1 for r in uns if r.get("narrow")) / max(1, len(uns)), 3),
    "classes (a pair can carry several)": {k: f"{v} ({v / max(1, len(uns)):.0%})" for k, v in cls.most_common()},
    "unsupported_prob_median": round(probs[len(probs) // 2], 3) if probs else None,
    "unsupported_prob_p90": round(probs[int(len(probs) * 0.9)], 3) if probs else None,
}
print(json.dumps(summary, indent=1))
random.seed(3)
print("\n== high-overlap misses (candidate false negatives) ==")
for p, c, d in random.sample(hi_samples, min(8, len(hi_samples))):
    print(f"p={p}\n  CLAIM: {c}\n  DOC:   {d}\n")
