#!/usr/bin/env python3
"""Embedder A/B on LIVE KG facts: bge-m3 (production) vs qwen3-embedding:0.6b.

Model program item 3 (2026-08-24). Reproduces the 2026-06-09 bake-off design —
paraphrase->fact retrieval, entity kept / relation paraphrased (the case
keyword search misses) — but on Nova's REAL current KG instead of a synthetic
set, and adds the config the June run likely missed: qwen3-embedding is an
ASYMMETRIC model whose queries want an instruction prefix. Running it
symmetric under-measures it (the "half-wired embedder" mistake class).

Docs are embedded EXACTLY as production indexes them:
    f"{subject} {predicate.replace('_', ' ')} {object}"

Run (inside the app container, which reaches the embed instance):
    docker exec -i nova-app python - < scripts/embedder_ab.py
"""
from __future__ import annotations

import json
import random
import sqlite3
import time
import urllib.request

import numpy as np

EMBED_URL = "http://nova-embed:11434/api/embed"
DB = "/data/nova.db"
SEED = 20260824
N_FILLERS = 400
PER_PREDICATE = 5

# Relation-paraphrase templates: {s}=subject, {o}=object. Entity kept,
# relation reworded to avoid the predicate's own tokens.
TEMPLATES = {
    "lives_in": "what city does {s} reside in?",
    "works_at": "who employs {s}?",
    "employed_by": "which organization has {s} on its payroll?",
    "leads": "who is in charge of {o}?",
    "capital_of": "which city is the seat of government of {o}?",
    "located_in": "where can {s} be found?",
    "created_by": "who made {s}?",
    "developed_by": "who built {s}?",
    "written_by": "who authored {s}?",
    "invented_by": "who came up with {s}?",
    "founded_in": "what year was {s} established?",
    "born_in": "what is {s}'s birthplace?",
    "acquired": "which company did {s} buy?",
    "owns": "what property does {s} hold?",
    "invested_in": "where did {s} put its money?",
    "partnered_with": "who teamed up with {s}?",
    "competes_with": "who are {s}'s rivals?",
    "launched": "what did {s} unveil?",
    "price_of": "how much does {s} cost?",
    "version_of": "which release is {s} at?",
    "regulates": "what does {s} oversee?",
    "produces": "what does {s} make?",
    "supplies": "who gets components from {s}?",
    "sued": "who did {s} take to court?",
    "is_a": "what type of thing is {s}?",
    "known_for": "what makes {s} notable?",
    "member_of": "which group counts {s} among its members?",
    "married_to": "who is {s}'s spouse?",
    "succeeded_by": "who came after {s}?",
    "successor_of": "who did {s} replace?",
    "subsidiary_of": "which parent company controls {s}?",
    "caused_by": "what brought about {s}?",
    "sanctioned": "who did {s} penalize?",
    "part_of": "what larger whole includes {s}?",
    "contains": "what is inside {s}?",
    "used_for": "what purpose does {s} serve?",
    "belongs_to": "who is the owner of {s}?",
    "spoken_in": "where do people speak {s}?",
    "studied_at": "where did {s} go to school?",
    "currency_of": "what money is used in {o}?",
}

CONFIGS = [
    # (label, model, query_prefix, doc_prefix)
    ("bge-m3 (prod, symmetric)", "bge-m3", "", ""),
    ("qwen3-emb symmetric", "qwen3-embedding:0.6b", "", ""),
    ("qwen3-emb instructed", "qwen3-embedding:0.6b",
     "Instruct: Given a question, retrieve the stored fact that answers it\nQuery: ", ""),
    # harrier-oss-v1-0.6b (2026-09-01, owner-approved pull): Microsoft's
    # multilingual decoder-embedding family (last-token pooling, 1024-dim —
    # same dim as bge-m3), the one research-surfaced candidate that survived
    # the 2026-08-30 measurement gate. Asymmetric family → also test the
    # instructed-query config so it isn't under-measured (the "half-wired
    # embedder" mistake class).
    ("harrier-0.6b symmetric", "hf.co/mradermacher/harrier-oss-v1-0.6b-GGUF:Q8_0", "", ""),
    ("harrier-0.6b instructed", "hf.co/mradermacher/harrier-oss-v1-0.6b-GGUF:Q8_0",
     "Instruct: Given a question, retrieve the stored fact that answers it\nQuery: ", ""),
]


def render(s: str, p: str, o: str) -> str:
    return f"{s} {p.replace('_', ' ')} {o}"


def embed(model: str, texts: list[str], prefix: str = "") -> np.ndarray:
    out = []
    payload_texts = [prefix + t for t in texts] if prefix else texts
    for i in range(0, len(payload_texts), 64):
        chunk = payload_texts[i:i + 64]
        body = json.dumps({"model": model, "input": chunk,
                           "options": {"num_gpu": 0}}).encode()
        req = urllib.request.Request(EMBED_URL, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode())
        embs = data.get("embeddings")
        if not embs or len(embs) != len(chunk):
            raise RuntimeError(f"embed returned {len(embs) if embs else 0}/{len(chunk)}")
        out.extend(embs)
    m = np.asarray(out, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def main():
    rng = random.Random(SEED)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, subject, predicate, object FROM kg_facts "
        "WHERE superseded_at IS NULL AND valid_to IS NULL AND quarantined = 0 "
        "AND length(subject) BETWEEN 3 AND 60 AND length(object) BETWEEN 2 AND 80"
    ).fetchall()
    con.close()
    print(f"live facts eligible: {len(rows)}")

    by_pred: dict[str, list] = {}
    for r in rows:
        by_pred.setdefault(r["predicate"], []).append(r)

    # Targets: up to PER_PREDICATE per templated predicate, unique (s,p) pairs.
    targets = []
    seen_sp = set()
    for pred, tmpl in TEMPLATES.items():
        pool = by_pred.get(pred, [])
        rng.shuffle(pool)
        picked = 0
        for r in pool:
            sp = (r["subject"].lower(), pred)
            if sp in seen_sp or picked >= PER_PREDICATE:
                continue
            seen_sp.add(sp)
            targets.append(r)
            picked += 1
    print(f"targets: {len(targets)} across {len({t['predicate'] for t in targets})} predicates")

    target_ids = {t["id"] for t in targets}
    # Fillers: random other live facts, EXCLUDING any that share a target's
    # (subject, predicate) — those would be equally-valid answers.
    fillers = [r for r in rows if r["id"] not in target_ids
               and (r["subject"].lower(), r["predicate"]) not in seen_sp]
    rng.shuffle(fillers)
    fillers = fillers[:N_FILLERS]

    corpus = targets + fillers
    rng.shuffle(corpus)
    docs = [render(r["subject"], r["predicate"], r["object"]) for r in corpus]
    pos_of = {r["id"]: i for i, r in enumerate(corpus)}

    queries, gold = [], []
    for t in targets:
        q = TEMPLATES[t["predicate"]].format(s=t["subject"], o=t["object"])
        queries.append(q)
        gold.append(pos_of[t["id"]])
    print(f"corpus: {len(docs)} docs, {len(queries)} queries\n")

    header = f"{'config':<28} {'r@1':>5} {'r@3':>5} {'r@5':>5} {'r@10':>5} {'MRR':>6} {'embed_s':>8}"
    print(header)
    print("-" * len(header))
    artifact_rows = []
    for label, model, qpfx, dpfx in CONFIGS:
        t0 = time.time()
        dv = embed(model, docs, dpfx)
        qv = embed(model, queries, qpfx)
        dt = time.time() - t0
        sims = qv @ dv.T
        ranks = []
        for qi, g in enumerate(gold):
            order = np.argsort(-sims[qi])
            ranks.append(int(np.where(order == g)[0][0]) + 1)
        ranks = np.asarray(ranks)
        r = {k: float((ranks <= k).mean()) for k in (1, 3, 5, 10)}
        mrr = float((1.0 / ranks).mean())
        print(f"{label:<28} {r[1]:>5.2f} {r[3]:>5.2f} {r[5]:>5.2f} {r[10]:>5.2f} {mrr:>6.3f} {dt:>8.1f}")
        artifact_rows.append({"config": label, "model": model,
                              "recall": r, "mrr": mrr, "embed_s": round(dt, 1)})

    # Persist the run (2026-08-25): the 08-24 "bge-m3 WINS" verdict existed
    # only as stdout + session memory — A/B evidence must outlive the session.
    import os
    from datetime import datetime, timezone
    out_dir = os.environ.get("AB_RESULTS_DIR", "/data/ceiling/results")
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"embedder_ab_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"seed": SEED, "n_queries": len(queries),
                       "n_docs": len(docs), "results": artifact_rows}, f, indent=2)
        print(f"\nartifact: {out_path}")
    except OSError as e:
        print(f"\n[warn] artifact not written ({e}) — set AB_RESULTS_DIR")

    print("\ndecision rule: promote only if a qwen3-emb config beats bge-m3 on "
          "r@3 AND MRR; instructed-query wins additionally require wiring "
          "query-prefix support through embedding.py before deploy.")


if __name__ == "__main__":
    main()
