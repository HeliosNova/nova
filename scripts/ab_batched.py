#!/usr/bin/env python3
"""Batched A/B eval — avoids the per-query model thrashing that timed out the
stock harness on the 27B (which cold-loads on every call).

Phases: (1) load candidate once, generate all; (2) load base once, generate
all; (3) load judge once, score all pairs. 3 model loads instead of 30.

Usage: ab_batched.py <queries.json> <base_model> <candidate_model> <judge_model> <output.json>
"""
from __future__ import annotations
import os
# Generous per-call ceiling — a 27B cold-load + long generation can take >10 min.
os.environ.setdefault("EVAL_GENERATION_TIMEOUT", "2400")

import asyncio
import json
import sys
from datetime import datetime

import httpx

sys.path.insert(0, "/mnt/f/Helios Project/nova_")
from scripts.eval_harness import _generate, _judge_pair, JUDGE_TIE_THRESHOLD  # noqa: E402

QUERIES_PATH, BASE, CAND, JUDGE, OUT = sys.argv[1:6]
URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

with open(QUERIES_PATH) as f:
    queries = json.load(f)

print(f"Batched A/B: base={BASE} candidate={CAND} judge={JUDGE}")
print(f"{len(queries)} queries | timeout={os.environ['EVAL_GENERATION_TIMEOUT']}s\n")


async def gen_all(client, model, label):
    out = []
    for i, q in enumerate(queries):
        t0 = datetime.now()
        r = await _generate(client, URL, model, q, temperature=0.3, max_tokens=1000)
        dt = (datetime.now() - t0).total_seconds()
        ok = not r.startswith("[ERROR")
        print(f"  [{label} {i+1}/{len(queries)}] {dt:.0f}s {'ok' if ok else 'FAIL: ' + r[:120]}")
        out.append(r)
    return out


async def main():
    async with httpx.AsyncClient() as client:
        print("=== Phase 1: candidate generations ===")
        cand = await gen_all(client, CAND, "cand")
        print("\n=== Phase 2: base generations ===")
        base = await gen_all(client, BASE, "base")
        print("\n=== Phase 3: judging ===")
        comparisons = []
        base_wins = cand_wins = ties = 0
        pref_sum = 0.0
        scored = 0
        for i, q in enumerate(queries):
            if base[i].startswith("[ERROR") or cand[i].startswith("[ERROR"):
                print(f"  [judge {i+1}/{len(queries)}] SKIP (generation error)")
                comparisons.append({"query": q[:200], "winner": "error",
                                    "preference_score": 0.0, "error": "generation failed"})
                continue
            winner, score, reasoning = await _judge_pair(client, URL, JUDGE, q, base[i], cand[i])
            scored += 1
            pref_sum += score
            if winner == "candidate":
                cand_wins += 1
            elif winner == "base":
                base_wins += 1
            else:
                ties += 1
            print(f"  [judge {i+1}/{len(queries)}] winner={winner} score={score:+.2f}")
            comparisons.append({
                "query": q[:200],
                "base_response": base[i][:600],
                "candidate_response": cand[i][:600],
                "winner": winner,
                "preference_score": round(score, 3),
                "judge_reasoning": reasoning[:300],
                "error": "",
            })

        avg_pref = pref_sum / scored if scored else 0.0
        win_rate = cand_wins / scored if scored else 0.0
        result = {
            "base_model": BASE,
            "candidate_model": CAND,
            "judge_model": JUDGE,
            "total_queries": len(queries),
            "scored": scored,
            "base_wins": base_wins,
            "candidate_wins": cand_wins,
            "ties": ties,
            "win_rate": round(win_rate, 4),
            "avg_preference": round(avg_pref, 4),
            "candidate_is_better": win_rate > 0.5 and avg_pref > 0,
            "evaluated_at": datetime.now().isoformat(),
            "comparisons": comparisons,
        }
        with open(OUT, "w") as f:
            json.dump(result, f, indent=2)

        print("\n" + "=" * 60)
        print(f"RESULT  base={BASE}  candidate={CAND}")
        print(f"  scored: {scored}/{len(queries)}")
        print(f"  candidate wins: {cand_wins} | base wins: {base_wins} | ties: {ties}")
        print(f"  win_rate (candidate): {win_rate:.1%}")
        print(f"  avg_preference: {avg_pref:+.3f}  (>0 favors candidate)")
        print(f"  candidate_is_better: {result['candidate_is_better']}")
        print("=" * 60)
        print(f"Saved: {OUT}")


asyncio.run(main())
