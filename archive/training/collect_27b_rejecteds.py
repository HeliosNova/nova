"""Collect 'rejected' responses by running each seed query through qwen3.6:27b via Ollama.

These are the responses the 27B naturally produces — the failure side of the DPO pairs.
The 'chosen' side will be filled in separately (by Claude, the assistant in conversation).

Output: rejecteds_27b.jsonl (one JSON per line:
  {"idx": int, "source": "...", "category": "...", "query": "...", "rejected": "...", "tokens": int, "sec": float}
)

Idempotent: if rejecteds_27b.jsonl exists, skips queries already processed.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "seeds_27b.jsonl"
OUT = ROOT / "rejecteds_27b.jsonl"
OLLAMA = "http://localhost:11434"
MODEL = "qwen3.6:27b"


def load_seeds() -> list[dict]:
    seeds = []
    with open(SEEDS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def load_done_indices() -> set[int]:
    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    pass
    return done


def query_ollama(prompt: str, timeout_s: int = 300) -> tuple[str, int, float]:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": 8192, "num_predict": 1024},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read().decode("utf-8"))
    el = time.time() - t0
    return data.get("response", ""), data.get("eval_count", 0), el


def main():
    seeds = load_seeds()
    done = load_done_indices()
    total = len(seeds)
    todo = [(i, s) for i, s in enumerate(seeds) if i not in done]
    print(f"Total seeds: {total}, already done: {len(done)}, todo: {len(todo)}", flush=True)

    with open(OUT, "a", encoding="utf-8") as fout:
        for n, (idx, seed) in enumerate(todo, 1):
            q = seed["query"]
            try:
                resp, tokens, sec = query_ollama(q)
            except Exception as e:
                print(f"[{n}/{len(todo)}] idx={idx} ERROR: {e}", flush=True)
                continue
            rec = {
                "idx": idx,
                "source": seed.get("source"),
                "category": seed.get("category"),
                "query": q,
                "rejected": resp,
                "tokens": tokens,
                "sec": round(sec, 1),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{n}/{len(todo)}] idx={idx} {seed.get('source')}/{seed.get('category')} {sec:.1f}s {tokens}tok", flush=True)


if __name__ == "__main__":
    main()
