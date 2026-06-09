"""Long-context degradation eval — needle-in-haystack at 4k, 16k, 64k, 128k tokens.

Measures at which context length Nova's baseline (nova-ft-v8-q8) actually starts
missing facts. Same harness can be re-run against nova-ft-v9 after training to
compare. This is a diagnostic, not a gating test — informs whether long-context
is a real problem for your workloads.

Usage:
    # Run against default model (from config_overrides / .env)
    python scripts/long_context_eval.py

    # Run against a specific model
    python scripts/long_context_eval.py --model nova-ft-v9

    # Compare two models
    python scripts/long_context_eval.py --model nova-ft-v8-q8 --compare nova-ft-v9

The needle is a unique fact placed at varying depths in a large filler corpus.
Model is asked to retrieve it. Pass = needle found; fail = missed or wrong answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Approximate tokens-per-char for Qwen tokenizers. Conservative.
CHARS_PER_TOKEN = 3.5

# The "needle" is a unique, unlikely-to-be-confused fact.
NEEDLE_FACT = (
    "The secret launch code for Project Helios version 42.7 is "
    "HELIOS-ZETA-9041-QUINN and it expires on March 19, 2027."
)
# The question targets the full fact — model must return exact code + date.
NEEDLE_QUESTION = (
    "What is the secret launch code for Project Helios version 42.7, "
    "and when does it expire? Quote the answer exactly."
)
EXPECTED_TOKENS_IN_ANSWER = ("HELIOS-ZETA-9041-QUINN", "March 19, 2027")

# Filler text — verbose but neutral. Long-form analytical prose on an unrelated topic.
FILLER_PARAGRAPH = (
    "The process of photosynthesis is a fundamental biological mechanism by which "
    "plants, algae, and certain bacteria convert light energy, typically from the sun, "
    "into chemical energy stored in glucose molecules. This process occurs primarily in "
    "the chloroplasts of plant cells, which contain chlorophyll — the green pigment "
    "responsible for capturing light. The overall reaction involves carbon dioxide from "
    "the atmosphere and water from the soil being transformed into glucose and oxygen, "
    "with the oxygen released as a byproduct into the atmosphere. The light-dependent "
    "reactions occur in the thylakoid membranes and produce ATP and NADPH, which are "
    "then used in the light-independent reactions, also known as the Calvin cycle, to "
    "fix carbon dioxide into organic molecules. Various factors affect the rate of "
    "photosynthesis, including light intensity, carbon dioxide concentration, temperature, "
    "and water availability. Different plant species have evolved various adaptations to "
    "optimize photosynthesis under specific environmental conditions. "
)


@dataclass
class ProbeResult:
    context_tokens: int
    depth_pct: float  # 0.0 = start, 0.5 = middle, 1.0 = end
    passed: bool
    model: str
    latency_s: float
    answer: str


def build_haystack(target_tokens: int, needle_depth_pct: float) -> str:
    """Build a haystack with the needle at a specific depth.

    depth_pct: 0.0 puts needle near start, 1.0 near end.
    """
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    # Repeat filler until we reach target, sized to leave room for the needle
    filler_needed = target_chars - len(NEEDLE_FACT)
    paras = []
    while sum(len(p) for p in paras) < filler_needed:
        paras.append(FILLER_PARAGRAPH)
    # Compute insertion index for needle
    insert_idx = max(0, min(len(paras) - 1, int(len(paras) * needle_depth_pct)))
    paras.insert(insert_idx, "\n\n" + NEEDLE_FACT + "\n\n")
    return "".join(paras)[:target_chars + len(NEEDLE_FACT) + 50]


async def run_probe(
    model: str,
    context_tokens: int,
    needle_depth_pct: float,
    ollama_url: str,
    timeout: float = 180.0,
) -> ProbeResult:
    import httpx
    haystack = build_haystack(context_tokens, needle_depth_pct)
    prompt = (
        "The following is a long document. Read it carefully, then answer the question at the end.\n\n"
        "=== DOCUMENT START ===\n"
        f"{haystack}\n"
        "=== DOCUMENT END ===\n\n"
        f"QUESTION: {NEEDLE_QUESTION}"
    )
    t0 = time.time()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"num_predict": 200, "temperature": 0.1, "num_ctx": max(context_tokens + 2048, 8192)},
            },
        )
    latency = time.time() - t0
    if r.status_code != 200:
        logger.warning("probe failed: %s %s", r.status_code, r.text[:200])
        return ProbeResult(context_tokens, needle_depth_pct, False, model, latency,
                           f"[HTTP {r.status_code}] {r.text[:120]}")
    answer = r.json().get("response", "")
    passed = all(tok in answer for tok in EXPECTED_TOKENS_IN_ANSWER)
    return ProbeResult(context_tokens, needle_depth_pct, passed, model, latency, answer)


async def run_sweep(model: str, ollama_url: str) -> list[ProbeResult]:
    sizes = [4000, 16000, 64000, 128000]
    depths = [0.1, 0.5, 0.9]  # near-start, middle, near-end
    results: list[ProbeResult] = []
    for size in sizes:
        for depth in depths:
            logger.info("Probe: model=%s tokens=%d depth=%.1f", model, size, depth)
            try:
                r = await run_probe(model, size, depth, ollama_url)
            except Exception as e:
                logger.warning("probe raised: %s", e)
                r = ProbeResult(size, depth, False, model, 0.0, f"[ERR] {e}")
            results.append(r)
            logger.info(
                "  -> %s (lat %.1fs, answer=%r)",
                "PASS" if r.passed else "FAIL", r.latency_s, r.answer[:80],
            )
    return results


def summarize(results: list[ProbeResult], model: str) -> str:
    lines = [f"\n=== {model} — long-context needle-in-haystack ==="]
    by_size: dict[int, list[ProbeResult]] = {}
    for r in results:
        by_size.setdefault(r.context_tokens, []).append(r)
    for size in sorted(by_size):
        rs = by_size[size]
        passed = sum(1 for r in rs if r.passed)
        avg_lat = sum(r.latency_s for r in rs) / len(rs)
        lines.append(f"  {size:>6} tok: {passed}/{len(rs)} passed, avg {avg_lat:.1f}s latency")
        for r in rs:
            mark = "OK  " if r.passed else "FAIL"
            lines.append(f"    {mark} depth={r.depth_pct:.1f} — {r.answer[:100]!r}")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> None:
    models = [args.model]
    if args.compare:
        models.append(args.compare)

    all_results: dict[str, list[ProbeResult]] = {}
    for m in models:
        all_results[m] = await run_sweep(m, args.ollama_url)

    out_lines = []
    for m, rs in all_results.items():
        out_lines.append(summarize(rs, m))
    print("\n".join(out_lines))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {m: [asdict(r) for r in rs] for m, rs in all_results.items()},
                f, indent=2,
            )
        logger.info("Report saved to %s", args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nova-ft-v8-q8", help="Model to test")
    parser.add_argument("--compare", default=None, help="Second model to compare against")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--output", default=None, help="Path to write JSON report")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
