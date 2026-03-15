"""A/B Evaluation Harness — compare base model vs fine-tuned model.

Takes a set of held-out queries and runs each through both models via
Ollama API. Uses a judge prompt (the same LLM) to compare responses.
Returns structured results with win rates and preference scores.

USAGE:
    # As a module (called from finetune_auto.py):
    from scripts.eval_harness import run_eval
    results = await run_eval(queries, base_model, ft_model, ollama_url)

    # Standalone:
    python scripts/eval_harness.py --base qwen3.5:27b --candidate nova-ft --queries queries.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
JUDGE_TEMPERATURE = 0.1
GENERATION_TIMEOUT = 120  # seconds per query


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    query: str
    base_response: str
    candidate_response: str
    winner: str              # 'base', 'candidate', 'tie'
    preference_score: float  # -1.0 (base much better) to 1.0 (candidate much better)
    judge_reasoning: str
    error: str = ""


@dataclass
class EvalResults:
    base_model: str
    candidate_model: str
    total_queries: int
    base_wins: int
    candidate_wins: int
    ties: int
    win_rate: float                          # candidate win rate (0.0 - 1.0)
    avg_preference: float                    # avg preference score (-1.0 to 1.0)
    candidate_is_better: bool
    comparisons: list[ComparisonResult] = field(default_factory=list)
    evaluated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "candidate_model": self.candidate_model,
            "total_queries": self.total_queries,
            "base_wins": self.base_wins,
            "candidate_wins": self.candidate_wins,
            "ties": self.ties,
            "win_rate": round(self.win_rate, 4),
            "avg_preference": round(self.avg_preference, 4),
            "candidate_is_better": self.candidate_is_better,
            "evaluated_at": self.evaluated_at,
            "comparisons": [
                {
                    "query": c.query[:200],
                    "base_response": c.base_response[:500],
                    "candidate_response": c.candidate_response[:500],
                    "winner": c.winner,
                    "preference_score": round(c.preference_score, 2),
                    "judge_reasoning": c.judge_reasoning[:300],
                    "error": c.error,
                }
                for c in self.comparisons
            ],
        }


# ---------------------------------------------------------------------------
# Ollama API helpers
# ---------------------------------------------------------------------------

async def _generate(
    client: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    json_mode: bool = False,
) -> str:
    """Generate a response from Ollama."""
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if json_mode:
        payload["format"] = "json"
        payload["options"]["repeat_penalty"] = 1.1

    try:
        resp = await client.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=GENERATION_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        logger.error("Ollama generate failed (model=%s): %s", model, e)
        return f"[ERROR: {e}]"


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an impartial judge evaluating two AI assistant responses.

USER QUERY:
{query}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Compare Response A and Response B. Consider:
1. Accuracy and correctness
2. Helpfulness and completeness
3. Clarity and natural tone
4. Relevance to the query

Respond with a single JSON object:
{{"winner": "A" or "B" or "tie", "score": <float from -1.0 to 1.0>, "reasoning": "<brief explanation>"}}

score: -1.0 means A is much better, 0.0 means equal, 1.0 means B is much better.
Only output the JSON object, nothing else."""


async def _judge_pair(
    client: httpx.AsyncClient,
    ollama_url: str,
    judge_model: str,
    query: str,
    base_response: str,
    candidate_response: str,
) -> tuple[str, float, str]:
    """Judge a pair of responses. Returns (winner, score, reasoning).

    Randomizes A/B assignment to avoid position bias.
    """
    # Randomize which response is A vs B to avoid position bias
    if random.random() < 0.5:
        response_a = base_response
        response_b = candidate_response
        a_is_base = True
    else:
        response_a = candidate_response
        response_b = base_response
        a_is_base = False

    prompt = JUDGE_PROMPT.format(
        query=query,
        response_a=response_a[:1500],
        response_b=response_b[:1500],
    )

    raw = await _generate(
        client, ollama_url, judge_model, prompt,
        temperature=JUDGE_TEMPERATURE, max_tokens=300, json_mode=True,
    )

    # Parse judge response
    try:
        # Try to extract JSON from the response
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to find JSON in the response
        import re
        match = re.search(r'\{[^{}]*"winner"[^{}]*\}', raw)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return "tie", 0.0, f"Could not parse judge response: {raw[:200]}"
        else:
            return "tie", 0.0, f"Could not parse judge response: {raw[:200]}"

    raw_winner = result.get("winner", "tie").upper()
    raw_score = float(result.get("score", 0.0))
    reasoning = result.get("reasoning", "")

    # Map back from A/B to base/candidate
    if raw_winner == "A":
        winner = "base" if a_is_base else "candidate"
    elif raw_winner == "B":
        winner = "candidate" if a_is_base else "base"
    else:
        winner = "tie"

    # Adjust score direction: positive = candidate better
    if a_is_base:
        preference_score = raw_score  # A=base, B=candidate, positive = B better = candidate better
    else:
        preference_score = -raw_score  # A=candidate, B=base, positive = B better = base better, so flip

    # Clamp
    preference_score = max(-1.0, min(1.0, preference_score))

    return winner, preference_score, reasoning


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def run_eval(
    queries: list[str],
    base_model: str,
    candidate_model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    judge_model: str | None = None,
) -> EvalResults:
    """Run A/B evaluation on a set of queries.

    Args:
        queries: List of query strings to evaluate.
        base_model: Name of the base model in Ollama.
        candidate_model: Name of the fine-tuned candidate model in Ollama.
        ollama_url: Ollama API URL.
        judge_model: Model to use as judge (defaults to base_model).

    Returns:
        EvalResults with detailed comparison data.
    """
    if not judge_model:
        judge_model = base_model

    logger.info(
        "Starting A/B evaluation: %s vs %s (%d queries, judge=%s)",
        base_model, candidate_model, len(queries), judge_model,
    )

    comparisons: list[ComparisonResult] = []

    async with httpx.AsyncClient() as client:
        for i, query in enumerate(queries, 1):
            logger.info("  [%d/%d] Evaluating: %s", i, len(queries), query[:80])

            # Generate responses from both models
            base_resp = await _generate(client, ollama_url, base_model, query)
            candidate_resp = await _generate(client, ollama_url, candidate_model, query)

            if base_resp.startswith("[ERROR") or candidate_resp.startswith("[ERROR"):
                error = base_resp if base_resp.startswith("[ERROR") else candidate_resp
                comparisons.append(ComparisonResult(
                    query=query,
                    base_response=base_resp,
                    candidate_response=candidate_resp,
                    winner="tie",
                    preference_score=0.0,
                    judge_reasoning="",
                    error=error,
                ))
                continue

            # Judge the pair
            winner, score, reasoning = await _judge_pair(
                client, ollama_url, judge_model,
                query, base_resp, candidate_resp,
            )

            comparisons.append(ComparisonResult(
                query=query,
                base_response=base_resp,
                candidate_response=candidate_resp,
                winner=winner,
                preference_score=score,
                judge_reasoning=reasoning,
            ))

            logger.info("    Winner: %s (score=%.2f)", winner, score)

    # Calculate aggregate metrics
    valid = [c for c in comparisons if not c.error]
    base_wins = sum(1 for c in valid if c.winner == "base")
    candidate_wins = sum(1 for c in valid if c.winner == "candidate")
    ties = sum(1 for c in valid if c.winner == "tie")

    total_valid = len(valid)
    win_rate = candidate_wins / total_valid if total_valid > 0 else 0.0
    avg_preference = sum(c.preference_score for c in valid) / total_valid if total_valid > 0 else 0.0

    # Candidate is better if win rate > 50% and average preference is positive
    candidate_is_better = win_rate > 0.5 and avg_preference > 0.0

    results = EvalResults(
        base_model=base_model,
        candidate_model=candidate_model,
        total_queries=len(queries),
        base_wins=base_wins,
        candidate_wins=candidate_wins,
        ties=ties,
        win_rate=win_rate,
        avg_preference=avg_preference,
        candidate_is_better=candidate_is_better,
        comparisons=comparisons,
        evaluated_at=datetime.now().isoformat(),
    )

    logger.info(
        "Evaluation complete: candidate wins %d/%d (%.0f%%), avg preference=%.2f, better=%s",
        candidate_wins, total_valid, win_rate * 100, avg_preference, candidate_is_better,
    )

    return results


def sample_holdout_queries(
    data_path: str,
    n: int = 10,
    seed: int | None = None,
) -> list[str]:
    """Sample n holdout queries from training data for evaluation.

    Removes them from the training set conceptually (returns the queries
    so the caller can exclude them from training if needed).
    """
    path = Path(data_path)
    if not path.exists():
        logger.warning("Training data not found: %s", data_path)
        return []

    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                query = entry.get("query", "").strip()
                if query and len(query) > 10:
                    entries.append(query)
            except json.JSONDecodeError:
                continue

    if not entries:
        return []

    if seed is not None:
        random.seed(seed)

    n = min(n, len(entries))
    return random.sample(entries, n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A/B Evaluation Harness — compare base vs fine-tuned model",
    )
    parser.add_argument("--base", required=True, help="Base model name in Ollama")
    parser.add_argument("--candidate", required=True, help="Candidate (fine-tuned) model name in Ollama")
    parser.add_argument("--judge", default=None, help="Judge model (defaults to base model)")
    parser.add_argument("--queries", help="Path to JSON file with list of query strings")
    parser.add_argument("--data", default="/data/training_data.jsonl", help="Training data to sample holdout queries from")
    parser.add_argument("--holdout", type=int, default=10, help="Number of holdout queries to sample (if --queries not given)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API URL")
    parser.add_argument("--output", default="/data/finetune/eval_results.json", help="Output file for results")
    args = parser.parse_args()

    # Load or sample queries
    if args.queries:
        with open(args.queries, encoding="utf-8") as f:
            queries = json.load(f)
    else:
        queries = sample_holdout_queries(args.data, n=args.holdout)

    if not queries:
        logger.error("No queries available for evaluation.")
        sys.exit(1)

    logger.info("Loaded %d evaluation queries", len(queries))

    # Run evaluation
    results = asyncio.run(run_eval(
        queries=queries,
        base_model=args.base,
        candidate_model=args.candidate,
        ollama_url=args.ollama_url,
        judge_model=args.judge,
    ))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info("Results saved to %s", output_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"A/B Evaluation Results")
    print(f"{'='*60}")
    print(f"Base model:      {results.base_model}")
    print(f"Candidate model: {results.candidate_model}")
    print(f"Total queries:   {results.total_queries}")
    print(f"Base wins:       {results.base_wins}")
    print(f"Candidate wins:  {results.candidate_wins}")
    print(f"Ties:            {results.ties}")
    print(f"Win rate:        {results.win_rate:.1%}")
    print(f"Avg preference:  {results.avg_preference:+.2f}")
    print(f"Candidate better: {results.candidate_is_better}")
    print(f"{'='*60}")

    if results.candidate_is_better:
        print("\nRECOMMENDATION: Deploy the fine-tuned model.")
    else:
        print("\nRECOMMENDATION: Keep the base model.")


if __name__ == "__main__":
    main()
