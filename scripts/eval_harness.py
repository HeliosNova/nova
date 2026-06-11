"""A/B Evaluation Harness — compare base model vs fine-tuned model.

Takes a set of held-out queries and runs each through both models via
Ollama API. An independent judge model (cross-family recommended — set
EVAL_JUDGE_MODEL) compares responses, scoring both A/B orders and requiring
agreement (position-swap) across four independent dimensions.
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
import os
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

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
JUDGE_TEMPERATURE = 0.1
# Per-call timeout. 600s is generous to cover cold-load of large models (e.g.,
# 27B Q4 = 17 GB; on a 24 GB GPU it swaps in/out between base/candidate/judge
# calls, so each generate may pay a disk-load cost before producing tokens).
GENERATION_TIMEOUT = int(os.getenv("EVAL_GENERATION_TIMEOUT", "600"))
# Tie threshold for the derived winner. |preference_score| below this counts as
# a tie. 0.15 catches obvious wins while keeping noise out — earlier "winner"
# field was self-inconsistent with score ~47% of the time on Qwen3.5:9b judge,
# so we treat score as the single source of truth and derive winner here.
JUDGE_TIE_THRESHOLD = float(os.getenv("EVAL_JUDGE_TIE_THRESHOLD", "0.15"))
# Position-swapped judging: score BOTH orders (base-first and candidate-first)
# and only declare a decisive winner when the orders agree — a flip means the
# judge is position-biased, which we score as a tie. Default on; set
# EVAL_JUDGE_SWAP=false for legacy single-order behavior.
JUDGE_SWAP = os.getenv("EVAL_JUDGE_SWAP", "true").strip().lower() not in (
    "0", "false", "no", "off", "",
)
# Independent (cross-family) judge model. Self-judging (judge == a model under
# test) inflates scores via self-preference bias — point this at a DIFFERENT
# model family running locally in Ollama (e.g. a Llama or Gemma).
DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL") or os.getenv("JUDGE_MODEL") or ""


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
        # Disable Qwen3.x thinking mode — without this the model emits everything
        # inside <think>...</think> and Ollama returns an empty `response`. The
        # eval harness needs the raw answer, not chain-of-thought.
        "think": False,
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

Score Response A vs Response B on FOUR dimensions independently. For each, use
a float from -1.0 to 1.0:
  -1.0 = A much better,  0.0 = equivalent,  +1.0 = B much better.

Respond with a single JSON object — no other text:
{{"accuracy": <float>, "completeness": <float>, "clarity": <float>, "relevance": <float>, "reasoning": "<brief explanation>"}}

Judge each dimension independently — do not let one strong dimension dominate
the others. Each score must reflect the magnitude of the difference you describe."""


# Dimensions averaged into the overall preference. Decomposing the judgment into
# independent dimensions measurably reduces self-preference bias (2026 studies).
_JUDGE_DIMENSIONS = ("accuracy", "completeness", "clarity", "relevance")


def _derive_winner(preference_score: float, threshold: float = JUDGE_TIE_THRESHOLD) -> str:
    """Map a continuous preference score (positive = candidate better) to
    a discrete winner label using a tie threshold.

    The judge is asked only for dimension scores; `winner` is derived here from
    the (sign-corrected) overall so the two are consistent by construction.
    """
    if preference_score > threshold:
        return "candidate"
    if preference_score < -threshold:
        return "base"
    return "tie"


def _parse_judge(raw: str) -> tuple[float | None, str]:
    """Parse a judge response into (overall_score, reasoning).

    overall_score is in [-1, 1] where positive means RESPONSE B is better.
    Accepts the multi-dimensional schema (dimensions averaged) and the legacy
    single `score` field. Returns (None, diagnostic) when no numeric score is
    present, so the caller can treat the judgment as missing rather than a tie
    with a fabricated 0.0.
    """
    result: dict | None = None
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Judge sometimes wraps JSON in prose despite format=json — extract it.
        import re
        match = re.search(r'\{[^{}]*"(?:score|accuracy)"[^{}]*\}', raw or "")
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                result = None

    if not isinstance(result, dict):
        return None, f"Could not parse judge response: {(raw or '')[:200]}"

    # Multi-dimensional: average whichever named dimensions are present.
    dim_vals: list[float] = []
    for d in _JUDGE_DIMENSIONS:
        if d in result:
            try:
                dim_vals.append(float(result[d]))
            except (TypeError, ValueError):
                return None, f"Non-numeric {d} from judge: {(raw or '')[:200]}"
    reasoning = str(result.get("reasoning", ""))
    if dim_vals:
        return sum(dim_vals) / len(dim_vals), reasoning

    # Legacy single-score schema.
    if "score" in result:
        try:
            return float(result["score"]), reasoning
        except (TypeError, ValueError):
            return None, f"Non-numeric score from judge: {(raw or '')[:200]}"

    return None, f"Judge response missing score fields: {(raw or '')[:200]}"


async def _judge_once_directed(
    client: httpx.AsyncClient,
    ollama_url: str,
    judge_model: str,
    query: str,
    base_response: str,
    candidate_response: str,
    base_first: bool,
) -> tuple[float | None, str]:
    """One directed judge call with a fixed A/B order.

    Returns (preference, reasoning) where preference is in [-1, 1], positive =
    CANDIDATE better; None on parse failure.
    """
    if base_first:
        response_a, response_b = base_response, candidate_response
    else:
        response_a, response_b = candidate_response, base_response

    prompt = JUDGE_PROMPT.format(
        query=query, response_a=response_a[:1500], response_b=response_b[:1500],
    )
    raw = await _generate(
        client, ollama_url, judge_model, prompt,
        temperature=JUDGE_TEMPERATURE, max_tokens=300, json_mode=True,
    )
    score, reasoning = _parse_judge(raw)
    if score is None:
        return None, reasoning
    # score is "B better positive". Convert to "candidate better positive":
    #   base_first  → B is candidate → pref = +score
    #   !base_first → B is base      → pref = -score
    pref = score if base_first else -score
    return max(-1.0, min(1.0, pref)), reasoning


async def _judge_pair(
    client: httpx.AsyncClient,
    ollama_url: str,
    judge_model: str,
    query: str,
    base_response: str,
    candidate_response: str,
) -> tuple[str, float, str]:
    """Judge a pair of responses. Returns (winner, preference, reasoning),
    preference positive = candidate better.

    Default (EVAL_JUDGE_SWAP): score BOTH orders and only declare a decisive
    winner when they agree — a decisive flip between orders is position bias and
    is scored a tie. With swap disabled, falls back to one randomized-order call.
    """
    if not JUDGE_SWAP:
        base_first = random.random() < 0.5
        pref, reasoning = await _judge_once_directed(
            client, ollama_url, judge_model, query,
            base_response, candidate_response, base_first,
        )
        if pref is None:
            return "tie", 0.0, reasoning
        return _derive_winner(pref), pref, reasoning

    pref1, r1 = await _judge_once_directed(
        client, ollama_url, judge_model, query, base_response, candidate_response, True,
    )
    pref2, r2 = await _judge_once_directed(
        client, ollama_url, judge_model, query, base_response, candidate_response, False,
    )

    valid = [p for p in (pref1, pref2) if p is not None]
    if not valid:
        return "tie", 0.0, f"Could not parse judge response (both orders): {r1}"
    if len(valid) == 1:
        p = valid[0]
        return _derive_winner(p), round(p, 4), (r1 if pref1 is not None else r2)

    avg = (pref1 + pref2) / 2.0
    w1, w2 = _derive_winner(pref1), _derive_winner(pref2)
    if {w1, w2} == {"base", "candidate"}:
        # Orders disagree decisively → position bias → tie.
        return "tie", round(avg, 4), (
            f"position-bias: order1={w1}({pref1:+.2f}) order2={w2}({pref2:+.2f})"
        )
    return _derive_winner(avg), round(avg, 4), (r1 or r2)


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
        judge_model = DEFAULT_JUDGE_MODEL or base_model
    if judge_model in (base_model, candidate_model):
        logger.warning(
            "Judge model %r is one of the models under test — self-preference "
            "bias is likely. Set EVAL_JUDGE_MODEL to an independent "
            "(different-family) local model for a credible result.",
            judge_model,
        )

    logger.info(
        "Starting A/B evaluation: %s vs %s (%d queries, judge=%s, swap=%s)",
        base_model, candidate_model, len(queries), judge_model, JUDGE_SWAP,
    )

    comparisons: list[ComparisonResult] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(float(GENERATION_TIMEOUT))) as client:
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
