"""Automated Fine-Tuning Orchestrator — end-to-end pipeline with A/B evaluation.

Orchestrates the full fine-tuning cycle:
  1. Check if enough new training pairs exist since last fine-tune
  2. Load and validate training data
  3. Stop Ollama via docker compose (free VRAM)
  4. Run DPO training via finetune.py train()
  5. Export to GGUF
  6. Register fine-tuned model with Ollama
  7. Restart Ollama
  8. Run A/B evaluation (base vs fine-tuned)
  9. If fine-tuned wins, update model config; otherwise keep base
  10. Record metadata about the training run

Called by the heartbeat monitor (when readiness is detected) or manually.

USAGE:
    python scripts/finetune_auto.py                          # Full auto pipeline
    python scripts/finetune_auto.py --force                  # Skip minimum pairs check
    python scripts/finetune_auto.py --skip-eval              # Train without A/B eval
    python scripts/finetune_auto.py --eval-only              # Just run eval on existing models
    python scripts/finetune_auto.py --data /path/to/data.jsonl

REQUIREMENTS:
    - RTX 3090 (24GB VRAM) or better
    - Docker compose for Ollama management
    - Unsloth + TRL (see requirements-finetune.txt)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import scripts.finetune
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.finetune import (
    DEFAULT_MODEL,
    load_training_data,
    show_data_stats,
    train,
    export_gguf,
)
from scripts.eval_harness import run_eval, sample_holdout_queries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Defaults (overridden by config when running inside Nova)
DEFAULT_DATA_PATH = "/data/training_data.jsonl"
DEFAULT_OUTPUT_DIR = "/data/finetune"
DEFAULT_MIN_NEW_PAIRS = 50
DEFAULT_EVAL_HOLDOUT = 10
DEFAULT_BASE_MODEL = "qwen3.5:27b"
DEFAULT_FT_MODEL_NAME = "nova-ft"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
METADATA_FILE = "run_history.json"


# ---------------------------------------------------------------------------
# Docker compose helpers (runs on host, not in container)
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str], *, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
        if result.stdout.strip():
            logger.info("  stdout: %s", result.stdout.strip()[:500])
        if result.stderr.strip():
            logger.info("  stderr: %s", result.stderr.strip()[:500])
        return result
    except subprocess.TimeoutExpired:
        logger.error("Command timed out after %ds: %s", timeout, " ".join(cmd))
        raise
    except subprocess.CalledProcessError as e:
        logger.error("Command failed (rc=%d): %s\n  stderr: %s", e.returncode, " ".join(cmd), e.stderr[:500])
        raise


def stop_ollama(compose_dir: str | None = None) -> bool:
    """Stop Ollama via docker compose to free VRAM."""
    logger.info("Stopping Ollama to free VRAM...")
    try:
        cmd = ["docker", "compose", "stop", "ollama"]
        if compose_dir:
            cmd = ["docker", "compose", "-f", os.path.join(compose_dir, "docker-compose.yml"), "stop", "ollama"]
        _run_cmd(cmd, timeout=60)
        # Wait for VRAM to be fully released
        time.sleep(5)
        return True
    except Exception as e:
        logger.error("Failed to stop Ollama: %s", e)
        return False


def start_ollama(compose_dir: str | None = None) -> bool:
    """Start Ollama via docker compose."""
    logger.info("Starting Ollama...")
    try:
        cmd = ["docker", "compose", "start", "ollama"]
        if compose_dir:
            cmd = ["docker", "compose", "-f", os.path.join(compose_dir, "docker-compose.yml"), "start", "ollama"]
        _run_cmd(cmd, timeout=60)
        # Wait for Ollama to be ready
        logger.info("Waiting for Ollama to initialize...")
        time.sleep(15)
        return True
    except Exception as e:
        logger.error("Failed to start Ollama: %s", e)
        return False


def register_model(model_name: str, modelfile_path: str) -> bool:
    """Register a GGUF model with Ollama via `ollama create`."""
    container = os.environ.get("OLLAMA_CONTAINER", "helios-ollama")
    logger.info("Registering model '%s' from %s (container: %s)", model_name, modelfile_path, container)
    try:
        _run_cmd(
            ["docker", "exec", container, "ollama", "create", model_name, "-f", modelfile_path],
            timeout=300,
        )
        return True
    except Exception as e:
        logger.error("Failed to register model: %s", e)
        return False


# ---------------------------------------------------------------------------
# Training run metadata
# ---------------------------------------------------------------------------

def _load_history(output_dir: str) -> list[dict]:
    """Load training run history."""
    path = Path(output_dir) / METADATA_FILE
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(output_dir: str, history: list[dict]) -> None:
    """Save training run history."""
    path = Path(output_dir) / METADATA_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _get_last_training_count(output_dir: str) -> int:
    """Get the number of training pairs used in the last fine-tune run."""
    history = _load_history(output_dir)
    if history:
        return history[-1].get("training_pairs", 0)
    return 0


def check_readiness(
    data_path: str,
    output_dir: str,
    min_new_pairs: int = DEFAULT_MIN_NEW_PAIRS,
) -> tuple[bool, int, int, str]:
    """Check if enough new training pairs exist for fine-tuning.

    Returns (ready, total_pairs, new_pairs, message).
    """
    data = load_training_data(data_path)
    total = len(data)
    last_count = _get_last_training_count(output_dir)
    new_pairs = total - last_count

    if total == 0:
        return False, 0, 0, "No training data available."

    if new_pairs < min_new_pairs:
        return (
            False, total, new_pairs,
            f"Only {new_pairs} new pairs since last training (need {min_new_pairs}). "
            f"Total: {total}, last training used: {last_count}.",
        )

    return (
        True, total, new_pairs,
        f"{new_pairs} new training pairs available (total: {total}). Ready for fine-tuning.",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

import asyncio


async def run_pipeline(
    *,
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    base_model_ollama: str = DEFAULT_BASE_MODEL,
    ft_model_name: str = DEFAULT_FT_MODEL_NAME,
    base_model_hf: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    min_new_pairs: int = DEFAULT_MIN_NEW_PAIRS,
    eval_holdout: int = DEFAULT_EVAL_HOLDOUT,
    force: bool = False,
    skip_eval: bool = False,
    compose_dir: str | None = None,
) -> dict:
    """Run the full automated fine-tuning pipeline.

    Returns a metadata dict describing the run outcome.
    """
    run_meta: dict = {
        "started_at": datetime.now().isoformat(),
        "data_path": data_path,
        "output_dir": output_dir,
        "base_model": base_model_ollama,
        "ft_model": ft_model_name,
        "status": "started",
    }

    # --- Step 1: Check readiness ---
    logger.info("=" * 60)
    logger.info("Step 1: Checking readiness")
    logger.info("=" * 60)

    ready, total, new_pairs, message = check_readiness(data_path, output_dir, min_new_pairs)
    run_meta["total_pairs"] = total
    run_meta["new_pairs"] = new_pairs
    logger.info(message)

    if not ready and not force:
        run_meta["status"] = "skipped"
        run_meta["reason"] = message
        run_meta["completed_at"] = datetime.now().isoformat()
        logger.info("Not enough new data. Use --force to override.")
        return run_meta

    # --- Step 2: Load and validate data ---
    logger.info("=" * 60)
    logger.info("Step 2: Loading training data")
    logger.info("=" * 60)

    all_data = load_training_data(data_path)
    show_data_stats(all_data)

    if not all_data:
        run_meta["status"] = "failed"
        run_meta["reason"] = "No valid training data"
        run_meta["completed_at"] = datetime.now().isoformat()
        return run_meta

    # Sample holdout queries for evaluation BEFORE training
    holdout_queries = sample_holdout_queries(data_path, n=eval_holdout)
    holdout_set = set(holdout_queries)
    training_data = [d for d in all_data if d["prompt"] not in holdout_set]

    run_meta["training_pairs"] = len(training_data)
    run_meta["holdout_queries"] = len(holdout_queries)
    logger.info("Training pairs: %d, Holdout queries: %d", len(training_data), len(holdout_queries))

    # --- Step 3: Stop Ollama ---
    logger.info("=" * 60)
    logger.info("Step 3: Stopping Ollama to free VRAM")
    logger.info("=" * 60)

    if not stop_ollama(compose_dir):
        logger.warning("Could not stop Ollama. Training may fail due to insufficient VRAM.")

    # --- Step 4: Run DPO training ---
    logger.info("=" * 60)
    logger.info("Step 4: Running DPO training")
    logger.info("=" * 60)

    try:
        adapter_dir = train(
            training_data,
            model_name=base_model_hf,
            output_dir=output_dir,
        )
        run_meta["adapter_dir"] = adapter_dir
        logger.info("Training complete. Adapter: %s", adapter_dir)
    except Exception as e:
        logger.error("Training failed: %s", e)
        run_meta["status"] = "failed"
        run_meta["reason"] = f"Training failed: {e}"
        run_meta["completed_at"] = datetime.now().isoformat()
        # Restart Ollama even on failure
        start_ollama(compose_dir)
        _record_run(output_dir, run_meta)
        return run_meta

    # --- Step 5: Export to GGUF ---
    logger.info("=" * 60)
    logger.info("Step 5: Exporting to GGUF")
    logger.info("=" * 60)

    try:
        gguf_path = export_gguf(adapter_dir, output_dir, model_name=base_model_hf)
        if not gguf_path:
            raise RuntimeError("GGUF export produced no file")
        run_meta["gguf_path"] = gguf_path
        logger.info("GGUF exported: %s", gguf_path)
    except Exception as e:
        logger.error("GGUF export failed: %s", e)
        run_meta["status"] = "failed"
        run_meta["reason"] = f"GGUF export failed: {e}"
        run_meta["completed_at"] = datetime.now().isoformat()
        start_ollama(compose_dir)
        _record_run(output_dir, run_meta)
        return run_meta

    # --- Step 6: Restart Ollama + register model ---
    logger.info("=" * 60)
    logger.info("Step 6: Restarting Ollama and registering model")
    logger.info("=" * 60)

    if not start_ollama(compose_dir):
        run_meta["status"] = "failed"
        run_meta["reason"] = "Failed to restart Ollama"
        run_meta["completed_at"] = datetime.now().isoformat()
        _record_run(output_dir, run_meta)
        return run_meta

    modelfile_path = os.path.join(output_dir, "Modelfile")
    if not register_model(ft_model_name, modelfile_path):
        logger.warning("Model registration failed. Evaluation may not work.")

    # --- Step 7: A/B Evaluation ---
    if skip_eval:
        logger.info("Skipping A/B evaluation (--skip-eval)")
        run_meta["status"] = "trained"
        run_meta["eval_skipped"] = True
        run_meta["completed_at"] = datetime.now().isoformat()
        _record_run(output_dir, run_meta)
        return run_meta

    logger.info("=" * 60)
    logger.info("Step 7: Running A/B evaluation")
    logger.info("=" * 60)

    if not holdout_queries:
        logger.warning("No holdout queries available. Skipping evaluation.")
        run_meta["status"] = "trained"
        run_meta["eval_skipped"] = True
        run_meta["completed_at"] = datetime.now().isoformat()
        _record_run(output_dir, run_meta)
        return run_meta

    try:
        eval_results = await run_eval(
            queries=holdout_queries,
            base_model=base_model_ollama,
            candidate_model=ft_model_name,
            ollama_url=ollama_url,
        )

        # Save eval results
        eval_path = Path(output_dir) / "eval_results.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results.to_dict(), f, indent=2)

        run_meta["eval_results"] = {
            "win_rate": eval_results.win_rate,
            "avg_preference": eval_results.avg_preference,
            "candidate_wins": eval_results.candidate_wins,
            "base_wins": eval_results.base_wins,
            "ties": eval_results.ties,
            "candidate_is_better": eval_results.candidate_is_better,
        }
        logger.info(
            "Evaluation complete: win_rate=%.1f%%, avg_pref=%.2f, better=%s",
            eval_results.win_rate * 100,
            eval_results.avg_preference,
            eval_results.candidate_is_better,
        )

    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        run_meta["status"] = "trained"
        run_meta["eval_error"] = str(e)
        run_meta["completed_at"] = datetime.now().isoformat()
        _record_run(output_dir, run_meta)
        return run_meta

    # --- Step 8: Decision ---
    logger.info("=" * 60)
    logger.info("Step 8: Making deployment decision")
    logger.info("=" * 60)

    if eval_results.candidate_is_better:
        logger.info("Fine-tuned model WINS. Deploying '%s' as active model.", ft_model_name)
        run_meta["status"] = "deployed"
        run_meta["deployed_model"] = ft_model_name

        # Write deployment marker for Nova to pick up
        deploy_marker = Path(output_dir) / "active_model.json"
        with open(deploy_marker, "w", encoding="utf-8") as f:
            json.dump({
                "model": ft_model_name,
                "deployed_at": datetime.now().isoformat(),
                "win_rate": eval_results.win_rate,
                "avg_preference": eval_results.avg_preference,
            }, f, indent=2)
        logger.info("Deployment marker written to %s", deploy_marker)
        logger.info(
            "To activate: set LLM_MODEL=%s in .env and restart Nova.",
            ft_model_name,
        )
    else:
        logger.info(
            "Base model wins or tie. Keeping '%s'. Fine-tuned model available as '%s'.",
            base_model_ollama, ft_model_name,
        )
        run_meta["status"] = "rejected"
        run_meta["reason"] = (
            f"Fine-tuned model did not outperform base "
            f"(win_rate={eval_results.win_rate:.1%}, avg_pref={eval_results.avg_preference:+.2f})"
        )

    run_meta["completed_at"] = datetime.now().isoformat()
    _record_run(output_dir, run_meta)
    return run_meta


def _record_run(output_dir: str, meta: dict) -> None:
    """Append a training run to the history file."""
    history = _load_history(output_dir)
    history.append(meta)
    _save_history(output_dir, history)
    logger.info("Training run recorded (status=%s)", meta.get("status", "unknown"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated Fine-Tuning Pipeline with A/B Evaluation",
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to training_data.jsonl")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base Ollama model name")
    parser.add_argument("--base-model-hf", default=DEFAULT_MODEL, help="Base HuggingFace model for training")
    parser.add_argument("--ft-model", default=DEFAULT_FT_MODEL_NAME, help="Fine-tuned model name in Ollama")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API URL")
    parser.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_NEW_PAIRS, help="Minimum new pairs threshold")
    parser.add_argument("--holdout", type=int, default=DEFAULT_EVAL_HOLDOUT, help="Number of holdout queries for eval")
    parser.add_argument("--force", action="store_true", help="Skip minimum pairs check")
    parser.add_argument("--skip-eval", action="store_true", help="Skip A/B evaluation after training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation (no training)")
    parser.add_argument("--compose-dir", default=None, help="Path to docker-compose.yml directory")
    parser.add_argument("--check", action="store_true", help="Only check readiness, don't train")
    args = parser.parse_args()

    # Check-only mode
    if args.check:
        ready, total, new_pairs, message = check_readiness(args.data, args.output, args.min_pairs)
        print(message)
        sys.exit(0 if ready else 1)

    # Eval-only mode
    if args.eval_only:
        queries = sample_holdout_queries(args.data, n=args.holdout)
        if not queries:
            logger.error("No queries available for evaluation.")
            sys.exit(1)
        results = asyncio.run(run_eval(
            queries=queries,
            base_model=args.base_model,
            candidate_model=args.ft_model,
            ollama_url=args.ollama_url,
        ))
        print(f"\nWin rate: {results.win_rate:.1%}")
        print(f"Avg preference: {results.avg_preference:+.2f}")
        print(f"Candidate is better: {results.candidate_is_better}")
        sys.exit(0)

    # Full pipeline
    result = asyncio.run(run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        base_model_ollama=args.base_model,
        ft_model_name=args.ft_model,
        base_model_hf=args.base_model_hf,
        ollama_url=args.ollama_url,
        min_new_pairs=args.min_pairs,
        eval_holdout=args.holdout,
        force=args.force,
        skip_eval=args.skip_eval,
        compose_dir=args.compose_dir,
    ))

    print(f"\nPipeline finished: {result.get('status', 'unknown')}")
    if result.get("eval_results"):
        er = result["eval_results"]
        print(f"  Win rate: {er['win_rate']:.1%}")
        print(f"  Avg preference: {er['avg_preference']:+.2f}")
        print(f"  Candidate better: {er['candidate_is_better']}")


if __name__ == "__main__":
    main()
