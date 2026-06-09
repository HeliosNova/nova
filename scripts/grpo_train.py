"""Nova GRPO Trainer — consume RLVR verifiable_signals into a fresh LoRA adapter.

Same hardware envelope as scripts/finetune.py: trains the 9B (Qwen3.5-9B) base.
The 27b base is NOT fine-tuned — to switch we'd need a separate training cycle.

Pipeline:
  1. Read unconsumed signals from `verifiable_signals` table (RLVR collection)
  2. Group by normalized query → GRPOGroup with completions + rewards
  3. If trl GRPOTrainer is available AND we have ≥ MIN_GRPO_GROUPS trainable groups:
       run proper GRPO over LoRA on the 9B base
     Otherwise:
       fall back to DPO pairs (best vs worst within each group) and reuse the
       existing finetune.py train() path
  4. On success: mark signals consumed, save adapter to /data/finetune/grpo_<ts>/
  5. Optionally export GGUF for Ollama deployment

USAGE:
    docker compose stop ollama         # Free 17GB VRAM
    python scripts/grpo_train.py --dry-run            # Preview groups + counts
    python scripts/grpo_train.py                      # Train (DPO fallback if needed)
    python scripts/grpo_train.py --force-grpo         # Insist on GRPO; abort if unavailable
    python scripts/grpo_train.py --export-gguf        # Train + GGUF export
    docker compose start ollama        # Restart inference

DATA FORMAT IN DB (verifiable_signals):
    Created by app.core.rlvr.record_signal() during normal Nova operation.
    Each row has (query, response, signal_type, signal_value, evidence).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Same VRAM-safety preamble as finetune.py
if sys.platform == "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["PYTHONUNBUFFERED"] = "1"

# Ensure app/* is on the path (we run from repo root or from /app inside container)
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_LOG_DIR = Path(os.environ.get("NOVA_DATA_DIR", "/data"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "grpo_progress.log"
_log_handlers: list[logging.Handler] = [logging.StreamHandler()]
try:
    _log_handlers.append(logging.FileHandler(str(_LOG_FILE), mode="w"))
except OSError:
    # Read-only filesystem (e.g. running inside the production container) —
    # stick with stdout-only logging instead of crashing.
    pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)
logger.info("=== GRPO log: %s ===", _LOG_FILE)


# Defaults — match finetune.py for hardware compatibility
_NOVA_DATA_DIR = os.environ.get("NOVA_DATA_DIR", "/data")
DEFAULT_OUTPUT_DIR = os.path.join(_NOVA_DATA_DIR, "finetune")
DEFAULT_MODEL = os.environ.get("GRPO_BASE_MODEL", "Qwen/Qwen3.5-9B")
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_LORA_RANK = 16
DEFAULT_EPOCHS = 1            # GRPO is on-policy; 1 epoch is the canonical recipe
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 4         # Smaller groups → bigger accumulation
DEFAULT_LR = 2e-6              # Higher than DPO's 5e-7 since group advantages are smaller
DEFAULT_BETA = 0.04            # KL penalty coefficient
DEFAULT_GRPO_GROUP_SIZE = 4    # Min completions per group to count as GRPO-trainable

MIN_GRPO_GROUPS = 8            # Below this, fall back to DPO
MIN_DPO_PAIRS = 8              # Below this, abort entirely


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Nova on RLVR verifiable_signals via GRPO (or DPO fallback)."
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Base model (default: {DEFAULT_MODEL})")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Output directory for adapter + GGUF")
    p.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    p.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA,
                   help="KL penalty coefficient (default 0.04)")
    p.add_argument("--group-size", type=int, default=DEFAULT_GRPO_GROUP_SIZE,
                   help="Min completions per group for GRPO. Below this, group is DPO-only.")
    p.add_argument("--signal-types", default="",
                   help="Comma-separated signal_type filter (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show stats + dataset preview, don't train")
    p.add_argument("--force-grpo", action="store_true",
                   help="Abort if GRPO unavailable; never fall back to DPO")
    p.add_argument("--force-dpo", action="store_true",
                   help="Skip GRPO entirely; go straight to DPO fallback")
    p.add_argument("--export-gguf", action="store_true",
                   help="After training, export merged GGUF for Ollama")
    p.add_argument("--no-mark-consumed", action="store_true",
                   help="Don't mark signals as consumed (useful for dry-runs)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(args: argparse.Namespace) -> dict:
    """Read RLVR signals, build groups, return a summary dict + paths."""
    from app.core import grpo_dataset

    types = [t.strip() for t in args.signal_types.split(",") if t.strip()] or None
    groups = grpo_dataset.build_groups(
        signal_types=types,
        only_unconsumed=True,
        limit=10_000,
    )
    s = grpo_dataset.stats(groups)
    logger.info(
        "[grpo_train] dataset: %d groups (%d trainable), %d total completions, by_type=%s, by_size=%s",
        s["n_groups"], s["n_trainable"], s["n_completions"],
        s["by_signal_type"], s["by_size_bucket"],
    )

    # Filter groups by GRPO group size
    grpo_eligible = [
        g for g in groups
        if g.is_trainable() and len(g.completions) >= args.group_size
    ]
    dpo_pairs = grpo_dataset.to_dpo_pairs(groups)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grpo_path = out_dir / "grpo_dataset.jsonl"
    dpo_path = out_dir / "grpo_to_dpo_pairs.jsonl"

    # Always write both for inspection
    grpo_flat = grpo_dataset.to_grpo_dataset(grpo_eligible)
    rows: list[dict] = []
    for i in range(len(grpo_flat["prompt"])):
        rows.append({
            "prompt": grpo_flat["prompt"][i],
            "completion": grpo_flat["completion"][i],
            "reward": grpo_flat["reward"][i],
            "advantage": grpo_flat["advantage"][i],
            "group_id": grpo_flat["group_id"][i],
            "signal_type": grpo_flat["signal_type"][i],
        })
    grpo_dataset.write_jsonl(rows, str(grpo_path))
    grpo_dataset.write_jsonl(dpo_pairs, str(dpo_path))

    return {
        "groups": groups,
        "grpo_eligible": grpo_eligible,
        "dpo_pairs": dpo_pairs,
        "grpo_path": str(grpo_path),
        "dpo_path": str(dpo_path),
        "stats": s,
    }


# ---------------------------------------------------------------------------
# Training paths
# ---------------------------------------------------------------------------

def _grpo_available() -> bool:
    """True iff trl.GRPOTrainer + Unsloth are importable."""
    try:
        # Unsloth must come first to monkey-patch trl
        import unsloth  # noqa: F401
        from trl import GRPOTrainer, GRPOConfig  # noqa: F401
        return True
    except Exception as e:
        logger.warning("[grpo_train] GRPO path unavailable: %s", e)
        return False


def train_grpo(prep: dict, args: argparse.Namespace) -> str:
    """Run real GRPO over LoRA on the 9B base. Returns adapter path.

    Loads the base model in 4-bit, attaches a LoRA, then runs GRPOTrainer
    with the precomputed advantages from each group. Reference model is the
    base (frozen) for the KL penalty.
    """
    import torch
    import unsloth  # patches trl imports below
    from datasets import Dataset
    from trl import GRPOTrainer, GRPOConfig
    from unsloth import FastLanguageModel

    logger.info("Loading base %s in 4-bit...", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # Build HF dataset from groups
    from app.core import grpo_dataset
    flat = grpo_dataset.to_grpo_dataset(prep["grpo_eligible"])
    ds = Dataset.from_dict(flat)
    logger.info("[grpo_train] dataset rows: %d", len(ds))

    # GRPO config
    out_dir = Path(args.output_dir) / f"grpo_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    cfg = GRPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_prompt_length=args.max_seq_length // 2,
        max_completion_length=args.max_seq_length // 2,
        beta=args.beta,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    # GRPOTrainer needs a reward_funcs callable. Since we have precomputed
    # rewards, wrap them: the trainer calls reward_funcs(prompts, completions)
    # and expects per-sample rewards. Look up by (prompt, completion) match.
    reward_lookup = {
        (flat["prompt"][i], flat["completion"][i]): flat["reward"][i]
        for i in range(len(flat["prompt"]))
    }

    def precomputed_reward(prompts, completions, **kwargs):
        out = []
        for p, c in zip(prompts, completions):
            r = reward_lookup.get((p, c))
            if r is None:
                # New rollout from the model — score with reward=0.5 as a
                # "no-information" prior. The training run is offline-style:
                # we expect existing prompt+completion pairs from the dataset.
                r = 0.5
            out.append(float(r))
        return out

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=[precomputed_reward],
        processing_class=tokenizer,
    )
    trainer.train()
    adapter_path = str(out_dir / "adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("[grpo_train] adapter saved: %s", adapter_path)
    return adapter_path


def train_dpo_fallback(prep: dict, args: argparse.Namespace) -> str:
    """Fall back to DPO with the (chosen, rejected) pairs. Returns adapter path."""
    if len(prep["dpo_pairs"]) < MIN_DPO_PAIRS:
        raise RuntimeError(
            f"Only {len(prep['dpo_pairs'])} DPO pairs available; "
            f"need at least {MIN_DPO_PAIRS}. Collect more RLVR signals first."
        )
    # Reuse the existing finetune.py path. It expects training_data.jsonl with
    # {prompt, chosen, rejected} per line. Write our pairs to a temp jsonl.
    out_dir = Path(args.output_dir) / f"grpo_dpo_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fallback_jsonl = out_dir / "input.jsonl"
    with open(fallback_jsonl, "w", encoding="utf-8") as f:
        for p in prep["dpo_pairs"]:
            f.write(json.dumps({
                "query": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
                "source": "grpo_to_dpo",
                "signal_type": p["signal_type"],
                "timestamp": datetime.utcnow().isoformat(),
            }, ensure_ascii=False) + "\n")
    logger.info("[grpo_train] wrote %d DPO pairs to %s", len(prep["dpo_pairs"]), fallback_jsonl)

    # Import lazily — avoids importing torch at module load
    sys.path.insert(0, str(_HERE))
    from finetune import load_training_data, train as dpo_train

    pairs = load_training_data(str(fallback_jsonl))
    if len(pairs) < MIN_DPO_PAIRS:
        raise RuntimeError(f"After load, only {len(pairs)} DPO pairs valid")

    adapter_path = dpo_train(
        pairs,
        model_name=args.model,
        output_dir=str(out_dir),
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        epochs=args.epochs if args.epochs > 0 else 3,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=5e-7,  # DPO LR (lower than GRPO)
    )
    return adapter_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    if args.force_grpo and args.force_dpo:
        logger.error("Cannot specify both --force-grpo and --force-dpo")
        return 2

    logger.info("=== Nova GRPO trainer ===")
    logger.info("Args: %s", vars(args))

    # Phase 1: prepare dataset
    prep = prepare_dataset(args)

    # Decide path
    n_grpo_groups = len(prep["grpo_eligible"])
    n_dpo_pairs = len(prep["dpo_pairs"])

    print(f"\n{'='*60}")
    print(f"Dataset summary")
    print(f"{'='*60}")
    print(f"  Total groups:      {prep['stats']['n_groups']}")
    print(f"  Trainable groups:  {prep['stats']['n_trainable']}")
    print(f"  GRPO-eligible:     {n_grpo_groups} (size >= {args.group_size})")
    print(f"  DPO pairs:         {n_dpo_pairs}")
    print(f"  By signal_type:    {prep['stats']['by_signal_type']}")
    print(f"  By group size:     {prep['stats']['by_size_bucket']}")
    print(f"  GRPO jsonl:        {prep['grpo_path']}")
    print(f"  DPO jsonl:         {prep['dpo_path']}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[dry-run] not training. Sample of first 2 DPO pairs:")
        for p in prep["dpo_pairs"][:2]:
            print(f"  prompt:   {p['prompt'][:80]!r}")
            print(f"  chosen:   {p['chosen'][:80]!r} (r={p['chosen_reward']:.2f})")
            print(f"  rejected: {p['rejected'][:80]!r} (r={p['rejected_reward']:.2f})")
            print()
        return 0

    # Decide GRPO vs DPO fallback
    use_grpo = (
        n_grpo_groups >= MIN_GRPO_GROUPS
        and not args.force_dpo
        and (args.force_grpo or _grpo_available())
    )
    if args.force_grpo and not _grpo_available():
        logger.error(
            "--force-grpo specified but trl/Unsloth GRPOTrainer not importable"
        )
        return 3

    if use_grpo:
        logger.info(
            "[grpo_train] running GRPO path (n_groups=%d, threshold=%d)",
            n_grpo_groups, MIN_GRPO_GROUPS,
        )
        adapter_path = train_grpo(prep, args)
    else:
        logger.info(
            "[grpo_train] running DPO fallback (grpo_groups=%d < %d, or trl unavailable)",
            n_grpo_groups, MIN_GRPO_GROUPS,
        )
        adapter_path = train_dpo_fallback(prep, args)

    # Mark consumed unless explicitly disabled
    if not args.no_mark_consumed:
        from app.core import rlvr
        consumed_ids = []
        for g in prep["grpo_eligible" if use_grpo else "groups"]:
            consumed_ids.extend(g.signal_ids)
        n = rlvr.mark_consumed(consumed_ids)
        logger.info("[grpo_train] marked %d signals as consumed_for_training=1", n)

    # GGUF export
    if args.export_gguf:
        logger.info("[grpo_train] export_gguf requested — running export step")
        sys.path.insert(0, str(_HERE))
        try:
            from finetune import export_gguf
            export_gguf(adapter_path, args.model, str(Path(args.output_dir)))
        except Exception as e:
            logger.error("[grpo_train] GGUF export failed: %s", e)
            return 4

    print(f"\nAdapter saved: {adapter_path}")
    print("Next: run scripts/finetune_auto.py --eval-only to A/B vs base.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
