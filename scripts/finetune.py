"""Nova Fine-Tuning Pipeline — DPO with Unsloth + QLoRA.

This script fine-tunes Qwen3.5:27b using Direct Preference Optimization (DPO)
on corrections Nova has collected. It produces a LoRA adapter that makes
Nova permanently better at the things its owner corrected.

REQUIREMENTS:
    - RTX 3090 (24GB VRAM) or better
    - CUDA 12.x
    - Unsloth + TRL installed (see requirements-finetune.txt)
    - Stop Ollama first: `docker compose stop ollama` (frees 17GB VRAM)

USAGE:
    python scripts/finetune.py                    # Train from default JSONL
    python scripts/finetune.py --data path.jsonl  # Custom data path
    python scripts/finetune.py --export-gguf      # Also export to GGUF for Ollama
    python scripts/finetune.py --dry-run          # Show data stats, don't train

DATA FORMAT (training_data.jsonl):
    {"query": "...", "chosen": "...", "rejected": "...", "timestamp": "..."}

OUTPUT:
    data/finetune/adapter/     — LoRA adapter (safetensors)
    data/finetune/merged/      — Merged model (if --export-gguf)
    data/finetune/nova-ft.gguf — GGUF for Ollama (if --export-gguf)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Defaults (configurable via env vars or CLI args)
_NOVA_DATA_DIR = os.environ.get("NOVA_DATA_DIR", "/data")
DEFAULT_DATA_PATH = os.getenv("TRAINING_DATA_PATH", os.path.join(_NOVA_DATA_DIR, "training_data.jsonl"))
DEFAULT_OUTPUT_DIR = os.getenv("FINETUNE_OUTPUT_DIR", os.path.join(_NOVA_DATA_DIR, "finetune"))
DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LORA_RANK = 16
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LR = 5e-5
MIN_TRAINING_PAIRS = 10  # Minimum pairs before training is worthwhile


def load_training_data(path: str) -> list[dict]:
    """Load DPO training pairs from JSONL file."""
    data = []
    p = Path(path)
    if not p.exists():
        logger.error("Training data not found: %s", path)
        return data

    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: invalid JSON: %s", i, e)
                continue

            # Validate required fields
            query = entry.get("query", "").strip()
            chosen = entry.get("chosen", "").strip()
            rejected = entry.get("rejected", "").strip()

            if not query or not chosen:
                logger.warning("Line %d: missing query or chosen answer, skipping", i)
                continue

            data.append({
                "prompt": query,
                "chosen": chosen,
                "rejected": rejected or "[No response]",
            })

    return data


def show_data_stats(data: list[dict]) -> None:
    """Print statistics about the training data."""
    print(f"\n{'='*60}")
    print(f"Training Data Statistics")
    print(f"{'='*60}")
    print(f"Total valid pairs: {len(data)}")

    if not data:
        print("No training data available!")
        return

    avg_prompt_len = sum(len(d["prompt"]) for d in data) / len(data)
    avg_chosen_len = sum(len(d["chosen"]) for d in data) / len(data)
    avg_rejected_len = sum(len(d["rejected"]) for d in data) / len(data)

    print(f"Avg prompt length:   {avg_prompt_len:.0f} chars")
    print(f"Avg chosen length:   {avg_chosen_len:.0f} chars")
    print(f"Avg rejected length: {avg_rejected_len:.0f} chars")

    print(f"\nSample entries:")
    for i, d in enumerate(data[:3], 1):
        print(f"\n  [{i}] Prompt:   {d['prompt'][:80]}")
        print(f"      Chosen:   {d['chosen'][:80]}")
        print(f"      Rejected: {d['rejected'][:80]}")

    print(f"\n{'='*60}")

    if len(data) < MIN_TRAINING_PAIRS:
        print(f"\nWARNING: Only {len(data)} pairs. Recommend at least {MIN_TRAINING_PAIRS}.")
        print("Continue collecting corrections before fine-tuning.\n")


def train(
    data: list[dict],
    *,
    model_name: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    lora_rank: int = DEFAULT_LORA_RANK,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    learning_rate: float = DEFAULT_LR,
) -> str:
    """Run DPO fine-tuning with QLoRA (vanilla HuggingFace stack).

    Uses transformers + PEFT + TRL directly (no Unsloth).
    Returns path to the saved adapter.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    # --- Step 1: Load model with 4-bit quantization ---
    logger.info("Loading %s with 4-bit quantization...", model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Step 2: Apply LoRA ---
    logger.info("Applying LoRA (rank=%d)...", lora_rank)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Step 3: Prepare dataset ---
    logger.info("Preparing DPO dataset (%d pairs)...", len(data))
    dataset = Dataset.from_list(data)

    # --- Step 4: DPO Training ---
    logger.info("Starting DPO training...")
    logger.info(
        "  Epochs: %d, Batch: %d, Grad accum: %d, LR: %s",
        epochs, batch_size, grad_accum, learning_rate,
    )

    training_args = DPOConfig(
        output_dir=adapter_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        optim="adamw_8bit",
        warmup_steps=2,
        logging_steps=1,
        save_strategy="epoch",
        max_length=max_seq_length,
        # max_prompt_length removed in TRL 0.16+ (auto-calculated)
        beta=0.1,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    logger.info("Training complete! Loss: %.4f", train_result.training_loss)

    # --- Step 5: Save LoRA adapter ---
    logger.info("Saving LoRA adapter to %s", adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save training metadata
    meta = {
        "model": model_name,
        "lora_rank": lora_rank,
        "training_pairs": len(data),
        "epochs": epochs,
        "final_loss": train_result.training_loss,
        "trained_at": datetime.now().isoformat(),
    }
    with open(os.path.join(adapter_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Adapter saved successfully!")
    return adapter_dir


def export_gguf(adapter_dir: str, output_dir: str, model_name: str = DEFAULT_MODEL) -> str:
    """Merge LoRA adapter and export to GGUF format for Ollama.

    Returns path to the GGUF file.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise RuntimeError("Unsloth not installed. Run: pip install -r scripts/requirements-finetune.txt")

    merged_dir = os.path.join(output_dir, "merged")
    gguf_path = os.path.join(output_dir, "nova-ft.gguf")

    logger.info("Loading base model + adapter for merge...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    logger.info("Exporting to GGUF (Q4_K_M quantization)...")
    model.save_pretrained_gguf(
        merged_dir,
        tokenizer,
        quantization_method="q4_k_m",
    )

    # The GGUF file will be in merged_dir with a standard name
    gguf_files = list(Path(merged_dir).glob("*.gguf"))
    if gguf_files:
        actual_path = str(gguf_files[0])
        logger.info("GGUF exported: %s", actual_path)

        # Create Ollama Modelfile
        modelfile_path = os.path.join(output_dir, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(f'FROM {actual_path}\n')
            f.write('TEMPLATE {{ .Prompt }}\n')
            f.write('RENDERER qwen3.5\n')
            f.write('PARSER qwen3.5\n')
            f.write('PARAMETER temperature 0.7\n')
            f.write('PARAMETER num_predict 2000\n')
            f.write('PARAMETER num_ctx 4096\n')
        logger.info("Ollama Modelfile created: %s", modelfile_path)
        logger.info(
            "To register with Ollama:\n"
            "  ollama create nova-ft -f %s",
            modelfile_path,
        )
        return actual_path

    logger.warning("No GGUF file found in %s", merged_dir)
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Nova Fine-Tuning Pipeline — DPO with Unsloth + QLoRA",
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA_PATH,
        help=f"Path to training_data.jsonl (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Base model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--rank", type=int, default=DEFAULT_LORA_RANK,
        help=f"LoRA rank (default: {DEFAULT_LORA_RANK})",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--export-gguf", action="store_true",
        help="Also export merged model to GGUF for Ollama",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show data stats only, don't train",
    )
    args = parser.parse_args()

    # Load data
    data = load_training_data(args.data)
    show_data_stats(data)

    if args.dry_run:
        return

    if len(data) < MIN_TRAINING_PAIRS:
        logger.warning(
            "Only %d training pairs. Need at least %d for meaningful fine-tuning.",
            len(data), MIN_TRAINING_PAIRS,
        )
        response = input(f"Continue anyway? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    # Check VRAM availability
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = torch.cuda.mem_get_info()[0] / 1e9
            logger.info("GPU: %s (%.1f GB total, %.1f GB free)", torch.cuda.get_device_name(0), vram, free)
            if free < 18:
                logger.warning(
                    "Only %.1f GB free VRAM. Ensure Ollama is stopped!\n"
                    "  docker compose stop ollama", free
                )
        else:
            raise RuntimeError("No CUDA GPU found! Fine-tuning requires an NVIDIA GPU.")
    except (ImportError, AttributeError):
        logger.warning("PyTorch not installed — can't check VRAM")

    # Train
    adapter_dir = train(
        data,
        model_name=args.model,
        output_dir=args.output,
        lora_rank=args.rank,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    print(f"\nLoRA adapter saved to: {adapter_dir}")

    # Optional GGUF export
    if args.export_gguf:
        gguf_path = export_gguf(adapter_dir, args.output, model_name=args.model)
        if gguf_path:
            print(f"GGUF exported to: {gguf_path}")
            print(f"\nTo use with Ollama:")
            print(f"  1. docker compose start ollama")
            print(f"  2. ollama create nova-ft -f {args.output}/Modelfile")
            print(f"  3. Update .env: LLM_MODEL=nova-ft")
            print(f"  4. docker compose restart nova")

    print("\nDone!")


if __name__ == "__main__":
    main()
