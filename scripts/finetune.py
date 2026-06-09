"""Nova Fine-Tuning Pipeline — DPO with Unsloth + QLoRA.

EXPERIMENTAL — off by default. This script runs a local DPO (Direct Preference
Optimization) fine-tune on the corrections Nova has collected (default base:
Qwen3.5-9B), producing a LoRA adapter. It is NOT how Nova learns facts: in honest
cross-family A/B evals these small-data fine-tunes tie the base model, and Nova's
production "learning" is the in-context memory loop (lessons + temporal KG). Use
this only for style/behavior experiments, and only deploy a result that wins an
A/B under an INDEPENDENT (different-family) judge — see scripts/eval_harness.py
and the deploy gate in finetune_auto.py.

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
import subprocess
import sys

# Kill any existing finetune processes to prevent RAM double-booking
if sys.platform == "win32":
    _my_pid = os.getpid()
    try:
        out = subprocess.check_output(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"], text=True)
        for line in out.splitlines()[1:]:
            parts = line.strip('"').split('","')
            if len(parts) >= 2:
                pid = int(parts[1])
                if pid != _my_pid:
                    try:
                        mem_kb = int(parts[4].replace(",", "").replace(" K", "").replace('"', ''))
                    except (ValueError, IndexError):
                        mem_kb = 0
                    # Kill any python process using >2GB RAM (likely a previous training)
                    if mem_kb > 2_000_000:
                        print(f"Killing old python process PID {pid} ({mem_kb // 1024}MB RAM)")
                        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    except Exception:
        pass  # Non-critical cleanup

# Fix CUDA fragmentation. On Linux, expandable_segments lets the allocator
# grow the address space dynamically and avoids fragmentation when running
# at the VRAM ceiling — critical for 9B fits in 24GB. Windows doesn't
# support expandable_segments; falls back to max_split_size_mb only.
if sys.platform == "win32":
    # 27B QLoRA OOM'd at max_split_size=128 with 3.68GB free but no contiguous 170MB block.
    # Lower split size = more granular allocation, less fragmentation. Windows has no
    # expandable_segments equivalent, so this is the only fragmentation lever.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
from datetime import datetime
from pathlib import Path

# Force unbuffered output so progress is visible in real-time
os.environ["PYTHONUNBUFFERED"] = "1"

# Log to both console AND a progress file
_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "finetune_progress.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger(__name__)
logger.info("=== Fine-tune log: %s ===", _LOG_FILE)

# Defaults (configurable via env vars or CLI args)
_NOVA_DATA_DIR = os.environ.get("NOVA_DATA_DIR", "/data")
DEFAULT_DATA_PATH = os.getenv("TRAINING_DATA_PATH", os.path.join(_NOVA_DATA_DIR, "training_data.jsonl"))
DEFAULT_OUTPUT_DIR = os.getenv("FINETUNE_OUTPUT_DIR", os.path.join(_NOVA_DATA_DIR, "finetune"))
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"  # Switched from Qwen3.5-27B after the converted Qwen3.5 GGUF failed to load in Ollama 0.17.5 (known qwen3next loader bug). Qwen3.6 architecture works in user's Ollama (qwen3.6:27b is already loaded and tested).
DEFAULT_MAX_SEQ_LENGTH = 192  # 2026-05-20: DPO processes chosen+rejected (~2x SFT activation memory). 192 keeps 27B QLoRA-DPO in 24GB and still covers prompt+rejected (~155 tok). SFT alone fit 256.
DEFAULT_LORA_RANK = 16  # 2026-05-20: raised from 4 for DPO. Attn-only adapter memory is trivial even at 16 (~21M params); rank drives adapter expressiveness, and the SFT-rank-4 run A/B-tied v16.
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8  # Effective batch 8 per the proven SFT config; larger than DPO's 2 because SFT has half the per-step memory.
DEFAULT_LR = 2e-4  # SFT-appropriate LR (DPO was 5e-7). Per the working config.
# trl 0.24.0 DPOTrainer/DPOConfig does NOT accept loss_type="simpo" — it raises
# `ValueError: Unknown loss type: ['simpo']. Should be one of [...]`. SimPO
# lives in CPOTrainer/CPOConfig (not DPO) as of trl 0.17+. To keep this
# pipeline working with stock DPOTrainer we default to "sigmoid" (the original
# DPO loss). Capability difference: standard DPO needs a reference model in
# VRAM (or precomputed log-probs — we do the latter). SimPO is reference-free
# and ~40% cheaper in VRAM — if we want that back, switch train() to CPOTrainer.
DEFAULT_LOSS_TYPE = "sigmoid"
DEFAULT_DPO_BETA = 0.1   # DPO temperature — standard value
DEFAULT_DPO_LR = 5e-6    # DPO LR is much lower than SFT's 2e-4; LoRA-DPO standard range 5e-6..1e-5
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
    loss_type: str = DEFAULT_LOSS_TYPE,
    use_dpo: bool | None = None,
) -> str:
    """Run QLoRA fine-tuning — DPO or SFT.

    DPO (use_dpo=True, or env FINETUNE_USE_DPO=true): trains on chosen vs
    rejected preference pairs via DPOTrainer. Contrastive signal, ~2x the
    activation memory of SFT (processes both responses). Needs the smaller
    max_seq (192) to fit 27B on 24GB. Reference model is the base with the
    LoRA adapter disabled (ref_model=None) — no second model in VRAM.

    SFT (default): trains only on the chosen response. Half the activation
    memory; loses the contrastive signal. Fit 27B on 24GB at max_seq 256.
    """
    if use_dpo is None:
        use_dpo = os.getenv("FINETUNE_USE_DPO", "false").strip().lower() == "true"
    # IMPORTANT: Unsloth must be imported before torch to install its patches.
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    # --- Step 1: Load model via Unsloth (4-bit, fused kernels for memory efficiency) ---
    logger.info("Loading %s via Unsloth FastLanguageModel (4-bit, max_seq=%d)...", model_name, max_seq_length)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Step 2: Apply LoRA via Unsloth (full target_modules, rank-16) ---
    logger.info("Applying Unsloth LoRA (rank=%d, alpha=%d, attn-only)...", lora_rank, lora_rank * 2)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        # Attn-only modules: dropped MLP (gate_proj/up_proj/down_proj) after rank-16 full-MLP
        # OOM'd at step 2 in WSL2. 27B MLP intermediate dim is huge; excluding cuts adapter
        # trainable params by ~75%. Still teaches identity/format/tool-call from chosens.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    model.print_trainable_parameters()

    # Patch for TRL + Qwen3.5 compatibility
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # Progress callback — writes every step to log file so training is trackable
    from transformers import TrainerCallback
    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.global_step > 0:
                loss = logs.get("loss", logs.get("train_loss", "?"))
                pct = (state.global_step / state.max_steps * 100) if state.max_steps else 0
                eta_s = 0
                if state.global_step > 0 and hasattr(state, 'log_history') and len(state.log_history) > 1:
                    import time
                    elapsed = time.time() - self._start
                    eta_s = elapsed / state.global_step * (state.max_steps - state.global_step)
                msg = (f"Step {state.global_step}/{state.max_steps} ({pct:.0f}%) | "
                       f"loss={loss} | ETA: {eta_s/60:.0f}m")
                logger.info(msg)
        def on_train_begin(self, *a, **kw):
            import time; self._start = time.time()
            logger.info("Training started!")

    if use_dpo:
        # --- Step 3 (DPO): Format preference dataset (prompt / chosen / rejected) ---
        # DPO keeps the contrastive signal SFT throws away: it learns chosen > rejected.
        logger.info("Formatting DPO dataset (%d preference pairs)...", len(data))
        def _fmt_dpo(ex):
            return {
                "prompt": [{"role": "user", "content": ex["prompt"]}],
                "chosen": [{"role": "assistant", "content": ex["chosen"]}],
                "rejected": [{"role": "assistant", "content": ex["rejected"]}],
            }
        raw_ds = Dataset.from_list(data)
        dataset = raw_ds.map(_fmt_dpo, remove_columns=raw_ds.column_names)

        # --- Step 4 (DPO): DPOTrainer (ref_model=None → base w/ adapter disabled) ---
        from trl import DPOTrainer, DPOConfig
        try:
            from unsloth import PatchDPOTrainer
            PatchDPOTrainer()
        except Exception as e:  # noqa: BLE001 — Unsloth auto-patches in recent versions
            logger.info("PatchDPOTrainer unavailable (%s) — relying on Unsloth auto-patch", e)

        dpo_lr = float(os.getenv("FINETUNE_DPO_LR", str(DEFAULT_DPO_LR)))
        dpo_beta = float(os.getenv("FINETUNE_DPO_BETA", str(DEFAULT_DPO_BETA)))

        training_args = DPOConfig(
            output_dir=adapter_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=dpo_lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            optim="paged_adamw_8bit",
            logging_steps=1,
            save_strategy="no",
            beta=dpo_beta,
            max_length=max_seq_length,
            max_prompt_length=max_seq_length // 2,
            loss_type=loss_type,
            bf16=True,
            report_to="none",
            gradient_checkpointing=True,
        )
        logger.info("Starting DPO training...")
        logger.info(
            "  Epochs: %d, Batch: %d, Grad accum: %d, LR: %s, beta: %s, max_seq: %d, rank: %d",
            epochs, batch_size, grad_accum, dpo_lr, dpo_beta, max_seq_length, lora_rank,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[ProgressCallback()],
        )
    else:
        # --- Step 3 (SFT): Format dataset (chosen-only, applied chat template) ---
        logger.info("Formatting SFT dataset (%d pairs, chosen-only)...", len(data))
        def _fmt(ex):
            msgs = [
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["chosen"]},
            ]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            return {"text": text}
        raw_ds = Dataset.from_list(data)
        dataset = raw_ds.map(_fmt, remove_columns=raw_ds.column_names)

        # --- Step 4 (SFT): SFTTrainer ---
        training_args = SFTConfig(
            output_dir=adapter_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            optim="paged_adamw_8bit",  # paged variant spills optimizer state to CPU under VRAM pressure
            logging_steps=1,
            # save_strategy="no": skip per-epoch checkpoint saves (serializing 27B model VRAM-spikes beyond training peak — caused step-23 OOM even on WSL2 with expandable_segments).
            save_strategy="no",
            max_length=max_seq_length,
            dataset_text_field="text",
            packing=False,
            bf16=True,
            report_to="none",
            gradient_checkpointing=True,
        )
        logger.info("Starting SFT training...")
        logger.info(
            "  Epochs: %d, Batch: %d, Grad accum: %d, LR: %s, max_seq: %d",
            epochs, batch_size, grad_accum, learning_rate, max_seq_length,
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[ProgressCallback()],
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
        "method": "dpo" if use_dpo else "sft",
        "lora_rank": lora_rank,
        "max_seq_length": max_seq_length,
        "training_pairs": len(data),
        "epochs": epochs,
        "final_loss": train_result.training_loss,
        "trained_at": datetime.now().isoformat(),
    }
    with open(os.path.join(adapter_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Adapter saved successfully!")

    # Optional: TIES continual-merge with prior adapter to preserve old corrections
    # while applying the new ones. Activated by ENABLE_LORA_CONTINUAL_MERGE; falls
    # back to the unmerged adapter on any failure (no harm done).
    if os.getenv("ENABLE_LORA_CONTINUAL_MERGE", "false").lower() == "true":
        try:
            from scripts.lora_merge import ties_merge, get_prior_adapter_path
            prior = get_prior_adapter_path()
            if prior:
                alpha = float(os.getenv("LORA_MERGE_ALPHA", "0.5"))
                merged_dir = ties_merge(adapter_dir, prior, alpha=alpha)
                if merged_dir:
                    logger.info("[LoRA-merge] using merged adapter: %s", merged_dir)
                    return merged_dir
                else:
                    logger.info("[LoRA-merge] merge skipped — using new adapter as-is")
            else:
                logger.info("[LoRA-merge] no prior adapter — this run is the seed")
        except Exception as e:
            logger.warning("[LoRA-merge] failed: %s — using unmerged adapter", e)

    return adapter_dir


def _find_convert_hf_to_gguf() -> str:
    """Locate the vanilla convert_hf_to_gguf.py script.

    Unsloth ships a copy at ~/.unsloth/llama.cpp/. Prefer it over
    unsloth_convert_hf_to_gguf.py — the unsloth wrapper has the same
    stale-cache and dynamic-module-loading bugs that motivated this
    manual path in the first place.
    """
    candidates = [
        Path.home() / ".unsloth" / "llama.cpp" / "convert_hf_to_gguf.py",
        # Fallback: a system-installed llama.cpp clone
        Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise RuntimeError(
        "convert_hf_to_gguf.py not found in any known location. "
        "Expected at ~/.unsloth/llama.cpp/convert_hf_to_gguf.py. "
        "Reinstall Unsloth or clone llama.cpp."
    )


def export_gguf(adapter_dir: str, output_dir: str, model_name: str = DEFAULT_MODEL) -> str:
    """Merge LoRA adapter and export to GGUF format for Ollama.

    Uses a manual PEFT merge + vanilla convert_hf_to_gguf.py rather than
    Unsloth's `save_pretrained_gguf`, which has two recurring bugs hit
    on v9, v10, and v17:
      (a) stale-cache: reuses cached merged shards from a prior run keyed
          on the snapshot path, producing a "fresh" GGUF that is actually
          the previous run's weights;
      (b) dynamic-module loading: downloads a temp script that fails
          with `ModuleNotFoundError: No module named 'conversion'`.
    See CLAUDE.md "Fine-Tuning -> Two quirks" for context.

    Returns path to the Q8_0 GGUF file.
    """
    import subprocess
    import sys as _sys
    import shutil as _shutil

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = os.path.join(output_dir, "merged_manual")
    gguf_path = os.path.join(output_dir, "nova-ft.gguf")

    # Clean any stale prior-run output to avoid the same kind of cache
    # confusion that bit Unsloth.
    if os.path.isdir(merged_dir):
        logger.info("Removing prior merged_manual at %s", merged_dir)
        _shutil.rmtree(merged_dir)
    os.makedirs(merged_dir, exist_ok=True)
    if os.path.exists(gguf_path):
        os.remove(gguf_path)

    logger.info("Loading base model (%s) for manual merge", model_name)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # CPU merge — merge is mostly IO, frees GPU for next run
        low_cpu_mem_usage=True,
    )
    logger.info("Applying LoRA adapter from %s", adapter_dir)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    logger.info("Merging weights (merge_and_unload)")
    merged = peft_model.merge_and_unload()
    logger.info("Saving merged model to %s", merged_dir)
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")

    # Carry tokenizer + chat template alongside the merged weights so
    # convert_hf_to_gguf.py can pick them up.
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    tok.save_pretrained(merged_dir)
    src_tpl = os.path.join(adapter_dir, "chat_template.jinja")
    if os.path.exists(src_tpl):
        _shutil.copy2(src_tpl, os.path.join(merged_dir, "chat_template.jinja"))

    convert_script = _find_convert_hf_to_gguf()
    logger.info("Converting merged model to Q8_0 GGUF via %s", convert_script)
    cmd = [
        _sys.executable, convert_script,
        merged_dir,
        "--outtype", "q8_0",
        "--outfile", gguf_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("convert_hf_to_gguf.py failed (exit=%d)\nstdout:\n%s\nstderr:\n%s",
                     result.returncode, result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError(f"GGUF conversion failed: {result.stderr[-500:]}")

    if not os.path.exists(gguf_path):
        raise RuntimeError(f"GGUF conversion reported success but file missing: {gguf_path}")

    sz_gb = os.path.getsize(gguf_path) / (1024 ** 3)
    logger.info("GGUF exported: %s (%.2f GiB)", gguf_path, sz_gb)

    # Create Ollama Modelfile
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(f'FROM {gguf_path}\n')
        f.write('TEMPLATE {{ .Prompt }}\n')
        f.write('RENDERER qwen3.5\n')
        f.write('PARSER qwen3.5\n')
        f.write('PARAMETER temperature 0.7\n')
        f.write('PARAMETER num_predict 4000\n')
        f.write('PARAMETER num_ctx 32768\n')
    logger.info("Ollama Modelfile created: %s", modelfile_path)
    logger.info(
        "To register with Ollama:\n"
        "  ollama create nova-ft -f %s",
        modelfile_path,
    )
    return gguf_path


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
        "--loss-type", default=DEFAULT_LOSS_TYPE,
        choices=["sigmoid", "hinge", "ipo", "robust"],
        help=(
            f"DPO loss type (default: {DEFAULT_LOSS_TYPE}). "
            "Note: 'simpo' is NOT a valid DPOTrainer loss in trl 0.24+ — "
            "use CPOTrainer instead if SimPO is needed."
        ),
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
        loss_type=args.loss_type,
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
