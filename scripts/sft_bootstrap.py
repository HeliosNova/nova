"""open-rs SFT pre-DPO bootstrap.

Runs one short SFT epoch on R1-style reasoning traces before the regular DPO
training step. Lifts the model's reasoning floor without RL infra by giving
it explicit chain-of-thought exemplars to mimic.

Source: github.com/knoveleng/open-rs (AAAI 2026), HuggingFace Mixture-of-Thoughts.

USAGE (called from scripts/finetune_auto.py before train()):
    if config.ENABLE_SFT_BOOTSTRAP:
        sft_adapter = run_sft_bootstrap(base_model_name, output_dir)
        # then DPO trains starting from sft_adapter, not the base model

The bootstrap is conservative by design:
- 1 epoch only (short, cheap, low overfit risk)
- Higher LR than DPO (2e-4 vs 5e-7) — SFT teaches new behavior, DPO only nudges
- Drops samples with chosen.length < 200 chars (no point SFT'ing on stub answers)

Defaults to using the project's existing DPO `chosen` field as the SFT target,
filtered to the subset with reasoning markers ("Step 1", "First,", "let's
break this down", numbered enumerations). For a more aggressive bootstrap,
the user can drop a `data/finetune/sft_data.jsonl` file with explicit traces.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


_REASONING_MARKERS = re.compile(
    r"\b(?:Step\s*\d+|First[,]?\s|Then[,]?\s|Next[,]?\s|Finally[,]?\s|"
    r"let'?s\s+(?:break|think|walk)|"
    r"^\s*\d+\.\s|^\s*[a-z]\.\s|"
    r"because|therefore|thus|since)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _is_reasoning_trace(text: str) -> bool:
    """Heuristic: at least 2 reasoning markers + length >= 200 chars."""
    if not text or len(text) < 200:
        return False
    return len(_REASONING_MARKERS.findall(text)) >= 2


def filter_reasoning_traces(dpo_data: list[dict]) -> list[dict]:
    """From DPO training pairs, pick only those whose `chosen` is a reasoning trace.

    Returns SFT records with `prompt` (= query) and `completion` (= chosen).
    """
    out: list[dict] = []
    for row in dpo_data:
        chosen = row.get("chosen", "")
        if _is_reasoning_trace(chosen):
            out.append({
                "prompt": row.get("query", row.get("prompt", "")),
                "completion": chosen,
            })
    return out


def load_sft_data(
    sft_data_path: str = "/data/finetune/sft_data.jsonl",
    fallback_dpo_path: str = "/data/finetune/training_data.jsonl",
) -> list[dict]:
    """Prefer an explicit sft_data.jsonl; fall back to filtering DPO data."""
    if os.path.exists(sft_data_path):
        records: list[dict] = []
        with open(sft_data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        logger.info("[SFT-bootstrap] loaded %d explicit SFT records", len(records))
        return records

    # Fallback: filter DPO pairs for reasoning traces
    if not os.path.exists(fallback_dpo_path):
        return []
    dpo_records: list[dict] = []
    with open(fallback_dpo_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dpo_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    sft_records = filter_reasoning_traces(dpo_records)
    logger.info(
        "[SFT-bootstrap] filtered %d reasoning traces from %d DPO pairs",
        len(sft_records), len(dpo_records),
    )
    return sft_records


def run_sft_bootstrap(
    base_model: str,
    output_dir: str,
    *,
    epochs: int = 1,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    max_seq_length: int = 1024,
    lora_rank: int = 16,
) -> str | None:
    """Run a single SFT epoch over reasoning traces. Returns path to SFT adapter or None.

    Caller should then load this adapter as the base for DPO training so DPO
    starts from a model that already has the reasoning behavior.
    """
    sft_data = load_sft_data()
    if len(sft_data) < 30:
        logger.info(
            "[SFT-bootstrap] only %d reasoning records — skipping (need >=30)",
            len(sft_data),
        )
        return None

    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as e:
        logger.warning("[SFT-bootstrap] training deps unavailable (%s) — skipping", e)
        return None

    sft_dir = os.path.join(output_dir, "sft_bootstrap")
    os.makedirs(sft_dir, exist_ok=True)

    logger.info(
        "[SFT-bootstrap] starting: %d records, %d epochs, lr=%.0e, lora_rank=%d",
        len(sft_data), epochs, learning_rate, lora_rank,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
    )

    dataset = Dataset.from_list([
        {"text": f"{r['prompt']}\n\n{r['completion']}"}
        for r in sft_data
    ])

    args = SFTConfig(
        output_dir=sft_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="no",      # we save adapter manually below
        bf16=True,
        max_seq_length=max_seq_length,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()
    logger.info("[SFT-bootstrap] complete — loss=%.4f", result.training_loss)

    model.save_pretrained(sft_dir)
    tokenizer.save_pretrained(sft_dir)
    return sft_dir
