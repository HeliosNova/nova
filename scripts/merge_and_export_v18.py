"""Merge checkpoint-23 LoRA into qwen3.5:27b base, export to GGUF Q8_0.

Salvage path after 13 OOMs in DPO/CPO/SFT training: attempt #1 saved a partial
adapter at checkpoint-23 (23 SFT steps, loss 1.97→1.22 ≈ 1 epoch). Use it.
"""
import os
import sys
import shutil
from pathlib import Path

# Must precede torch import per Unsloth recommendation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ.setdefault("HF_HOME", "/mnt/c/Users/sysadmin/.cache/huggingface")

from unsloth import FastLanguageModel
import torch

BASE = "unsloth/Qwen3.5-27B"
ADAPTER_DIR = "/mnt/f/Helios Project/nova_/finetune/adapter/checkpoint-23"
MERGED_DIR = "/mnt/f/Helios Project/nova_/finetune/v18_merged"
GGUF_DIR = "/mnt/f/Helios Project/nova_/finetune/v18_gguf"
MAX_SEQ = 512  # match training config

print("=" * 70)
print("Merge + GGUF export — v18 (27B SFT, checkpoint-23, partial 1-epoch)")
print("=" * 70)
print(f"BASE: {BASE}")
print(f"ADAPTER: {ADAPTER_DIR}")
print(f"MERGED: {MERGED_DIR}")
print(f"GGUF: {GGUF_DIR}")
print()

if not Path(ADAPTER_DIR).exists():
    print(f"FATAL: adapter dir not found: {ADAPTER_DIR}")
    sys.exit(1)

Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)
Path(GGUF_DIR).mkdir(parents=True, exist_ok=True)

print("Loading base model + adapter via Unsloth FastLanguageModel...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,  # Unsloth resolves adapter_config.json's base_model_name_or_path
    max_seq_length=MAX_SEQ,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)
print("Model + adapter loaded.")
print()

print("Saving merged FP16 model to disk (for GGUF conversion)...")
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit",  # full precision merge so GGUF Q8_0 quant is clean
)
print(f"Merged model saved to {MERGED_DIR}")
print()

print("Exporting GGUF Q8_0 quant via Unsloth (uses llama.cpp under the hood)...")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q8_0",
)
print(f"GGUF saved to {GGUF_DIR}")
print()

# List output files
print("Output files:")
for p in sorted(Path(GGUF_DIR).iterdir()):
    sz = p.stat().st_size
    print(f"  {p.name}  {sz/1e9:.2f} GB")
print()
print("DONE.")
