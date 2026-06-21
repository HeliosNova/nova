#!/usr/bin/env bash
# Launcher for finetune_auto.py inside WSL2.
# - Sets HF_HOME to reuse the existing 52GB Windows-side HuggingFace cache (no re-download)
# - Sets NOVA_DATA_DIR to the F: project root so the script finds training_data_27b.jsonl
#   and writes adapter output to /mnt/f/Helios Project/nova_/finetune/
# - Sets PYTORCH_CUDA_ALLOC_CONF to use expandable_segments (Linux-only) which is the
#   fragmentation fix that's unavailable on Windows and was the root cause of the 8 OOMs.

set -euo pipefail

export HF_HOME="/home/sysadmin/hf_cache"
# Was /mnt/f/Helios Project/nova_/hf_cache but 9P interop caused 10 consecutive OOMs at training step 0-2
# (v9-v17 in run_history.json, 2026-05-17 to 2026-05-19). Moving to WSL-native ext4 (862G free) fixes it.
# C: is full so /mnt/c not an option.
export NOVA_DATA_DIR="/mnt/f/Helios Project/nova_"
export TRAINING_DATA_PATH="/mnt/f/Helios Project/nova_/training_data_27b.jsonl"
export FINETUNE_OUTPUT_DIR="/mnt/f/Helios Project/nova_/finetune_output_27b"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM="false"
# DPO mode — train on chosen-vs-rejected preference pairs (rank 16, max_seq 192).
export FINETUNE_USE_DPO="true"

VENV="$HOME/finetune_env"
SCRIPT="/mnt/f/Helios Project/nova_/scripts/finetune_auto.py"

echo "=== WSL2 fine-tune launcher ==="
echo "HF_HOME=$HF_HOME"
echo "NOVA_DATA_DIR=$NOVA_DATA_DIR"
echo "TRAINING_DATA_PATH=$TRAINING_DATA_PATH"
echo "FINETUNE_OUTPUT_DIR=$FINETUNE_OUTPUT_DIR"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "VENV=$VENV"
echo "SCRIPT=$SCRIPT"
echo ""

mkdir -p "$FINETUNE_OUTPUT_DIR"

exec "$VENV/bin/python" "$SCRIPT" --force --data "$TRAINING_DATA_PATH"
