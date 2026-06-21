#!/usr/bin/env bash
# Nova Weekly Fine-Tune Pipeline
# Scheduled via Windows Task Scheduler: 11 PM Sunday Pacific
#
# Orchestrates: readiness check → stop Ollama → DPO train → GGUF export → deploy
set -euo pipefail

NOVA_DIR="C:/Users/sysadmin/Desktop/Helios Project/nova_"
LOG_DIR="$NOVA_DIR/logs"
LOG_FILE="$LOG_DIR/finetune_$(date +%Y%m%d_%H%M%S).log"
MIN_NEW_PAIRS=10
COMPOSE_CMD="docker compose -f $NOVA_DIR/docker-compose.yml"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Nova Weekly Fine-Tune — $(date) ==="

# ---------------------------------------------------------------
# Step 1: Check readiness
# ---------------------------------------------------------------
echo "[Step 1] Checking readiness..."
export MSYS_NO_PATHCONV=1

NEW_PAIRS=$(docker exec nova-app python3 -c "
import json
from pathlib import Path
total = 0
path = Path('/data/training_data.jsonl')
if path.exists():
    for line in open(path):
        line = line.strip()
        if not line: continue
        try:
            e = json.loads(line)
            if e.get('query','').strip() and e.get('chosen','').strip():
                total += 1
        except: pass
history = Path('/data/finetune/run_history.json')
last = 0
if history.exists():
    try:
        h = json.loads(history.read_text())
        if h: last = h[-1].get('training_pairs', 0)
    except: pass
print(total - last)
")

echo "  New pairs since last run: $NEW_PAIRS"
if [ "$NEW_PAIRS" -lt "$MIN_NEW_PAIRS" ]; then
    echo "  Not enough new pairs (need $MIN_NEW_PAIRS). Skipping."
    echo "=== Skipped at $(date) ==="
    exit 0
fi
echo "  Proceeding with training."

# ---------------------------------------------------------------
# Step 2: Stop Ollama to free VRAM
# ---------------------------------------------------------------
echo "[Step 2] Stopping Ollama..."
$COMPOSE_CMD stop ollama
sleep 5
echo "  VRAM freed: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo 'unknown')"

# ---------------------------------------------------------------
# Step 3: DPO training (vanilla HuggingFace, ~3.5h)
# ---------------------------------------------------------------
echo "[Step 3] Running DPO training..."
docker run --rm --gpus all \
  -v nova__nova_data:/data \
  -v "$NOVA_DIR/scripts:/app/scripts" \
  -e HF_HOME=/data/hf_cache \
  nova-finetune \
  python /app/scripts/finetune.py \
    --data /data/training_data.jsonl \
    --output /data/finetune

TRAIN_EXIT=$?
if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "  ERROR: Training failed (exit $TRAIN_EXIT)"
    $COMPOSE_CMD start ollama
    echo "=== Failed at $(date) ==="
    exit 1
fi
echo "  Training complete."

# ---------------------------------------------------------------
# Step 4: GGUF export via Unsloth (~30min)
# ---------------------------------------------------------------
echo "[Step 4] Exporting to GGUF..."
docker run --rm --gpus all \
  --entrypoint bash \
  --user root \
  -v nova__nova_data:/data \
  -e HF_HOME=/data/hf_cache \
  unsloth/unsloth \
  -c 'pip install -q "transformers>=5.2.0" 2>&1 | tail -1
python3 -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=\"/data/finetune/adapter\",
    max_seq_length=2048,
    load_in_4bit=True,
)
model.save_pretrained_gguf(
    \"/data/finetune/gguf\",
    tokenizer,
    quantization_method=\"q4_k_m\",
)
print(\"GGUF export complete\")
"'

EXPORT_EXIT=$?
if [ "$EXPORT_EXIT" -ne 0 ]; then
    echo "  ERROR: GGUF export failed (exit $EXPORT_EXIT)"
    $COMPOSE_CMD start ollama
    echo "=== Failed at $(date) ==="
    exit 1
fi

# Find the GGUF file
GGUF_FILE=$(docker run --rm -v nova__nova_data:/data alpine find /data/finetune/gguf_gguf -name "*.Q4_K_M.gguf" 2>/dev/null | head -1)
if [ -z "$GGUF_FILE" ]; then
    echo "  ERROR: No Q4_K_M GGUF file found"
    $COMPOSE_CMD start ollama
    exit 1
fi
echo "  GGUF ready: $GGUF_FILE"

# ---------------------------------------------------------------
# Step 5: Deploy — start Ollama + register model
# ---------------------------------------------------------------
echo "[Step 5] Deploying fine-tuned model..."
# Copy GGUF to Ollama volume
docker run --rm -v nova__nova_data:/data -v nova__ollama_data:/ollama alpine \
  cp "$GGUF_FILE" /ollama/nova-ft.gguf

$COMPOSE_CMD start ollama
sleep 15

# Create Modelfile and register
docker exec nova-ollama sh -c "
cat > /tmp/Modelfile << 'EOF'
FROM /root/.ollama/nova-ft.gguf
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5
EOF
ollama create nova-ft -f /tmp/Modelfile
"

REGISTER_EXIT=$?
if [ "$REGISTER_EXIT" -ne 0 ]; then
    echo "  ERROR: Model registration failed"
    echo "=== Failed at $(date) ==="
    exit 1
fi
echo "  Model 'nova-ft' registered."

# ---------------------------------------------------------------
# Step 6: Record run metadata
# ---------------------------------------------------------------
echo "[Step 6] Recording run..."
docker exec nova-app python3 -c "
import json
from datetime import datetime
from pathlib import Path
history_path = Path('/data/finetune/run_history.json')
history = []
if history_path.exists():
    try: history = json.loads(history_path.read_text())
    except: pass
total = 0
for line in open('/data/training_data.jsonl'):
    line = line.strip()
    if not line: continue
    try:
        e = json.loads(line)
        if e.get('query','').strip() and e.get('chosen','').strip():
            total += 1
    except: pass
history.append({
    'started_at': datetime.now().isoformat(),
    'completed_at': datetime.now().isoformat(),
    'status': 'deployed',
    'training_pairs': total,
    'automated': True,
})
history_path.parent.mkdir(parents=True, exist_ok=True)
history_path.write_text(json.dumps(history, indent=2))
print(f'Recorded: {total} training pairs')
"

echo "=== Nova Weekly Fine-Tune COMPLETE — $(date) ==="
