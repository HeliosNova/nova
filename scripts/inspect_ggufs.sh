#!/usr/bin/env bash
# Inspect the GGUF metadata for the working base vs our broken v19.
# Run via PowerShell: wsl -d Ubuntu -- bash "/mnt/f/Helios Project/nova_/scripts/inspect_ggufs.sh"

PY=/home/sysadmin/finetune_env/bin/python
GGUF_PY="/home/sysadmin/llama.cpp/gguf-py"

echo "=== llama.cpp gguf-py available? ==="
ls "$GGUF_PY" 2>&1 | head -3

# Get blob SHAs for each model from Ollama
QWEN_SHA=$(docker exec nova-ollama sh -c "cat /root/.ollama/models/manifests/registry.ollama.ai/library/qwen3.6/27b 2>/dev/null | python3 -c 'import json,sys; m=json.load(sys.stdin); [print(l[\"digest\"]) for l in m[\"layers\"] if l[\"mediaType\"]==\"application/vnd.ollama.image.model\"]'" 2>&1)
V19_SHA=$(docker exec nova-ollama sh -c "cat /root/.ollama/models/manifests/registry.ollama.ai/library/nova-ft-v19-q8/latest 2>/dev/null | python3 -c 'import json,sys; m=json.load(sys.stdin); [print(l[\"digest\"]) for l in m[\"layers\"] if l[\"mediaType\"]==\"application/vnd.ollama.image.model\"]'" 2>&1)

echo "qwen3.6:27b blob: $QWEN_SHA"
echo "v19 blob: $V19_SHA"

# Dump GGUF metadata via docker exec
echo ""
echo "=== qwen3.6:27b GGUF header (working) ==="
docker exec nova-ollama sh -c "head -c 8000 /root/.ollama/models/blobs/$QWEN_SHA | strings -n 4 | grep -E 'general\\.architecture|block_count|attention|qwen|tokenizer|gguf' | head -30"

echo ""
echo "=== nova-ft-v19-q8 GGUF header (broken) ==="
docker exec nova-ollama sh -c "head -c 8000 /root/.ollama/models/blobs/$V19_SHA | strings -n 4 | grep -E 'general\\.architecture|block_count|attention|qwen|tokenizer|gguf' | head -30"
