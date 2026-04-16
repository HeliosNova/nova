# Nova Training Signal Export

## Quick Start

```bash
# Export all signal types
python scripts/export_training_signal.py --db data/nova.db --output training_export.jsonl

# Stats only (no file written)
python scripts/export_training_signal.py --db data/nova.db --stats

# Export specific types
python scripts/export_training_signal.py --db data/nova.db -o dpo_only.jsonl --types correction_dpo,reflexion_neg
```

## Signal Types

| Type | Source | Training Use | Format |
|------|--------|-------------|--------|
| `correction_dpo` | User corrections + lessons | DPO/ORPO | prompt + chosen + rejected |
| `reflexion_pos` | High-quality responses (score >= 0.8) | SFT | prompt + chosen |
| `reflexion_neg` | Low-quality responses (score < 0.4) | DPO negative | prompt + rejected |
| `skill_procedure` | Successful multi-step tool chains | SFT instruction-following | prompt + chosen (procedure steps) |
| `reasoning_trace` | Reflexion critiques with analysis | SFT chain-of-thought | prompt + chosen (reasoning) |
| `lesson_knowledge` | Learned lessons (high confidence) | SFT knowledge distillation | prompt + chosen |
| `tool_chain` | Multi-round conversation tool sequences | SFT tool use | prompt + chosen (tool chain + answer) |

## JSONL Format

Each line is a JSON object:

```json
{
  "signal_type": "correction_dpo",
  "prompt": "What is the capital of Australia?",
  "chosen": "The capital of Australia is Canberra...",
  "rejected": "The capital of Australia is Sydney.",
  "metadata": {
    "source_table": "lessons",
    "source_id": 42,
    "confidence": 0.9,
    "quality_score": 0.85,
    "tools_used": ["web_search"],
    "timestamp": "2026-04-15T10:30:00"
  }
}
```

## Training Recipes

### DPO (Direct Preference Optimization)
Best for: teaching the model to prefer correct answers over incorrect ones.

```bash
# Export DPO pairs
python scripts/export_training_signal.py --db data/nova.db -o dpo_data.jsonl --types correction_dpo,reflexion_neg

# Filter in Python
import json
dpo_pairs = []
with open("dpo_data.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["chosen"] and r["rejected"]:
            dpo_pairs.append(r)
```

**Unsloth:**
```python
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig

# Load your JSONL and map to DPO format:
# {"prompt": ..., "chosen": ..., "rejected": ...}
dataset = load_dataset("json", data_files="dpo_data.jsonl")
```

**Axolotl** (`config.yml`):
```yaml
datasets:
  - path: dpo_data.jsonl
    type: dpo
    field_map:
      prompt: prompt
      chosen: chosen
      rejected: rejected
```

### SFT (Supervised Fine-Tuning)
Best for: teaching tool use, reasoning, and knowledge.

```bash
python scripts/export_training_signal.py --db data/nova.db -o sft_data.jsonl --types reflexion_pos,skill_procedure,reasoning_trace,lesson_knowledge
```

**Unsloth:**
```python
# Map to chat format
dataset = [
    {"messages": [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r["chosen"]}
    ]}
    for r in records if r["chosen"]
]
```

### ORPO (Odds Ratio Preference Optimization)
Combines SFT + preference in one pass. Use all signal types.

```bash
python scripts/export_training_signal.py --db data/nova.db -o orpo_data.jsonl
```

## Metadata Filtering

Use metadata to filter by quality, recency, or source:

```python
import json

# Only high-confidence corrections
records = []
with open("training_export.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["signal_type"] == "correction_dpo" and r["metadata"].get("confidence", 0) >= 0.8:
            records.append(r)

# Only recent data (last 30 days)
from datetime import datetime, timedelta
cutoff = (datetime.now() - timedelta(days=30)).isoformat()
recent = [r for r in records if r["metadata"].get("timestamp", "") >= cutoff]
```

## Integration with Existing Pipeline

This export complements the existing `scripts/finetune.py` which reads from `training_data.jsonl`. The new exporter adds:

1. **Broader signal** — not just corrections, but also reflexions, skills, reasoning traces
2. **Metadata tags** — filter by signal type, quality score, confidence
3. **Framework-agnostic** — standard JSONL with prompt/chosen/rejected fields
4. **Reasoning traces** — teaches HOW to think, not just what to output
