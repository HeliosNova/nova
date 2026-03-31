---
license: apache-2.0
base_model: Qwen/Qwen3.5-27B
tags:
  - dpo
  - qlora
  - personal-ai
  - self-improving
  - nova
library_name: transformers
pipeline_tag: text-generation
language:
  - en
---

# Nova-FT — DPO Fine-Tuned Qwen3.5-27B

A DPO (Direct Preference Optimization) fine-tuned version of [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), trained on real user corrections collected by [Nova](https://github.com/HeliosNova/nova), a self-improving personal AI assistant.

## What This Model Is

Nova is a personal AI that learns from its mistakes. When you correct it, it extracts a structured lesson and generates a DPO training pair (`{query, chosen, rejected}`). When enough pairs accumulate, Nova fine-tunes its own base model using this data.

This model is the result of that pipeline — a Qwen3.5-27B that has been aligned to prefer correct, concise answers over the verbose, hedging, or incorrect responses it originally gave.

## Training Details

| Parameter | Value |
|-----------|-------|
| **Base model** | `Qwen/Qwen3.5-27B` |
| **Method** | DPO (Direct Preference Optimization) |
| **Quantization** | QLoRA (4-bit NF4) via Unsloth |
| **LoRA rank** | 16 |
| **LoRA alpha** | 32 |
| **Max sequence length** | 1,024 tokens |
| **Batch size** | 1 (gradient accumulation: 2, effective batch: 2) |
| **Learning rate** | 5e-5 |
| **Epochs** | 3 |
| **Training pairs** | 341 |
| **Hardware** | NVIDIA RTX 3090 (24GB VRAM) |
| **Framework** | Unsloth + TRL + Transformers |

### Training Data

The training data consists of 341 DPO pairs from two sources:

1. **Real corrections** — actual user corrections collected during Nova conversations. Each pair captures what the model said wrong (`rejected`) and what the correct response should have been (`chosen`).

2. **Curriculum pairs** — expert-crafted reasoning traces across 8 categories: factual accuracy, tool use, conciseness, correction handling, uncertainty acknowledgment, computation delegation, evidence grounding, and temporal awareness.

Data format (JSONL):
```json
{
  "query": "What's the capital of Australia?",
  "chosen": "The capital of Australia is Canberra.",
  "rejected": "The capital of Australia is Sydney.",
  "timestamp": "2026-03-15T14:23:01"
}
```

### Evaluation

Before deployment, the fine-tuned model is evaluated against the base model using an A/B evaluation harness:

- 10 holdout queries sampled before training (never seen during training)
- Both base and fine-tuned models generate responses to each query
- LLM-as-judge compares responses with randomized A/B ordering to prevent position bias
- Deployment only if fine-tuned model wins >50% with positive average preference score

## Intended Use

This model is designed to run as the inference backend for a Nova instance via Ollama. It can also be used as a general-purpose Qwen3.5-27B with improved factual accuracy and reduced verbosity.

**Primary use:** Personal AI assistant (question answering, tool use, conversation)

**Not intended for:** Code generation (use a coding-specific model), safety-critical applications, medical/legal advice

## How to Use

### With Ollama (recommended)
```bash
# If you have the GGUF file:
ollama create nova-ft -f Modelfile

# Then in Nova's .env:
LLM_MODEL=nova-ft
```

### With Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("HeliosNova/nova-ft")
tokenizer = AutoTokenizer.from_pretrained("HeliosNova/nova-ft")
```

### With Unsloth (for further fine-tuning)
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="HeliosNova/nova-ft",
    max_seq_length=1024,
    load_in_4bit=True,
)
```

## Limitations

- Trained on 341 pairs — this is a lightweight alignment, not a fundamental capability change
- Training data reflects one user's corrections and preferences — may not generalize to all use cases
- Base model limitations (Qwen3.5-27B) still apply: knowledge cutoff, potential hallucinations, language biases
- Best results when used within the Nova system where lessons and knowledge graph provide additional context

## License

This model inherits the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) from the base Qwen3.5-27B model. The Nova framework that produced the training data is licensed under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html).

## Citation

If you use this model or the Nova fine-tuning pipeline:

```bibtex
@software{nova2026,
  title={Nova: A Self-Improving Personal AI Assistant},
  url={https://github.com/HeliosNova/nova},
  year={2026},
}
```

## Links

- [Nova GitHub](https://github.com/HeliosNova/nova)
- [Base Model: Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
