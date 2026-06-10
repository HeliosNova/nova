---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
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

# Nova-FT — DPO Fine-Tuned Qwen3.5-9B (experimental)

A small-data DPO (Direct Preference Optimization) fine-tune of [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), built from real user corrections collected by [Nova](https://github.com/HeliosNova/nova), a sovereign personal AI assistant. **This adapter is an experiment, not Nova's primary learning mechanism** — see *Evaluation* below.

## What This Model Is

Nova learns primarily by **memory**: when you correct it, it stores a lesson and a knowledge-graph fact that are retrieved into the prompt on future queries (durable, in-context learning). Separately and experimentally, it can also export `{query, chosen, rejected}` pairs and run a local DPO fine-tune — this model is the result of that optional path.

It is a *lightweight behavioral alignment* on a small correction set, **not** a capability upgrade: in independent, position-swapped A/B evaluation it performs **on par with the base model** (see *Evaluation*). Use it as a drop-in Qwen3.5-9B; its real value is operating inside the Nova system, where the memory loop does the learning.

## Training Details

| Parameter | Value |
|-----------|-------|
| **Base model** | `Qwen/Qwen3.5-9B` |
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

**Result (honest):** under an *independent, different-family* judge (e.g. Llama-3.1-8B), position-swapped across both A/B orders and scored on four dimensions, Nova-FT and the base Qwen3.5-9B come out **statistically tied** (≈8/10 ties, near-zero average preference). The fine-tune does **not** beat its base, so by the deploy rule above it is not promoted on capability grounds. This matches the literature: RAG/memory beats fine-tuning for factual recall, and small models tend to degrade under small-data tuning. Nova's actual learning is the retrieval-based **memory loop**, validated separately by the `memory-learning` eval.

## Intended Use

This model is designed to run as the inference backend for a Nova instance via Ollama. It can also be used as a general-purpose Qwen3.5-9B with improved factual accuracy and reduced verbosity.

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

- Trained on a few hundred preference pairs — a lightweight alignment, not a fundamental capability change
- In independent A/B evaluation it **ties** (does not beat) the base model — prefer the base unless you specifically want Nova's behavioral nudges
- Training data reflects one user's corrections and preferences — may not generalize
- Base model limitations (Qwen3.5-9B) still apply: knowledge cutoff, potential hallucinations, language biases
- Best results when used within the Nova system, where the memory loop (lessons + knowledge graph) does the real learning

## License

This model inherits the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) from the base Qwen3.5-9B model. The Nova framework that produced the training data is licensed under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html).

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
- [Base Model: Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
