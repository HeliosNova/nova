"""MiniCheck entailment sidecar — CPU-only claim-verification service (#48).

Serves lytang/MiniCheck-Flan-T5-Large (MIT, <1B params, GPT-4-level on
LLM-AggreFact) via the official `minicheck` package so the input format is
exactly what the model was trained on. CPU-only by design: the GPU is fully
committed to the 27B synthesis model + the owner's 3 GiB reserve.

POST /check_batch {"pairs": [{"doc": "...", "claim": "..."}, ...]}
  -> {"results": [{"supported": bool, "prob": float}, ...]}
GET /health -> {"ok": true, "loaded": bool}

The model lazy-loads on first request (weights cached in the /cache volume so
restarts don't re-download). Single worker: scoring is CPU-bound and callers
(nova-app's entailment gate) are batched, bounded, and fail-open.
"""
from __future__ import annotations

import logging
import os
import threading

from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minicheck")

app = FastAPI()
_scorer = None
_lock = threading.Lock()

_MAX_DOC_CHARS = 8000
_MAX_CLAIM_CHARS = 1200
_MAX_BATCH = 64

# Scoring serialization (2026-08-26). FastAPI runs these sync endpoints in a
# THREADPOOL, so concurrent digest/eval batches interleaved on the CPU and
# thrashed torch inference — under that contention nova-app's 12-pair chat
# entail batch blew its 60s deadline and failed open 27x/48h. One lock
# serializes inference (fastest on CPU anyway), and large batches release it
# between sub-chunks so a small waiting batch slips in after ≤8 items
# instead of behind a whole 64-pair digest batch.
_score_lock = threading.Lock()
_SUBCHUNK = 8


def _score_serialized(scorer, docs: list[str], claims: list[str]):
    labels_all: list = []
    probs_all: list = []
    for j in range(0, len(docs), _SUBCHUNK):
        with _score_lock:
            out = scorer.score(docs=docs[j:j + _SUBCHUNK],
                               claims=claims[j:j + _SUBCHUNK])
        labels_all.extend(out[0])
        probs_all.extend(out[1])
    return labels_all, probs_all


def _get_scorer():
    global _scorer
    with _lock:
        if _scorer is None:
            from minicheck.minicheck import MiniCheck
            model = os.getenv("MINICHECK_MODEL", "flan-t5-large")
            logger.info("loading MiniCheck model %s (CPU)…", model)
            _scorer = MiniCheck(model_name=model, cache_dir="/cache")
            logger.info("MiniCheck loaded")
        return _scorer


class Pair(BaseModel):
    doc: str = Field(max_length=200_000)
    claim: str = Field(max_length=10_000)


class Batch(BaseModel):
    pairs: list[Pair] = Field(max_length=_MAX_BATCH)


@app.get("/health")
def health():
    return {"ok": True, "loaded": _scorer is not None}


@app.post("/check_batch")
def check_batch(batch: Batch):
    if not batch.pairs:
        return {"results": []}
    scorer = _get_scorer()
    docs = [p.doc[:_MAX_DOC_CHARS] for p in batch.pairs]
    claims = [p.claim[:_MAX_CLAIM_CHARS] for p in batch.pairs]
    labels, probs = _score_serialized(scorer, docs, claims)
    return {"results": [{"supported": bool(int(l)), "prob": float(p)}
                        for l, p in zip(labels, probs)]}
