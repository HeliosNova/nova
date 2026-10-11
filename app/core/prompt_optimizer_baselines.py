"""Baseline content for each self-modifiable prompt module.

These are the immutable reference versions seeded into prompt_modules on first
startup.  They mirror the current hardcoded constants exactly.

IMPORTANT: Do NOT import this module in a hot path — it imports from several
core modules and is only called during startup seeding or for drift checks.
"""

from __future__ import annotations


def _critique_baseline() -> str:
    from app.core.reflexion import _CRITIQUE_PROMPT
    return _CRITIQUE_PROMPT


def _extraction_baseline() -> str:
    from app.core.learning import _EXTRACTION_PROMPT
    return _EXTRACTION_PROMPT


def _merge_parallel_baseline() -> str:
    return (
        "Synthesize the parallel research findings ({roles}) into a "
        "single, direct, coherent answer. Original question: {query}"
    )


def _merge_sequential_baseline() -> str:
    return (
        "Synthesize the sequentially gathered findings ({roles}) into a "
        "complete, coherent answer. Original question: {query}"
    )


def _kg_extraction_baseline() -> str:
    return (
        "Extract factual (subject, predicate, object) triples from this Q&A.\n"
        "Use ONLY these predicates: {predicates}\n"
        "Return a JSON array. Max 5 triples. Only verifiable facts, not opinions.\n"
        "Rate each triple's confidence: 0.3 (uncertain/speculative) to 0.95 (well-established fact).\n"
        'Example: [{{"subject": "python", "predicate": "created_by", "object": "guido van rossum", "confidence": 0.9}}]\n\n'
        "Q: {query}\nA: {answer}"
    )


#: All baseline contents, keyed by module_name.
MODULE_BASELINES: dict[str, str] = {
    "critique_prompt": _critique_baseline(),
    "extraction_prompt": _extraction_baseline(),
    "merge_instruction_parallel": _merge_parallel_baseline(),
    "merge_instruction_sequential": _merge_sequential_baseline(),
    "kg_extraction_prompt": _kg_extraction_baseline(),
}
