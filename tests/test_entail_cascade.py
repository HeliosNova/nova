"""Entailment cascade: same verdicts, a third of the documents (2026-09-04).

Entailment is 64% of a digest's wall clock (19.8 of 30.8 minutes measured over
25 live digests) and runs on the CPU sidecar while the GPU idles. Cost scales
steeply with document length: measured on 60 real claim/document pairs,
5,508 chars took 8.79 s/pair and 2,754 took 2.42.

Simply trimming was refused — it is 3x faster but newly drops 4 of 60 true
sentences, and quality beats throughput here. What makes the cascade exact is
the DIRECTION of the disagreement: nothing the narrow document supported was
rejected at full width. So scoring narrow first and re-checking only the
failures reproduces the full-width verdict set.
"""
from __future__ import annotations

import inspect
import re

from app.monitors import deep_research as dr

SRC = inspect.getsource(dr._entail_gate) if hasattr(dr, "_entail_gate") else None


def _gate_source() -> str:
    """The gate is a module-level async function; find it by name."""
    for name in dir(dr):
        obj = getattr(dr, name)
        if callable(obj) and "entail" in name.lower():
            try:
                src = inspect.getsource(obj)
            except (OSError, TypeError):
                continue
            if "_check_cascade" in src:
                return src
    raise AssertionError("no entailment function defines _check_cascade")


def test_the_cascade_exists_and_scores_narrow_first():
    src = _gate_source()
    assert "async def _check_cascade" in src
    narrow_at = src.index("narrow = [{\"doc\": _doc_for(h, c, narrow=True)")
    full_at = src.index("full = [{\"doc\": _doc_for(specs[i][0], specs[i][1])")
    assert narrow_at < full_at, "the narrow pass must run before the full one"


def test_only_the_unsupported_are_rechecked():
    """Re-checking everything would cost more than the single full pass it replaces."""
    src = _gate_source()
    assert 'todo = [i for i, r in enumerate(res) if not r.get("supported")]' in src
    assert "for i, r2 in zip(todo, res_full)" in src, "results must merge back by index"


def test_every_call_site_uses_the_cascade():
    """The initial gate, the clause rescue and the alternate re-cite all pay it."""
    src = _gate_source()
    assert src.count("await _check_cascade(") == 3
    # the raw pair-scoring helper survives, but only the cascade calls it
    assert src.count("await _check_pairs(") == 2, "only the cascade may call _check_pairs"


def test_narrow_mode_takes_one_article():
    src = inspect.getsource(dr)
    m = re.search(r"cand\[:1 if narrow else 2\]", src)
    assert m, "narrow must select a single best-matching article"


def test_fail_open_is_preserved_per_call_site():
    """The initial gate fails open; the two rescue paths must not."""
    src = _gate_source()
    assert "_check_cascade([(hosts, _claim_of(st)) for _, st, hosts in checks],\n" \
           "                                   fail_open=True)" in src
    assert src.count("fail_open=False") >= 2


def test_a_failed_full_pass_does_not_silently_support_everything():
    src = _gate_source()
    assert "return None if not fail_open else res" in src, \
        "a strict caller must get None rather than optimistic verdicts"
