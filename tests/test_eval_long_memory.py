"""The long-memory eval category seeds several facts in order (2026-09-23).

LongMemEval's abilities, on Nova's own memory: a knowledge UPDATE (a later
fact on a functional predicate replaces an earlier one), MULTI-SESSION
aggregation (several facts on one subject), TEMPORAL questions (a dated
earlier fact), and ABSTENTION (an unknown entity must not be invented). The
seeded path is the kg-retrieval before/seed/after runner, so a pass still
means the seeds CAUSED the answer; every seed is retired afterwards.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.monitors import eval_harness as eh
from app.monitors.eval_harness import EvalHarness, EvalTask

SEEDS = [
    {"subject": "Ilse Varga", "predicate": "lives_in", "object": "Tromso", "valid_from": "2026-03-01"},
    {"subject": "Ilse Varga", "predicate": "lives_in", "object": "Valletta", "valid_from": "2026-08-01"},
]


def _task(**kw):
    base = dict(id="lm_update", category="long-memory", query="Where does Ilse Varga live now?",
                assertions=[{"type": "answer_contains", "value": "Valletta"}], timeout=5,
                seed_facts=SEEDS)
    base.update(kw)
    return EvalTask(**base)


def _inv(text):
    return eh._Invocation(response_text=text, tools_invoked=[], skill_used=None, decomposed=False,
                          max_decomposition_depth=0, reflexion_score=None, latency_seconds=0.1,
                          error=None, timed_out=False)


@pytest.mark.asyncio
async def test_seeds_are_added_in_order_with_dates_and_all_retired():
    kg = SimpleNamespace(add_fact=AsyncMock(return_value=True), delete_fact=AsyncMock(return_value=True))
    h = object.__new__(EvalHarness)
    answers = iter([_inv("I have no record of Ilse Varga."), _inv("Ilse Varga lives in Valletta.")])
    with patch("app.core.brain.get_services", return_value=SimpleNamespace(kg=kg)), \
         patch.object(EvalHarness, "_invoke_brain", AsyncMock(side_effect=lambda *a, **k: next(answers))):
        res = await h._run_kg_task(_task())
    added = [(c.args, c.kwargs.get("valid_from")) for c in kg.add_fact.await_args_list]
    assert added == [(("Ilse Varga", "lives_in", "Tromso"), "2026-03-01"),
                     (("Ilse Varga", "lives_in", "Valletta"), "2026-08-01")]
    retired = [c.args for c in kg.delete_fact.await_args_list]
    # pre-clean + final clean, newest first each time
    assert retired[-2:] == [("Ilse Varga", "lives_in", "Valletta"), ("Ilse Varga", "lives_in", "Tromso")]
    assert res.passed and res.memory_caused_fix is True and res.category == "long-memory"


@pytest.mark.asyncio
async def test_a_single_seed_fact_still_works_the_old_way():
    kg = SimpleNamespace(add_fact=AsyncMock(return_value=True), delete_fact=AsyncMock(return_value=True))
    h = object.__new__(EvalHarness)
    answers = iter([_inv("unknown"), _inv("Mara Quill leads it.")])
    task = EvalTask(id="kg1", category="kg-retrieval", query="Who leads the Aetherion Guild?",
                    assertions=[{"type": "answer_contains", "value": "Quill"}], timeout=5,
                    seed_fact={"subject": "Aetherion Guild", "predicate": "led_by", "object": "Mara Quill"})
    with patch("app.core.brain.get_services", return_value=SimpleNamespace(kg=kg)), \
         patch.object(EvalHarness, "_invoke_brain", AsyncMock(side_effect=lambda *a, **k: next(answers))):
        res = await h._run_kg_task(task)
    assert kg.add_fact.await_count == 1 and kg.add_fact.await_args.kwargs.get("valid_from") is None
    assert res.passed


@pytest.mark.asyncio
async def test_a_seed_missing_its_object_is_a_setup_error_not_a_crash():
    kg = SimpleNamespace(add_fact=AsyncMock(), delete_fact=AsyncMock())
    h = object.__new__(EvalHarness)
    with patch("app.core.brain.get_services", return_value=SimpleNamespace(kg=kg)):
        res = await h._run_kg_task(_task(seed_facts=[{"subject": "X", "predicate": "lives_in"}]))
    assert res.error == "setup_error" and kg.add_fact.await_count == 0


def test_the_suite_carries_the_long_memory_category():
    from pathlib import Path
    suite = Path(eh.__file__).resolve().parents[2] / "evals" / "suite.yaml"
    if not suite.exists():
        pytest.skip("suite.yaml not found")
    tasks = EvalHarness(suite_path=str(suite)).load_suite()
    lm = [t for t in tasks if t.category == "long-memory"]
    assert len(lm) >= 4, "update, temporal, multi-session and abstention tasks"
    assert any(t.seed_facts and len(t.seed_facts) >= 2 for t in lm), "a knowledge-update task seeds two facts in order"
    assert any(not t.seed_facts and not t.seed_fact for t in lm), "an abstention task seeds nothing"
    for t in lm:
        for s in t.seed_facts:
            assert s.get("subject") and s.get("predicate") and s.get("object")
