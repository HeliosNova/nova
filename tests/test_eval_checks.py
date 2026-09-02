"""Research-quality eval checks (audit 2026-09-01).

The nightly suite was 9/11 categories saturated at 100% for 7-46 runs, alerted
REGRESSION every night on two chronic chat tasks, and measured nothing about
digests, dossiers, forecasts or the KG. Now: 16 saturated tasks retired (one
canary per category kept), the fixture-collision paraphrase reworded, and
seven deterministic research-quality checks run every night without GPU.
"""
from __future__ import annotations

import pytest

from app.monitors import eval_checks
from app.monitors.eval_harness import EvalHarness, EvalTask


@pytest.mark.parametrize("name", sorted(eval_checks.CHECKS))
def test_each_check_passes_on_current_code(name):
    passed, detail = eval_checks.CHECKS[name]()
    assert passed, f"{name}: {detail}"


def test_suite_has_research_quality_tasks_and_no_saturated_filler():
    h = EvalHarness()
    tasks = h.load_suite()
    by_cat: dict[str, list[str]] = {}
    for t in tasks:
        by_cat.setdefault(t.category, []).append(t.id)
    checks = {t.check for t in tasks if t.check}
    assert set(eval_checks.CHECKS) <= checks, f"missing research-quality tasks: {set(eval_checks.CHECKS) - checks}"
    assert all(t.category == "research-quality" for t in tasks if t.check)
    retired = {"skill_match_crypto_price", "semantic_crypto_paraphrase", "retrieval_exact_keyword",
               "reflexion_good_simple", "knowing_dossier_direct"}
    assert not retired & {t.id for t in tasks}
    assert len(by_cat.get("skill-match", [])) <= 2 and len(by_cat.get("semantic-match", [])) <= 2
    sched = next(t for t in tasks if t.id == "mem_scheduler_codename_paraphrase")
    assert "background jobs on a timer" not in sched.query, "fixture collides with Nova's real monitors"


@pytest.mark.asyncio
async def test_check_task_runs_without_the_brain(monkeypatch):
    h = EvalHarness()

    async def _boom(*a, **k):
        raise AssertionError("a deterministic check must not invoke the brain")

    monkeypatch.setattr(h, "_invoke_brain", _boom)
    task = EvalTask(id="rq_probe", category="research-quality", query="check", assertions=[],
                    check="priming_key")
    res = await h.run_task(task)
    assert res.passed and res.tools_invoked == [] and res.reflexion_score is None
    assert "resolve" in res.response_text


@pytest.mark.asyncio
async def test_unknown_check_fails_loudly():
    h = EvalHarness()
    task = EvalTask(id="rq_bad", category="research-quality", query="check", assertions=[], check="nope")
    res = await h.run_task(task)
    assert not res.passed and "unknown check" in (res.error or res.response_text)
