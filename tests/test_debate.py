"""Tests for app.core.debate — A-HMAD style role-specialized debate."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.core import debate


@pytest.fixture
def enable_debate(monkeypatch):
    monkeypatch.setenv("ENABLE_DEBATE", "true")
    from app.config import reset_config
    reset_config()


# ---- is_enabled / config gate -------------------------------------------

def test_is_enabled_default_false():
    # Conftest doesn't set ENABLE_DEBATE — default false
    assert debate.is_enabled() is False


def test_is_enabled_when_flag_set(enable_debate):
    assert debate.is_enabled() is True


# ---- should_debate gate -------------------------------------------------

def test_should_debate_disabled_returns_false():
    assert debate.should_debate("medical advice please?", "general", "X" * 500) is False


def test_should_debate_skips_non_general_intent(enable_debate):
    assert debate.should_debate("question " * 30, "greeting", "X" * 500) is False
    assert debate.should_debate("question " * 30, "correction", "X" * 500) is False


def test_should_debate_skips_short_input(enable_debate):
    assert debate.should_debate("short", "general", "draft") is False


def test_should_debate_fires_on_medical(enable_debate):
    q = "Is 800mg of ibuprofen a safe daily dosage for chronic pain?"
    draft = "Daily 800mg ibuprofen has known kidney and GI side effects." * 4
    assert debate.should_debate(q, "general", draft) is True


def test_should_debate_fires_on_strong_numeric_claim(enable_debate):
    q = "Tell me about that protocol throughput claim history " * 2
    draft = (
        "The Acme protocol definitely processes 5000 requests in 50 days. "
        "This is proven across benchmarks and never falls below this rate."
    ) * 3
    assert debate.should_debate(q, "general", draft) is True


def test_should_debate_skips_already_hedged(enable_debate):
    q = "What is the best medication for headaches?" * 2
    draft = (
        "I'm not sure exactly. It may depend on individual factors and "
        "I think doses might vary. Perhaps consult a doctor." * 3
    )
    # Already hedged — debate would just re-hedge
    assert debate.should_debate(q, "general", draft) is False


# ---- run_debate (LLM-mocked) -------------------------------------------

class _FakeLLM:
    """Mockable replacement for app.core.llm.invoke_nothink."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    async def invoke_nothink(self, *args, **kwargs):
        self.call_count += 1
        if not self._responses:
            return None
        return self._responses.pop(0)


def _patch_llm(monkeypatch, fake):
    """Patch llm.invoke_nothink to return scripted responses."""
    import app.core.debate as debate_mod
    original = debate_mod.__dict__.get("_orig_llm")
    monkeypatch.setattr(
        "app.core.llm.invoke_nothink", fake.invoke_nothink, raising=True,
    )


def test_run_debate_returns_keep_when_disabled():
    out = asyncio.run(debate.run_debate("q" * 50, "draft" * 50))
    assert out.action == "keep"
    assert "disabled" in out.rationale.lower()


def test_run_debate_keep_when_all_critics_ok(enable_debate, monkeypatch):
    # All three critics return verdict='ok'
    ok = json.dumps({"verdict": "ok", "issues": [], "severity": "low"})
    fake = _FakeLLM([ok, ok, ok])  # judge not called when all ok
    _patch_llm(monkeypatch, fake)
    out = asyncio.run(debate.run_debate("q" * 50, "draft text " * 50))
    assert out.action == "keep"
    assert out.final_answer == "draft text " * 50
    # judge should not have been invoked
    assert fake.call_count == 3


def test_run_debate_runs_judge_when_critics_have_issues(enable_debate, monkeypatch):
    issues = json.dumps({
        "verdict": "issues",
        "issues": ["claim X is unsupported"],
        "severity": "high",
    })
    ok = json.dumps({"verdict": "ok", "issues": [], "severity": "low"})
    judge = json.dumps({
        "action": "amend",
        "final_answer": "AMENDED ANSWER",
        "rationale": "fixed unsupported claim",
    })
    fake = _FakeLLM([issues, ok, ok, judge])
    _patch_llm(monkeypatch, fake)
    out = asyncio.run(debate.run_debate("q" * 50, "original draft " * 50))
    assert out.action == "amend"
    assert out.final_answer == "AMENDED ANSWER"
    assert fake.call_count == 4


def test_run_debate_judge_failure_falls_back_to_keep(enable_debate, monkeypatch):
    issues = json.dumps({"verdict": "issues", "issues": ["x"], "severity": "high"})
    fake = _FakeLLM([issues, issues, issues, None])  # judge returns None
    _patch_llm(monkeypatch, fake)
    out = asyncio.run(debate.run_debate("q" * 50, "original " * 50))
    assert out.action == "keep"  # fallback


def test_run_debate_malformed_critic_json_treated_as_ok(enable_debate, monkeypatch):
    fake = _FakeLLM(["not-json-at-all", "{invalid", ""])  # all bad → all 'ok'
    _patch_llm(monkeypatch, fake)
    out = asyncio.run(debate.run_debate("q" * 50, "draft " * 50))
    # All malformed → no issues → keep
    assert out.action == "keep"


def test_run_debate_empty_inputs(enable_debate):
    out = asyncio.run(debate.run_debate("", "draft"))
    assert out.action == "keep"
    out = asyncio.run(debate.run_debate("q", ""))
    assert out.action == "keep"


def test_run_debate_decomposed_runs_only_fact_verifier(enable_debate, monkeypatch):
    # When decomposed=True the sub-agents have already validated reasoning
    # and tool-strategy through their own reflexion passes — debate runs
    # ONLY the fact verifier on the merged synthesis. That's 1 critic
    # call total when all OK (vs 3 for the full debate path).
    ok = json.dumps({"verdict": "ok", "issues": [], "severity": "low"})
    fake = _FakeLLM([ok])
    _patch_llm(monkeypatch, fake)
    out = asyncio.run(
        debate.run_debate("q" * 50, "draft text " * 50, decomposed=True),
    )
    assert out.action == "keep"
    assert fake.call_count == 1                          # 1 critic, no judge
    assert len(out.critics) == 1
    assert out.critics[0].role == "fact"


def test_run_debate_decomposed_invokes_judge_on_fact_issue(enable_debate, monkeypatch):
    # If the fact verifier finds issues, the judge still runs to decide
    # keep/amend/replace/hedge — provenance check survives the early-out.
    issues = json.dumps({
        "verdict": "issues",
        "issues": ["claim X is unsupported"],
        "severity": "high",
    })
    judge = json.dumps({
        "action": "amend",
        "final_answer": "AMENDED",
        "rationale": "removed unsupported claim",
    })
    fake = _FakeLLM([issues, judge])
    _patch_llm(monkeypatch, fake)
    out = asyncio.run(
        debate.run_debate("q" * 50, "draft text " * 50, decomposed=True),
    )
    assert out.action == "amend"
    assert out.final_answer == "AMENDED"
    assert fake.call_count == 2                          # 1 critic + 1 judge
