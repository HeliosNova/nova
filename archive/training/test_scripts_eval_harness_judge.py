"""Tests for scripts/eval_harness.py judge-derivation logic.

Covers the score → winner derivation introduced 2026-05-16 to fix the
~47% winner/score inconsistency seen with small Qwen3.x judges. The
judge is now asked only for `score`; `winner` is derived in code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not a package — load by file path.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import eval_harness  # noqa: E402


class TestDeriveWinner:
    def test_positive_above_threshold_picks_candidate(self):
        assert eval_harness._derive_winner(0.5) == "candidate"

    def test_negative_above_threshold_picks_base(self):
        assert eval_harness._derive_winner(-0.5) == "base"

    def test_zero_is_tie(self):
        assert eval_harness._derive_winner(0.0) == "tie"

    def test_exactly_at_threshold_is_tie(self):
        # Boundary: |score| == threshold counts as tie (strict inequality)
        assert eval_harness._derive_winner(eval_harness.JUDGE_TIE_THRESHOLD) == "tie"
        assert eval_harness._derive_winner(-eval_harness.JUDGE_TIE_THRESHOLD) == "tie"

    def test_just_above_threshold_picks_winner(self):
        t = eval_harness.JUDGE_TIE_THRESHOLD
        assert eval_harness._derive_winner(t + 0.001) == "candidate"
        assert eval_harness._derive_winner(-(t + 0.001)) == "base"

    def test_extreme_values_clamp_logically(self):
        # Caller should already clamp, but if it doesn't the winner still
        # comes out correctly (we only check sign vs threshold).
        assert eval_harness._derive_winner(1.5) == "candidate"
        assert eval_harness._derive_winner(-1.5) == "base"

    def test_custom_threshold_override(self):
        # Higher threshold → score 0.2 becomes tie instead of candidate
        assert eval_harness._derive_winner(0.2, threshold=0.5) == "tie"
        assert eval_harness._derive_winner(0.6, threshold=0.5) == "candidate"


class TestParseJudge:
    """Parsing robustness: multi-dimensional + legacy schema, and a clean None
    on unparseable / non-numeric input (so callers don't fabricate 0.0 ties)."""

    def test_multidim_averaged(self):
        raw = json.dumps({"accuracy": 0.8, "completeness": 0.8, "clarity": 0.8,
                          "relevance": 0.8, "reasoning": "B better"})
        score, reason = eval_harness._parse_judge(raw)
        assert score == pytest.approx(0.8)
        assert reason == "B better"

    def test_multidim_partial_dimensions_averaged(self):
        score, _ = eval_harness._parse_judge(
            json.dumps({"accuracy": 1.0, "relevance": 0.0, "reasoning": "mixed"}))
        assert score == pytest.approx(0.5)

    def test_legacy_single_score(self):
        score, _ = eval_harness._parse_judge(json.dumps({"score": 0.5, "reasoning": "x"}))
        assert score == pytest.approx(0.5)

    def test_unparseable_returns_none(self):
        score, reason = eval_harness._parse_judge("not even close to JSON")
        assert score is None
        assert "Could not parse" in reason

    def test_prose_embedded_json_is_parsed(self):
        score, _ = eval_harness._parse_judge('Here: {"score": 0.7, "reasoning": "y"} done.')
        assert score == pytest.approx(0.7)

    def test_non_numeric_returns_none(self):
        score, reason = eval_harness._parse_judge(json.dumps({"accuracy": "great"}))
        assert score is None
        assert "Non-numeric" in reason

    def test_missing_score_fields_returns_none(self):
        score, reason = eval_harness._parse_judge(json.dumps({"reasoning": "no score"}))
        assert score is None


class TestJudgeOnceDirected:
    """Sign convention: positive preference always means CANDIDATE is better,
    regardless of which response sat in the A vs B slot."""

    @pytest.fixture
    def stub(self, monkeypatch):
        async def _stub(client, ollama_url, model, prompt, **kwargs):
            return _stub.payload  # type: ignore[attr-defined]
        _stub.payload = ""  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_harness, "_generate", _stub)
        return _stub

    @pytest.mark.asyncio
    async def test_base_first_positive_means_candidate(self, stub):
        stub.payload = json.dumps({"score": 0.8, "reasoning": "B better"})
        pref, _ = await eval_harness._judge_once_directed(
            None, "x", "x", "q", "base", "cand", base_first=True)
        assert pref == pytest.approx(0.8)  # B == candidate

    @pytest.mark.asyncio
    async def test_candidate_first_positive_means_base(self, stub):
        stub.payload = json.dumps({"score": 0.8, "reasoning": "B better"})
        pref, _ = await eval_harness._judge_once_directed(
            None, "x", "x", "q", "base", "cand", base_first=False)
        assert pref == pytest.approx(-0.8)  # B == base → candidate worse

    @pytest.mark.asyncio
    async def test_parse_failure_returns_none(self, stub):
        stub.payload = "garbage"
        pref, reason = await eval_harness._judge_once_directed(
            None, "x", "x", "q", "base", "cand", base_first=True)
        assert pref is None
        assert "Could not parse" in reason


class TestJudgePairSwap:
    """Position-swap combine: agreement → decisive winner; a decisive flip →
    tie (position bias); single parse failure → fall back to the other order."""

    @staticmethod
    def _seq_generate(monkeypatch, payloads):
        it = iter(payloads)
        async def _stub(client, ollama_url, model, prompt, **kwargs):
            return next(it)
        monkeypatch.setattr(eval_harness, "_generate", _stub)
        monkeypatch.setattr(eval_harness, "JUDGE_SWAP", True)

    @pytest.mark.asyncio
    async def test_both_orders_agree_candidate(self, monkeypatch):
        # order1 (base_first): B=cand, +0.8 → candidate
        # order2 (cand_first): B=base, -0.8 → -(-0.8)=+0.8 → candidate
        self._seq_generate(monkeypatch, [
            json.dumps({"score": 0.8, "reasoning": "r1"}),
            json.dumps({"score": -0.8, "reasoning": "r2"}),
        ])
        winner, pref, _ = await eval_harness._judge_pair(
            None, "x", "x", "q", "base", "cand")
        assert winner == "candidate"
        assert pref == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_position_bias_scored_tie(self, monkeypatch):
        # Judge ALWAYS says "B is better" regardless of order → decisive flip → tie
        self._seq_generate(monkeypatch, [
            json.dumps({"score": 0.8, "reasoning": "always B"}),
            json.dumps({"score": 0.8, "reasoning": "always B"}),
        ])
        winner, _, reason = await eval_harness._judge_pair(
            None, "x", "x", "q", "base", "cand")
        assert winner == "tie"
        assert "position-bias" in reason

    @pytest.mark.asyncio
    async def test_single_parse_failure_uses_other_order(self, monkeypatch):
        # order1 unparseable; order2 (cand_first) -0.8 → +0.8 candidate
        self._seq_generate(monkeypatch, [
            "garbage",
            json.dumps({"score": -0.8, "reasoning": "r2"}),
        ])
        winner, pref, _ = await eval_harness._judge_pair(
            None, "x", "x", "q", "base", "cand")
        assert winner == "candidate"
        assert pref == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_both_unparseable_is_tie(self, monkeypatch):
        self._seq_generate(monkeypatch, ["nope", "still nope"])
        winner, pref, _ = await eval_harness._judge_pair(
            None, "x", "x", "q", "base", "cand")
        assert winner == "tie"
        assert pref == 0.0
