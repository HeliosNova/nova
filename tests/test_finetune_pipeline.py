"""Tests for the fine-tuning pipeline — data loading, stats, eval harness logic.

Tests everything that can run without a GPU:
  - Training data loading/validation
  - Data stats computation
  - Eval harness data types + judging logic
  - Holdout query sampling
  - End-to-end DPO pair export from learning engine

This proves the pipeline code is correct even though actual training
requires an RTX 3090 with CUDA.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# ===========================================================================
# Training Data Loading
# ===========================================================================

class TestTrainingDataLoading:
    @pytest.fixture
    def training_data_file(self, tmp_path):
        """Create a valid JSONL training data file."""
        data = [
            {"query": "What is Python?", "chosen": "A programming language", "rejected": "A snake", "timestamp": "2026-03-14T10:00:00"},
            {"query": "Capital of France?", "chosen": "Paris", "rejected": "London", "timestamp": "2026-03-14T10:01:00"},
            {"query": "2+2?", "chosen": "4", "rejected": "5", "timestamp": "2026-03-14T10:02:00"},
        ]
        path = tmp_path / "training_data.jsonl"
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        return str(path)

    def test_load_valid_data(self, training_data_file):
        from scripts.finetune import load_training_data
        data = load_training_data(training_data_file)
        assert len(data) == 3
        assert data[0]["prompt"] == "What is Python?"
        assert data[0]["chosen"] == "A programming language"
        assert data[0]["rejected"] == "A snake"

    def test_load_missing_file(self, tmp_path):
        from scripts.finetune import load_training_data
        data = load_training_data(str(tmp_path / "nonexistent.jsonl"))
        assert data == []

    def test_load_skips_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        with open(path, "w") as f:
            f.write('{"query": "valid", "chosen": "yes", "rejected": "no"}\n')
            f.write("not json at all\n")
            f.write('{"query": "also valid", "chosen": "ok", "rejected": "nah"}\n')
        from scripts.finetune import load_training_data
        data = load_training_data(str(path))
        assert len(data) == 2

    def test_load_skips_missing_fields(self, tmp_path):
        path = tmp_path / "incomplete.jsonl"
        with open(path, "w") as f:
            f.write('{"query": "has query", "chosen": "has chosen"}\n')
            f.write('{"query": "", "chosen": "empty query"}\n')  # empty query
            f.write('{"chosen": "no query field"}\n')  # missing query
        from scripts.finetune import load_training_data
        data = load_training_data(str(path))
        assert len(data) == 1
        assert data[0]["rejected"] == "[No response]"  # missing rejected gets default

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.touch()
        from scripts.finetune import load_training_data
        data = load_training_data(str(path))
        assert data == []

    def test_show_data_stats(self, training_data_file, capsys):
        from scripts.finetune import load_training_data, show_data_stats
        data = load_training_data(training_data_file)
        show_data_stats(data)
        output = capsys.readouterr().out
        assert "Total valid pairs: 3" in output
        assert "Avg prompt length:" in output
        assert "Sample entries:" in output

    def test_show_data_stats_empty(self, capsys):
        from scripts.finetune import show_data_stats
        show_data_stats([])
        output = capsys.readouterr().out
        assert "No training data available!" in output


# ===========================================================================
# Eval Harness — Data Types
# ===========================================================================

class TestEvalDataTypes:
    def test_comparison_result(self):
        from scripts.eval_harness import ComparisonResult
        cr = ComparisonResult(
            query="test query",
            base_response="base resp",
            candidate_response="candidate resp",
            winner="candidate",
            preference_score=0.7,
            judge_reasoning="candidate was better",
        )
        assert cr.winner == "candidate"
        assert cr.preference_score == 0.7
        assert cr.error == ""

    def test_eval_results_to_dict(self):
        from scripts.eval_harness import EvalResults, ComparisonResult
        cr = ComparisonResult(
            query="test", base_response="a", candidate_response="b",
            winner="candidate", preference_score=0.5, judge_reasoning="ok",
        )
        results = EvalResults(
            base_model="base",
            candidate_model="ft",
            total_queries=1,
            base_wins=0,
            candidate_wins=1,
            ties=0,
            win_rate=1.0,
            avg_preference=0.5,
            candidate_is_better=True,
            comparisons=[cr],
            evaluated_at="2026-03-14T12:00:00",
        )
        d = results.to_dict()
        assert d["candidate_is_better"] is True
        assert d["win_rate"] == 1.0
        assert len(d["comparisons"]) == 1
        assert d["comparisons"][0]["winner"] == "candidate"

    def test_candidate_better_criteria(self):
        from scripts.eval_harness import EvalResults
        # Candidate better requires both: win_rate > 50% AND avg_preference > 0
        # Win rate high but negative preference → not better
        r1 = EvalResults(
            base_model="b", candidate_model="c", total_queries=10,
            base_wins=3, candidate_wins=7, ties=0,
            win_rate=0.7, avg_preference=-0.1,
            candidate_is_better=False,
        )
        assert not r1.candidate_is_better

        # Both conditions met → better
        r2 = EvalResults(
            base_model="b", candidate_model="c", total_queries=10,
            base_wins=2, candidate_wins=8, ties=0,
            win_rate=0.8, avg_preference=0.4,
            candidate_is_better=True,
        )
        assert r2.candidate_is_better


# ===========================================================================
# Holdout Query Sampling
# ===========================================================================

class TestHoldoutSampling:
    @pytest.fixture
    def large_training_file(self, tmp_path):
        path = tmp_path / "large.jsonl"
        with open(path, "w") as f:
            for i in range(100):
                entry = {"query": f"This is query number {i} with enough text", "chosen": f"answer {i}", "rejected": f"wrong {i}"}
                f.write(json.dumps(entry) + "\n")
        return str(path)

    def test_sample_holdout(self, large_training_file):
        from scripts.eval_harness import sample_holdout_queries
        queries = sample_holdout_queries(large_training_file, n=10, seed=42)
        assert len(queries) == 10
        assert all(isinstance(q, str) for q in queries)
        assert all(len(q) > 10 for q in queries)

    def test_sample_deterministic(self, large_training_file):
        from scripts.eval_harness import sample_holdout_queries
        q1 = sample_holdout_queries(large_training_file, n=5, seed=42)
        q2 = sample_holdout_queries(large_training_file, n=5, seed=42)
        assert q1 == q2

    def test_sample_different_seeds(self, large_training_file):
        from scripts.eval_harness import sample_holdout_queries
        q1 = sample_holdout_queries(large_training_file, n=5, seed=1)
        q2 = sample_holdout_queries(large_training_file, n=5, seed=2)
        assert q1 != q2

    def test_sample_missing_file(self, tmp_path):
        from scripts.eval_harness import sample_holdout_queries
        queries = sample_holdout_queries(str(tmp_path / "nope.jsonl"))
        assert queries == []

    def test_sample_caps_at_available(self, tmp_path):
        path = tmp_path / "small.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"query": "Only one query with enough text here"}) + "\n")
        from scripts.eval_harness import sample_holdout_queries
        queries = sample_holdout_queries(str(path), n=10)
        assert len(queries) == 1


# ===========================================================================
# DPO Pair Export — End-to-End from LearningEngine
# ===========================================================================

class TestDPOPairExport:
    @pytest.mark.asyncio
    async def test_save_training_pair_writes_jsonl(self, db, tmp_path):
        """Full chain: correction → save_training_pair → file written."""
        from app.core.learning import LearningEngine

        engine = LearningEngine(db)

        # Monkey-patch the training data path
        data_file = str(tmp_path / "training_data.jsonl")
        with patch("app.core.learning.config") as mock_config:
            mock_config.TRAINING_DATA_PATH = data_file
            mock_config.TRAINING_DATA_CHANNELS = "api"
            mock_config.TRAINING_DATA_MAX_PAIRS = 10000
            await engine.save_training_pair(
                query="What is the capital of France?",
                bad_answer="London is the capital",
                good_answer="Paris is the capital of France, located on the Seine river.",
                channel="api",
                confidence=1.0,
            )

        # Verify JSONL was written
        assert os.path.exists(data_file)
        with open(data_file) as f:
            line = f.readline().strip()
            entry = json.loads(line)
            assert entry["query"] == "What is the capital of France?"
            assert entry["chosen"] == "Paris is the capital of France, located on the Seine river."
            assert entry["rejected"] == "London is the capital"

    @pytest.mark.asyncio
    async def test_channel_gating_blocks_unauthorized(self, db, tmp_path):
        """Training pairs from unauthorized channels should be silently dropped."""
        from app.core.learning import LearningEngine

        engine = LearningEngine(db)
        data_file = str(tmp_path / "training_data.jsonl")

        with patch("app.core.learning.config") as mock_config:
            mock_config.TRAINING_DATA_PATH = data_file
            mock_config.TRAINING_DATA_CHANNELS = "api"  # only API allowed
            mock_config.TRAINING_DATA_MAX_PAIRS = 10000
            await engine.save_training_pair(
                query="Injected query",
                bad_answer="bad",
                good_answer="good",
                channel="discord",  # NOT allowed
                confidence=1.0,
            )

        # File should not exist (no data written)
        assert not os.path.exists(data_file)

    @pytest.mark.asyncio
    async def test_low_confidence_external_blocked(self, db, tmp_path):
        """Low-confidence corrections from external channels should be blocked."""
        from app.core.learning import LearningEngine

        engine = LearningEngine(db)
        data_file = str(tmp_path / "training_data.jsonl")

        with patch("app.core.learning.config") as mock_config:
            mock_config.TRAINING_DATA_PATH = data_file
            mock_config.TRAINING_DATA_CHANNELS = "api,discord"
            mock_config.TRAINING_DATA_MAX_PAIRS = 10000
            await engine.save_training_pair(
                query="Questionable correction",
                bad_answer="old",
                good_answer="new",
                channel="discord",
                confidence=0.5,  # Below 0.8 threshold
            )

        assert not os.path.exists(data_file)


# ===========================================================================
# Eval Harness — Judge Logic (Mocked)
# ===========================================================================

class TestJudgeLogic:
    @pytest.mark.asyncio
    async def test_run_eval_with_mocked_ollama(self):
        """Eval harness works end-to-end with mocked Ollama responses."""
        from scripts.eval_harness import run_eval

        call_count = 0
        async def mock_post(url, *, json=None, timeout=None, **kwargs):
            nonlocal call_count
            call_count += 1

            class MockResp:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    if "judge" in (json or {}).get("prompt", "").lower() or call_count % 3 == 0:
                        return {"response": '{"winner": "B", "score": 0.5, "reasoning": "B was better"}'}
                    elif call_count % 3 == 1:
                        return {"response": "Base model answer here."}
                    else:
                        return {"response": "Candidate model answer here."}
            return MockResp()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            results = await run_eval(
                queries=["test query 1", "test query 2"],
                base_model="base",
                candidate_model="ft",
                ollama_url="http://fake:11434",
            )

        assert results.total_queries == 2
        assert results.base_wins + results.candidate_wins + results.ties == 2
