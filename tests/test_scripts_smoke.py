"""Item 66: Smoke test for finetune/eval scripts.

Import main functions from scripts/finetune.py and scripts/eval_harness.py,
verify they import without crashing. Test load_training_data() with an
empty/missing file. Test show_data_stats() with empty list.
"""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path
from io import StringIO

# Ensure project root is on sys.path so `from scripts.X import ...` works
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest

# Skip all tests if scripts/ directory isn't available (e.g., in Docker container)
_scripts_available = os.path.isdir(os.path.join(_project_root, "scripts"))
pytestmark = pytest.mark.skipif(not _scripts_available, reason="scripts/ not in container")


class TestFinetuneImport:
    """Verify finetune.py imports cleanly and functions work with edge cases."""

    def test_import_finetune_module(self):
        """scripts/finetune.py should import without errors."""
        from scripts.finetune import load_training_data, show_data_stats
        assert callable(load_training_data)
        assert callable(show_data_stats)

    def test_load_training_data_missing_file(self, tmp_path):
        """load_training_data with a missing file should return empty list."""
        from scripts.finetune import load_training_data
        missing = str(tmp_path / "nonexistent.jsonl")
        result = load_training_data(missing)
        assert result == []

    def test_load_training_data_empty_file(self, tmp_path):
        """load_training_data with an empty file should return empty list."""
        from scripts.finetune import load_training_data
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        result = load_training_data(str(empty_file))
        assert result == []

    def test_load_training_data_valid_entries(self, tmp_path):
        """load_training_data should parse valid JSONL entries."""
        from scripts.finetune import load_training_data
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            '{"query": "What is 2+2?", "chosen": "4", "rejected": "5"}\n'
            '{"query": "Who made Python?", "chosen": "Guido", "rejected": "Gosling"}\n'
        )
        result = load_training_data(str(data_file))
        assert len(result) == 2
        assert result[0]["prompt"] == "What is 2+2?"
        assert result[0]["chosen"] == "4"

    def test_load_training_data_invalid_json_skipped(self, tmp_path):
        """Invalid JSON lines should be skipped without crashing."""
        from scripts.finetune import load_training_data
        data_file = tmp_path / "bad.jsonl"
        data_file.write_text(
            '{"query": "valid", "chosen": "yes", "rejected": "no"}\n'
            'not valid json\n'
            '{"query": "also valid", "chosen": "answer", "rejected": "wrong"}\n'
        )
        result = load_training_data(str(data_file))
        assert len(result) == 2

    def test_show_data_stats_empty_list(self):
        """show_data_stats with empty list should not crash."""
        from scripts.finetune import show_data_stats
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            show_data_stats([])
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "No training data" in output or "0" in output

    def test_show_data_stats_with_data(self):
        """show_data_stats with valid data should print statistics."""
        from scripts.finetune import show_data_stats
        data = [
            {"prompt": "q1", "chosen": "a1", "rejected": "r1"},
            {"prompt": "q2", "chosen": "a2", "rejected": "r2"},
        ]
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            show_data_stats(data)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "2" in output  # Total valid pairs


class TestEvalHarnessImport:
    """Verify eval_harness.py imports cleanly."""

    def test_import_eval_harness_module(self):
        """scripts/eval_harness.py should import without errors."""
        from scripts.eval_harness import ComparisonResult, EvalResults
        assert ComparisonResult is not None
        assert EvalResults is not None

    def test_comparison_result_creation(self):
        """ComparisonResult dataclass should instantiate correctly."""
        from scripts.eval_harness import ComparisonResult
        result = ComparisonResult(
            query="test",
            base_response="base answer",
            candidate_response="candidate answer",
            winner="candidate",
            preference_score=0.8,
            judge_reasoning="candidate was better",
        )
        assert result.winner == "candidate"
        assert result.preference_score == 0.8

    def test_eval_results_to_dict(self):
        """EvalResults.to_dict() should return a dictionary."""
        from scripts.eval_harness import EvalResults
        results = EvalResults(
            base_model="base",
            candidate_model="candidate",
            total_queries=10,
            base_wins=3,
            candidate_wins=6,
            ties=1,
            win_rate=0.6,
            avg_preference=0.3,
            candidate_is_better=True,
        )
        d = results.to_dict()
        assert d["total_queries"] == 10
        assert d["candidate_is_better"] is True
        assert d["win_rate"] == 0.6
