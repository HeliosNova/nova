"""Tests for the automated evaluation harness.

Unit tests cover pure-function helpers (metric computation, regression
detection, report rendering, assertion evaluation).  The integration test
runs the harness against a tiny in-memory stub suite using a mocked
brain.think(), asserting the report is correctly shaped.

Nothing here hits the real LLM or network.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from app.monitors.eval_harness import (
    CategoryMetrics,
    EvalHarness,
    EvalReport,
    EvalTask,
    RegressionFlag,
    TaskResult,
    check_assertion,
    compute_category_metrics,
    detect_regressions,
    percentile,
    render_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    task_id="t1",
    category="reasoning",
    passed=True,
    latency=1.0,
    reflexion_score=0.8,
    tools_invoked=None,
    skill_used=None,
    response_text="The answer is 42.",
    failed_assertions=None,
    timed_out=False,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        category=category,
        query="what is the answer?",
        passed=passed,
        response_text=response_text,
        tools_invoked=tools_invoked or [],
        skill_used=skill_used,
        reflexion_score=reflexion_score,
        latency_seconds=latency,
        failed_assertions=failed_assertions or [],
        timed_out=timed_out,
    )


def _make_report(
    passed=8,
    failed=2,
    categories=None,
    regressions=None,
) -> EvalReport:
    cats = categories or {
        "reasoning": CategoryMetrics(
            category="reasoning", total=5, passed=5, pass_rate=1.0,
            latency_p50=1.0, latency_p95=2.0,
            reflexion_mean=0.8, reflexion_std=0.05,
            reflexion_p10=0.7, reflexion_p90=0.9,
        ),
    }
    return EvalReport(
        run_id="20260415_220000",
        suite_path="evals/suite.yaml",
        suite_version="1",
        total_tasks=passed + failed,
        passed=passed,
        failed=failed,
        skipped=0,
        pass_rate=passed / (passed + failed),
        duration_seconds=30.0,
        categories=cats,
        task_results=[],
        regressions=regressions or [],
        baseline_run_id=None,
        config_snapshot={},
        timestamp="2026-04-15T22:00:00+00:00",
    )


# ===========================================================================
# percentile()
# ===========================================================================

class TestPercentile:
    def test_empty_list(self):
        assert percentile([], 50) == 0.0

    def test_single_element(self):
        assert percentile([5.0], 50) == 5.0

    def test_median_even(self):
        values = [1.0, 2.0, 3.0, 4.0]
        result = percentile(values, 50)
        assert 2.0 <= result <= 3.0

    def test_p0_is_min(self):
        values = [3.0, 1.0, 2.0]
        assert percentile(values, 0) == 1.0

    def test_p100_is_max(self):
        values = [3.0, 1.0, 2.0]
        assert percentile(values, 100) == 3.0

    def test_p95_above_p50(self):
        values = list(range(1, 101))
        assert percentile(values, 95) > percentile(values, 50)


# ===========================================================================
# check_assertion()
# ===========================================================================

class TestCheckAssertion:
    def test_answer_contains_passes(self):
        assert check_assertion(
            {"type": "answer_contains", "value": "391"},
            "17 * 23 = 391", [], None, None,
        )

    def test_answer_contains_case_insensitive(self):
        assert check_assertion(
            {"type": "answer_contains", "value": "PYTHON"},
            "The answer involves python.", [], None, None,
        )

    def test_answer_contains_fails(self):
        assert not check_assertion(
            {"type": "answer_contains", "value": "391"},
            "The answer is 392.", [], None, None,
        )

    def test_answer_matches_regex(self):
        assert check_assertion(
            {"type": "answer_matches", "value": "\\b\\d+\\b"},
            "The value is 42.", [], None, None,
        )

    def test_answer_not_contains(self):
        assert check_assertion(
            {"type": "answer_not_contains", "value": "error"},
            "The calculation succeeded.", [], None, None,
        )
        assert not check_assertion(
            {"type": "answer_not_contains", "value": "error"},
            "An error occurred.", [], None, None,
        )

    def test_tool_invoked_passes(self):
        assert check_assertion(
            {"type": "tool_invoked", "value": "calculator"},
            "", ["calculator", "web_search"], None, None,
        )

    def test_tool_invoked_fails(self):
        assert not check_assertion(
            {"type": "tool_invoked", "value": "calculator"},
            "", ["web_search"], None, None,
        )

    def test_no_tool_invoked_passes(self):
        assert check_assertion({"type": "no_tool_invoked"}, "", [], None, None)

    def test_no_tool_invoked_fails(self):
        assert not check_assertion({"type": "no_tool_invoked"}, "", ["calculator"], None, None)

    def test_skill_matched_passes(self):
        assert check_assertion(
            {"type": "skill_matched"}, "", [], "Eval: Crypto Price Probe", None
        )

    def test_skill_matched_fails_when_none(self):
        assert not check_assertion({"type": "skill_matched"}, "", [], None, None)

    def test_skill_name_matches(self):
        assert check_assertion(
            {"type": "skill_name_matches", "value": "Crypto"},
            "", [], "Eval: Crypto Price Probe", None,
        )
        assert not check_assertion(
            {"type": "skill_name_matches", "value": "Weather"},
            "", [], "Eval: Crypto Price Probe", None,
        )

    def test_reflexion_in_range_passes(self):
        assert check_assertion(
            {"type": "reflexion_in_range", "min": 0.3, "max": 0.8},
            "", [], None, 0.65,
        )

    def test_reflexion_in_range_fails_outside(self):
        assert not check_assertion(
            {"type": "reflexion_in_range", "min": 0.3, "max": 0.8},
            "", [], None, 0.9,
        )
        assert not check_assertion(
            {"type": "reflexion_in_range", "min": 0.3, "max": 0.8},
            "", [], None, 0.1,
        )

    def test_reflexion_in_range_none_score_fails(self):
        assert not check_assertion(
            {"type": "reflexion_in_range", "min": 0.0, "max": 1.0},
            "", [], None, None,
        )

    def test_reflexion_above(self):
        assert check_assertion(
            {"type": "reflexion_above", "value": 0.5}, "", [], None, 0.7
        )
        assert not check_assertion(
            {"type": "reflexion_above", "value": 0.5}, "", [], None, 0.4
        )

    def test_reflexion_below(self):
        assert check_assertion(
            {"type": "reflexion_below", "value": 0.5}, "", [], None, 0.3
        )
        assert not check_assertion(
            {"type": "reflexion_below", "value": 0.5}, "", [], None, 0.5
        )

    def test_response_not_empty_passes(self):
        assert check_assertion(
            {"type": "response_not_empty"}, "a" * 20, [], None, None
        )

    def test_response_not_empty_fails_short(self):
        assert not check_assertion(
            {"type": "response_not_empty"}, "Hi", [], None, None
        )

    def test_unknown_type_returns_false(self):
        assert not check_assertion(
            {"type": "nonexistent_type"}, "anything", [], None, None
        )

    def test_retrieval_recall_literal_pass(self):
        assert check_assertion(
            {"type": "retrieval_recall", "value": "parameter"},
            "Gradient descent updates each parameter to reduce loss.",
            [], None, None,
        )

    def test_retrieval_recall_literal_fail(self):
        assert not check_assertion(
            {"type": "retrieval_recall", "value": "parameter"},
            "Gradient descent updates each weight to reduce loss.",
            [], None, None,
        )

    def test_retrieval_recall_synonym_alternation_passes_either(self):
        # Pipe-delimited alternatives: passes if ANY appears.
        assert check_assertion(
            {"type": "retrieval_recall", "value": "parameter|weight"},
            "Gradient descent updates each weight to reduce loss.",
            [], None, None,
        )
        assert check_assertion(
            {"type": "retrieval_recall", "value": "parameter|weight"},
            "Gradient descent updates each parameter to reduce loss.",
            [], None, None,
        )

    def test_retrieval_recall_synonym_fails_when_none_match(self):
        assert not check_assertion(
            {"type": "retrieval_recall", "value": "parameter|weight"},
            "Gradient descent uses neurons and biases.",
            [], None, None,
        )

    def test_retrieval_recall_synonym_handles_blanks(self):
        # 'parameter|' has an empty alternative — must not match empty
        assert check_assertion(
            {"type": "retrieval_recall", "value": "parameter|"},
            "model parameter update", [], None, None,
        )
        assert not check_assertion(
            {"type": "retrieval_recall", "value": "parameter|"},
            "the model adjusts weights", [], None, None,
        )


# ===========================================================================
# compute_category_metrics()
# ===========================================================================

class TestComputeCategoryMetrics:
    def test_basic_pass_rate(self):
        results = [
            _make_result(task_id="r1", category="reasoning", passed=True),
            _make_result(task_id="r2", category="reasoning", passed=True),
            _make_result(task_id="r3", category="reasoning", passed=False),
        ]
        metrics = compute_category_metrics(results)
        assert "reasoning" in metrics
        cm = metrics["reasoning"]
        assert cm.total == 3
        assert cm.passed == 2
        assert abs(cm.pass_rate - 2 / 3) < 0.001

    def test_reflexion_stats(self):
        results = [
            _make_result(task_id="r1", category="reasoning", reflexion_score=0.6),
            _make_result(task_id="r2", category="reasoning", reflexion_score=0.8),
            _make_result(task_id="r3", category="reasoning", reflexion_score=1.0),
        ]
        metrics = compute_category_metrics(results)
        cm = metrics["reasoning"]
        assert cm.reflexion_mean is not None
        assert abs(cm.reflexion_mean - 0.8) < 0.001
        assert cm.reflexion_std is not None
        assert cm.reflexion_p10 is not None
        assert cm.reflexion_p90 is not None

    def test_skill_match_hit_rate(self):
        results = [
            _make_result(task_id="s1", category="skill-match", skill_used="MySkill"),
            _make_result(task_id="s2", category="skill-match", skill_used=None),
            _make_result(task_id="s3", category="skill-match", skill_used="MySkill"),
        ]
        metrics = compute_category_metrics(results)
        cm = metrics["skill-match"]
        assert cm.hit_rate is not None
        assert abs(cm.hit_rate - 2 / 3) < 0.001

    def test_semantic_match_recall(self):
        results = [
            _make_result(task_id="e1", category="semantic-match", skill_used="Skill"),
            _make_result(task_id="e2", category="semantic-match", skill_used=None),
        ]
        metrics = compute_category_metrics(results)
        cm = metrics["semantic-match"]
        assert cm.recall_at_threshold == 0.5

    def test_autonomous_tool_multi_tool_rate(self):
        results = [
            _make_result(task_id="a1", category="autonomous-tool",
                         tools_invoked=["web_search", "calculator"]),  # 2 tools
            _make_result(task_id="a2", category="autonomous-tool",
                         tools_invoked=["web_search"]),  # 1 tool
        ]
        metrics = compute_category_metrics(results)
        cm = metrics["autonomous-tool"]
        assert cm.multi_tool_rate == 0.5

    def test_multiple_categories(self):
        results = [
            _make_result(task_id="r1", category="reasoning"),
            _make_result(task_id="t1", category="tool-use",
                         tools_invoked=["calculator"]),
        ]
        metrics = compute_category_metrics(results)
        assert "reasoning" in metrics
        assert "tool-use" in metrics

    def test_no_results(self):
        metrics = compute_category_metrics([])
        assert metrics == {}

    def test_reflexion_none_values_excluded(self):
        results = [
            _make_result(task_id="r1", category="reasoning", reflexion_score=None),
        ]
        metrics = compute_category_metrics(results)
        assert metrics["reasoning"].reflexion_mean is None


# ===========================================================================
# detect_regressions()
# ===========================================================================

class TestDetectRegressions:
    def _make_baseline(self, pass_rate=0.9, hit_rate=0.8, recall=0.75) -> dict:
        return {
            "categories": {
                "skill-match": {
                    "pass_rate": pass_rate,
                    "hit_rate": hit_rate,
                    "recall_at_threshold": None,
                },
                "semantic-match": {
                    "pass_rate": 0.8,
                    "hit_rate": None,
                    "recall_at_threshold": recall,
                },
            }
        }

    def test_no_regression_within_tolerance(self):
        baseline = self._make_baseline(pass_rate=0.9, hit_rate=0.8)
        current = {
            "skill-match": CategoryMetrics(
                category="skill-match", total=5, passed=4, pass_rate=0.85,
                latency_p50=1.0, latency_p95=2.0, hit_rate=0.75,
            ),
        }
        flags = detect_regressions(current, baseline, tolerance=0.10)
        assert all(not f.flagged for f in flags)

    def test_regression_detected_on_large_drop(self):
        baseline = self._make_baseline(pass_rate=0.9, hit_rate=0.8)
        current = {
            "skill-match": CategoryMetrics(
                category="skill-match", total=5, passed=2, pass_rate=0.4,
                latency_p50=1.0, latency_p95=2.0, hit_rate=0.2,
            ),
        }
        flags = detect_regressions(current, baseline, tolerance=0.10)
        flagged = [f for f in flags if f.flagged]
        assert len(flagged) >= 1
        metrics_flagged = {f.metric for f in flagged}
        assert "skill-match.pass_rate" in metrics_flagged or "skill-match.hit_rate" in metrics_flagged

    def test_improvement_not_flagged(self):
        baseline = self._make_baseline(pass_rate=0.5, hit_rate=0.5)
        current = {
            "skill-match": CategoryMetrics(
                category="skill-match", total=5, passed=5, pass_rate=1.0,
                latency_p50=1.0, latency_p95=2.0, hit_rate=1.0,
            ),
        }
        flags = detect_regressions(current, baseline, tolerance=0.10)
        assert all(not f.flagged for f in flags)

    def test_missing_baseline_category_skipped(self):
        baseline = {"categories": {}}
        current = {
            "reasoning": CategoryMetrics(
                category="reasoning", total=5, passed=5, pass_rate=1.0,
                latency_p50=1.0, latency_p95=2.0,
            ),
        }
        flags = detect_regressions(current, baseline, tolerance=0.10)
        assert flags == []

    def test_empty_baseline_returns_no_flags(self):
        flags = detect_regressions({}, {}, tolerance=0.10)
        assert flags == []

    def test_semantic_recall_regression(self):
        """Key regression: SKILL_SEMANTIC_THRESHOLD too high tanks recall."""
        baseline = {"categories": {
            "semantic-match": {"pass_rate": 0.8, "hit_rate": None,
                               "recall_at_threshold": 0.8,
                               "multi_tool_rate": None, "reflexion_mean": None},
        }}
        # After raising threshold to 0.99, recall drops to 0.0
        current = {
            "semantic-match": CategoryMetrics(
                category="semantic-match", total=5, passed=0, pass_rate=0.0,
                latency_p50=1.0, latency_p95=2.0, recall_at_threshold=0.0,
            ),
        }
        flags = detect_regressions(current, baseline, tolerance=0.10)
        flagged = [f for f in flags if f.flagged]
        assert any(f.metric == "semantic-match.recall_at_threshold" for f in flagged)


# ===========================================================================
# render_markdown()
# ===========================================================================

class TestRenderMarkdown:
    def test_ok_report_has_ok_header(self):
        report = _make_report()
        md = render_markdown(report)
        assert "[OK]" in md

    def test_regression_report_has_regression_header(self):
        regression = RegressionFlag(
            metric="skill-match.hit_rate",
            baseline=0.9, current=0.5, delta=-0.4,
            tolerance=0.10, flagged=True,
        )
        report = _make_report(regressions=[regression])
        md = render_markdown(report)
        assert "REGRESSION" in md
        assert "skill-match.hit_rate" in md

    def test_pass_rate_shown(self):
        report = _make_report(passed=8, failed=2)
        md = render_markdown(report)
        assert "80.0%" in md  # format is :.1% → "80.0%"

    def test_category_table_present(self):
        report = _make_report()
        md = render_markdown(report)
        assert "reasoning" in md
        assert "Per-Category" in md

    def test_all_tasks_passed_shows_message(self):
        report = _make_report()
        md = render_markdown(report)
        assert "All tasks passed" in md

    def test_failed_tasks_listed(self):
        result = _make_result(
            task_id="fail_001", passed=False,
            failed_assertions=["answer_contains('391') — got: 'nope'"],
        )
        cats = {"reasoning": CategoryMetrics(
            category="reasoning", total=1, passed=0, pass_rate=0.0,
            latency_p50=1.0, latency_p95=1.0,
        )}
        report = _make_report(passed=0, failed=1, categories=cats)
        report.task_results.append(result)
        md = render_markdown(report)
        assert "fail_001" in md
        assert "answer_contains" in md


# ===========================================================================
# EvalHarness — suite loading
# ===========================================================================

class TestSuiteLoading:
    def test_load_real_suite(self):
        """The shipped evals/suite.yaml must parse cleanly into >=30 tasks."""
        import os
        suite_path = Path(__file__).parent.parent / "evals" / "suite.yaml"
        if not suite_path.exists():
            pytest.skip("suite.yaml not found")
        harness = EvalHarness(suite_path=suite_path, report_dir="/tmp/eval_test")
        tasks = harness.load_suite()
        assert len(tasks) >= 30

    def test_load_real_suite_categories(self):
        """All 7 required categories must be represented."""
        suite_path = Path(__file__).parent.parent / "evals" / "suite.yaml"
        if not suite_path.exists():
            pytest.skip("suite.yaml not found")
        harness = EvalHarness(suite_path=suite_path, report_dir="/tmp/eval_test")
        tasks = harness.load_suite()
        categories = {t.category for t in tasks}
        required = {
            "reasoning", "tool-use", "skill-match",
            "semantic-match", "autonomous-tool", "reflexion-calibration",
            "multi-agent",
        }
        assert required.issubset(categories), f"Missing categories: {required - categories}"

    def test_load_minimal_yaml(self, tmp_path):
        """EvalHarness.load_suite() parses a minimal valid YAML."""
        suite = tmp_path / "suite.yaml"
        suite.write_text(textwrap.dedent("""\
            version: "1"
            tasks:
              - id: t1
                category: reasoning
                query: "What is 2+2?"
                timeout: 30
                assertions:
                  - type: answer_contains
                    value: "4"
        """))
        harness = EvalHarness(suite_path=suite, report_dir=tmp_path / "reports")
        tasks = harness.load_suite()
        assert len(tasks) == 1
        assert tasks[0].id == "t1"
        assert tasks[0].category == "reasoning"
        assert tasks[0].assertions[0]["type"] == "answer_contains"

    def test_seed_skill_parsed(self, tmp_path):
        """seed_skill fields are correctly normalized."""
        suite = tmp_path / "suite.yaml"
        suite.write_text(textwrap.dedent("""\
            version: "1"
            tasks:
              - id: s1
                category: skill-match
                query: "eval-probe: something"
                timeout: 30
                seed_skill:
                  name: "Eval: Test Skill"
                  trigger_pattern: "(?i)eval-probe"
                  steps:
                    - tool: web_search
                      args_template: {q: "{query}"}
                      output_key: result
                  answer_template: "Result: {result}"
                assertions:
                  - type: skill_matched
        """))
        harness = EvalHarness(suite_path=suite, report_dir=tmp_path / "reports")
        tasks = harness.load_suite()
        assert tasks[0].seed_skill is not None
        assert tasks[0].seed_skill["name"] == "Eval: Test Skill"
        assert len(tasks[0].seed_skill["steps"]) == 1


# ===========================================================================
# EvalHarness — report persistence
# ===========================================================================

class TestReportPersistence:
    def test_write_report_creates_files(self, tmp_path):
        harness = EvalHarness(suite_path="evals/suite.yaml", report_dir=tmp_path)
        report = _make_report()
        json_path, md_path = harness.write_report(report)
        assert json_path.exists()
        assert md_path.exists()

    def test_json_report_parseable(self, tmp_path):
        harness = EvalHarness(suite_path="evals/suite.yaml", report_dir=tmp_path)
        report = _make_report()
        json_path, _ = harness.write_report(report)
        with open(json_path) as f:
            data = json.load(f)
        assert data["run_id"] == report.run_id
        assert data["total_tasks"] == report.total_tasks
        assert "categories" in data

    def test_append_history_creates_jsonl(self, tmp_path):
        harness = EvalHarness(suite_path="evals/suite.yaml", report_dir=tmp_path)
        report = _make_report()
        harness.append_history(report)
        history = tmp_path / "eval_history.jsonl"
        assert history.exists()
        with open(history) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["run_id"] == report.run_id
        assert "pass_rate" in entry
        assert "categories" in entry

    def test_append_history_accumulates(self, tmp_path):
        harness = EvalHarness(suite_path="evals/suite.yaml", report_dir=tmp_path)
        for _ in range(3):
            harness.append_history(_make_report())
        with open(tmp_path / "eval_history.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_baseline_written_on_first_run(self, tmp_path):
        harness = EvalHarness(suite_path="evals/suite.yaml", report_dir=tmp_path)
        report = _make_report()
        harness.write_baseline(report)
        baseline = tmp_path / "eval_baseline.json"
        assert baseline.exists()
        with open(baseline) as f:
            data = json.load(f)
        assert data["run_id"] == report.run_id

    def test_load_baseline_returns_empty_when_missing(self, tmp_path):
        harness = EvalHarness(suite_path="evals/suite.yaml", report_dir=tmp_path)
        data, run_id = harness._load_baseline()
        assert data == {}
        assert run_id is None


# ===========================================================================
# EvalHarness — integration test with stubbed brain.think()
#
# We mock brain.think() to return a fixed sequence of events.  The harness
# still exercises the real assertion evaluation, metric computation, and
# regression detection code paths.
# ===========================================================================

class TestEvalHarnessIntegration:
    """Integration test: minimal stub suite + mocked brain.think().

    Verifies:
    - Report is shaped correctly (keys, counts, categories)
    - Pass/fail is computed from actual assertions
    - Regression detection fires correctly against a stale baseline
    - History appended, files written
    """

    @staticmethod
    def _make_stub_suite(tmp_path: Path) -> Path:
        """Write a tiny 4-task YAML suite covering 2 categories."""
        suite = tmp_path / "stub_suite.yaml"
        suite.write_text(textwrap.dedent("""\
            version: "stub-1"
            tasks:
              - id: r1
                category: reasoning
                query: "What is 2 plus 2?"
                timeout: 10
                assertions:
                  - type: answer_contains
                    value: "4"
              - id: r2
                category: reasoning
                query: "Capital of France?"
                timeout: 10
                assertions:
                  - type: answer_contains
                    value: "Paris"
              - id: t1
                category: tool-use
                query: "calculate 10 * 5"
                timeout: 10
                assertions:
                  - type: tool_invoked
                    value: calculator
              - id: t2
                category: tool-use
                query: "run python code"
                timeout: 10
                assertions:
                  - type: tool_invoked
                    value: code_exec
        """))
        return suite

    @staticmethod
    async def _good_brain(query: str, ephemeral: bool = True, **_):
        """Stub think() that answers correctly and invokes tools."""
        from app.schema import EventType, StreamEvent
        # Emit response tokens
        if "2 plus 2" in query.lower() or "2 + 2" in query.lower():
            text = "The answer is 4."
        elif "france" in query.lower():
            text = "The capital of France is Paris."
        elif "10 * 5" in query.lower() or "10*5" in query.lower():
            text = "50"
        else:
            text = "Python code executed."

        yield StreamEvent(type=EventType.TOKEN, data={"text": text})

        # Emit tool_use for tool-use tasks
        if "calculate" in query.lower():
            yield StreamEvent(
                type=EventType.TOOL_USE,
                data={"tool": "calculator", "args": {}, "status": "executing", "tool_call_id": "c1"},
            )
        elif "python code" in query.lower():
            yield StreamEvent(
                type=EventType.TOOL_USE,
                data={"tool": "code_exec", "args": {}, "status": "executing", "tool_call_id": "c2"},
            )

        yield StreamEvent(
            type=EventType.DONE,
            data={"conversation_id": "stub", "intent": "general",
                  "skill_used": None, "tool_results_count": 0},
        )

    @staticmethod
    async def _bad_brain(query: str, ephemeral: bool = True, **_):
        """Stub think() that gives wrong answers and no tool calls."""
        from app.schema import EventType, StreamEvent
        yield StreamEvent(type=EventType.TOKEN, data={"text": "I don't know."})
        yield StreamEvent(
            type=EventType.DONE,
            data={"conversation_id": "stub", "intent": "general",
                  "skill_used": None, "tool_results_count": 0},
        )

    @pytest.mark.asyncio
    async def test_integration_all_pass(self, tmp_path):
        """Good brain → all 4 tasks pass → report correctly shaped."""
        suite_path = self._make_stub_suite(tmp_path)
        harness = EvalHarness(suite_path=suite_path, report_dir=tmp_path / "reports")

        with patch("app.monitors.eval_harness.EvalHarness._seed_skills"):
            with patch("app.core.brain.think", side_effect=self._good_brain):
                report = await harness.run_all()

        assert report.total_tasks == 4
        assert report.passed == 4
        assert report.failed == 0
        assert report.pass_rate == 1.0
        assert "reasoning" in report.categories
        assert "tool-use" in report.categories
        assert report.categories["reasoning"].pass_rate == 1.0
        assert report.categories["tool-use"].pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_integration_partial_fail(self, tmp_path):
        """Bad brain → tool-use tasks fail (no tool called) → regression flagged."""
        suite_path = self._make_stub_suite(tmp_path)
        harness = EvalHarness(suite_path=suite_path, report_dir=tmp_path / "reports",
                              regression_tolerance=0.10)

        # Establish baseline with good brain first
        with patch("app.monitors.eval_harness.EvalHarness._seed_skills"):
            with patch("app.core.brain.think", side_effect=self._good_brain):
                baseline_report = await harness.run_all()
        harness.write_baseline(baseline_report)

        # Now run with bad brain — tool-use fails, regression should be caught
        with patch("app.monitors.eval_harness.EvalHarness._seed_skills"):
            with patch("app.core.brain.think", side_effect=self._bad_brain):
                bad_report = await harness.run_all()

        assert bad_report.passed < 4
        # At least tool-use tasks fail (no calculator/code_exec emitted)
        tool_use = bad_report.categories.get("tool-use")
        assert tool_use is not None
        assert tool_use.pass_rate < 1.0

        # Regression detector should flag tool-use.pass_rate
        flagged = [r for r in bad_report.regressions if r.flagged]
        flagged_metrics = {r.metric for r in flagged}
        assert "tool-use.pass_rate" in flagged_metrics, (
            f"Expected tool-use.pass_rate regression; got: {flagged_metrics}"
        )

    @pytest.mark.asyncio
    async def test_integration_files_written(self, tmp_path):
        """run_and_persist() creates JSON, MD, history.jsonl, baseline.json."""
        suite_path = self._make_stub_suite(tmp_path)
        harness = EvalHarness(suite_path=suite_path, report_dir=tmp_path / "reports")

        with patch("app.monitors.eval_harness.EvalHarness._seed_skills"):
            with patch("app.core.brain.think", side_effect=self._good_brain):
                report, json_path, md_path = await harness.run_and_persist()

        assert json_path.exists()
        assert md_path.exists()
        assert (tmp_path / "reports" / "eval_history.jsonl").exists()
        assert (tmp_path / "reports" / "eval_baseline.json").exists()

        # JSON report is valid
        with open(json_path) as f:
            data = json.load(f)
        assert data["total_tasks"] == 4
        assert len(data["task_results"]) == 4

    @pytest.mark.asyncio
    async def test_regression_evidence_semantic_threshold(self, tmp_path):
        """Empirical regression proof: simulate SKILL_SEMANTIC_THRESHOLD=0.99.

        Baseline: semantic-match recall = 1.0 (skill matches for all tasks)
        Broken:   semantic-match recall = 0.0 (threshold too high, no matches)
        Expected: harness flags semantic-match.recall_at_threshold regression
        """
        # Build a suite with 3 semantic-match tasks
        suite = tmp_path / "semantic_suite.yaml"
        suite.write_text(textwrap.dedent("""\
            version: "regression-test"
            tasks:
              - id: sem1
                category: semantic-match
                query: "how much does BTC cost"
                timeout: 10
                assertions:
                  - type: skill_matched
              - id: sem2
                category: semantic-match
                query: "ethereum price please"
                timeout: 10
                assertions:
                  - type: skill_matched
              - id: sem3
                category: semantic-match
                query: "bitcoin value now"
                timeout: 10
                assertions:
                  - type: skill_matched
        """))

        harness = EvalHarness(suite_path=suite, report_dir=tmp_path / "reports",
                              regression_tolerance=0.10)

        async def _brain_with_skill_match(query, ephemeral=True, **_):
            """Stub: skill always matches (baseline behavior)."""
            from app.schema import EventType, StreamEvent
            yield StreamEvent(type=EventType.TOKEN, data={"text": "Price is $X."})
            yield StreamEvent(type=EventType.DONE, data={
                "conversation_id": "stub", "intent": "general",
                "skill_used": "Eval: Crypto Price Probe", "tool_results_count": 0,
            })

        async def _brain_no_skill_match(query, ephemeral=True, **_):
            """Stub: threshold too high, no skill matches (broken behavior)."""
            from app.schema import EventType, StreamEvent
            yield StreamEvent(type=EventType.TOKEN, data={"text": "Price is $X."})
            yield StreamEvent(type=EventType.DONE, data={
                "conversation_id": "stub", "intent": "general",
                "skill_used": None, "tool_results_count": 0,
            })

        # Establish baseline with good skill matching
        with patch("app.monitors.eval_harness.EvalHarness._seed_skills"):
            with patch("app.core.brain.think", side_effect=_brain_with_skill_match):
                baseline = await harness.run_all()
        harness.write_baseline(baseline)

        # Simulate broken config: semantic threshold too high → no skill matches
        with patch("app.monitors.eval_harness.EvalHarness._seed_skills"):
            with patch("app.core.brain.think", side_effect=_brain_no_skill_match):
                broken = await harness.run_all()

        # Baseline recall should be 1.0, broken should be 0.0
        assert baseline.categories["semantic-match"].recall_at_threshold == 1.0
        assert broken.categories["semantic-match"].recall_at_threshold == 0.0

        # Harness must flag the regression
        flagged = [r for r in broken.regressions if r.flagged]
        assert any(r.metric == "semantic-match.recall_at_threshold" for r in flagged), (
            f"Expected semantic-match.recall_at_threshold to be flagged; got: "
            f"{[r.metric for r in flagged]}"
        )
        # Verify delta magnitude
        recall_flag = next(
            r for r in flagged if r.metric == "semantic-match.recall_at_threshold"
        )
        assert recall_flag.delta == -1.0
        assert recall_flag.baseline == 1.0
        assert recall_flag.current == 0.0


# ===========================================================================
# Multi-agent harness tests
# ===========================================================================

class TestCheckAssertionDecomposition:
    """check_assertion() handles decomposition_fired / decomposition_not_fired."""

    def test_decomposition_fired_passes_when_decomposed(self):
        assert check_assertion(
            {"type": "decomposition_fired"}, "", [], None, None, decomposed=True
        )

    def test_decomposition_fired_fails_when_not_decomposed(self):
        assert not check_assertion(
            {"type": "decomposition_fired"}, "", [], None, None, decomposed=False
        )

    def test_decomposition_not_fired_passes_when_not_decomposed(self):
        assert check_assertion(
            {"type": "decomposition_not_fired"}, "", [], None, None, decomposed=False
        )

    def test_decomposition_not_fired_fails_when_decomposed(self):
        assert not check_assertion(
            {"type": "decomposition_not_fired"}, "", [], None, None, decomposed=True
        )

    def test_default_decomposed_is_false(self):
        """Without passing decomposed, decomposition_fired must fail."""
        assert not check_assertion({"type": "decomposition_fired"}, "", [], None, None)

    def test_default_decomposed_not_fired_passes(self):
        """Without passing decomposed, decomposition_not_fired must pass."""
        assert check_assertion({"type": "decomposition_not_fired"}, "", [], None, None)


class TestComputeCategoryMetricsMultiAgent:
    """compute_category_metrics() handles multi-agent category."""

    def _make_ma_result(self, task_id, passed, decomposed):
        r = _make_result(task_id=task_id, category="multi-agent", passed=passed)
        r.decomposed = decomposed
        return r

    def test_decomposition_rate_computed(self):
        results = [
            self._make_ma_result("m1", passed=True, decomposed=True),
            self._make_ma_result("m2", passed=True, decomposed=True),
            self._make_ma_result("m3", passed=True, decomposed=False),
        ]
        metrics = compute_category_metrics(results)
        cm = metrics.get("multi-agent")
        assert cm is not None
        assert cm.decomposition_rate is not None
        assert abs(cm.decomposition_rate - 2 / 3) < 0.001

    def test_decomposition_rate_all_decomposed(self):
        results = [
            self._make_ma_result("m1", passed=True, decomposed=True),
            self._make_ma_result("m2", passed=True, decomposed=True),
        ]
        metrics = compute_category_metrics(results)
        assert metrics["multi-agent"].decomposition_rate == 1.0

    def test_decomposition_rate_none_decomposed(self):
        results = [
            self._make_ma_result("m1", passed=False, decomposed=False),
            self._make_ma_result("m2", passed=True, decomposed=False),
        ]
        metrics = compute_category_metrics(results)
        assert metrics["multi-agent"].decomposition_rate == 0.0

    def test_non_multi_agent_has_no_decomposition_rate(self):
        results = [_make_result("r1", category="reasoning")]
        metrics = compute_category_metrics(results)
        assert metrics["reasoning"].decomposition_rate is None


class TestDetectRegressionsMultiAgent:
    """detect_regressions() flags decomposition_rate drops."""

    def test_decomposition_rate_regression_flagged(self):
        baseline = {"categories": {
            "multi-agent": {
                "pass_rate": 1.0,
                "hit_rate": None,
                "recall_at_threshold": None,
                "multi_tool_rate": None,
                "reflexion_mean": None,
                "decomposition_rate": 0.67,
            }
        }}
        current = {
            "multi-agent": CategoryMetrics(
                category="multi-agent", total=3, passed=2, pass_rate=0.67,
                latency_p50=1.0, latency_p95=2.0,
                decomposition_rate=0.0,  # everything stopped decomposing
            )
        }
        flags = detect_regressions(current, baseline, tolerance=0.10)
        flagged_metrics = {f.metric for f in flags if f.flagged}
        assert "multi-agent.decomposition_rate" in flagged_metrics

    def test_decomposition_rate_improvement_not_flagged(self):
        baseline = {"categories": {
            "multi-agent": {
                "pass_rate": 0.5,
                "hit_rate": None,
                "recall_at_threshold": None,
                "multi_tool_rate": None,
                "reflexion_mean": None,
                "decomposition_rate": 0.33,
            }
        }}
        current = {
            "multi-agent": CategoryMetrics(
                category="multi-agent", total=3, passed=3, pass_rate=1.0,
                latency_p50=1.0, latency_p95=2.0,
                decomposition_rate=0.67,
            )
        }
        flags = detect_regressions(current, baseline, tolerance=0.10)
        ma_flags = [f for f in flags if "multi-agent" in f.metric]
        assert all(not f.flagged for f in ma_flags)


class TestMultiAgentEmpiricalRegression:
    """Empirical regression proof: MULTI_AGENT_TRIGGER_THRESHOLD=1 → everything decomposes
    → non-decomposable task fails decomposition_not_fired → decomposition_rate drifts
    → regression flagged automatically.
    """

    @staticmethod
    def _make_multi_agent_suite(tmp_path: Path) -> Path:
        suite = tmp_path / "ma_suite.yaml"
        suite.write_text(textwrap.dedent("""\
            version: "ma-regression-test"
            tasks:
              - id: ma_parallel
                category: multi-agent
                query: "Compare Python and JavaScript performance"
                timeout: 30
                tags: [parallel-decomposable]
                assertions:
                  - type: response_not_empty
                  - type: decomposition_fired

              - id: ma_sequential
                category: multi-agent
                query: "First search for Python version then calculate years since 1994"
                timeout: 30
                tags: [sequential-decomposable]
                assertions:
                  - type: response_not_empty
                  - type: decomposition_fired

              - id: ma_no_decompose
                category: multi-agent
                query: "What is 2 plus 2?"
                timeout: 30
                tags: [should-not-decompose]
                assertions:
                  - type: answer_contains
                    value: "4"
                  - type: decomposition_not_fired
        """))
        return suite

    @staticmethod
    async def _brain_decomposed(query: str, ephemeral: bool = True, **_):
        """Stub: always reports decomposed=True (threshold=1 behavior)."""
        from app.schema import EventType, StreamEvent
        if "2 plus 2" in query.lower():
            text = "The answer is 4."
        else:
            text = "Comparison result: Python and JavaScript differ in typing, execution, and use cases."
        yield StreamEvent(type=EventType.TOKEN, data={"text": text})
        yield StreamEvent(type=EventType.DONE, data={
            "conversation_id": "stub", "intent": "general",
            "skill_used": None, "tool_results_count": 0,
            "decomposed": True, "agent_count": 2,
        })

    @staticmethod
    async def _brain_calibrated(query: str, ephemeral: bool = True, **_):
        """Stub: reports decomposed correctly — True for compare/search, False for simple."""
        from app.schema import EventType, StreamEvent
        is_complex = "compare" in query.lower() or "search" in query.lower()
        if "2 plus 2" in query.lower():
            text = "The answer is 4."
        else:
            text = "Comparison result: Python and JavaScript differ in typing, execution, and use cases."
        yield StreamEvent(type=EventType.TOKEN, data={"text": text})
        yield StreamEvent(type=EventType.DONE, data={
            "conversation_id": "stub", "intent": "general",
            "skill_used": None, "tool_results_count": 0,
            "decomposed": is_complex, "agent_count": 2 if is_complex else 0,
        })

    @pytest.mark.asyncio
    async def test_threshold_too_low_triggers_regression(self, tmp_path):
        """Baseline: calibrated decomposition. Broken: everything decomposes.
        The non-decomposable task fails decomposition_not_fired.
        decomposition_rate drifts up → regression flagged.
        """
        suite_path = self._make_multi_agent_suite(tmp_path)
        harness = EvalHarness(
            suite_path=suite_path,
            report_dir=tmp_path / "reports",
            regression_tolerance=0.10,
        )

        # Establish baseline with correctly calibrated decomposition
        with patch("app.monitors.eval_harness.EvalHarness._seed_skills"):
            with patch("app.core.brain.think", side_effect=self._brain_calibrated):
                baseline = await harness.run_all()
        harness.write_baseline(baseline)

        # Calibrated: 2 decomposed, 1 not → rate = 0.67; all 3 tasks pass
        assert baseline.categories["multi-agent"].decomposition_rate is not None
        assert baseline.passed == 3

        # Now simulate threshold=1 (everything decomposes, including simple query)
        with patch("app.monitors.eval_harness.EvalHarness._seed_skills"):
            with patch("app.core.brain.think", side_effect=self._brain_decomposed):
                broken = await harness.run_all()

        # The no-decompose task must fail (decomposition_not_fired fails when decomposed=True)
        assert broken.passed < 3
        ma_results = [r for r in broken.task_results if r.category == "multi-agent"]
        no_decompose_result = next(
            (r for r in ma_results if r.task_id == "ma_no_decompose"), None
        )
        assert no_decompose_result is not None
        assert not no_decompose_result.passed  # decomposition_not_fired fails

        # decomposition_rate drifts up to 1.0 (was ~0.67)
        broken_rate = broken.categories["multi-agent"].decomposition_rate
        assert broken_rate == 1.0

        # Harness must flag the decomposition_rate regression
        flagged = [r for r in broken.regressions if r.flagged]
        flagged_metrics = {r.metric for r in flagged}
        assert "multi-agent.decomposition_rate" in flagged_metrics or broken.passed < 3, (
            "Either decomposition_rate regression or pass_rate regression must be flagged "
            f"when threshold fires everything; flagged: {flagged_metrics}"
        )


# ===========================================================================
# Heartbeat integration — seed count
# ===========================================================================

class TestHeartbeatSeedCount:
    def test_eval_monitor_seeded(self, db):
        """seed_defaults() must include 'Quality Eval Harness'."""
        from app.monitors.heartbeat import MonitorStore
        store = MonitorStore(db)
        store.seed_defaults()
        monitor = store.get_by_name("Quality Eval Harness")
        assert monitor is not None
        assert monitor.check_type == "eval"
        assert monitor.schedule_seconds == 86400

    def test_seed_count_includes_eval(self, db):
        """Total seeded monitors = 69 (61 catalog + 3 system + 4 loop closers + 1 added later)."""
        from app.monitors.heartbeat import MonitorStore
        store = MonitorStore(db)
        count = store.seed_defaults()
        all_monitors = store.list_all()
        assert len(all_monitors) == 69


class TestMemoryLearningEval:
    """The learning-loop PROOF: a seeded lesson must CAUSE a corrected answer.

    Unit-tests the orchestration of _run_memory_task (before / seed / after /
    cleanup / caused_fix) and the memory_causal_fix_rate metric, with the
    brain invocation mocked. The live causal numbers come from the in-container
    eval run against real Ollama.
    """

    @staticmethod
    def _inv(text, error=None):
        from app.monitors.eval_harness import _Invocation
        return _Invocation(
            response_text=text, tools_invoked=[], skill_used=None,
            decomposed=False, max_decomposition_depth=0,
            reflexion_score=0.8 if text else None,
            latency_seconds=0.01, error=error,
        )

    @staticmethod
    def _mem_task():
        return EvalTask(
            id="mem_x", category="memory-learning",
            query="What is the codename of Nova's scheduler?",
            assertions=[{"type": "answer_contains", "value": "Chronos"}],
            seed_lesson={"topic": "t", "correct_answer": "It is Chronos.",
                         "lesson_text": "codename Chronos", "confidence": 0.95},
        )

    def _harness_with_learning(self, learning):
        svc = MagicMock()
        svc.learning = learning
        return svc

    @pytest.mark.asyncio
    async def test_lesson_causes_fix(self):
        """before wrong, after correct → caused_fix=True, passed=True, lesson cleaned up."""
        task = self._mem_task()
        learning = MagicMock()
        learning.add_knowledge_lesson.return_value = 42
        learning._db.fetchall.return_value = [{"id": 42}]
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=self._harness_with_learning(learning)):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(side_effect=[self._inv("I don't know."),
                                           self._inv("It is Chronos.")]),
            ):
                result = await harness._run_memory_task(task)
        assert result.memory_before_correct is False
        assert result.memory_after_correct is True
        assert result.memory_caused_fix is True
        assert result.passed is True
        # seeded with the eval context marker (so cleanup is scoped + safe)
        assert learning.add_knowledge_lesson.call_args.kwargs["context"] == "eval-mem:mem_x"
        # and cleaned up afterward
        learning.delete_lesson.assert_any_call(42)

    @pytest.mark.asyncio
    async def test_no_fix_when_retrieval_misses(self):
        """before wrong, after STILL wrong → caused_fix=False, passed=False."""
        task = self._mem_task()
        learning = MagicMock()
        learning.add_knowledge_lesson.return_value = 7
        learning._db.fetchall.return_value = []
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=self._harness_with_learning(learning)):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(side_effect=[self._inv("No idea."), self._inv("Still no idea.")]),
            ):
                result = await harness._run_memory_task(task)
        assert result.memory_before_correct is False
        assert result.memory_after_correct is False
        assert result.memory_caused_fix is False
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_already_known_is_not_a_causal_fix(self):
        """before already correct → caused_fix=False (excluded from the proof)."""
        task = self._mem_task()
        learning = MagicMock()
        learning.add_knowledge_lesson.return_value = 1
        learning._db.fetchall.return_value = []
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=self._harness_with_learning(learning)):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(side_effect=[self._inv("It is Chronos already."),
                                           self._inv("It is Chronos.")]),
            ):
                result = await harness._run_memory_task(task)
        assert result.memory_before_correct is True
        assert result.memory_caused_fix is False

    @pytest.mark.asyncio
    async def test_missing_learning_engine_is_setup_error(self):
        task = self._mem_task()
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=self._harness_with_learning(None)):
            result = await harness._run_memory_task(task)
        assert result.passed is False
        assert result.error == "setup_error"

    def test_purge_only_deletes_marked_rows(self):
        """_purge_eval_lessons scopes the SELECT to the exact marker and never touches others."""
        learning = MagicMock()
        learning._db.fetchall.return_value = [{"id": 11}, {"id": 12}]
        EvalHarness._purge_eval_lessons(learning, "eval-mem:abc")
        sql, params = learning._db.fetchall.call_args[0]
        assert "context = ?" in sql
        assert params == ("eval-mem:abc",)
        learning.delete_lesson.assert_any_call(11)
        learning.delete_lesson.assert_any_call(12)

    def test_memory_metric_counts_only_testable(self):
        """memory_causal_fix_rate denominator excludes pairs the model already knew."""
        def tr(tid, before, after):
            return TaskResult(
                task_id=tid, category="memory-learning", query="q", passed=after,
                response_text="", tools_invoked=[], skill_used=None,
                reflexion_score=None, latency_seconds=0.0, failed_assertions=[],
                memory_before_correct=before, memory_after_correct=after,
                memory_caused_fix=bool(after and not before),
            )
        rows = [tr("a", False, True), tr("b", False, True),
                tr("c", False, False), tr("d", True, True)]
        m = compute_category_metrics(rows)["memory-learning"]
        assert m.memory_testable == 3
        assert m.memory_causal_fix_rate == pytest.approx(2 / 3)
        assert m.memory_already_known_rate == pytest.approx(1 / 4)

    def test_shipped_suite_memory_tasks_have_seed(self):
        """The shipped suite's memory-learning tasks parse with a seed_lesson."""
        mem = [t for t in EvalHarness().load_suite() if t.category == "memory-learning"]
        assert len(mem) >= 5
        assert all(t.seed_lesson and t.seed_lesson.get("correct_answer") for t in mem)


class TestTimeoutSeparation:
    """Timeouts are budget exhaustion, NOT wrong answers.

    A run that hits its time budget without proving correctness is outcome
    timeout: excluded from every correctness denominator, counted separately,
    and visible in latency stats. Slow-but-correct still counts as a pass.
    """

    # --- metrics level -----------------------------------------------------

    def test_timeout_excluded_from_pass_rate(self):
        rows = [
            _make_result("a", passed=True),
            _make_result("b", passed=False),
            _make_result("c", passed=False, timed_out=True, latency=60.0),
        ]
        m = compute_category_metrics(rows)["reasoning"]
        assert m.total == 3
        assert m.timeouts == 1
        # pass_rate over the 2 completed runs, not 3
        assert m.pass_rate == pytest.approx(0.5)

    def test_all_timeouts_no_correctness_signal(self):
        rows = [
            _make_result("a", passed=False, timed_out=True, latency=60.0),
            _make_result("b", passed=False, timed_out=True, latency=60.0),
        ]
        m = compute_category_metrics(rows)["reasoning"]
        assert m.timeouts == 2
        assert m.pass_rate == 0.0

    def test_timeout_latency_still_counts(self):
        """Latency percentiles include timed-out runs — a timeout IS latency truth."""
        rows = [
            _make_result("a", passed=True, latency=1.0),
            _make_result("b", passed=False, timed_out=True, latency=60.0),
        ]
        m = compute_category_metrics(rows)["reasoning"]
        assert m.latency_p95 >= 30.0

    def test_skill_match_hit_rate_over_completed_only(self):
        rows = [
            _make_result("a", category="skill-match", passed=True, skill_used="crypto"),
            _make_result("b", category="skill-match", passed=False, timed_out=True),
        ]
        m = compute_category_metrics(rows)["skill-match"]
        # the timed-out run never got its DONE event — it can't count against hit rate
        assert m.hit_rate == pytest.approx(1.0)

    def test_untestable_memory_pair_excluded_from_causal_rate(self):
        def tr(tid, before, after, timed_out=False):
            return TaskResult(
                task_id=tid, category="memory-learning", query="q",
                passed=bool(after), response_text="", tools_invoked=[],
                skill_used=None, reflexion_score=None, latency_seconds=0.0,
                failed_assertions=[], timed_out=timed_out,
                memory_before_correct=before, memory_after_correct=after,
                memory_caused_fix=(bool(after and not before) if before is not None else None),
            )
        rows = [
            tr("a", False, True),                      # causal fix
            tr("b", None, None, timed_out=True),       # untestable — leg timed out
        ]
        m = compute_category_metrics(rows)["memory-learning"]
        assert m.memory_testable == 1
        assert m.memory_causal_fix_rate == pytest.approx(1.0)
        assert m.timeouts == 1

    # --- run_task level ----------------------------------------------------

    @staticmethod
    def _inv(text, timed_out=False):
        from app.monitors.eval_harness import _Invocation
        return _Invocation(
            response_text=text, tools_invoked=[], skill_used=None,
            decomposed=False, max_decomposition_depth=0,
            reflexion_score=0.8 if text else None,
            latency_seconds=60.0 if timed_out else 0.01,
            error="Timeout after 60s" if timed_out else None,
            timed_out=timed_out,
        )

    @pytest.mark.asyncio
    async def test_run_task_timeout_is_not_a_failure(self):
        task = EvalTask(
            id="t_slow", category="reasoning", query="hard question",
            assertions=[{"type": "answer_contains", "value": "42"}],
        )
        harness = EvalHarness()
        svc = MagicMock()
        with patch("app.core.brain.get_services", return_value=svc):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(return_value=self._inv("", timed_out=True)),
            ):
                result = await harness.run_task(task)
        assert result.timed_out is True
        assert result.passed is False
        # budget exhaustion must NOT be written to the reflexion store as a failure
        svc.reflexions.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_task_slow_but_correct_is_a_pass(self):
        task = EvalTask(
            id="t_slow_ok", category="reasoning", query="hard question",
            assertions=[{"type": "answer_contains", "value": "42"}],
        )
        harness = EvalHarness()
        with patch.object(
            EvalHarness, "_invoke_brain",
            new=AsyncMock(return_value=self._inv("The answer is 42.", timed_out=True)),
        ):
            result = await harness.run_task(task)
        assert result.passed is True
        assert result.timed_out is False  # outcome is pass, not timeout

    @pytest.mark.asyncio
    async def test_memory_pair_with_timed_out_after_leg_is_untestable(self):
        task = EvalTask(
            id="mem_t", category="memory-learning",
            query="What is the codename of Nova scheduler?",
            assertions=[{"type": "answer_contains", "value": "Chronos"}],
            seed_lesson={"topic": "t", "correct_answer": "It is Chronos.",
                         "lesson_text": "codename Chronos", "confidence": 0.95},
        )
        learning = MagicMock()
        learning.add_knowledge_lesson.return_value = 1
        learning._db.fetchall.return_value = []
        svc = MagicMock()
        svc.learning = learning
        harness = EvalHarness()
        with patch("app.core.brain.get_services", return_value=svc):
            with patch.object(
                EvalHarness, "_invoke_brain",
                new=AsyncMock(side_effect=[self._inv("No idea."),
                                           self._inv("", timed_out=True)]),
            ):
                result = await harness._run_memory_task(task)
        assert result.timed_out is True
        assert result.passed is False
        assert result.memory_before_correct is None
        assert result.memory_caused_fix is None

    # --- report level ------------------------------------------------------

    def test_report_renders_timeout_counts(self):
        cats = {
            "reasoning": CategoryMetrics(
                category="reasoning", total=3, passed=1, pass_rate=0.5,
                latency_p50=2.0, latency_p95=60.0, timeouts=1,
            ),
        }
        report = EvalReport(
            run_id="r", suite_path="s", suite_version="1",
            total_tasks=3, passed=1, failed=1, skipped=0, pass_rate=0.5,
            duration_seconds=10.0, categories=cats, task_results=[],
            regressions=[], baseline_run_id=None, config_snapshot={},
            timestamp="2026-06-10T00:00:00+00:00", timeouts=1,
        )
        md = render_markdown(report)
        assert "Timeouts (excluded from pass rate) | 1" in md
        assert "Pass rate (over completed)" in md


class TestCompletionRateRegressionVisibility:
    """Timeouts must be visible to regression detection via completion_rate.

    pass_rate excludes timeouts from its denominator, so before this metric a
    build that timed out on every task flagged zero regressions.
    """

    def test_completion_rate_computed(self):
        rows = [
            _make_result("a", passed=True),
            _make_result("b", passed=False),
            _make_result("c", passed=False, timed_out=True, latency=60.0),
            _make_result("d", passed=False, timed_out=True, latency=60.0),
        ]
        m = compute_category_metrics(rows)["reasoning"]
        assert m.completion_rate == pytest.approx(0.5)

    def test_all_timeouts_completion_rate_zero(self):
        rows = [
            _make_result("a", passed=False, timed_out=True, latency=60.0),
            _make_result("b", passed=False, timed_out=True, latency=60.0),
        ]
        m = compute_category_metrics(rows)["reasoning"]
        assert m.completion_rate == 0.0

    def test_timeout_spike_flags_regression(self):
        """The previously-invisible failure mode: pass_rate steady, all timeouts."""
        baseline = {
            "categories": {
                "reasoning": {"pass_rate": 1.0, "completion_rate": 1.0},
            }
        }
        current = {
            "reasoning": CategoryMetrics(
                category="reasoning", total=4, passed=1, pass_rate=1.0,
                latency_p50=60.0, latency_p95=60.0, timeouts=3,
                completion_rate=0.25,
            ),
        }
        flags = detect_regressions(current, baseline, tolerance=0.10)
        flagged = {f.metric for f in flags if f.flagged}
        assert "reasoning.completion_rate" in flagged
        assert "reasoning.pass_rate" not in flagged  # the blind spot this closes

    def test_old_baseline_without_completion_rate_skipped(self):
        """Baselines written before this metric simply don't compare it."""
        baseline = {"categories": {"reasoning": {"pass_rate": 1.0}}}
        current = {
            "reasoning": CategoryMetrics(
                category="reasoning", total=2, passed=1, pass_rate=1.0,
                latency_p50=60.0, latency_p95=60.0, timeouts=1,
                completion_rate=0.5,
            ),
        }
        flags = detect_regressions(current, baseline, tolerance=0.10)
        assert not any(f.flagged for f in flags)


class TestMultiTurnTasks:
    """Multi-turn eval: prior `turns` share an ephemeral conversation with the
    graded final `query` via brain's in-process history."""

    @staticmethod
    def _inv(text, error=None, timed_out=False):
        from app.monitors.eval_harness import _Invocation
        return _Invocation(
            response_text=text, tools_invoked=[], skill_used=None,
            decomposed=False, max_decomposition_depth=0,
            reflexion_score=0.7 if text else None,
            latency_seconds=0.01, error=error, timed_out=timed_out,
        )

    @staticmethod
    def _mt_task(turns=None):
        return EvalTask(
            id="mt_x", category="multi-turn",
            query="What is my colleague's name?",
            assertions=[{"type": "answer_contains", "value": "Lindqvist"}],
            timeout=30,
            turns=turns or ["My colleague is Dr. Verena Lindqvist."],
        )

    @pytest.mark.asyncio
    async def test_turns_share_conversation_and_record_history(self):
        task = self._mt_task()
        harness = EvalHarness()
        invoke = AsyncMock(side_effect=[self._inv("Noted."), self._inv("Dr. Verena Lindqvist.")])
        recorded = []
        with patch.object(EvalHarness, "_invoke_brain", new=invoke), \
             patch("app.core.brain.record_ephemeral_turn",
                   side_effect=lambda c, q, a: recorded.append((c, q, a))), \
             patch("app.core.brain.clear_ephemeral_history") as clear_mock:
            result = await harness.run_task(task)
        assert result.passed is True
        # both turns went through the same conversation id
        conv_ids = {call.kwargs.get("conversation_id") or call.args[2] if len(call.args) > 2 else call.kwargs.get("conversation_id") for call in invoke.call_args_list}
        assert conv_ids == {"eval-multiturn-mt_x"}
        # the setup turn was recorded into history before the final turn
        assert recorded == [("eval-multiturn-mt_x", task.turns[0], "Noted.")]
        # history cleared before the run and after (finally)
        assert clear_mock.call_count >= 2

    @pytest.mark.asyncio
    async def test_setup_turn_failure_aborts(self):
        task = self._mt_task()
        harness = EvalHarness()
        invoke = AsyncMock(side_effect=[self._inv("", error="Timeout after 30s", timed_out=True)])
        with patch.object(EvalHarness, "_invoke_brain", new=invoke), \
             patch("app.core.brain.record_ephemeral_turn") as rec_mock, \
             patch("app.core.brain.clear_ephemeral_history"):
            result = await harness.run_task(task)
        assert result.passed is False
        assert "setup turn 1/1 failed" in (result.error or "")
        rec_mock.assert_not_called()
        assert invoke.call_count == 1  # final turn never ran

    @pytest.mark.asyncio
    async def test_single_turn_tasks_unaffected(self):
        task = EvalTask(
            id="single", category="reasoning", query="2+2?",
            assertions=[{"type": "answer_contains", "value": "4"}], timeout=30,
        )
        harness = EvalHarness()
        invoke = AsyncMock(return_value=self._inv("The answer is 4."))
        with patch.object(EvalHarness, "_invoke_brain", new=invoke):
            result = await harness.run_task(task)
        assert result.passed is True
        assert invoke.call_count == 1
        # no conversation_id for single-turn tasks
        assert invoke.call_args.kwargs.get("conversation_id") is None
        assert len(invoke.call_args.args) <= 2

    def test_suite_parses_turns(self):
        harness = EvalHarness()
        tasks = harness.load_suite()
        mt = [t for t in tasks if t.category == "multi-turn"]
        assert len(mt) == 4
        assert all(t.turns for t in mt)
        consistency = next(t for t in mt if t.id == "multiturn_consistency")
        assert len(consistency.turns) == 2


class TestEphemeralHistory:
    """brain's in-process ephemeral conversation history."""

    def setup_method(self):
        from app.core import brain
        brain._EPHEMERAL_HISTORIES.clear()

    teardown_method = setup_method

    def test_record_and_clear(self):
        from app.core.brain import (
            _EPHEMERAL_HISTORIES,
            clear_ephemeral_history,
            record_ephemeral_turn,
        )
        record_ephemeral_turn("c1", "hello", "hi there")
        assert _EPHEMERAL_HISTORIES["c1"] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        clear_ephemeral_history("c1")
        assert "c1" not in _EPHEMERAL_HISTORIES

    def test_default_ephemeral_id_never_recorded(self):
        from app.core import brain
        brain.record_ephemeral_turn("ephemeral", "q", "a")
        brain.record_ephemeral_turn("", "q", "a")
        assert brain._EPHEMERAL_HISTORIES == {}

    def test_message_cap(self):
        from app.core import brain
        for i in range(60):
            brain.record_ephemeral_turn("c2", f"q{i}", f"a{i}")
        hist = brain._EPHEMERAL_HISTORIES["c2"]
        assert len(hist) == brain._EPHEMERAL_MAX_MESSAGES
        assert hist[-1] == {"role": "assistant", "content": "a59"}

    def test_conversation_cap_evicts_oldest(self):
        from app.core import brain
        for i in range(brain._EPHEMERAL_MAX_CONVS + 5):
            brain.record_ephemeral_turn(f"conv{i}", "q", "a")
        assert len(brain._EPHEMERAL_HISTORIES) == brain._EPHEMERAL_MAX_CONVS
        assert "conv0" not in brain._EPHEMERAL_HISTORIES
