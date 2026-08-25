"""Eval-harness fixes from the 2026-08-25 pass.

1. multi_tool_rate re-scoped: the old "any autonomous-tool task that used
   >=2 tools" definition collapsed 0.5→0.0 when forced math routing
   correctly made pure-math tasks single-tool — an improvement flagged as
   a regression. Now it is computed only over tasks DECLARING >=2
   expect_tools, as "used all expected tools".
2. Chronic-failure escalation: baseline-delta regression can never flag a
   task that was already red at baseline time (multiturn_recall_name sat
   red 5 consecutive nightly runs invisibly). 3+ consecutive reds now get
   their own channel.
"""

from __future__ import annotations

import json

import pytest


def _result(task_id="t", category="autonomous-tool", passed=True,
            tools=None, expect=None, timed_out=False):
    from app.monitors.eval_harness import TaskResult

    return TaskResult(
        task_id=task_id, category=category, query="q", passed=passed,
        response_text="r", tools_invoked=tools or [], skill_used=None,
        reflexion_score=0.8, latency_seconds=1.0, failed_assertions=[],
        timed_out=timed_out, expect_tools=expect or [],
    )


class TestMultiToolRateScoping:
    def test_single_tool_correct_tasks_do_not_dilute(self):
        from app.monitors.eval_harness import compute_category_metrics

        results = [
            _result("compound", tools=["calculator"]),          # correctly 1-tool
            _result("stock", tools=["calculator"]),             # correctly 1-tool
            _result("weather", tools=["web_search"]),           # correctly 1-tool
            _result("multi", tools=["web_search", "calculator"],
                    expect=["web_search", "calculator"]),
        ]
        cm = compute_category_metrics(results)["autonomous-tool"]
        assert cm.multi_tool_rate == 1.0

    def test_missing_expected_tool_is_a_miss(self):
        from app.monitors.eval_harness import compute_category_metrics

        results = [
            _result("multi", tools=["web_search"],              # calculator skipped
                    expect=["web_search", "calculator"]),
        ]
        cm = compute_category_metrics(results)["autonomous-tool"]
        assert cm.multi_tool_rate == 0.0

    def test_no_declared_tasks_yields_none_not_zero(self):
        from app.monitors.eval_harness import compute_category_metrics

        results = [_result("solo", tools=["calculator"])]
        cm = compute_category_metrics(results)["autonomous-tool"]
        assert cm.multi_tool_rate is None


class TestChronicFailureDetection:
    def _harness(self, tmp_path):
        from app.monitors.eval_harness import EvalHarness

        return EvalHarness(suite_path="evals/suite.yaml",
                           report_dir=str(tmp_path))

    def _write_prior(self, tmp_path, run_id, failed_ids):
        data = {"run_id": run_id, "task_results": [
            {"task_id": tid, "passed": False, "timed_out": False}
            for tid in failed_ids
        ] + [{"task_id": "green", "passed": True, "timed_out": False}]}
        (tmp_path / f"eval_{run_id}.json").write_text(json.dumps(data))

    def _report(self, run_id, failed_ids):
        from app.monitors.eval_harness import EvalReport

        return EvalReport(
            run_id=run_id, suite_path="s", suite_version="v",
            total_tasks=2, passed=1, failed=1, skipped=0, pass_rate=0.5,
            duration_seconds=1.0, categories={},
            task_results=[_result(tid, category="multi-turn", passed=False)
                          for tid in failed_ids],
            regressions=[], baseline_run_id=None, config_snapshot={},
            timestamp="2026-08-25T00:00:00Z",
        )

    def test_three_consecutive_reds_flagged(self, tmp_path):
        h = self._harness(tmp_path)
        self._write_prior(tmp_path, "20260823_000000", ["multiturn_recall_name"])
        self._write_prior(tmp_path, "20260824_000000", ["multiturn_recall_name"])
        report = self._report("20260825_000000", ["multiturn_recall_name"])
        assert h.detect_chronic_failures(report) == ["multiturn_recall_name"]

    def test_recovered_task_not_flagged(self, tmp_path):
        h = self._harness(tmp_path)
        self._write_prior(tmp_path, "20260823_000000", ["multiturn_recall_name"])
        self._write_prior(tmp_path, "20260824_000000", [])  # passed yesterday
        report = self._report("20260825_000000", ["multiturn_recall_name"])
        assert h.detect_chronic_failures(report) == []

    def test_insufficient_history_not_flagged(self, tmp_path):
        h = self._harness(tmp_path)
        self._write_prior(tmp_path, "20260824_000000", ["multiturn_recall_name"])
        report = self._report("20260825_000000", ["multiturn_recall_name"])
        assert h.detect_chronic_failures(report) == []


class TestDigestHealthCanary:
    def test_healthy_week_is_info(self):
        from app.monitors.heartbeat_loop import _digest_health_verdict

        status, summary = _digest_health_verdict(
            [8000] * 50, linkish=0, checked=1400, dropped=600)
        assert status == "info"

    def test_no_digests_is_error(self):
        from app.monitors.heartbeat_loop import _digest_health_verdict

        status, _ = _digest_health_verdict([], 0, 0, 0)
        assert status == "error"

    def test_link_only_share_is_error(self):
        from app.monitors.heartbeat_loop import _digest_health_verdict

        status, _ = _digest_health_verdict([8000] * 8 + [300] * 2, linkish=2,
                                           checked=0, dropped=0)
        assert status == "error"

    def test_rising_drop_rate_is_warning(self):
        from app.monitors.heartbeat_loop import _digest_health_verdict

        status, _ = _digest_health_verdict(
            [8000] * 50, linkish=0, checked=1000, dropped=700)
        assert status == "warning"

    def test_gate_line_regex_parses_live_format(self):
        from app.monitors.heartbeat_loop import _ENTAIL_GATE_LINE_RE

        line = ("[entail-gate] Supply Chain: 24 checked, 12 unsupported → "
                "5 anchored (clause), 2 re-cited, 1 de-cited (analysis), 4 dropped")
        m = _ENTAIL_GATE_LINE_RE.search(line)
        assert m and m.group(1) == "24" and m.group(2) == "4"
