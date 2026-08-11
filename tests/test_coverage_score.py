"""Deterministic coverage metric — recall proxy for deep-research (task #65)."""

from __future__ import annotations

from app.monitors.report_grader import coverage_score, _anchors


class TestAnchors:
    def test_extracts_multiword_entities(self):
        a = _anchors("Nvidia and Taiwan Semiconductor beat estimates; the S&P 500 rose.")
        assert any("nvidia" in x for x in a)
        assert any("taiwan semiconductor" in x for x in a)

    def test_drops_sentence_initial_common_words(self):
        a = _anchors("The Federal Reserve held rates.")
        # "The" stripped → "Federal Reserve" kept, not "The Federal Reserve"
        assert any("federal reserve" == x for x in a)
        assert not any(x.startswith("the ") for x in a)

    def test_ignores_weekdays_and_months(self):
        a = _anchors("Monday markets fell. In January the deal closed.")
        assert "monday" not in a
        assert "january" not in a


class TestCoverageScore:
    def _findings(self):
        return [
            ("t1", "u1", "Nvidia reported record data-center revenue of $30 billion."),
            ("t2", "u2", "Taiwan Semiconductor raised its capex guidance."),
            ("t3", "u3", "The European Central Bank signaled a rate cut."),
        ]

    def test_full_coverage(self):
        report = ("Nvidia posted record results. Taiwan Semiconductor lifted capex. "
                  "The European Central Bank leaned dovish.")
        cov = coverage_score(report, self._findings())
        assert cov["coverage"] == 1.0
        assert cov["missed"] == []

    def test_partial_coverage_reports_missed(self):
        report = "Nvidia posted record results and nothing else mattered."
        cov = coverage_score(report, self._findings())
        assert cov["coverage"] < 1.0

    def test_core_coverage_only_counts_multiply_sourced(self):
        # Nvidia appears in 2 findings (core); the others once (peripheral).
        findings = [
            ("t1", "u1", "Nvidia beat estimates on data-center demand."),
            ("t2", "u2", "Nvidia guided higher for next quarter."),
            ("t3", "u3", "Acme Corp announced a minor product refresh."),
        ]
        # Report covers only the core story, omits the peripheral one.
        report = "Nvidia beat estimates and guided higher."
        cov = coverage_score(report, findings)
        assert cov["core_anchors"] >= 1
        assert cov["core_coverage"] == 1.0        # the multiply-sourced story is covered
        assert cov["coverage"] < 1.0              # raw coverage dinged for omitting Acme
        # dropping the CORE story is the real miss:
        cov2 = coverage_score("Acme Corp refreshed a product.", findings)
        assert cov2["core_coverage"] < 1.0
        assert any("nvidia" in m for m in cov2["missed"])

    def test_entity_superset_counts_as_covered(self):
        findings = [("t", "u", "Nvidia beat estimates.")]
        report = "Nvidia Corp beat estimates."   # superset token match
        cov = coverage_score(report, findings)
        assert cov["coverage"] == 1.0

    def test_empty_pool_is_full_coverage(self):
        cov = coverage_score("anything", [])
        assert cov["coverage"] == 1.0
        assert cov["pool_anchors"] == 0
