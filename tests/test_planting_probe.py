"""P1: the planting probe samples rescued vs first-pass claims, plants each in an
uncited passage and reports whether the verdict migrates."""

from __future__ import annotations

from scripts import planting_probe as pp


def _rec(site, claim, hosts, doc, supported=True):
    return {"site": site, "claim": claim, "hosts": hosts, "doc": doc, "supported": supported}


RECORDS = [
    _rec("gate", "Acme bought Beta in 2026.", ["reuters.com"], "Reuters: Acme has acquired Beta this year. The deal closed in March."),
    _rec("gate", "Gamma raised prices.", ["bbc.com"], "BBC: Gamma lifted its prices on Monday. Analysts expected it."),
    _rec("clause", "Delta cut staff.", ["ft.com"], "FT: Delta reduced headcount by a tenth. Unions objected loudly."),
    _rec("alt", "Epsilon opened a plant.", ["apnews.com"], "AP: Epsilon opened a new factory in Ohio. Hiring starts soon."),
    _rec("clause", "Zeta lost a suit.", ["ft.com"], "FT: Zeta lost in court. Appeals are planned.", supported=False),
    _rec("gate", "Eta merged.", ["bbc.com"], "", supported=True),
]


def test_sample_picks_supported_rescued_and_first_pass_with_docs():
    rescued, first = pp.sample(RECORDS, 10)
    assert {r["claim"] for r in rescued} == {"Delta cut staff.", "Epsilon opened a plant."}
    assert {r["claim"] for r in first} == {"Acme bought Beta in 2026.", "Gamma raised prices."}


def test_sample_is_bounded_and_deterministic():
    a, _ = pp.sample(RECORDS, 1)
    b, _ = pp.sample(RECORDS, 1)
    assert len(a) == 1 and a == b


def test_plant_inserts_the_claim_mid_passage():
    out = pp.plant("Delta cut staff", "Reuters: Acme has acquired Beta this year. The deal closed in March.")
    assert "Delta cut staff." in out
    assert out.startswith("Reuters: Acme has acquired Beta this year.")
    assert out.endswith("The deal closed in March.")


def test_uncited_doc_comes_from_a_non_overlapping_host():
    rec = RECORDS[2]  # ft.com
    doc = pp.uncited_doc_for(rec, RECORDS)
    assert doc and "FT:" not in doc


def test_run_reports_migration_and_the_rule():
    def scorer(pairs):
        # the planted claim's words are in the doc: a surface-keyed model says yes to everything
        return [{"supported": True, "prob": 0.9} for _ in pairs]
    out = pp.run(RECORDS, 10, scorer)
    assert out["rescued_migration"] == 1.0 and out["first_pass_migration"] == 1.0
    assert out["delta_pp"] == 0.0 and out["rule_p1_disable_rescue"] is False

    def rescue_only(pairs):
        return [{"supported": "Delta" in p["claim"] or "Epsilon" in p["claim"]} for p in pairs]
    out = pp.run(RECORDS, 10, rescue_only)
    assert out["rescued_migration"] == 1.0 and out["first_pass_migration"] == 0.0
    assert out["delta_pp"] == 100.0 and out["rule_p1_disable_rescue"] is True


def test_run_with_no_records_reports_none():
    out = pp.run([], 10, lambda pairs: [])
    assert out["rescued_migration"] is None and out["delta_pp"] is None
    assert out["rule_p1_disable_rescue"] is False
