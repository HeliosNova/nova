"""Source-independence layer (2026-08-12) — the anti-laundering fix for the
credibility-not-provenance gap (full-system exploration 2026-07-09 #1 risk).

Mirror/syndication networks must count as ONE source in every corroboration
path: figure support, the lead-credibility gate, and the evidence-pack tags
the synthesizer calibrates on. Authority floors close the paraphrase edge for
figures. All deterministic — no LLM, no network."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


WIRE_COPY = (
    "Federal regulators approved the merger on Monday. The combined company "
    "will control 1,234,567 subscriber accounts across nine states, executives "
    "said, with annual revenue projected at $8,765,432 for the first year. "
    "Consumer groups criticized the decision and promised court challenges."
)
ORIGINAL_ANALYSIS = (
    "Our review of the filing shows the merged entity reaches 1,234,567 "
    "subscriber accounts, a figure regulators confirmed. Unlike the wire "
    "reports, we obtained the internal integration memo describing layoffs "
    "in three regional offices and a two-year systems migration plan."
)
UNRELATED = (
    "The city council voted to expand the harbor dredging project, citing "
    "shipping delays. Contractors estimate completion in eighteen months, "
    "with environmental monitoring required at every stage of the work."
)


def _mirror(text: str) -> str:
    """A light mirror: same body with a trivial prefix (how farms restamp)."""
    return "Breaking news update: " + text


class TestClustering:
    def test_verbatim_mirrors_share_cluster(self):
        from app.monitors.deep_research import _independence_clusters
        arts = [("A", "https://farm-one.com/x", WIRE_COPY),
                ("B", "https://farm-two.com/y", WIRE_COPY),
                ("C", "https://indie.com/z", UNRELATED)]
        art_c, host_c = _independence_clusters(arts)
        assert art_c[0] == art_c[1] != art_c[2]
        assert host_c["farm-one.com"] == host_c["farm-two.com"] != host_c["indie.com"]

    def test_near_duplicate_restamp_caught(self):
        from app.monitors.deep_research import _independence_clusters
        arts = [("A", "https://farm-one.com/x", WIRE_COPY),
                ("B", "https://farm-two.com/y", _mirror(WIRE_COPY))]
        art_c, _ = _independence_clusters(arts)
        assert art_c[0] == art_c[1]

    def test_original_analysis_stays_independent(self):
        from app.monitors.deep_research import _independence_clusters
        arts = [("A", "https://wire-copy-a.com/x", WIRE_COPY),
                ("B", "https://real-outlet.com/y", ORIGINAL_ANALYSIS)]
        art_c, host_c = _independence_clusters(arts)
        assert art_c[0] != art_c[1]
        assert host_c["wire-copy-a.com"] != host_c["real-outlet.com"]

    def test_empty_bodies_safe(self):
        from app.monitors.deep_research import _independence_clusters
        arts = [("A", "https://a.com/x", ""), ("B", "https://b.com/y", None)]
        art_c, host_c = _independence_clusters(arts)
        assert art_c[0] != art_c[1]          # nothing to compare → independent


class TestFigureSupport:
    def test_mirrors_count_once(self):
        from app.monitors.deep_research import _figure_support
        arts = [("A", "https://farm-one.com/x", WIRE_COPY),
                ("B", "https://farm-two.com/y", _mirror(WIRE_COPY)),
                ("C", "https://farm-three.com/z", WIRE_COPY)]
        sup = _figure_support("Subscribers hit 1,234,567 this quarter.", arts)
        (count, hosts), = [v for k, v in sup.items() if "1,234,567" in k]
        assert count == 1                    # three domains, ONE independent source
        assert len(hosts) == 3               # display set keeps the hosts

    def test_wire_plus_original_counts_two(self):
        from app.monitors.deep_research import _figure_support
        arts = [("A", "https://wire-copy-a.com/x", WIRE_COPY),
                ("B", "https://real-outlet.com/y", ORIGINAL_ANALYSIS)]
        sup = _figure_support("Subscribers hit 1,234,567 this quarter.", arts)
        (count, _hosts), = [v for k, v in sup.items() if "1,234,567" in k]
        assert count == 2


class TestCorroborateNumbers:
    def _arts_independent(self):
        return [("A", "https://alpha-site.com/x", WIRE_COPY),
                ("B", "https://beta-site.com/y", ORIGINAL_ANALYSIS)]

    def test_mirror_pair_never_confirms(self):
        from app.monitors import deep_research as dr
        arts = [("A", "https://farm-one.com/x", WIRE_COPY),
                ("B", "https://farm-two.com/y", _mirror(WIRE_COPY))]
        _, confirmed = asyncio.run(dr._corroborate_numbers(
            "Subscribers hit 1,234,567 this quarter.", arts))
        assert confirmed == set()

    def test_independent_pair_with_credible_host_confirms(self):
        from app.monitors import deep_research as dr
        with patch.object(dr, "_sa_authority", lambda h: 0.9):
            _, confirmed = asyncio.run(dr._corroborate_numbers(
                "Subscribers hit 1,234,567 this quarter.", self._arts_independent()))
        assert any("1,234,567" in f for f in confirmed)

    def test_junk_only_pair_never_confirms(self):
        # Paraphrase-farm edge: two INDEPENDENT texts, both junk-tier — text
        # dedup can't see it, the authority floor must.
        from app.monitors import deep_research as dr
        with patch.object(dr, "_sa_authority", lambda h: 0.1):
            _, confirmed = asyncio.run(dr._corroborate_numbers(
                "Subscribers hit 1,234,567 this quarter.", self._arts_independent()))
        assert confirmed == set()

    def test_three_independent_junk_clusters_confirm(self):
        from app.monitors import deep_research as dr
        arts = self._arts_independent() + [
            ("C", "https://gamma-site.com/z",
             "Officials counted 1,234,567 subscriber accounts in the audit, "
             "a total disputed by two advocacy organizations this week.")]
        with patch.object(dr, "_sa_authority", lambda h: 0.1):
            _, confirmed = asyncio.run(dr._corroborate_numbers(
                "Subscribers hit 1,234,567 this quarter.", arts))
        assert any("1,234,567" in f for f in confirmed)


class TestLeadGate:
    LEAD = ("**Lead Development — Big Merger Approved**\n\n"
            "The merger was approved with 1,234,567 subscribers affected "
            "(farm-one.com). Regulators signed off on Monday (farm-two.com).\n\n"
            "**Secondary Developments**\n- x")

    def test_mirror_citations_get_gated(self):
        from app.monitors import deep_research as dr
        clusters = {"farm-one.com": 7, "farm-two.com": 7}
        with patch.object(dr, "_sa_authority", lambda h: 0.4):
            out, gated = dr._gate_lead_credibility(self.LEAD, host_clusters=clusters)
        assert gated is True and "Sourcing note" in out

    def test_independent_citations_pass(self):
        from app.monitors import deep_research as dr
        clusters = {"farm-one.com": 7, "farm-two.com": 8}
        with patch.object(dr, "_sa_authority", lambda h: 0.4):
            _, gated = dr._gate_lead_credibility(self.LEAD, host_clusters=clusters)
        assert gated is False

    def test_no_cluster_map_is_legacy_behavior(self):
        from app.monitors import deep_research as dr
        with patch.object(dr, "_sa_authority", lambda h: 0.4):
            _, gated = dr._gate_lead_credibility(self.LEAD, host_clusters=None)
        assert gated is False                 # two hosts, no map → counts as two


class TestEvidenceTags:
    def test_mirror_hosts_tag_one_source(self):
        from app.monitors.deep_research import _annotated_evidence
        findings = [("Regulators approve giant telecom merger deal", "https://farm-one.com/x", "f1"),
                    ("Regulators approve giant telecom merger deal", "https://farm-two.com/y", "f2")]
        clusters = {"farm-one.com": 7, "farm-two.com": 7}
        out = _annotated_evidence(findings, host_clusters=clusters)
        assert "1 source" in out and "2 sources" not in out

    def test_independent_hosts_tag_two_sources(self):
        from app.monitors.deep_research import _annotated_evidence
        findings = [("Regulators approve giant telecom merger deal", "https://a-site.com/x", "f1"),
                    ("Regulators approve giant telecom merger deal", "https://b-site.com/y", "f2")]
        out = _annotated_evidence(findings, host_clusters={"a-site.com": 1, "b-site.com": 2})
        assert "2 sources" in out


class TestTemporalNetworks:
    @pytest.fixture()
    def db(self, tmp_path):
        from app.database import SafeDB
        d = SafeDB(str(tmp_path / "cooc_test.db"))
        d.init_schema()
        yield d
        d.close()

    def test_record_and_flag_junk_network(self, db):
        from app.monitors import deep_research as dr
        # Two junk hosts co-occur in 8 straight digests → flagged.
        for _ in range(8):
            dr._record_host_cooccurrence(db, ["shady-a.com", "shady-b.com", "reuters.com"])
        with patch.object(dr, "_sa_authority", lambda h: 0.2):
            pairs = dr._network_pairs(db)
        assert ("shady-a.com", "shady-b.com") in pairs
        # reuters pairs are excluded by the hand-tier guard even at 100% ratio.
        assert all("reuters.com" not in p for p in pairs)

    def test_low_ratio_pair_not_flagged(self, db):
        from app.monitors import deep_research as dr
        for _ in range(8):
            dr._record_host_cooccurrence(db, ["shady-a.com", "shady-b.com"])
        for _ in range(8):   # shady-a appears in 8 more digests WITHOUT shady-b
            dr._record_host_cooccurrence(db, ["shady-a.com", "other.com"])
        for _ in range(8):   # ...and shady-b in 8 more without shady-a
            dr._record_host_cooccurrence(db, ["shady-b.com", "another.com"])
        with patch.object(dr, "_sa_authority", lambda h: 0.2):
            pairs = dr._network_pairs(db)
        assert ("shady-a.com", "shady-b.com") not in pairs   # ratio 8/16 = 0.5 < 0.8

    def test_apply_merges_clusters(self, db):
        from app.monitors import deep_research as dr
        for _ in range(8):
            dr._record_host_cooccurrence(db, ["shady-a.com", "shady-b.com"])
        clusters = {"shady-a.com": 1, "shady-b.com": 2, "indie.com": 3}
        with patch.object(dr, "_sa_authority", lambda h: 0.2):
            out = dr._apply_network_pairs(db, clusters)
        assert out["shady-a.com"] == out["shady-b.com"]
        assert out["indie.com"] not in (out["shady-a.com"],)


class TestAuthorityFolding:
    def test_dataset_reputable_becomes_lead_anchor(self):
        from app.monitors import deep_research as dr
        with patch.object(dr, "_sa_authority", lambda h: 0.9):
            assert dr._source_quality("https://some-quality-outlet-xyz.com/a") == 2.0

    def test_dataset_farm_drops_below_floor(self):
        from app.monitors import deep_research as dr
        with patch.object(dr, "_sa_authority", lambda h: 0.1):
            assert dr._source_quality("https://some-shady-farm-xyz.com/a") == 0.4

    def test_unknown_keeps_generic_floor(self):
        from app.monitors import deep_research as dr
        with patch.object(dr, "_sa_authority", lambda h: 0.5):
            assert dr._source_quality("https://some-bland-site-xyz.com/a") == 1.0

    def test_hand_tiers_still_win(self):
        from app.monitors import deep_research as dr
        with patch.object(dr, "_sa_authority", lambda h: 0.1):
            # A tier-1 wire keeps 3.0 regardless of any dataset value.
            assert dr._source_quality("https://reuters.com/article") == 3.0
