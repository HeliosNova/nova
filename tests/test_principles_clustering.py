"""Principle-distillation clustering (2026-08-27).

Path B's old cluster key was the ALPHABETICALLY-FIRST two topic keywords,
which scattered natural lesson families across distinct keys — verified
against the live corpus: max cluster size 2 forever, min_cluster=3 never
reached, zero principles minted in the system's lifetime. The overlap-based
grouping joins lessons sharing >=2 substantive tokens.
"""
from app.core.principles import _cluster_by_overlap


def _rows(*topics):
    return [{"topic": t} for t in topics]


class TestClusterByOverlap:
    def test_family_with_shared_pair_clusters(self):
        # All share {art, history}: the old alphabetical-first-2 keying
        # scattered these into {art,factual}/{art,famous}/{art,questions}...
        clusters = _cluster_by_overlap(_rows(
            "Factual Art History Questions",
            "Famous Art History Questions",
            "Art History Verification",
            "Nova scheduler codename",
        ))
        sizes = sorted(len(m) for _k, m in clusters)
        assert sizes == [1, 3]

    def test_label_is_most_common_tokens(self):
        clusters = _cluster_by_overlap(_rows(
            "Factual Art History Questions",
            "Famous Art History Questions",
            "Art History Verification",
        ))
        (label, members), = [c for c in clusters if len(c[1]) == 3]
        assert label == frozenset({"art", "history"})

    def test_disjoint_topics_stay_separate(self):
        clusters = _cluster_by_overlap(_rows(
            "Calculator usage for math",
            "Fed policy under chair Warsh",
            "SpaceX Starship regulatory milestones",
        ))
        assert all(len(m) == 1 for _k, m in clusters)

    def test_single_keyword_topics_skipped(self):
        clusters = _cluster_by_overlap(_rows("Math", "Math", "Math"))
        assert clusters == []
