"""Regression tests for the 2026-08-16 source-breadth fixes:
- The Economic Times (legit business daily) was scored 0.216 in the PC1 dataset
  and demoted under the 0.3 credibility gate, starving finance/India digests.
- The Open Source/GitHub monitor lacked the real GitHub-trending feed.
"""

from __future__ import annotations


def test_economictimes_allowlisted_above_demotion_gate():
    from app.core.source_authority import authority
    # allow-listed legit financial paper is now above the 0.3 demote gate
    assert authority("economictimes.indiatimes.com") >= 0.3
    assert authority("m.economictimes.com") >= 0.3


def test_genuine_lowcred_still_demoted():
    from app.core.source_authority import authority
    # the allow-list is surgical — a real low-quality finance blog stays demoted
    assert authority("zerohedge.com") < 0.3


def test_github_trending_feed_wired():
    from app.monitors.rss_feeds import feeds_for
    feeds = feeds_for("Domain Study: Open Source and GitHub")
    assert any("GitHubTrendingRSS" in f for f in feeds), \
        "GitHub trending feed should be wired into the open-source/github monitor"
