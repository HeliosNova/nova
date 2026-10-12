"""Item 1 (2026-10-12): the per-domain feeds are the breadth now that the
search layer is down to two general engines."""

from __future__ import annotations

from app.monitors.rss_feeds import _FEEDS, feeds_for

DEAD_OR_BLOCKED = {
    "https://english.kyodonews.net/rss/news.xml", "https://focustaiwan.tw/rss",
    "https://kyivindependent.com/feed/", "https://mg.co.za/feed/", "https://news.ycombinator.com/rss",
    "https://www.nature.com/subjects/quantum-information.rss", "https://www.nature.com/subjects/medical-research.rss",
    "https://www.nature.com/subjects/physics.rss", "https://openai.com/blog/rss/", "https://www.lawfaremedia.org/all-content.rss",
    "https://www.fiercepharma.com/rss/xml", "https://www.fiercebiotech.com/rss/xml", "https://devops.com/feed/",
    "https://www.crunchbase.com/feed", "https://www.symmetrymagazine.org/feed", "https://www.politico.com/rss/politicopicks.xml",
    "https://www.euractiv.com/feed/", "https://www.barrons.com/feed/rss",
}


def test_no_known_dead_feed_is_listed():
    listed = {u for v in _FEEDS.values() for u in v}
    assert not (listed & DEAD_OR_BLOCKED), listed & DEAD_OR_BLOCKED


def test_every_economics_domain_study_has_feeds():
    for name in ("Domain Study: Macroeconomics", "Domain Study: Microeconomics", "Domain Study: Economics and Markets"):
        assert len(feeds_for(name)) >= 5, name


def test_no_feed_url_is_listed_twice_within_a_domain():
    for k, v in _FEEDS.items():
        assert len(v) == len(set(v)), k


def test_sources_per_digest_reads_the_overview_line(tmp_path):
    from datetime import date
    from app.monitors.engineering_report import sources_per_digest
    today = date.today().isoformat()
    log = tmp_path / "nova-app.log"
    log.write_text(
        f"{today} 01:00:00,000 [INFO] app.monitors.deep_research []: [DeepResearch] finance overview: read 18 sources from 9 distinct hosts (a, b), learned 4 facts\n"
        f"{today} 02:00:00,000 [INFO] app.monitors.deep_research []: [DeepResearch] China tech overview: read 3 sources from 2 distinct hosts (a, b), learned 1 facts\n"
        f"{today} 02:30:00,000 [INFO] app.monitors.deep_research []: [DeepResearch] China tech: read 2 sources (x, y), learned 0 facts\n",
        encoding="utf-8")
    out = sources_per_digest(1, log_glob=str(log), today=today)
    assert out == {"digests": 2, "median_hosts": 5.5, "min_hosts": 2, "thin": ["China tech"]}
