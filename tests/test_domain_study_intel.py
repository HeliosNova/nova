"""Cross-referencing + insight for domain-study digests (the 'sources,
cross-references, actual insight' upgrade). Cross-reference is deterministic
and unit-tested; insight is LLM-backed so we only test its gating."""
import asyncio
from app.monitors.domain_study_runner import (
    _cross_reference, _story_key_tokens, _synthesize_insight,
    _render_items_deterministic,
)
from datetime import datetime, timezone


class TestCrossReference:
    def test_same_story_different_outlets_corroborate(self):
        items = [
            {"title": "OpenAI signs $10B compute deal with Oracle", "outlet": "reuters.com"},
            {"title": "Oracle and OpenAI ink $10 billion cloud compute agreement", "outlet": "bloomberg.com"},
        ]
        _cross_reference(items)
        assert "bloomberg.com" in items[0]["_corroborating"]
        assert "reuters.com" in items[1]["_corroborating"]

    def test_unrelated_stories_do_not_match(self):
        items = [
            {"title": "OpenAI signs compute deal with Oracle", "outlet": "reuters.com"},
            {"title": "Starlink users pay monthly hardware rental fee", "outlet": "cnet.com"},
        ]
        _cross_reference(items)
        assert not items[0].get("_corroborating")
        assert not items[1].get("_corroborating")

    def test_same_outlet_does_not_self_corroborate(self):
        items = [
            {"title": "Fed holds rates steady amid inflation data", "outlet": "wsj.com"},
            {"title": "Fed holds interest rates steady on inflation", "outlet": "wsj.com"},
        ]
        _cross_reference(items)
        assert not items[0].get("_corroborating")

    def test_story_key_drops_generic_news_words(self):
        toks = _story_key_tokens("New report: latest update on the news today")
        assert "report" not in toks and "news" not in toks and "today" not in toks

    def test_render_shows_corroboration_and_insight(self):
        items = [
            {"title": "OpenAI signs compute deal with Oracle", "outlet": "reuters.com",
             "snippet": "OpenAI agreed to a multiyear cloud compute deal.", "date_str": "June 11, 2026",
             "url": "https://reuters.com/x", "_title_only": False},
            {"title": "Oracle inks compute agreement with OpenAI", "outlet": "bloomberg.com",
             "snippet": "Oracle will supply cloud capacity to OpenAI.", "date_str": "June 11, 2026",
             "url": "https://bloomberg.com/y", "_title_only": False},
        ]
        out = _render_items_deterministic("AI", "🤖", items, datetime.now(timezone.utc),
                                          insight="Compute access is the new moat.")
        assert "Confirmed by 2 outlets" in out
        assert "💡 **Insight**" in out
        assert "Compute access is the new moat." in out


class TestInsightGating:
    def test_too_few_items_returns_empty(self):
        items = [{"title": "Only one thing happened"}, {"title": "Second thing"}]
        assert asyncio.run(_synthesize_insight("AI", items)) == ""


class TestNewsworthiness:
    def test_drops_coupon_and_deal_content(self):
        from app.monitors.domain_study_runner import _is_newsworthy
        for title, url in [
            ("Dell Coupon Codes: 20% Off for June 2026", "https://wired.com/story/dell-coupon-code/"),
            ("The best time-tracking software of 2026: Expert tested", "https://zdnet.com/article/best-time-tracking-software/"),
            ("Today's best deals: 4K TVs and more", "https://engadget.com/deals/x"),
            ("Save $600 off Alienware laptops with this promo code", "https://x.com/a"),
            ("Father's Day gift guide 2026", "https://x.com/buying-guide/g"),
        ]:
            assert _is_newsworthy({"title": title, "url": url}) is False, title

    def test_keeps_real_news_including_ma_deal(self):
        from app.monitors.domain_study_runner import _is_newsworthy
        for title, url in [
            ("OpenAI signs $10B compute deal with Oracle", "https://reuters.com/tech/x"),
            ("US and EU reach trade deal on semiconductors", "https://bloomberg.com/y"),
            ("Deezer launches an AI music detector", "https://theverge.com/ai/z"),
            ("Fed holds rates steady amid inflation data", "https://wsj.com/econ/w"),
        ]:
            assert _is_newsworthy({"title": title, "url": url}) is True, title


class TestSourceAuthority:
    """Dataset-backed authority (Lin et al. PNAS Nexus 2023) + primary-source rules."""
    def test_wire_and_primary_top(self):
        from app.core.source_authority import authority
        assert authority("reuters.com") >= 0.97
        assert authority("apnews.com") >= 0.95
        assert authority("sec.gov") == 1.0
        assert authority("fda.gov") == 1.0
        assert authority("anytown.courts.gov") == 1.0  # .gov suffix rule

    def test_reputable_vs_lowcred(self):
        from app.core.source_authority import authority
        assert authority("nytimes.com") > 0.7
        assert authority("infowars.com") < 0.2
        assert authority("breitbart.com") < 0.4

    def test_unknown_is_neutral(self):
        from app.core.source_authority import authority
        assert authority("some-unknown-blog-xyz123.net") == 0.5

    def test_subdomain_and_cctld_resolve(self):
        from app.core.source_authority import authority
        assert authority("www.theverge.com") > 0.7
        assert authority("https://www.bbc.co.uk/news") > 0.5  # registrable cctld


class TestSyndicationCollapse:
    """Wire reprints = ONE source, not N (the corroboration correctness fix)."""
    def _wire(self, outlet):
        return {"title": "OpenAI Oracle ten billion compute deal", "outlet": outlet,
                "snippet": "(Reuters) OpenAI agreed a multiyear cloud compute deal with Oracle worth ten billion dollars, sources said Thursday."}

    def test_reprints_detected(self):
        from app.monitors.domain_study_runner import _is_syndicated
        assert _is_syndicated(self._wire("yahoo.com"), self._wire("msn.com")) is True

    def test_independent_not_syndicated(self):
        from app.monitors.domain_study_runner import _is_syndicated
        indep = {"title": "OpenAI signs Oracle cloud compute agreement", "outlet": "bloomberg.com",
                 "snippet": "Bloomberg analysis finds the Oracle pact reshapes the compute market and pressures rivals to lock in scarce capacity ahead of the next training cycle."}
        assert _is_syndicated(self._wire("yahoo.com"), indep) is False

    def test_collapse_keeps_one_highest_authority(self):
        from app.monitors.domain_study_runner import _collapse_syndication
        items = [self._wire("yahoo.com"), self._wire("msn.com"), self._wire("reuters.com")]
        out = _collapse_syndication(items)
        assert len(out) == 1
        assert out[0]["outlet"] == "reuters.com"  # highest authority survives

    def test_corroboration_counts_independent_only(self):
        from app.monitors.domain_study_runner import _collapse_syndication, _cross_reference
        indep = {"title": "OpenAI Oracle compute agreement reshapes market", "outlet": "bloomberg.com",
                 "snippet": "Bloomberg's independent analysis of the Oracle pact and its market consequences for compute-hungry rivals."}
        items = _collapse_syndication([self._wire("yahoo.com"), self._wire("msn.com"), indep])
        _cross_reference(items)
        # after collapse: 1 wire + 1 independent = the wire is corroborated by exactly 1 independent outlet
        wire = [i for i in items if "reuters" in (i.get("snippet","").lower())][0]
        assert len(wire.get("_corroborating") or []) == 1


class TestImportanceRanking:
    def test_authority_and_corroboration_win(self):
        from app.monitors.domain_study_runner import _importance_rank
        items = [
            {"title": "minor blog item", "outlet": "random-blog.xyz", "_corroborating": []},
            {"title": "big story", "outlet": "reuters.com", "_corroborating": ["bbc.com", "nytimes.com"]},
            {"title": "mid", "outlet": "theverge.com", "_corroborating": []},
        ]
        ranked = _importance_rank(items)
        assert ranked[0]["outlet"] == "reuters.com"  # authority + corroboration
        assert ranked[-1]["outlet"] == "random-blog.xyz"


class TestDirectedDeepDive:
    """Helios analyst pattern: directed per-story follow-up → primary source +
    independent corroboration. The jump from clipping service to analyst."""

    def _run(self, item, fake_results):
        import asyncio
        from unittest.mock import patch, AsyncMock
        import app.tools.native_search as ns
        import app.monitors.domain_study_runner as dr
        with patch.object(ns, "search", new=AsyncMock(return_value=fake_results)):
            asyncio.run(dr._directed_followup("Finance", item))

    def test_traces_to_primary_source(self):
        from types import SimpleNamespace
        item = {"title": "Federal Reserve holds interest rates steady at June meeting",
                "outlet": "randomblog.xyz", "snippet": "The Fed kept rates unchanged."}
        fake = [SimpleNamespace(
            url="https://federalreserve.gov/newsevents/pressreleases/monetary20260611a.htm",
            title="Federal Reserve issues FOMC statement holding interest rates steady",
            snippet="The Committee decided to maintain the target range.", engine="x", published_date="")]
        self._run(item, fake)
        assert item.get("_primary_source", {}).get("outlet") == "federalreserve.gov"
        assert item["_primary_source"]["authority"] == 1.0

    def test_adds_independent_corroboration(self):
        from types import SimpleNamespace
        item = {"title": "OpenAI signs ten billion Oracle compute deal",
                "outlet": "techblog.example", "snippet": "OpenAI and Oracle struck a deal."}
        fake = [SimpleNamespace(
            url="https://reuters.com/x", title="OpenAI signs Oracle ten billion compute deal",
            snippet="Reuters' own reporting on the Oracle compute agreement and its market impact across the cloud sector.",
            engine="x", published_date="")]
        self._run(item, fake)
        assert "reuters.com" in (item.get("_corroborating") or [])

    def test_search_failure_is_silent(self):
        import asyncio
        from unittest.mock import patch, AsyncMock
        import app.tools.native_search as ns
        import app.monitors.domain_study_runner as dr
        item = {"title": "Some major story about a thing", "outlet": "x.com", "snippet": "body"}
        with patch.object(ns, "search", new=AsyncMock(side_effect=RuntimeError("searxng down"))):
            asyncio.run(dr._directed_followup("Finance", item))  # must not raise
        assert "_primary_source" not in item
