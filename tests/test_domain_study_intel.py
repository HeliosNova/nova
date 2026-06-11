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
