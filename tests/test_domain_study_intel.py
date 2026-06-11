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
