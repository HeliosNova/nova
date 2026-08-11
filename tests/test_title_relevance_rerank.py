"""Topic-relevance reranker for deep-research read-set selection (#38, 2026-07-08).

Ordering-only + fails-open. These pin the contract: on-topic titles score
higher, and any embedder failure yields an empty map (no effect on ranking).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.monitors import deep_research as dr


def _pick(url, title):
    return SimpleNamespace(url=url, title=title)


class TestTitleRelevance:
    @pytest.mark.asyncio
    async def test_on_topic_title_scores_higher(self):
        picks = [
            _pick("https://a.com/1", "Central bank raises interest rates amid inflation"),
            _pick("https://b.com/2", "Local bakery wins dessert award"),
        ]

        # Deterministic fake embedder: vector = [presence of 'rate', 'inflation',
        # 'bakery'] so cosine reflects topical overlap with the subject.
        def fake_ef(texts):
            def vec(t):
                t = t.lower()
                return [
                    1.0 if ("rate" in t or "inflation" in t or "monetary" in t) else 0.0,
                    1.0 if "bakery" in t or "dessert" in t else 0.0,
                    0.1,
                ]
            return [vec(t) for t in texts]

        with patch("app.core.embedding.get_embedding_function", return_value=fake_ef):
            scores = await dr._title_relevance(picks, ["monetary policy and inflation"])
        assert scores["https://a.com/1"] > scores["https://b.com/2"]

    @pytest.mark.asyncio
    async def test_fails_open_when_embedder_none(self):
        picks = [_pick("https://a.com/1", "Something")]
        with patch("app.core.embedding.get_embedding_function", return_value=None):
            scores = await dr._title_relevance(picks, ["topic"])
        assert scores == {}

    @pytest.mark.asyncio
    async def test_fails_open_on_exception(self):
        picks = [_pick("https://a.com/1", "Something")]
        def boom(texts):
            raise RuntimeError("embedder down")
        with patch("app.core.embedding.get_embedding_function", return_value=boom):
            scores = await dr._title_relevance(picks, ["topic"])
        assert scores == {}

    @pytest.mark.asyncio
    async def test_empty_inputs_no_call(self):
        assert await dr._title_relevance([], ["topic"]) == {}
        assert await dr._title_relevance([_pick("u", "t")], []) == {}

    @pytest.mark.asyncio
    async def test_negative_cosine_floored_to_zero(self):
        picks = [_pick("https://a.com/1", "anti-correlated")]

        def fake_ef(texts):
            # query vec [1,0], title vec [-1,0] -> cosine -1 -> floored to 0
            return [[1.0, 0.0] if i == 0 else [-1.0, 0.0] for i, _ in enumerate(texts)]

        with patch("app.core.embedding.get_embedding_function", return_value=fake_ef):
            scores = await dr._title_relevance(picks, ["topic"])
        assert scores["https://a.com/1"] == 0.0
