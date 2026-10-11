"""G2 (plan 2026-10-09): researched facts carry world time.

Measured 2026-10-09: valid_from == created_at on 25,089 of 25,091 kg_facts —
the KG had transaction time only. deep_research already records each read
article's publication date (_PUB_DATES); _learn_facts now passes the earliest
publication date among the digest's articles as valid_from, and the digest
date when none is known — never None — and the triple prompt reads the
evidence AS OF that date.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.monitors import deep_research as dr

RAW = '{"facts": [{"subject": "Reuters", "predicate": "reports_on", "object": "copper tariffs"}]}'
FINDINGS = [
    ("t1", "https://reuters.com/a", "Reuters reports on copper tariffs."),
    ("t2", "https://apnews.com/b", "AP: copper tariffs reported by Reuters."),
]
ARTICLES = [
    ("t1", "https://reuters.com/a", "Reuters reports on copper tariffs in detail."),
    ("t2", "https://apnews.com/b", "Copper tariffs, reported by Reuters, rise."),
]


async def _run(kg):
    with patch.object(dr, "_invoke_bg", AsyncMock(return_value=RAW)):
        return await dr._learn_facts("copper", "Reuters reports on copper tariffs.",
                                     FINDINGS, kg, articles=ARTICLES)


@pytest.mark.asyncio
async def test_earliest_publication_date_becomes_valid_from():
    kg = AsyncMock(); kg.add_fact = AsyncMock(return_value=True)
    with patch.dict(dr._PUB_DATES, {"https://reuters.com/a": "2026-10-03",
                                    "https://apnews.com/b": "2026-09-30T08:00:00Z"}, clear=False):
        assert await _run(kg) == 1
    assert kg.add_fact.await_args.kwargs["valid_from"] == "2026-09-30"


@pytest.mark.asyncio
async def test_no_known_date_falls_back_to_the_digest_date_never_none():
    kg = AsyncMock(); kg.add_fact = AsyncMock(return_value=True)
    with patch.dict(dr._PUB_DATES, {}, clear=True):
        assert await _run(kg) == 1
    vf = kg.add_fact.await_args.kwargs["valid_from"]
    assert vf == datetime.now(timezone.utc).strftime("%Y-%m-%d")


@pytest.mark.asyncio
async def test_prompt_reads_the_evidence_as_of_the_date():
    kg = AsyncMock(); kg.add_fact = AsyncMock(return_value=True)
    seen = {}

    async def _capture(messages, **kw):
        seen["prompt"] = messages[0]["content"]
        return RAW

    with patch.dict(dr._PUB_DATES, {"https://reuters.com/a": "2026-10-03"}, clear=True), \
         patch.object(dr, "_invoke_bg", AsyncMock(side_effect=_capture)):
        await dr._learn_facts("copper", "Reuters reports on copper tariffs.", FINDINGS, kg, articles=ARTICLES)
    assert "AS OF: 2026-10-03" in seen["prompt"]


def test_earliest_date_helper_ignores_unparseable_and_uncited():
    with patch.dict(dr._PUB_DATES, {"https://reuters.com/a": "garbage",
                                    "https://apnews.com/b": "2026-09-30",
                                    "https://other.example/c": "2020-01-01"}, clear=True):
        assert dr._earliest_pub_date(ARTICLES, FINDINGS) == "2026-09-30"
    with patch.dict(dr._PUB_DATES, {}, clear=True):
        assert dr._earliest_pub_date(ARTICLES, FINDINGS) is None
