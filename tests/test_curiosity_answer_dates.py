"""(b) Date discipline in curiosity answers (2026-10-17, staged for k10).

A resolved curiosity answer in October presented a Diet session "scheduled to
end July 17, 2026" as current: the evidence carried no dates and the prompt
said nothing about what a past date means. On the curiosity path only, each
finding's tag now ends with the article's publication date, and the prompt
says a date before today has happened. The digest chain's tags are unchanged.
"""

from __future__ import annotations

import inspect

from app.monitors import deep_research as dr

FINDINGS = [("Diet session ends", "https://japantimes.co.jp/a", "The session was scheduled to end July 17."),
            ("Budget passes", "https://nhk.or.jp/b", "The budget passed on July 15.")]


def test_curiosity_tags_carry_the_publication_date(monkeypatch):
    monkeypatch.setitem(dr._PUB_DATES, "https://japantimes.co.jp/a", "2026-07-14")
    monkeypatch.delitem(dr._PUB_DATES, "https://nhk.or.jp/b", raising=False)
    ev = dr._annotated_evidence(FINDINGS, with_dates=True)
    first, second = ev.split("\n\n")
    assert first.startswith("[Diet session ends] (japantimes.co.jp") and "· published 2026-07-14)" in first
    assert "· publication date unknown)" in second


def test_digest_tags_are_unchanged(monkeypatch):
    monkeypatch.setitem(dr._PUB_DATES, "https://japantimes.co.jp/a", "2026-07-14")
    ev = dr._annotated_evidence(FINDINGS)
    assert "published" not in ev and "publication date" not in ev
    assert ev.startswith("[Diet session ends] (japantimes.co.jp")


def test_research_question_asks_for_dated_answers():
    src = inspect.getsource(dr.research_question)
    assert "_annotated_evidence(findings, with_dates=True)" in src
    assert "already happened" in src and "as of <month> <year>" in src
