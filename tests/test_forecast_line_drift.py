"""The FORECAST line the model actually writes, replayed (2026-09-07).

44 storyline forecasts were lost in eleven days to "FORECAST line present but
not stored", and the warning never showed the line. Replaying the storyline
update prompt twelve times on the live 9B reproduced the loss at the same
rate (3 of 12) and showed the two shapes, both verbatim below:

1. STATE and FORECAST on ONE line - the model runs the two tail lines
   together, so `^FORECAST:` never matches, and the STATE parser swallows the
   forecast into the thread's status.
2. The `| resolves YYYY-MM-DD` segment omitted when the deadline is already
   inside the claim ("...by December 31, 2026 | 0.9 confidence").

The prompt is not changed: naming a forbidden form hands the model an exact
string to route around (CLAUDE.md, 2026-09-04). The parser reads what the
model writes. 'FORECAST: none' still mints nothing, and a claim with no
deadline anywhere still mints nothing - the discipline is the date, not the
punctuation.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from app.core import forecasts
from app.core.storylines import _STATE_RE


@pytest.fixture
def db(tmp_path):
    from app.database import SafeDB
    d = SafeDB(str(tmp_path / "f.db"))
    d.init_schema()
    yield d
    d.close()


def _resolves(db, fid) -> datetime:
    row = db.fetchone("SELECT resolves_at FROM forecasts WHERE id=?", (fid,))
    return datetime.strptime(row["resolves_at"][:19], "%Y-%m-%d %H:%M:%S")


def _claim(db, fid) -> str:
    return db.fetchone("SELECT claim FROM forecasts WHERE id=?", (fid,))["claim"]


# --- shape 1: STATE and FORECAST run together on one line --------------------

RUN_TOGETHER = (
    "The Treasury has begun buybacks.\n"
    "CHANGED: intervention started\n"
    "STATE: U.S. Sovereign Debt Crisis | Active intervention initiated FORECAST: "
    "Market yields for 30-year Treasuries remain below 5% by October 15, 2026 "
    "| resolves 2026-10-15 | 0.7"
)


def test_a_forecast_written_after_the_state_on_the_same_line_is_stored(db):
    fid = forecasts.parse_and_store_forecast(db, RUN_TOGETHER)
    assert fid, "the replayed line was dropped"
    assert _claim(db, fid).startswith("Market yields for 30-year Treasuries")
    assert _resolves(db, fid).date() == datetime(2026, 10, 15).date()


def test_the_state_parser_does_not_swallow_the_forecast():
    m = _STATE_RE.search(RUN_TOGETHER)
    assert m, "the STATE line itself is still a state"
    assert m.group("entity") == "U.S. Sovereign Debt Crisis"
    assert m.group("status") == "Active intervention initiated"


# --- shape 2: the resolves segment omitted, the deadline inside the claim -----

NO_RESOLVES = (
    "The global economy has locked into a stagflationary trap.\n"
    "FORECAST: Pantheon Macroeconomics will not revise the Philippines' 2026 GDP "
    "forecast above 3% by December 31, 2026 | 0.9 confidence"
)


def test_a_claim_carrying_its_own_deadline_needs_no_resolves_segment(db):
    fid = forecasts.parse_and_store_forecast(db, NO_RESOLVES)
    assert fid, "the replayed line was dropped"
    assert _resolves(db, fid).date() == datetime(2026, 12, 31).date()
    row = db.fetchone("SELECT confidence FROM forecasts WHERE id=?", (fid,))
    assert abs(row["confidence"] - 0.9) < 1e-6


def test_a_claim_with_no_deadline_anywhere_still_mints_nothing(db):
    assert forecasts.parse_and_store_forecast(
        db, "FORECAST: The consortium publishes its charter | 0.6 confidence") is None


# --- the opt-out and the small variants ------------------------------------

def test_forecast_none_still_mints_nothing(db):
    assert forecasts.parse_and_store_forecast(db, "CHANGED: x\nFORECAST: none") is None
    assert forecasts.parse_and_store_forecast(db, "STATE: X | quiet FORECAST: none") is None


def test_confidence_word_before_the_number_and_a_trailing_period(db):
    fid = forecasts.parse_and_store_forecast(
        db, "FORECAST: The widget ships to customers | resolves 2026-12-15 | confidence 0.7.")
    assert fid
    row = db.fetchone("SELECT confidence FROM forecasts WHERE id=?", (fid,))
    assert abs(row["confidence"] - 0.7) < 1e-6
    assert _resolves(db, fid).date() == datetime(2026, 12, 15).date()


def test_the_shapes_that_already_worked_still_work(db):
    for line in (
        "FORECAST: Hyperscalers will report capex above 100% of operating cash flow in Q4 2026 earnings | resolves 2027-03-15 | 0.8",
        "FORECAST: US PCE inflation will remain above 3% through Q4 2026 despite Fed interventions | resolves 2027-01-15 | 0.8 confidence",
        "FORECAST: The consortium publishes its charter | 90 days | 0.6",
    ):
        assert forecasts.parse_and_store_forecast(db, line), line
