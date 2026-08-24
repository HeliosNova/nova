"""SEC Form 4 intelligence: URL parsing, XML transaction parsing, signal rendering,
and cluster-buy detection. The parse turns a raw filing-list dump into buy/sell + $
value + a bullish-cluster signal (owner: "links instead of what we need")."""
from __future__ import annotations

import pytest

from app.monitors import domain_study_runner as dsr


def test_form4_dir_extracts_cik_and_accession():
    url = ("https://www.sec.gov/Archives/edgar/data/1033767/000119312526291233/"
           "0001193125-26-291233-index.htm")
    assert dsr._form4_dir(url) == ("1033767", "000119312526291233")
    assert dsr._form4_dir("https://example.com/nope") is None


def test_fmt_usd_compact():
    assert dsr._fmt_usd(1_700_000) == "$1.7M"
    assert dsr._fmt_usd(87_000) == "$87K"
    assert dsr._fmt_usd(2.4e9) == "$2.4B"
    assert dsr._fmt_usd(920) == "$920"


def test_sec_signal_line_buy_sell_and_routine():
    buy = {"direction": "buy", "buy_value": 1_700_000, "buy_shares": 10_000,
           "sell_value": 0, "sell_shares": 0, "codes": ["P"]}
    assert dsr._sec_signal_line(buy) == "🟢 **BUY** $1.7M (10,000 sh)"
    sell = {"direction": "sell", "sell_value": 500_000, "sell_shares": 4_000,
            "buy_value": 0, "buy_shares": 0, "codes": ["S"]}
    assert dsr._sec_signal_line(sell) == "🔴 **SELL** $500K (4,000 sh)"
    # routine comp events must be labeled honestly, NOT shown as a trade
    exercise = {"direction": "other", "buy_value": 0, "sell_value": 0, "codes": ["M", "M"]}
    assert dsr._sec_signal_line(exercise) == "⚪ option exercise"
    grant = {"direction": "other", "buy_value": 0, "sell_value": 0, "codes": ["A"]}
    assert dsr._sec_signal_line(grant) == "⚪ grant/award"
    assert dsr._sec_signal_line(None) == ""


def test_detect_sec_clusters_flags_multi_insider_buys():
    class _It:
        def __init__(self, title, f4):
            self.title = title
            self.meta = {"form4": f4} if f4 else {}
    buy = lambda v: {"direction": "buy", "buy_value": v, "buy_shares": v / 10}
    items = [
        _It("Acme Corp — insider: A (Form 4)", buy(1_000_000)),
        _It("Acme Corp — insider: B (Form 4)", buy(2_000_000)),   # 2nd insider → cluster
        _It("Beta Inc — insider: C (Form 4)", buy(5_000_000)),    # lone buyer → not a cluster
        _It("Gamma Ltd — insider: D (Form 4)", {"direction": "other", "buy_value": 0}),
    ]
    clusters = dsr._detect_sec_clusters(items)
    assert len(clusters) == 1
    assert clusters[0]["issuer"] == "Acme Corp"
    assert clusters[0]["insiders"] == 2
    assert clusters[0]["total_value"] == 3_000_000


@pytest.mark.asyncio
async def test_fetch_form4_txn_parses_transactions():
    # P (open-market buy) → buy $; a derivative M → recorded as a code (labelable).
    xml = ("<?xml version='1.0'?><ownershipDocument><nonDerivativeTable>"
           "<nonDerivativeTransaction><transactionCoding><transactionCode>P</transactionCode>"
           "</transactionCoding><transactionAmounts>"
           "<transactionShares><value>1000</value></transactionShares>"
           "<transactionPricePerShare><value>50.00</value></transactionPricePerShare>"
           "</transactionAmounts></nonDerivativeTransaction></nonDerivativeTable>"
           "<derivativeTable><derivativeTransaction><transactionCoding>"
           "<transactionCode>M</transactionCode></transactionCoding></derivativeTransaction>"
           "</derivativeTable></ownershipDocument>")

    class _Resp:
        def __init__(self, payload, is_json):
            self._p, self._j = payload, is_json
        @property
        def text(self):
            return self._p
        def json(self):
            return self._p

    class _Client:
        async def get(self, url):
            if url.endswith("index.json"):
                return _Resp({"directory": {"item": [{"name": "form4.xml"}]}}, True)
            return _Resp(xml, False)

    out = await dsr._fetch_form4_txn(
        "https://www.sec.gov/Archives/edgar/data/1/000000000000000001/x-index.htm", _Client())
    assert out["direction"] == "buy"
    assert out["buy_shares"] == 1000 and out["buy_value"] == 50_000
    assert "P" in out["codes"] and "M" in out["codes"]   # derivative code captured for labeling


# --- GitHub advisory roll-up + gov-contract parsing ---------------------------

class _MetaIt:
    def __init__(self, meta):
        self.meta = meta


def test_rollup_advisories_counts_and_act_on():
    items = [
        _MetaIt({"advisory": {"severity": "critical", "packages": ["openssl"], "cvss": 9.8}}),
        _MetaIt({"advisory": {"severity": "high", "packages": ["twig/twig"], "cvss": 7.5}}),
        _MetaIt({"advisory": {"severity": "high", "packages": ["lodash"]}}),
        _MetaIt({"advisory": {"severity": "low", "packages": ["foo"]}}),
    ]
    line = dsr._rollup_advisories(items)
    assert "1 critical" in line and "2 high" in line and "1 low" in line
    assert "patch now" in line and "openssl" in line          # critical package surfaced
    assert dsr._rollup_advisories([]) is None


def test_advisory_badge():
    assert dsr._advisory_badge({"cvss": 9.8, "cve": "CVE-2026-1"}) == "🔺 CVSS 9.8  ·  CVE-2026-1"
    assert dsr._advisory_badge({}) == ""


def test_parse_dod_contracts_aggregates():
    body = ("Lockheed Martin Corp., Bethesda, Maryland, is awarded a $500,000,000 modification "
            "for Army systems support. Raytheon Co. is awarded $1.2 billion for Navy radar "
            "development. A small line item of $50 should be ignored. Boeing gets $250 million "
            "for an Air Force contract this quarter.")
    d = dsr._parse_dod_contracts(body)
    assert d["count"] == 3                                    # $50 excluded (< $100k)
    assert abs(d["total"] - 1.95e9) < 1.0
    assert d["branches"] == {"Army": 1, "Navy": 1, "Air Force": 1}
    assert dsr._parse_dod_contracts("too short") is None


def test_contracts_rollup_line():
    items = [_MetaIt({"contracts": {"total": 1e9, "count": 2, "branches": {"Navy": 2}}}),
             _MetaIt({"contracts": {"total": 5e8, "count": 1, "branches": {"Army": 1}}})]
    line = dsr._contracts_rollup_line(items)
    assert "$1.5B ceiling across 3 awards" in line and "Navy 2" in line
