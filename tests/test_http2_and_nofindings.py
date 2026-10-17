"""(d) HTTP/2 fetch + (c-ext) curiosity 0-findings parking (2026-10-17, k10).

Measured in the live log 10-10..10-17: Chromium's HTTP/2 negotiation failed
43-73x/day against a cluster of major news/wire servers (usnews.com 226,
washingtonpost.com 52, businesswire.com 25...), losing each as an article; and
23 of 79 curiosity attempts gathered >=3 sources yet extracted no finding, each
otherwise burning three research passes before requeueing.
"""

from __future__ import annotations

import inspect

import app.tools.browser as browser
import app.monitors.heartbeat_loop as hb


def test_chromium_is_launched_without_http2():
    src = inspect.getsource(browser.BrowserTool._launch)
    assert "--disable-http2" in src
    # and it sits inside the launch args list, not only a comment
    assert '"--disable-http2"' in src


def test_no_findings_with_sources_is_parked_before_the_requeue():
    src = inspect.getsource(hb.HeartbeatLoop._research_one_curiosity)
    i_park = src.index("reason=unsettled_no_findings")
    # the branch guards on reason and a source floor
    block = src[:i_park]
    assert 'grounding.get("reason") == "no_findings"' in block
    assert 'grounding.get("sources", 0) or 0) >= 3' in block
    assert "svc.curiosity.defer" in block
    # it is decided before the think()/closure path that would burn an attempt
    i_closure = src.index("closure check failed — requeued")
    assert i_park < i_closure


def test_the_no_findings_park_follows_the_degraded_search_guard():
    """<3 sources stays on the existing no_sources/degraded-search handling;
    the new branch only fires when sources were actually read (>=3)."""
    src = inspect.getsource(hb.HeartbeatLoop._research_one_curiosity)
    assert src.index('"no_sources"') < src.index("reason=unsettled_no_findings")
