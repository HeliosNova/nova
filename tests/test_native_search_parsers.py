"""Bing HTML parser — pinned against the 2026-08-26 markup drift.

Bing moved to `<li class="b_algo" data-id iid=...>` + `<h2 class="">` and
wraps result URLs in a base64url redirect (`/ck/a?...&u=a1<b64>`). The old
exact-match regex parsed 0 results from a page carrying 10.
"""
import base64

from app.tools.native_search import _bing_real_url, _parse_bing


def _wrap(url: str) -> str:
    b64 = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    return f"https://www.bing.com/ck/a?!&amp;&amp;p=abc123&amp;u=a1{b64}&amp;ntb=1"


FIXTURE = (
    '<ol id="b_results" class=""><li class="b_algo" data-id iid=SERP.5339>'
    '<div class="b_tpcn"><div class="b_attribution" tabindex="-1"></div></div>'
    f'<h2 class=""><a target="_blank" href="{_wrap("https://www.federalreserve.gov/newsevents.htm")}"'
    ' h="ID=SERP,5127.2"><strong>Federal Reserve</strong> Board - <strong>News</strong> &amp; Events</a></h2>'
    '<div class="b_caption"><p class="b_lineclamp2" data-rslinkclamp-iid="">'
    '<span>The Fed announced...</span></p></div></li>'
    '<li class="b_algo" data-id iid=SERP.5340>'
    f'<h2 class=""><a href="{_wrap("https://www.reuters.com/markets/us/federal-reserve/")}">Reuters Fed page</a></h2>'
    '<div class="b_caption"><p class="b_lineclamp2">Latest stories.</p></div></li></ol>'
)


class TestBingParser:
    def test_parses_current_markup(self):
        results = _parse_bing(FIXTURE)
        assert len(results) == 2
        assert results[0].title == "Federal Reserve Board - News & Events"
        assert results[0].snippet.startswith("The Fed announced")

    def test_unwraps_redirect_urls(self):
        results = _parse_bing(FIXTURE)
        assert results[0].url == "https://www.federalreserve.gov/newsevents.htm"
        assert results[1].url == "https://www.reuters.com/markets/us/federal-reserve/"

    def test_real_url_passthrough_and_garbage_safe(self):
        assert _bing_real_url("https://example.com/x") == "https://example.com/x"
        assert _bing_real_url(
            "https://www.bing.com/ck/a?u=a1notbase64!!!"
        ).startswith("https://www.bing.com/ck/a")
