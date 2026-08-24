"""Probe candidate feeds for the 4 mislabeled native-list monitors — see what each
ACTUALLY returns before changing config (this project has a dead-feed history)."""
import asyncio
import re

import httpx

FEEDS = {
    "GH advisories (want: GHSA pkg CVEs)": "https://github.com/advisories.atom",
    "CISA all (currently fills GH slot)": "https://www.cisa.gov/cybersecurity-advisories/all.xml",
    "FDA press-releases (want: approvals)": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    "FDA drugs feed?": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs/rss.xml",
    "FDA MedWatch safety?": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml",
    "statnews biotech (business noise)": "https://www.statnews.com/category/biotech/feed/",
    "DoD contracts CT=400 (want: awards)": "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=400",
    "DoD contracts war.gov?": "https://www.war.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=400",
    "SAM.gov? (no rss expected)": "https://sam.gov/api/prod/sgs/v1/search/feed?index=opp",
    "SEC Form4 owner=only (dedup?)": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=only&count=40&output=atom",
}
_TITLE = re.compile(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", re.S | re.I)
_LINK = re.compile(r"<link[^>]*?href=\"([^\"]+)\"|<link>\s*(https?://[^<\s]+)", re.I)
_ACC = re.compile(r"accession[_-]?number=([\d-]+)|/(\d{10}-\d\d-\d{6})", re.I)


async def probe(name, url):
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                     headers={"User-Agent": "Nova-Monitor/1.0 (research@example.com)"}) as c:
            r = await c.get(url)
        txt = r.text
        titles = [re.sub(r"\s+", " ", t).strip() for t in _TITLE.findall(txt)]
        titles = [t for t in titles if t][1:]  # drop channel title
        n_items = max(txt.count("<item"), txt.count("<entry"))
        accs = [a or b for a, b in _ACC.findall(txt)]
        uniq = len(set(accs))
        extra = f"  | entries={len(accs)} uniqAccession={uniq}" if accs else ""
        print(f"\n[{r.status_code}] {name}\n    items={n_items}{extra}")
        for t in titles[:4]:
            print(f"    • {t[:96]}")
    except Exception as e:
        print(f"\n[ERR] {name}: {type(e).__name__}: {e}")


async def main():
    for name, url in FEEDS.items():
        await probe(name, url)


asyncio.run(main())
