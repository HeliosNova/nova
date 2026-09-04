"""Curated RSS feed ingestion for Domain Studies.

For niche/high-value domains, generic web search returns SEO landing pages
instead of news. RSS feeds from known authoritative outlets give us real
fresh items with reliable timestamps. Used as a primary source by
domain_study_runner.run_domain_study before falling back to SearXNG.

Feeds chosen for:
  - Reliability (named outlet, RSS still maintained)
  - Reasonable update cadence
  - Date metadata in the feed (pubDate / updated)
  - Variety per domain (don't lean on one source)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx

logger = logging.getLogger(__name__)


# Curated feeds per domain. Order matters — items from higher-priority feeds
# show up first when the merger picks top items.
_FEEDS: dict[str, list[str]] = {
    "ai and ml": [
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://venturebeat.com/category/ai/feed/",
        "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",
        "https://huggingface.co/blog/feed.xml",
        "https://openai.com/blog/rss/",
        "https://blog.research.google/feeds/posts/default",
        "https://machinelearningmastery.com/feed/",
        "https://thegradient.pub/rss/",
        "https://syncedreview.com/feed/",
        "https://aibusiness.com/rss.xml",
        "https://stratechery.com/feed/",
        # CNBC tech / AI section — covers the BIG business stories (Amazon-Anthropic $25B)
        "https://www.cnbc.com/id/19854910/device/rss/rss.html",  # CNBC Technology
        "https://www.cnbc.com/id/100727362/device/rss/rss.html",  # CNBC AI
        # MIT Tech Review — major AI features
        "https://www.technologyreview.com/feed/",
        # Guardian AI (Reuters discontinued public RSS — 401/404)
        "https://www.theguardian.com/technology/artificialintelligenceai/rss",
        # The Information (limited RSS but high-signal)
        "https://www.theinformation.com/feed",
        # Substack AI — Latent Space, Stratechery, others
        "https://www.latent.space/feed",
        "https://www.aisnakeoil.com/feed",
        # ScienceDaily AI section
        "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        # Anthropic blog        # The Verge / Wired AI
        "https://www.wired.com/feed/tag/ai/latest/rss",    ],
    "technology": [
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://arstechnica.com/feed/",
        "https://www.wired.com/feed/rss",
        "https://www.engadget.com/rss.xml",
        "https://news.ycombinator.com/rss",
        "https://www.zdnet.com/news/rss.xml",
        "https://www.cnet.com/rss/news/",
        "https://gizmodo.com/rss",
        "https://restofworld.org/feed/",
    ],
    "crypto and web3": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://thedefiant.io/feed",
        "https://blog.ethereum.org/feed.xml",
        "https://www.theblock.co/rss.xml",
        "https://blockworks.co/feed",
        "https://protos.com/feed/",
        "https://cryptoslate.com/feed/",
        "https://news.bitcoin.com/feed/",
        "https://bitcoinmagazine.com/.rss/full/",
    ],
    "finance": [
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.ft.com/markets?format=rss",
        "https://www.theguardian.com/uk/business/rss",
        "https://www.investing.com/rss/news.rss",
        "https://www.marketwatch.com/rss/topstories",
        "https://www.fool.com/feeds/index.aspx",
        "https://seekingalpha.com/feed.xml",
        "https://feeds.feedburner.com/zerohedge/feed",
        "https://www.barrons.com/feed/rss",
        "https://www.forbes.com/business/feed/",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    ],
    "geopolitics": [
        # http-friendly, server-rendered sources lead — BBC/AlJazeera are JS-shells
        # that fail the article-body fetch, so we anchor on readable outlets.
        # DW World + Atlantic Council are high-volume, server-rendered, and read
        # cleanly via http (verified) — they lift Geopolitics' readable yield.
        "https://www.theguardian.com/world/rss",
        "https://rss.dw.com/xml/rss-en-world",
        "https://www.atlanticcouncil.org/feed/",
        # High-volume, browser-readable (Cloudflare-gated) defense/geopolitics —
        # verified render OK; they fill the breaking-news gap the analysis sites
        # miss. (19FortyFive deliberately excluded — it mixes in evergreen
        # military-history explainers that read as filler in a current digest.)
        "https://www.defenseone.com/rss/all/",
        "https://balkaninsight.com/feed/",
        "https://www.al-monitor.com/rss",
        "https://theconversation.com/global/articles.atom",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://foreignpolicy.com/feed/",
        "https://www.justsecurity.org/feed/",
        "https://thediplomat.com/feed/",
        "https://www.lawfaremedia.org/all-content.rss",
        "https://warontherocks.com/feed/",
        "https://www.economist.com/international/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://feeds.npr.org/1004/rss.xml",        # Non-Western / realist / multipolar perspectives — balance the Western
        # establishment think-tank lean above (bias audit 2026-06-21).
        "https://asiatimes.com/feed/",
        "https://responsiblestatecraft.org/feeds/feed.rss",
        # moderndiplomacy.eu replaced 2026-08-26: its /feed/ now 302s to a
        # "server-security-challenge" bot wall — every fetch failed (silent
        # coverage gap). SCMP world news keeps the non-Western slot
        # (validated from the container: parses clean, 50 items).
        "https://www.scmp.com/rss/91/feed",
        "https://geopoliticalfutures.com/feed/",
    ],
    # Topic-filter regex — items whose title/summary doesn't match these
    # keyword patterns are dropped from the feed for the named domain.
    # Empty / missing → no filter (accept all items from those feeds).
    "cybersecurity": [
        "https://www.bleepingcomputer.com/feed/",
        "https://krebsonsecurity.com/feed/",
        "https://thehackernews.com/feeds/posts/default",
        "https://www.darkreading.com/rss.xml",
        "https://www.schneier.com/feed/atom/",
        "https://www.cisa.gov/news.xml",
        "https://www.securityweek.com/feed/",
        "https://www.welivesecurity.com/en/rss/feed/",
        "https://www.csoonline.com/feed/",        "https://threatpost.com/feed/",
        "https://www.helpnetsecurity.com/feed/",
        "https://unit42.paloaltonetworks.com/feed/",
        "https://grahamcluley.com/feed/",
    ],
    "quantum computing": [
        "https://thequantuminsider.com/feed/",
        "https://quantumcomputingreport.com/feed/",
        "https://blog.research.google/feeds/posts/default/-/Quantum",
        "https://phys.org/rss-feed/physics-news/quantum-physics/",
        "https://www.nature.com/subjects/quantum-information.rss",
        "https://physicsworld.com/c/quantum/feed/",
        "https://hpcwire.com/feed/",
        "https://www.quantamagazine.org/feed/",
    ],
    "space and astronomy": [
        "https://spacenews.com/feed/",
        "https://www.space.com/feeds/all",
        "https://www.nasa.gov/news-release/feed/",
        "https://www.universetoday.com/feed/",
        "https://www.spaceflightnow.com/feed/",
    ],
    "health and medicine": [
        "https://www.statnews.com/feed/",
        "https://www.medpagetoday.com/rss/headlines.xml",
        "https://www.medscape.com/cx/rssfeeds/2700.xml",
        "https://www.nature.com/subjects/medical-research.rss",
        "https://www.fiercepharma.com/rss/xml",
    ],
    "energy and climate": [
        "https://www.utilitydive.com/feeds/news/",
        "https://insideclimatenews.org/feed/",
        "https://www.carbonbrief.org/feed/",
        "https://cleantechnica.com/feed/",
        # verified parse+read 2026-06-23 — high-volume, server-rendered
        "https://www.canarymedia.com/articles.rss",
        "https://www.pv-magazine.com/feed/",
    ],
    "semiconductors": [
        "https://semiengineering.com/feed/",
        "https://www.eetimes.com/feed/",
        "https://www.tomshardware.com/feeds/all",
        "https://www.semiconductor-digest.com/feed/",
        # verified parse+read 2026-06-23 (browser-readable, high-volume)
        "https://wccftech.com/feed/",
        "https://www.datacenterdynamics.com/en/rss/",
    ],
    "robotics and autonomy": [
        "https://www.therobotreport.com/feed/",
        "https://spectrum.ieee.org/feeds/topic/robotics.rss",
        "https://techcrunch.com/tag/robotics/feed/",
        "https://www.therobotreport.com/category/news/feed/",
        # fatten thin domain (2026-06-29): reputable robotics/autonomy outlets
        "https://roboticsandautomationnews.com/feed/",
        "https://www.therobotreport.com/category/research/feed/",
        "https://techxplore.com/rss-feed/robotics-news/",
    ],
    "biotech and genetics": [
        "https://www.fiercebiotech.com/rss/xml",
        "https://www.statnews.com/category/biotech/feed/",
        "https://www.biopharmadive.com/feeds/news/",
    ],
    "open source and github": [
        # The actual daily trending repos (GitHub killed its official
        # trending.atom; this community feed is what the stargazer monitor uses).
        # Was missing here, so the monitor leaned on web_search and shipped soft,
        # secondary "about GitHub" blogs (~10 sources, 2026-08-16).
        "https://mshibanami.github.io/GitHubTrendingRSS/daily/all.xml",
        "https://news.ycombinator.com/rss",
        "https://github.blog/feed/",
        "https://opensource.com/feed",
    ],
    "developer ecosystem": [
        "https://hnrss.org/frontpage",
        "https://github.blog/feed/",
        "https://stackoverflow.blog/feed/",
        "https://www.infoq.com/feed/",
        # fatten thin domain (2026-06-29): major developer-news outlets
        "https://thenewstack.io/feed/",
        "https://devops.com/feed/",
        "https://www.theregister.com/software/headlines.atom",
    ],
    "earnings and corporate events": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Top News
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.globenewswire.com/rssfeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20about%20Public%20Companies",
        "https://seekingalpha.com/feed.xml",
        "https://www.fool.com/feeds/index.aspx",
    ],
    "science": [
        "https://www.sciencedaily.com/rss/all.xml",
        "https://www.nature.com/nature.rss",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "https://www.scientificamerican.com/platform/syndication/rss/",
        "https://www.newscientist.com/feed/home/",
        "https://www.quantamagazine.org/feed/",
    ],
    "startups and vc": [
        "https://techcrunch.com/category/startups/feed/",
        "https://www.crunchbase.com/feed",
        "https://news.crunchbase.com/feed/",
        "https://www.theinformation.com/feed",
        "https://stratechery.com/feed/",
        "https://www.fortune.com/feed",    ],
    "physics and mathematics": [
        "https://physicsworld.com/feed/",
        "https://www.nature.com/subjects/physics.rss",
        "https://www.quantamagazine.org/feed/",
        "https://www.symmetrymagazine.org/feed",
        "http://export.arxiv.org/rss/physics",
        "http://export.arxiv.org/rss/math",
    ],
    "economics and markets": [
        # http-friendly leads (Economist/Bloomberg/FT below are paywalled shells).
        "https://www.cnbc.com/id/20910258/device/rss/rss.html",  # CNBC Economy
        "https://www.theguardian.com/business/economics/rss",
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://www.economist.com/finance-and-economics/rss.xml",
        "https://feeds.bloomberg.com/economics/news.rss",
        "https://www.ft.com/economics?format=rss",
    ],
    "whale watch": [
        "https://cryptoslate.com/feed/",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://www.theblock.co/rss.xml",
        "https://decrypt.co/feed",
        "https://protos.com/feed/",
    ],
    "top trades and positioning": [
        "https://feeds.feedburner.com/zerohedge/feed",
        "https://seekingalpha.com/feed.xml",
        "https://www.barrons.com/feed/rss",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.fool.com/feeds/index.aspx",
    ],
    "sec insider trading": [
        # owner=only returns actual Form 4 INSIDER trades (the bare type=4 feed returns
        # generic 424B2 prospectuses). EDGAR lists a SEPARATE entry per filer (issuer +
        # each reporting person) for the SAME filing — merged by accession in
        # _render_native_list (was showing one trade 2-3×). count bumped 40→80 so there
        # are enough UNIQUE filings after the merge. (Dropped pressreleases.rss — it
        # added the non-trade SEC personnel/RFC items the 2026-06-29 audit flagged.)
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=only&count=80&output=atom",
    ],
    "fomc and fed watch": [
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://www.federalreserve.gov/feeds/speeches.xml",
        "https://www.federalreserve.gov/feeds/h41.xml",  # Fed balance sheet
        "https://feeds.bloomberg.com/economics/news.rss",
    ],
    "fda drug approvals": [
        # Drug-SPECIFIC FDA feeds. The drugs feed carries Novel Drug Approvals + Drug
        # Trials Snapshots (the approved drugs); press-releases carries the "FDA Approves
        # ..." announcements. (Dropped statnews/fiercebiotech — biotech BUSINESS feeds
        # that were the source of the off-topic "STAT+" funding/opinion filler the
        # 2026-06-29 audit found instead of actual approvals.)
        "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drugs/rss.xml",
        "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    ],
    "government contract awards": [
        # The DoD daily "Contracts for <date>" rollup IS the contract-award feed (each
        # item is a day's awarded contracts, $7.5M+). (Dropped fedscoop/federalnewsnetwork/
        # gao — generic federal-news feeds that diluted awards with RIFs, GAO reports, and
        # commentary so only ~4/15 items were actually awards in the 2026-06-29 audit.)
        "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=400",
    ],
    "hacker news top stories": [
        "https://news.ycombinator.com/rss",
        "https://hnrss.org/frontpage",
        "https://hnrss.org/best",
    ],
    "product hunt trending": [
        "https://www.producthunt.com/feed",
    ],
    "github security advisories": [
        # GitHub's advisories.atom is Cloudflare-blocked server-side (HTTP 406) — which
        # is why this monitor shipped 100% off-topic CISA ICS advisories in the
        # 2026-06-29 audit. The REST API returns the real REVIEWED GHSA database (package
        # CVEs) unauthenticated (60/hr); handled by the JSON branch in _fetch_one_feed.
        # (CISA's ICS/OT feed removed — different domain; the monitor name promises GitHub.)
        "https://api.github.com/advisories?per_page=20&sort=published&type=reviewed",
    ],
    "github stargazer counts": [
        # GitHub killed its official trending.atom (returns empty); this
        # community-maintained feed gives the actual daily trending repos.
        "https://mshibanami.github.io/GitHubTrendingRSS/daily/all.xml",
        "https://github.blog/feed/",
    ],
    "morning check-in": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://feeds.npr.org/1001/rss.xml",
        "https://www.theguardian.com/world/rss",
        "https://rss.dw.com/xml/rss-en-world",
        "https://www.france24.com/en/rss",
    ],
    "us policy and regulation": [
        "https://www.politico.com/rss/politicopicks.xml",
        "https://www.axios.com/feeds/feed.rss",
        "https://thehill.com/feed/",
        # verified parse+read 2026-06-23 — Congress + federal-gov, high-volume
        "https://rollcall.com/feed/",
        "https://federalnewsnetwork.com/feed/",
        # Right + libertarian sources balance the center-establishment lean above
        # (bias audit 2026-06-21) — span the spectrum, not one side.
        "https://www.nationalreview.com/feed/",
        "https://thedispatch.com/feed/",
        "https://reason.com/feed/",
    ],
    "current events": [
        "https://www.theguardian.com/world/rss",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.npr.org/rss/rss.php?id=1001",
        "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
        "https://rss.dw.com/rdf/rss-en-all",
        "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml",
    ],
    "world awareness": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.theguardian.com/world/rss",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://rss.dw.com/xml/rss-en-world",
        "https://www.france24.com/en/rss",
        "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml",
    ],
    "middle east": [
        # Lead with http-friendly outlets; AlJazeera is a JS shell for body reads.
        "https://www.timesofisrael.com/feed/",
        "https://www.al-monitor.com/rss",
        "https://www.middleeastmonitor.com/feed/",
        "https://www.theguardian.com/world/middleeast/rss",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
    "china tech and economy": [
        "https://technode.com/feed/",
        "https://www.scmp.com/rss/91/feed",   # SCMP tech
        "https://www.scmp.com/rss/2/feed",    # SCMP china
        "https://pandaily.com/feed/",          # China tech, reduce SCMP over-weight
    ],
    # NEW region (2026-06-29): East Asia ex-China — Japan / Korea / Taiwan / ASEAN,
    # an economically major blind spot (semis, BOJ/BOK, supply chains) with only
    # China + India covered before.
    "east asia": [
        "https://www.japantimes.co.jp/feed/",
        "https://english.kyodonews.net/rss/news.xml",
        "https://en.yna.co.kr/RSS/news.xml",
        "https://www.koreatimes.co.kr/www/rss/nation.xml",
        "https://focustaiwan.tw/rss",
        "https://asia.nikkei.com/rss/feed/nar",
    ],
    "russia and eastern europe": [
        "https://meduza.io/rss/en/all",
        "https://www.kyivpost.com/feed",
        "https://www.themoscowtimes.com/rss/news",
        # fatten thin domain (2026-06-29)
        "https://kyivindependent.com/feed/",
        "https://www.intellinews.com/feed",
        "https://jamestown.org/feed/",
    ],
    "europe and eu": [
        "https://www.politico.eu/feed/",
        "https://www.euractiv.com/feed/",
        "https://www.theguardian.com/world/europe-news/rss",  # thin domain — broaden
        "https://rss.dw.com/xml/rss-en-eu",
    ],
    "india": [
        # Verified parse+read via the engine (2026-06-23). The indiatimes family
        # (TOI, Economic Times) + News18 + Moneycontrol are server-rendered and
        # read cleanly; the old business-standard feed is dead and indianexpress
        # is bot-gated, so they were dropped.
        "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
        "https://www.news18.com/rss/india.xml",
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://www.thehindu.com/news/feeder/default.rss",
        "https://feeds.feedburner.com/ndtvnews-india-news",
    ],
    "africa and emerging markets": [
        "https://www.theafricareport.com/feed/",
        "https://restofworld.org/feed/",
        "https://www.aljazeera.com/xml/rss/all.xml",
        # fatten thin domain (2026-06-29)
        "https://african.business/feed/",
        "https://mg.co.za/feed/",
        "https://www.premiumtimesng.com/feed",
    ],
    "latin america": [
        "https://restofworld.org/feed/",
        "https://en.mercopress.com/rss",
        "https://www.batimes.com.ar/feed",
        "https://riotimesonline.com/feed/",
    ],
    "supply chain and trade": [
        "https://www.supplychaindive.com/feeds/news/",
        "https://www.freightwaves.com/news/feed",
        "https://theloadstar.com/feed/",
        "https://www.supplychainbrain.com/rss/articles",
    ],
    "research frontiers": [
        "http://export.arxiv.org/rss/cs.AI",
        "http://export.arxiv.org/rss/cs.LG",
        "https://www.nature.com/nature.rss",
        "https://www.quantamagazine.org/feed/",   # browser-verified 2026-06-24
    ],
    "defense and military tech": [
        "https://breakingdefense.com/feed/",
        "https://www.defensenews.com/arc/outboundfeeds/rss/",
        "https://www.twz.com/feed",            # The War Zone (current home of thedrive)
        "https://www.aspistrategist.org.au/feed/",
        # verified parse+read 2026-06-23 — high-volume defense news
        "https://www.defenseone.com/rss/all/",
        "https://www.militarytimes.com/arc/outboundfeeds/rss/",
    ],
    "defi and protocols": [
        "https://thedefiant.io/feed",
        "https://www.theblock.co/rss.xml",
        "https://blockworks.co/feed",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ],
    "commodities and forex": [
        "https://oilprice.com/rss/main",
        "https://www.mining.com/feed/",
        "https://www.investing.com/rss/commodities.rss",
        "https://www.fxstreet.com/rss/news",
    ],
}


@dataclass
class FeedItem:
    title: str
    url: str
    summary: str
    published: datetime
    source_host: str
    # Cross-source verification: list of additional source hosts that
    # reported the same story. Items reported by ≥2 outlets get a
    # ✓ verified badge in the rendered output and rank higher.
    corroborating_sources: list[str] = None  # type: ignore
    # Structured, parsed metadata for specialized monitors that carry more than a
    # headline — e.g. SEC Form 4 transaction detail (buy/sell, shares, $ value),
    # GitHub-advisory severity/CVSS, parsed gov-contract awards. Kept generic so the
    # native-list renderer can surface real signal instead of just a title + link.
    meta: dict = None  # type: ignore

    def __post_init__(self):
        if self.corroborating_sources is None:
            self.corroborating_sources = []
        if self.meta is None:
            self.meta = {}

    @property
    def date_str(self) -> str:
        return self.published.strftime("%B %d, %Y")

    @property
    def is_verified(self) -> bool:
        return len(self.corroborating_sources) >= 1


_TIMEOUT = 10.0
_USER_AGENT = "Mozilla/5.0 (compatible; NovaBot/1.0; +https://github.com/HeliosNova/nova)"
# SEC enforces a fair-access policy that 403s any UA that doesn't include
# real contact info. Format must be "<name/tool> <email>".
_SEC_USER_AGENT = "Nova Personal Assistant espinozarogelio323@gmail.com"


# Per-domain topic filters. An item must match at least one keyword (case-
# insensitive substring in title or summary) to pass. Mixed-topic feeds
# (BBC World, Al Jazeera) get filtered to the topic that matters.
_TOPIC_FILTERS: dict[str, tuple[str, ...]] = {
    "geopolitics": (
        "ukraine", "russia", "putin", "nato", "china", "taiwan", "xi ",
        "iran", "israel", "gaza", "lebanon", "hamas", "hezbollah",
        "north korea", "kim", "saudi", "syria", "yemen", "afghanistan",
        "diplomatic", "diplomacy", "summit", "sanction", "treaty",
        "election", "coup", "geopoliti", "foreign polic", "alliance",
        "military", "war", "ceasefire", "negotia",
    ),
    "current events": (
        "election", "court", "ruling", "policy", "crisis", "summit",
        "investigation", "sentenced", "indictment", "law", "regulation",
        "minister", "president", "congress", "senate", "govern",
    ),
    "world awareness": (
        "election", "summit", "crisis", "minister", "president", "war",
        "ceasefire", "natural disaster", "earthquake", "flood",
    ),
    # NB (2026-08-29): "middle east" and "us policy and regulation" were each
    # defined TWICE in this dict — once here and again ~100/200 lines below.
    # Python keeps the LAST definition, so these earlier tuples were discarded
    # at parse time and never filtered anything. Same silent-collision class as
    # the duplicate _CHECK_DISPATCH key that killed scheduled Dream
    # Consolidation (2026-08-17). Both duplicates are removed here and their
    # unique keywords folded into the surviving definitions below.
    "finance": (
        "stock", "market", "fed ", "powell", "treasury", "bond", "yield",
        "rate", "inflation", "gdp", "earnings", "jobs", "unemployment",
        "dow", "s&p", "nasdaq", "etf", "ipo", "trader", "hedge fund",
    ),
    "earnings and corporate events": (
        "earnings", "revenue", "profit", "loss", "guidance", "ceo",
        "merger", "acquisition", "ipo", "spinoff", "layoff", "buyback",
    ),
    "ai and ml": (
        "ai", "artificial intelligence", "machine learning", "ml ", "llm",
        "large language", "neural", "deep learning", "anthropic", "openai",
        "claude", "gpt", "gemini", "chatgpt", "deepmind", "deepseek",
        "huggingface", "mistral", "meta ai", "llama", "perplexity",
        "cohere", "stability ai", "midjourney", "diffusion", "transformer",
        "model release", "model training", "agent", "rag ",
        "training", "fine-tune", "inference", "embedding", "tokens",
        "compute", "gpu cluster", "data center",
        "nvidia",  # central AI infra
    ),
    "technology": (
        "tech", "software", "hardware", "chip", "semiconductor", "platform",
        "release", "launch", "update", "announcement", "feature",
        "smartphone", "laptop", "browser", "operating system", "kernel",
        "framework", "library", "developer", "api", "sdk", "open source",
        "cloud", "saas", "startup", "ipo", "acquisition",
    ),
    "crypto and web3": (
        "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
        "defi", "nft", "stablecoin", "altcoin", "token", "wallet",
        "exchange", "binance", "coinbase", "ftx", "tether", "ripple",
        "solana", "polygon", "layer 2", "zksync", "arbitrum",
        "smart contract", "dao", "web3", "halving", "etf",
    ),
    "cybersecurity": (
        "breach", "hack", "ransomware", "malware", "phishing", "exploit",
        "vulnerability", "cve", "zero-day", "zero day", "patch",
        "cybersecurity", "security", "attack", "threat", "incident",
        "firewall", "encryption", "ddos", "compromise", "leak",
    ),
    "africa and emerging markets": (
        "africa", "nigeria", "kenya", "south africa", "egypt", "ethiopia",
        "ghana", "morocco", "tunisia", "algeria", "tanzania", "uganda",
        "rwanda", "senegal", "ivory coast", "cameroon", "zambia",
        "emerging market", "fintech", "mobile money", "m-pesa",
        "africa fintech", "african startup", "ecobank",
        "vietnam", "indonesia", "philippines", "thailand", "malaysia",
        "asean", "frontier market",
    ),
    "russia and eastern europe": (
        "russia", "putin", "kremlin", "moscow", "ukraine", "kyiv",
        "zelensky", "belarus", "lukashenko", "poland", "warsaw",
        "hungary", "budapest", "orban", "czech", "slovakia", "romania",
        "bulgaria", "serbia", "kosovo", "moldova", "georgia",
        "lithuania", "latvia", "estonia", "baltic", "nato",
    ),
    "china tech and economy": (
        "china", "chinese", "beijing", "shanghai", "shenzhen", "hong kong",
        "taiwan", "xi jinping", "ccp", "pboc", "yuan",
        "alibaba", "baidu", "tencent", "huawei", "byd", "bytedance",
        "deepseek", "qwen", "tiktok", "wechat", "didi", "jd.com",
    ),
    "east asia": (
        "japan", "japanese", "tokyo", "korea", "korean", "seoul", "north korea",
        "pyongyang", "kim jong", "taiwan", "taipei", "samsung", "sk hynix", "tsmc",
        "softbank", "toyota", "sony", "nintendo", "boj", "bank of japan", "yen",
        "won", "nikkei", "kospi", "asean", "vietnam", "indonesia", "philippines",
        "thailand", "singapore", "malaysia",
    ),
    "india": (
        "india", "indian", "modi", "mumbai", "delhi", "bangalore",
        "rupee", "rbi", "sensex", "nifty", "tata", "reliance", "infosys",
        "tcs", "wipro", "hcl", "adani", "jio", "upi",
    ),
    "middle east": (
        "israel", "iran", "saudi", "uae", "qatar", "syria", "yemen",
        "lebanon", "gaza", "hamas", "hezbollah", "kuwait", "bahrain",
        "egypt", "jordan", "iraq", "opec", "houthi", "tehran",
        "riyadh", "doha", "abu dhabi", "dubai",
    ),
    "latin america": (
        "brazil", "mexico", "argentina", "chile", "colombia", "peru",
        "venezuela", "ecuador", "bolivia", "uruguay", "paraguay",
        "lula", "amlo", "milei", "boric", "petro", "real ", "peso",
        "petrobras", "mercado libre", "nubank", "lithium",
    ),
    "europe and eu": (
        "european union", "ecb", "european central bank", "european commission",
        "eurozone", "eu ", "germany", "france", "italy", "spain", "uk ",
        "britain", "merz", "macron", "meloni", "starmer", "scholz",
        "poland", "netherlands", "sweden", "denmark", "finland",
        "asml", "sap", "lvmh", "siemens", "airbus",
    ),
    "semiconductors": (
        "semiconductor", "chip ", "fab", "wafer", "lithography", "euv",
        "nvidia", "amd", "intel", "tsmc", "samsung", "qualcomm",
        "broadcom", "asml", "applied materials", "lam research",
        "memory chip", "dram", "nand", "gpu", "cpu", "asic",
        "export control", "chip act",
    ),
    "commodities and forex": (
        "oil", "wti", "brent", "crude", "natural gas", "lng",
        "gold", "silver", "platinum", "copper", "lithium", "uranium",
        "wheat", "corn", "soybean",
        "forex", "fx ", "currency", "dollar", "euro", "yen", "yuan",
        "pound", "swiss franc", "rupee", "real ", "peso",
    ),
    "open source and github": (
        "open source", "github", "gitlab", "bitbucket", "open-source",
        "license", "license change", "fork", "pull request", "release",
        "kubernetes", "docker", "terraform", "ansible", "react",
        "vue", "svelte", "django", "flask", "fastapi", "rust",
        "python", "rust", "go ", "typescript",
    ),
    "developer ecosystem": (
        "developer", "framework", "library", "language", "compiler",
        "vscode", "jetbrains", "github copilot", "cursor", "ide",
        "package", "npm", "pip", "cargo", "maven", "gradle",
        "python", "rust", "javascript", "typescript", "go ", "java ",
        "release", "version", "v1.0", "v2.0",
    ),
    "supply chain and trade": (
        "supply chain", "shipping", "container", "freight", "port",
        "tariff", "trade war", "sanction", "export", "import",
        "logistics", "warehouse", "fedex", "ups", "maersk", "msc",
        "red sea", "panama canal", "suez",
        "rare earth", "critical mineral", "lithium", "cobalt",
    ),
    "defense and military tech": (
        "defense", "military", "pentagon", "lockheed", "raytheon",
        "northrop", "boeing defense", "general dynamics", "bae",
        "weapon", "missile", "drone", "uav", "fighter jet",
        "f-35", "submarine", "destroyer", "patriot",
        "darpa", "army", "navy", "air force", "marines", "space force",
    ),
    "defi and protocols": (
        "defi", "decentralized finance", "tvl", "uniswap", "aave",
        "maker", "compound", "curve", "synthetix", "yearn",
        "liquidity", "yield", "vault", "lending", "borrowing",
        "bridge", "rollup", "layer 2", "arbitrum", "optimism",
        "base", "zksync", "polygon",
    ),
    "biotech and genetics": (
        "biotech", "biotechnology", "crispr", "gene therapy", "gene editing",
        "synthetic biology", "longevity", "stem cell", "mrna",
        "clinical trial", "phase 1", "phase 2", "phase 3", "fda approval",
        "moderna", "biontech", "pfizer", "regeneron", "vertex",
    ),
    "health and medicine": (
        "health", "medicine", "medical", "drug", "fda", "vaccine",
        "clinical", "trial", "patient", "doctor", "hospital",
        "disease", "outbreak", "cancer", "diabetes", "alzheimer",
        "heart", "stroke", "covid", "flu", "infection",
    ),
    "space and astronomy": (
        "space", "spacex", "nasa", "esa", "rocket", "satellite",
        "starship", "falcon", "iss", "moon", "lunar", "mars",
        "asteroid", "comet", "exoplanet", "telescope", "jwst",
        "galaxy", "nebula", "black hole", "cosmic",
    ),
    "energy and climate": (
        "energy", "climate", "carbon", "emission", "renewable",
        "solar", "wind", "battery", "ev ", "electric vehicle",
        "lithium", "hydrogen", "nuclear", "fusion", "reactor",
        "oil", "gas", "coal", "grid", "utility",
    ),
    "physics and mathematics": (
        "physics", "physicist", "particle", "quantum", "relativity",
        "experiment", "theory", "equation", "math", "mathematic",
        "proof", "theorem", "conjecture", "fields medal",
        "cern", "lhc", "fermilab", "neutrino", "boson",
    ),
    "robotics and autonomy": (
        "robot", "robotics", "humanoid", "autonomous", "self-driving",
        "tesla", "waymo", "cruise", "lidar", "perception",
        "boston dynamics", "figure", "1x", "agility",
        "drone", "uav", "embodied",
    ),
    "quantum computing": (
        "quantum", "qubit", "superconducting", "ion trap", "photonic",
        "ibm quantum", "google quantum", "ionq", "rigetti", "psiquantum",
        "quantinuum", "atom computing", "pasqal",
        "error correction", "logical qubit", "quantum advantage",
        "quantum supremacy", "post-quantum", "qkd",
    ),
    "research frontiers": (
        "arxiv", "preprint", "paper", "research", "study",
        "nature", "science", "cell", "physical review",
        "breakthrough", "novel", "discovery", "experiment",
    ),
    "us policy and regulation": (
        "white house", "biden", "trump", "harris", "congress", "senate",
        "house of representatives", "supreme court", "executive order",
        "federal register", "doj", "fda", "ftc", "fcc", "sec ",
        "regulation", "bill ", "act ", "law ", "policy", "lawsuit",
        "antitrust", "tariff", "sanction",
        # Recovered 2026-08-29 from the duplicate definition that Python was
        # silently discarding — election coverage was matching nothing here.
        # Bare "house" is deliberately NOT restored: this list already carries
        # the precise "house of representatives" and "white house", and the
        # broad token would pull "housing market" stories into US politics.
        "election", "governor",
    ),
    "startups and vc": (
        "startup", "raises", "raised", "funding", "series a", "series b",
        "series c", "series d", "seed round", "venture", "vc ",
        "ipo", "spac", "exit", "acquisition", "valuation",
        "y combinator", "a16z", "sequoia", "benchmark", "founders fund",
    ),
    "earnings and corporate events specific": (
        "earnings beat", "earnings miss", "revenue", "guidance",
        "ceo step down", "ceo named", "layoff", "buyback",
        "stock split", "dividend", "ex-dividend",
    ),
}


def _topic_filter_for(monitor_name: str) -> tuple[str, ...]:
    return _TOPIC_FILTERS.get(_profile_label(monitor_name), ())


_WORD_KEYWORD_CACHE: dict[str, "re.Pattern"] = {}


def _matches_topic(item_title: str, item_summary: str, keywords: tuple[str, ...]) -> bool:
    """Check if the item matches at least one topic keyword. Uses
    word-boundary matching for short keywords (≤4 chars) so 'ai' doesn't
    match 'said' / 'main' / 'claim'. Multi-word phrases use plain
    substring matching since they're already specific.
    """
    if not keywords:
        return True
    blob = f"{item_title or ''} {item_summary or ''}".lower()
    cache_key = "|".join(keywords)
    pattern = _WORD_KEYWORD_CACHE.get(cache_key)
    if pattern is None:
        # Build one regex with all keywords. Short ones get \b boundaries.
        parts: list[str] = []
        for kw in keywords:
            kw_low = kw.lower().strip()
            if not kw_low:
                continue
            # Multi-word phrase or long keyword → plain substring (escaped)
            if " " in kw_low or len(kw_low) >= 6:
                parts.append(re.escape(kw_low))
            else:
                # Short keyword → word-boundary match
                parts.append(r"\b" + re.escape(kw_low) + r"\b")
        pattern = re.compile("|".join(parts)) if parts else re.compile(r"$^")
        _WORD_KEYWORD_CACHE[cache_key] = pattern
    return bool(pattern.search(blob))


def _profile_label(monitor_name: str) -> str:
    return monitor_name.replace("Domain Study:", "").strip().lower()


def feeds_for(monitor_name: str) -> list[str]:
    """Return the curated list of RSS URLs for this monitor's topic."""
    return _FEEDS.get(_profile_label(monitor_name), [])


# Strip HTML tags from RSS summaries
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = _HTML_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# Below this many characters of actual prose, a GitHub advisory description is a
# stub rather than a description. The CVE cross-reference that made every row a
# link scores 40; a real one-sentence description scores well over 100. Compared
# against an absolute bar, NOT against the summary's length: a real description
# that is merely SHORTER than the summary is still the better body.
_ADVISORY_MIN_PROSE = 60


def _prose_len(text: str) -> int:
    """Characters a reader learns something from: markdown links, bold markers
    and bare identifiers do not count."""
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", " ", text or "")   # masked links
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"(?i)cve-[\d-]+|GHSA-[\w-]+", " ", t)
    t = re.sub(r"[*_`#]", "", t)
    return len(re.sub(r"\s+", " ", t).strip())


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    # Try RFC 2822 first (RSS standard)
    try:
        d = parsedate_to_datetime(raw)
        if d.tzinfo is not None:
            d = d.replace(tzinfo=None)
        return d
    except (TypeError, ValueError):
        pass
    # ISO 8601 (Atom)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d",
    ):
        try:
            d = datetime.strptime(raw, fmt)
            if d.tzinfo is not None:
                d = d.replace(tzinfo=None)
            return d
        except ValueError:
            continue
    return None


def _strip_xmlns(tag: str) -> str:
    """ElementTree returns tags like '{http://...}entry' — strip the namespace."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


async def _fetch_github_advisories(url: str, *, max_items: int = 8) -> list[FeedItem]:
    """GitHub's advisories.atom is server-blocked (HTTP 406); the REST API returns the
    same REVIEWED GHSA database (package CVEs) as JSON, unauthenticated. Map each to a
    FeedItem so the monitor delivers actual GitHub advisories, not the wrong feed."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT,
                                                  "Accept": "application/vnd.github+json"})
        if resp.status_code >= 400:
            logger.info("[RSS] github advisories API HTTP %d", resp.status_code)
            return []
        data = resp.json()
    except Exception as e:
        logger.info("[RSS] github advisories API failed: %s", str(e)[:120])
        return []
    if not isinstance(data, list):
        return []
    out: list[FeedItem] = []
    for a in data[:max_items]:
        if not isinstance(a, dict):
            continue
        ghsa = (a.get("ghsa_id") or "").strip()
        summary = (a.get("summary") or "").strip()
        sev = (a.get("severity") or "").strip().lower()
        pub = _parse_date(a.get("published_at") or a.get("updated_at") or "")
        link = (a.get("html_url") or (f"https://github.com/advisories/{ghsa}" if ghsa else "")).strip()
        if not (summary and link and pub):
            continue
        pkgs = []
        for v in (a.get("vulnerabilities") or []):
            name = ((v.get("package") or {}) if isinstance(v, dict) else {}).get("name")
            if name and name not in pkgs:
                pkgs.append(name)
        title = (f"[{sev.upper()}] " if sev else "") + summary
        # GitHub's `description` is often nothing but a pointer at NVD - all 15
        # advisories on 2026-09-04 read "**CVE:** This vulnerability corresponds
        # to [CVE-...]" and nothing else, so the digest row was an id, a package
        # and two links (owner: "hyperlinks instead of report"). The advisory's
        # own `summary` is the real one-liner; it is already the item TITLE,
        # where the renderer truncates it. So when the description carries less
        # prose than the summary, show the summary instead of a cross-reference.
        body = (a.get("description") or "").strip()
        if _prose_len(body) < _ADVISORY_MIN_PROSE:
            body = summary
        body = body or summary
        if pkgs:
            body = f"Affected: {', '.join(pkgs[:6])}. " + body
        if ghsa:
            body = f"{ghsa} · {body}"
        # Keep the structured signal (severity/CVSS/CVE/packages) so the renderer can
        # roll up "N critical · M high · patch now: <pkgs>" instead of a flat list.
        cvss = a.get("cvss") if isinstance(a.get("cvss"), dict) else {}
        meta = {"advisory": {
            "severity": sev,
            "cvss": (cvss or {}).get("score"),
            "cve": (a.get("cve_id") or "").strip(),
            "ghsa": ghsa,
            "packages": pkgs[:6],
        }}
        out.append(FeedItem(title=_clean_text(title)[:200], url=link,
                            summary=_clean_text(body)[:1500], published=pub,
                            source_host="github.com", meta=meta))
    return out


async def _fetch_one_feed(url: str, *, max_items: int = 8) -> list[FeedItem]:
    """Fetch one RSS/Atom feed and return parsed items."""
    if "api.github.com/advisories" in url:          # JSON REST API (the atom feed is 406-blocked)
        return await _fetch_github_advisories(url, max_items=max_items)
    # SEC requires a UA with real contact info or 403s the request
    ua = _SEC_USER_AGENT if "sec.gov" in url else _USER_AGENT
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": ua, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml"})
            if resp.status_code >= 400:
                logger.info("[RSS] %s returned HTTP %d", url, resp.status_code)
                return []
            xml_text = resp.text
    except Exception as e:
        logger.info("[RSS] %s fetch failed: %s", url, str(e)[:120])
        return []

    if not xml_text or len(xml_text) < 100:
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.info("[RSS] %s parse failed: %s", url, e)
        return []

    items: list[FeedItem] = []
    feed_host = ""
    try:
        feed_host = urlparse(url).netloc.lower()
        if feed_host.startswith("www."):
            feed_host = feed_host[4:]
    except Exception:
        pass

    # RSS 2.0: <rss><channel><item>...
    # Atom: <feed><entry>...
    # Walk children looking for items/entries. Track the feed-level date (channel
    # pubDate/lastBuildDate, seen before any item) as a fallback for items that
    # omit their own date — some valid feeds (e.g. the GitHub trending RSS) only
    # date the channel; skipping those dateless items loses the whole feed.
    feed_date = None
    seen_item = False
    for elem in root.iter():
        tag = _strip_xmlns(elem.tag).lower()
        if tag not in ("item", "entry"):
            if not seen_item and tag in ("lastbuilddate", "pubdate", "updated", "date"):
                d = _parse_date((elem.text or "").strip())
                if d:
                    feed_date = d
            continue
        seen_item = True
        title = ""
        link = ""
        summary = ""
        pub_raw = ""
        for child in elem:
            ctag = _strip_xmlns(child.tag).lower()
            if ctag == "title":
                title = (child.text or "").strip()
            elif ctag == "link":
                # RSS uses text content; Atom uses href attribute
                link = (child.get("href") or child.text or "").strip()
            elif ctag in ("description", "summary", "content"):
                summary = (child.text or "")
            elif ctag in ("pubdate", "published", "updated", "date", "dc:date"):
                pub_raw = (child.text or "").strip()
        if not title or not link:
            continue
        pub = _parse_date(pub_raw) or feed_date
        if not pub:
            continue
        items.append(FeedItem(
            title=_clean_text(title)[:200],
            url=link,
            summary=_clean_text(summary)[:1500],
            published=pub,
            source_host=feed_host or "rss",
        ))
        if len(items) >= max_items:
            break

    return items


async def fetch_recent_items(
    monitor_name: str, *, hours: int = 72, max_total: int = 8, per_feed: int = 8,
) -> list[FeedItem]:
    """Fetch recent items from all curated feeds for this monitor's topic.
    Returns deduped, sorted-newest-first list. Empty if no feeds configured
    or all feeds returned nothing fresh.

    `per_feed` caps items pulled from EACH feed (default 8). Single-authoritative-
    source native-list monitors (SEC Form-4, DoD contracts, GHSA) pass a larger value
    so one feed can fill the whole list — and SEC needs the headroom because its
    issuer/reporting double-rows collapse ~3:1 after the accession merge.
    """
    feeds = feeds_for(monitor_name)
    if not feeds:
        return []

    # Fan out — fetch all feeds in parallel
    results = await asyncio.gather(
        *[_fetch_one_feed(u, max_items=per_feed) for u in feeds],
        return_exceptions=False,
    )

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    topic_keywords = _topic_filter_for(monitor_name)

    # First pass: collect all items, dedupe within-feed by URL
    raw_items: list[FeedItem] = []
    for items in results:
        for it in items:
            if it.published < cutoff:
                continue
            if not _matches_topic(it.title, it.summary, topic_keywords):
                continue
            raw_items.append(it)

    # Second pass: cross-source clustering. Two items from DIFFERENT outlets
    # talking about the same story (overlapping ≥3 distinctive keywords from
    # the title) get merged into one item with corroborating_sources marking
    # the additional outlets. The story with the most outlets wins.
    clusters: list[list[FeedItem]] = []
    for it in raw_items:
        keywords = _title_keywords(it.title)
        if not keywords:
            continue
        merged = False
        for cluster in clusters:
            other = cluster[0]
            other_keys = _title_keywords(other.title)
            # Require at least 3 distinctive keyword overlap (only when both titles
            # are long enough — short titles get exact-only matching)
            min_overlap = 3 if len(keywords) >= 4 and len(other_keys) >= 4 else max(len(keywords), len(other_keys))
            if len(keywords & other_keys) >= min_overlap and it.source_host != other.source_host:
                cluster.append(it)
                merged = True
                break
        if not merged:
            clusters.append([it])

    # Build representative items: from each cluster, pick the most-reputable
    # source (by feed order — earlier feeds in the curated list are higher
    # priority). Decorate with corroborating_sources.
    final_items: list[FeedItem] = []
    seen_urls: set[str] = set()
    for cluster in clusters:
        # Within a cluster, sort by publish recency
        cluster.sort(key=lambda x: x.published, reverse=True)
        primary = cluster[0]
        url_key = primary.url.split("#")[0].rstrip("/").lower()
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)
        # Mark cross-source verification
        primary.corroborating_sources = sorted({c.source_host for c in cluster[1:] if c.source_host != primary.source_host})
        final_items.append(primary)

    # Rank: verified items first (most corroborating sources wins), then by recency
    final_items.sort(key=lambda x: (-len(x.corroborating_sources), -x.published.timestamp()))

    # Outlet diversity: no single outlet contributes more than 2 items in
    # the top N. Forces variety in the final feed instead of TechCrunch
    # dominating AI/ML or Bloomberg dominating Finance. Verified items
    # bypass the cap because cross-source confirmation IS the value signal.
    diversified: list[FeedItem] = []
    per_outlet: dict[str, int] = {}
    deferred: list[FeedItem] = []
    for it in final_items:
        if it.is_verified:
            diversified.append(it)
            per_outlet[it.source_host] = per_outlet.get(it.source_host, 0) + 1
        elif per_outlet.get(it.source_host, 0) < 2:
            diversified.append(it)
            per_outlet[it.source_host] = per_outlet.get(it.source_host, 0) + 1
        else:
            deferred.append(it)
        if len(diversified) >= max_total:
            break
    # Top up from deferred if we didn't hit max_total with the diversity cap
    if len(diversified) < max_total:
        diversified.extend(deferred[:max_total - len(diversified)])
    return diversified[:max_total]


_TITLE_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "as", "from", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "what", "when", "where", "why",
    "how", "who", "which", "after", "before", "during", "while", "but",
    "his", "her", "its", "their", "our", "your", "him", "she", "he",
    "they", "them", "we", "us",
    "new", "now", "also", "more", "most", "some", "all", "no", "not",
    "say", "says", "said", "report", "reports", "reported",
})


def _title_keywords(title: str) -> set[str]:
    """Extract distinctive keywords from a title for cross-source matching."""
    if not title:
        return set()
    words = re.findall(r"\b[A-Za-z][a-zA-Z0-9'-]{2,}\b", title.lower())
    # Drop stopwords; keep words ≥4 chars (proper nouns + content nouns)
    return {w for w in words if w not in _TITLE_STOP and len(w) >= 4}
