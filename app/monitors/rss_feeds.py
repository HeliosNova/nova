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
        # Reuters tech
        "https://www.reuters.com/technology/artificial-intelligence/rss",
        # The Information (limited RSS but high-signal)
        "https://www.theinformation.com/feed",
        # Substack AI — Latent Space, Stratechery, others
        "https://www.latent.space/feed",
        "https://www.aisnakeoil.com/feed",
        # ScienceDaily AI section
        "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
        # Anthropic blog
        "https://www.anthropic.com/news/rss.xml",
        # The Verge / Wired AI
        "https://www.wired.com/feed/category/business/artificial-intelligence/latest/rss",
        "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
    ],
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
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "https://www.investing.com/rss/news.rss",
        "https://www.marketwatch.com/rss/topstories",
        "https://www.fool.com/feeds/index.aspx",
        "https://seekingalpha.com/feed.xml",
        "https://www.zerohedge.com/fullrss2.xml",
        "https://www.barrons.com/feed/rss",
    ],
    "geopolitics": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://foreignpolicy.com/feed/",
        "https://www.cfr.org/rss-feeds/all-feeds.xml",
        "https://thediplomat.com/feed/",
        "https://www.lawfaremedia.org/all-content.rss",
        "https://warontherocks.com/feed/",
        "https://www.economist.com/international/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://feeds.npr.org/1004/rss.xml",
        "https://www.brookings.edu/feed/",
        "https://carnegieendowment.org/rss/all/en",
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
        "https://www.scmagazine.com/feed",
        "https://www.csoonline.com/feed/",
        "https://nakedsecurity.sophos.com/feed/",
        "https://threatpost.com/feed/",
        "https://www.helpnetsecurity.com/feed/",
        "https://blog.talosintelligence.com/feeds/posts/default",
        "https://grahamcluley.com/feed/",
    ],
    "quantum computing": [
        "https://thequantuminsider.com/feed/",
        "https://quantumcomputingreport.com/feed/",
        "https://blog.research.google/feeds/posts/default/-/Quantum",
        "https://www.ibm.com/quantum/blog/rss",
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
        "https://www.medscape.com/rss/public/all-medscape-headlines.xml",
        "https://www.nature.com/subjects/medical-research.rss",
        "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds",
    ],
    "energy and climate": [
        "https://www.utilitydive.com/feeds/news/",
        "https://insideclimatenews.org/feed/",
        "https://www.carbonbrief.org/feed/",
        "https://cleantechnica.com/feed/",
    ],
    "semiconductors": [
        "https://semiengineering.com/feed/",
        "https://www.eetimes.com/feed/",
        "https://www.tomshardware.com/feeds/all",
        "https://www.anandtech.com/rss/",
    ],
    "robotics and autonomy": [
        "https://www.therobotreport.com/feed/",
        "https://spectrum.ieee.org/topic/robotics/feeds.rss",
        "https://www.therobotreport.com/category/news/feed/",
    ],
    "biotech and genetics": [
        "https://www.fiercebiotech.com/rss/xml",
        "https://www.statnews.com/category/biotech/feed/",
        "https://www.biopharmadive.com/feeds/news/",
    ],
    "open source and github": [
        "https://news.ycombinator.com/rss",
        "https://github.blog/feed/",
        "https://opensource.com/feed",
    ],
    "developer ecosystem": [
        "https://hnrss.org/frontpage",
        "https://github.blog/feed/",
        "https://stackoverflow.blog/feed/",
        "https://www.infoq.com/feed/",
    ],
    "earnings and corporate events": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Top News
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.businesswire.com/portal/site/home/rss/",
        "https://seekingalpha.com/feed.xml",
        "https://www.fool.com/feeds/index.aspx",
    ],
    "science": [
        "https://www.sciencedaily.com/rss/all.xml",
        "https://www.nature.com/nature.rss",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "https://www.scientificamerican.com/rss/news/",
        "https://www.newscientist.com/feed/home/",
        "https://www.quantamagazine.org/feed/",
    ],
    "startups and vc": [
        "https://techcrunch.com/category/startups/feed/",
        "https://www.crunchbase.com/feed",
        "https://news.crunchbase.com/feed/",
        "https://www.theinformation.com/feed",
        "https://stratechery.com/feed/",
        "https://www.fortune.com/feed",
        "https://thisweekinstartups.com/feed/",
    ],
    "physics and mathematics": [
        "https://physicsworld.com/feed/",
        "https://www.nature.com/subjects/physics.rss",
        "https://www.quantamagazine.org/feed/",
        "https://www.symmetrymagazine.org/feed",
        "http://export.arxiv.org/rss/physics",
        "http://export.arxiv.org/rss/math",
    ],
    "economics and markets": [
        "https://www.economist.com/finance-and-economics/rss.xml",
        "https://feeds.bloomberg.com/economics/news.rss",
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://www.ft.com/economics?format=rss",
        "https://www.reutersagency.com/feed/?best-topics=economy&post_type=best",
        "https://www.imf.org/en/News/RSS",
    ],
    "whale watch": [
        "https://cryptoslate.com/feed/",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://www.theblock.co/rss.xml",
        "https://decrypt.co/feed",
        "https://protos.com/feed/",
    ],
    "top trades and positioning": [
        "https://www.zerohedge.com/fullrss2.xml",
        "https://seekingalpha.com/feed.xml",
        "https://www.barrons.com/feed/rss",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.fool.com/feeds/index.aspx",
    ],
    "sec insider trading": [
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&output=atom",
        "https://www.sec.gov/news/pressreleases.rss",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom",
    ],
    "fomc and fed watch": [
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://www.federalreserve.gov/feeds/speeches.xml",
        "https://www.federalreserve.gov/feeds/h41.xml",  # Fed balance sheet
        "https://feeds.bloomberg.com/economics/news.rss",
    ],
    "fda drug approvals": [
        "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
        "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/cder-news/rss.xml",
        "https://www.fiercebiotech.com/rss/xml",
        "https://www.statnews.com/category/biotech/feed/",
    ],
    "government contract awards": [
        "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=400",
        "https://sam.gov/api/prod/sgs/v1/search/feed",
        "https://www.gao.gov/rss/reports.xml",
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
        "https://github.com/advisories.atom",
        "https://www.cisa.gov/cybersecurity-advisories/all.xml",
        "https://nvd.nist.gov/feeds/xml/cve/misc/nvd-rss.xml",
    ],
    "github stargazer counts": [
        "https://github.com/trending.atom",
        "https://github.blog/feed/",
    ],
    "morning check-in": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://feeds.npr.org/1001/rss.xml",
        "https://www.reutersagency.com/feed/?best-topics=top-news&post_type=best",
    ],
    "us policy and regulation": [
        "https://www.politico.com/rss/politicopicks.xml",
        "https://www.axios.com/feeds/feed.rss",
        "https://thehill.com/feed/",
    ],
    "current events": [
        "https://www.reutersagency.com/feed/?best-topics=top-news&post_type=best",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.npr.org/rss/rss.php?id=1001",
    ],
    "world awareness": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.reutersagency.com/feed/?best-topics=world&post_type=best",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
    "middle east": [
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.middleeastmonitor.com/feed/",
        "https://www.timesofisrael.com/feed/",
    ],
    "china tech and economy": [
        "https://technode.com/feed/",
        "https://www.scmp.com/rss/91/feed",   # SCMP tech
        "https://www.scmp.com/rss/2/feed",    # SCMP china
    ],
    "russia and eastern europe": [
        "https://meduza.io/rss/en/all",
        "https://www.kyivpost.com/feed",
        "https://www.themoscowtimes.com/rss/news",
    ],
    "europe and eu": [
        "https://www.politico.eu/feed/",
        "https://www.euractiv.com/feed/",
    ],
    "india": [
        "https://www.thehindu.com/news/feeder/default.rss",
        "https://www.business-standard.com/rss/latest.rss",
    ],
    "africa and emerging markets": [
        "https://www.theafricareport.com/feed/",
        "https://restofworld.org/feed/",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
    "latin america": [
        "https://restofworld.org/feed/",
        "https://riotimesonline.com/feed/",
        "https://www.bnamericas.com/feed",
    ],
    "supply chain and trade": [
        "https://www.supplychaindive.com/feeds/news/",
        "https://www.freightwaves.com/news/feed",
    ],
    "research frontiers": [
        "http://export.arxiv.org/rss/cs.AI",
        "http://export.arxiv.org/rss/cs.LG",
        "https://www.nature.com/nature.rss",
    ],
    "defense and military tech": [
        "https://breakingdefense.com/feed/",
        "https://www.defensenews.com/arc/outboundfeeds/rss/",
        "https://www.thedrive.com/the-war-zone/feed",
    ],
    "defi and protocols": [
        "https://thedefiant.io/feed",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ],
    "commodities and forex": [
        "https://oilprice.com/rss/main",
        "https://www.fxstreet.com/rss/news",
        "https://www.kitco.com/news/rss.xml",
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

    def __post_init__(self):
        if self.corroborating_sources is None:
            self.corroborating_sources = []

    @property
    def date_str(self) -> str:
        return self.published.strftime("%B %d, %Y")

    @property
    def is_verified(self) -> bool:
        return len(self.corroborating_sources) >= 1


_TIMEOUT = 10.0
_USER_AGENT = "Mozilla/5.0 (compatible; NovaBot/1.0; +https://github.com/anthropics/nova)"
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
    "middle east": (
        "israel", "iran", "saudi", "uae", "qatar", "syria", "yemen",
        "lebanon", "gaza", "hamas", "hezbollah", "kuwait", "bahrain",
        "egypt", "jordan", "iraq", "opec", "houthi",
    ),
    "us policy and regulation": (
        "trump", "biden", "harris", "congress", "senate", "house",
        "supreme court", "sec ", "fcc", "fda", "ftc", "doj", "executive order",
        "regulation", "bill ", "law ", "policy", "election", "governor",
    ),
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


async def _fetch_one_feed(url: str, *, max_items: int = 8) -> list[FeedItem]:
    """Fetch one RSS/Atom feed and return parsed items."""
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
    # Walk children looking for items/entries
    for elem in root.iter():
        tag = _strip_xmlns(elem.tag).lower()
        if tag not in ("item", "entry"):
            continue
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
        pub = _parse_date(pub_raw)
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
    monitor_name: str, *, hours: int = 72, max_total: int = 8,
) -> list[FeedItem]:
    """Fetch recent items from all curated feeds for this monitor's topic.
    Returns deduped, sorted-newest-first list. Empty if no feeds configured
    or all feeds returned nothing fresh.
    """
    feeds = feeds_for(monitor_name)
    if not feeds:
        return []

    # Fan out — fetch all feeds in parallel
    results = await asyncio.gather(
        *[_fetch_one_feed(u) for u in feeds],
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
