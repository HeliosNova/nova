"""Source authority — a reusable, research-grounded reliability weight for a
news/source domain. Replaces hand-picked guesses.

Design (from a 2025-2026 review of source-credibility practice):
- **Anchor on the Lin et al. (PNAS Nexus 2023) consensus quality score** — an
  ensemble (imputation + PCA, "PC1") that reconciles NewsGuard, MBFC, Ad Fontes,
  Iffy, and fact-checker ratings across 11,520 domains. PC1 explains ~68% of the
  variance across raters, so a single domain→quality number is defensible.
  Vendored at app/data/domain_quality_pc1.csv (offline-safe for the giveaway).
- **Anchor on reliability, not bias** — bias ratings are noisier and politically
  contested; the dataset scores factual quality.
- **Domain-TYPE overrides for the primary-source tier** — official/government,
  regulators, courts, and wire services are primary records and rank top
  regardless of (or absent from) the dataset. This is the OSINT "original
  document" preference and the Admiralty Code's source-reliability axis.
- Unknown domains get a neutral 0.5 (below reputable, above a content farm) so an
  un-rated source is neither trusted nor dismissed.

Keep this SEPARATE from corroboration: per the Admiralty Code, a domain's
authority and a claim's cross-source corroboration are independent axes — a
trusted source can't launder a weak claim, and a weak source's corroborated
claim isn't auto-dismissed.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent.parent / "data" / "domain_quality_pc1.csv"

# Primary / official source TYPES → top weight, by domain suffix or known host.
# These are records, not journalism, so they sit above even wire services.
_PRIMARY_SUFFIXES = (".gov", ".mil")
_PRIMARY_SUBSTR = (
    ".gov.", ".mil.", "europa.eu", "eur-lex", "sec.gov", "federalreserve.gov",
    "ecb.europa.eu", "bls.gov", "treasury.gov", "cisa.gov", "fda.gov",
    "courtlistener.com", "supremecourt.gov", "uscourts.gov", "imf.org",
    "worldbank.org", "oecd.org", "who.int", "un.org",
)
# Wire services — top of factual-reliability reputation (RSP "generally
# reliable", Admiralty A). Slightly below primary records.
_WIRE = {"reuters.com": 1.0, "apnews.com": 0.98, "afp.com": 0.97, "ap.org": 0.98}

_NEUTRAL_DEFAULT = 0.5  # unknown / un-rated domain

_cache: dict[str, float] | None = None


def _load() -> dict[str, float]:
    global _cache
    if _cache is not None:
        return _cache
    table: dict[str, float] = {}
    try:
        with open(_DATA, encoding="utf-8") as f:
            next(f, None)  # header: domain,pc1
            for line in f:
                line = line.strip()
                if not line or "," not in line:
                    continue
                dom, _, score = line.partition(",")
                try:
                    table[dom.strip().lower()] = float(score)
                except ValueError:
                    continue
        logger.info("Source authority: loaded %d domain quality ratings", len(table))
    except FileNotFoundError:
        logger.warning("Source authority dataset missing at %s — using type rules + neutral default", _DATA)
    _cache = table
    return table


_HOST_RE = re.compile(r"^(?:https?://)?(?:www\.)?([^/:?#]+)", re.IGNORECASE)


def _registrable(host: str) -> str:
    """Best-effort registrable domain: strip scheme/www and leading subdomains,
    keeping the last 2 labels (or 3 for known 2-level ccTLDs like co.uk)."""
    m = _HOST_RE.match(host or "")
    h = (m.group(1) if m else host or "").lower().strip(".")
    if not h:
        return ""
    parts = h.split(".")
    if len(parts) <= 2:
        return h
    two_level = {"co.uk", "co.jp", "com.au", "co.in", "com.br", "co.nz", "or.jp", "ne.jp", "org.uk", "gov.uk", "ac.uk"}
    if ".".join(parts[-2:]) in two_level and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def authority(host: str) -> float:
    """Reliability weight in [0, 1] for a source host/domain.
    1.0 = primary/official record or top wire; ~0.9 reputable national; ~0.5
    unknown; <0.3 low-credibility / content farm (from the dataset)."""
    if not host:
        return _NEUTRAL_DEFAULT
    h = host.lower()
    # Primary-source TYPE overrides (suffix or known host substring)
    if any(h.endswith(s) or s + "/" in h for s in _PRIMARY_SUFFIXES) or any(s in h for s in _PRIMARY_SUBSTR):
        return 1.0
    reg = _registrable(h)
    if reg in _WIRE:
        return _WIRE[reg]
    table = _load()
    # exact host, then registrable domain
    if h in table:
        return table[h]
    if reg in table:
        return table[reg]
    return _NEUTRAL_DEFAULT


def tier(host: str) -> str:
    """Human-readable tier label for a host (for digest annotations/logs)."""
    a = authority(host)
    if a >= 0.97:
        return "primary/wire"
    if a >= 0.8:
        return "reputable"
    if a >= 0.5:
        return "general"
    if a >= 0.3:
        return "mixed"
    return "low-credibility"
