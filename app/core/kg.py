"""Knowledge graph — structured facts as (subject, predicate, object) triples.

SQLite-only, no NetworkX. 1-hop graph queries via recursive CTE.
Predicate normalization to 31 canonical forms.
Temporal tracking: facts have valid_from/valid_to for historical queries.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_PRUNE_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    id: int
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str
    created_at: str
    valid_from: str | None = None
    valid_to: str | None = None
    provenance: str = ""
    superseded_by: int | None = None


# ---------------------------------------------------------------------------
# Predicate normalization
# ---------------------------------------------------------------------------

CANONICAL_PREDICATES = frozenset({
    "is_a", "part_of", "located_in", "created_by", "used_for",
    "known_for", "related_to", "belongs_to", "has_property",
    "born_in", "founded_in", "capital_of", "currency_of",
    "spoken_in", "developed_by", "written_by", "caused_by",
    "contains", "produces", "leads",
    "works_at", "employed_by", "lives_in", "studied_at",
    "married_to", "member_of", "invented_by", "successor_of",
    "succeeded_by", "price_of", "version_of", "has_status",
    # News/business/geopolitics relations (2026-06-29): the analytical verbs the
    # monitors ingest constantly that USED to flatten to `related_to` — discarding
    # the relationship and disabling change-tracking for the bulk of monitor facts.
    # Preserving the verb makes the KG queryable by real relationship ("what did X
    # acquire", "who partnered with Y", "who sued Z").
    "acquired", "owns", "subsidiary_of", "invested_in", "partnered_with",
    "competes_with", "supplies", "sued", "sanctioned", "launched", "regulates",
    # Self-distilled principles (principles.distill) — the durable facts that
    # survive lesson decay. Canonical so they are NOT flattened to related_to,
    # which had made distilled principles indistinguishable from generic
    # associations in retrieval (audit 2026-08-23). Functional (not multi-valued):
    # a refined principle for a topic supersedes the prior one.
    "principle_says", "principle_consensus",
})

# Predicates where a subject legitimately holds MANY simultaneous objects, so a
# new differing object is an ADDITIONAL fact — NOT a contradiction. add_fact
# must not mechanically supersede these; doing so silently collapsed
# multi-valued knowledge ("Python contains lists AND dicts", co-authors of a
# paper, a thing born_in Ulm AND Germany at different scales). Genuine
# contradictions on these are caught upstream by the LLM resolver
# (check_and_resolve_contradictions), which sets valid_to BEFORE add_fact runs.
# `related_to` is included deliberately: it is the degrade target for every
# non-canonical predicate (see normalize_predicate), so collapsing it would
# defeat the whole "degrade, don't orphan" design and merge unrelated facts.
# Everything NOT listed here stays functional (single current value that a new
# value supersedes): lives_in, capital_of, price_of, married_to, leads,
# currency_of, version_of, successor_of/succeeded_by — residence/role/price
# style facts whose change-over-time IS the supersession the bitemporal store
# is built to track.
MULTI_VALUED_PREDICATES = frozenset({
    "is_a", "part_of", "located_in", "created_by", "used_for", "known_for",
    "related_to", "belongs_to", "has_property", "born_in", "founded_in",
    "spoken_in", "developed_by", "written_by", "caused_by", "contains",
    "produces", "works_at", "employed_by", "studied_at", "member_of",
    "invented_by",
    # News relations that legitimately accumulate — a company acquires MANY
    # firms, invests in many, partners with many, sues many; a regulator
    # regulates many domains. A new object is an ADDITIONAL fact, never a
    # supersession. (subsidiary_of is deliberately NOT here: a company has one
    # current parent, so a new parent SUPERSEDES — the ownership-change the
    # bitemporal store exists to track. owns stays multi-valued: a holder owns
    # many things.)
    "acquired", "owns", "invested_in", "partnered_with", "competes_with",
    "supplies", "sued", "sanctioned", "launched", "regulates",
})

# Predicates where only ONE subject may hold the relation to a given object, so
# a new subject for the same object supersedes the prior holder (one current
# leader per org, one capital per country). Replaces the old `_UNIQUE_PREDICATES`
# set, which mostly listed non-canonical strings (is_ceo_of, is_president_of)
# that normalize_predicate degrades to `related_to` — so they never matched and
# the inverse-functional guard was effectively dead for everything but `leads`.
INVERSE_FUNCTIONAL_PREDICATES = frozenset({"leads", "capital_of", "married_to"})

# Natural verb phrases for prompt rendering (task #63). Every phrase keeps
# strict "SUBJECT <phrase> OBJECT" order — brain._kg_answers_query parses
# "SUBJECT <verb>" from these lines (see format_for_prompt). Extend the gate's
# verb alternation when adding phrases here.
_PRED_PHRASES = {
    "price_of": "is currently priced at",
    "located_in": "is located in",
    "based_in": "is based in",
    "capital_of": "is the capital of",
    "leads": "leads",
    "married_to": "is married to",
    "works_for": "works for",
    "created_by": "was created by",
    "developed_by": "was developed by",
    "owned_by": "is owned by",
    "acquired": "acquired",
    "acquired_by": "was acquired by",
    "member_of": "is a member of",
    "part_of": "is part of",
    "has_status": "currently has status",
    "related_to": "is connected to",
    "lives_in": "lives in",
    "currency_of": "is the currency of",
    "version_of": "is a version of",
}

_PREDICATE_ALIASES: dict[str, str] = {
    "is a": "is_a", "is an": "is_a", "type of": "is_a",
    "is part of": "part_of", "part of": "part_of",
    "located in": "located_in", "is in": "located_in", "is located in": "located_in",
    "created by": "created_by", "made by": "created_by", "built by": "created_by",
    "used for": "used_for", "used in": "used_for",
    "known for": "known_for", "famous for": "known_for",
    "related to": "related_to",
    "belongs to": "belongs_to",
    "has property": "has_property", "has": "has_property",
    "born in": "born_in",
    "founded in": "founded_in", "established in": "founded_in",
    "capital of": "capital_of", "is capital of": "capital_of",
    "currency of": "currency_of",
    "spoken in": "spoken_in",
    "developed by": "developed_by",
    "written by": "written_by", "authored by": "written_by",
    "caused by": "caused_by",
    "contains": "contains", "includes": "contains",
    "produces": "produces",
    "leads": "leads",
    "works at": "works_at", "works for": "works_at", "employed at": "works_at",
    "employed by": "employed_by", "hired by": "employed_by",
    "lives in": "lives_in", "resides in": "lives_in",
    "studied at": "studied_at", "graduated from": "studied_at", "attended": "studied_at",
    "married to": "married_to", "spouse of": "married_to",
    "member of": "member_of",
    "invented by": "invented_by", "discovered by": "invented_by",
    "successor of": "successor_of", "succeeded by": "succeeded_by", "replaced by": "succeeded_by",
    "price of": "price_of", "cost of": "price_of", "costs": "price_of",
    "version of": "version_of", "variant of": "version_of",
    "has status": "has_status", "status of": "has_status", "status": "has_status",
    "current status": "has_status", "state of": "has_status",
    # News/business/geopolitics verbs (2026-06-29)
    "acquired": "acquired", "acquires": "acquired", "bought": "acquired",
    "purchased": "acquired", "to acquire": "acquired", "acquisition of": "acquired",
    "owns": "owns", "owner of": "owns",
    "subsidiary of": "subsidiary_of", "unit of": "subsidiary_of",
    "division of": "subsidiary_of", "owned by": "subsidiary_of",
    "invested in": "invested_in", "invests in": "invested_in", "backed": "invested_in",
    "funded": "invested_in", "stake in": "invested_in",
    "partnered with": "partnered_with", "partners with": "partnered_with",
    "partnership with": "partnered_with", "teamed up with": "partnered_with",
    "collaborates with": "partnered_with",
    "competes with": "competes_with", "competitor of": "competes_with",
    "rival of": "competes_with", "rivals": "competes_with",
    "supplies": "supplies", "supplier of": "supplies", "supplies to": "supplies",
    "sued": "sued", "sues": "sued", "lawsuit against": "sued",
    "filed suit against": "sued",
    "sanctioned": "sanctioned", "sanctions": "sanctioned",
    "imposed sanctions on": "sanctioned",
    "launched": "launched", "launches": "launched", "unveiled": "launched",
    "rolled out": "launched",
    "regulates": "regulates",
}


def normalize_predicate(pred: str) -> str:
    """Normalize a predicate to a canonical form."""
    p = pred.strip().lower()

    # Check alias map (before underscore conversion)
    if p in _PREDICATE_ALIASES:
        return _PREDICATE_ALIASES[p]

    # Underscores
    p = p.replace(" ", "_")

    # Already canonical?
    if p in CANONICAL_PREDICATES:
        return p

    # Strip common prefixes and re-check
    for prefix in ("is_", "has_", "was_", "does_", "are_"):
        if p.startswith(prefix):
            short = p[len(prefix):]
            if short in CANONICAL_PREDICATES:
                return short
            # Check with common suffixes
            for alias_key, canon in _PREDICATE_ALIASES.items():
                if short == alias_key.replace(" ", "_"):
                    return canon

    # Hard allow-list — only canonical predicates and explicit aliases pass.
    # Permissive custom-predicate matching was removed 2026-05-13: LLM
    # extractions like "founded_in_year" or "custom_metric_v2" would orphan
    # facts (stored under a unique key that no canonical query ever hits).
    # The 43 canonical predicates + alias phrases cover the common shapes;
    # anything else degrades to `related_to`, preserving the relationship
    # without splintering the predicate space.
    return "related_to"


# ---------------------------------------------------------------------------
# Stop words and normalization — shared via text_utils
# ---------------------------------------------------------------------------

from app.core.text_utils import normalize_words as _base_normalize  # noqa: E402


def _normalize_words(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word set (min length 2)."""
    return _base_normalize(text, min_length=2)


def normalize_entity(name: str) -> str:
    """Light cleanup only — strip and collapse whitespace. Casing is PRESERVED.

    The old implementation title-cased every word, which mangled the extractor's
    correct casing (BlackRock->Blackrock, OpenAI->Openai, iPhone->Iphone) and
    diverged on acronyms ("AMD" kept but "amd"->"Amd"), fragmenting entities into
    casing-only variants. Cross-variant consistency is now handled by
    KnowledgeGraph._canonical_entity (a kg_entity_aliases registry), which maps
    every casing variant to one canonical form chosen by casing richness.
    """
    return " ".join(name.split()) if name else name


def _casing_score(s: str) -> int:
    """Rank casing 'intentionality' for choosing a canonical form. Counts
    uppercase letters: BlackRock=2 > Blackrock=1; AMD=3 > Amd=1; OpenAI=3 >
    Openai=1. Naive .capitalize() artifacts (one leading capital) score lowest."""
    return sum(1 for c in s if c.isupper())


# ---------------------------------------------------------------------------
# Triple quality gate — heuristic pre-filter
# ---------------------------------------------------------------------------

_GARBAGE_PATTERNS = [
    re.compile(r"^[\d\s\.\+\-\*\/\=\(\)]+$"),        # math expressions
    re.compile(r"[/\\][\w/\\]+\.\w+"),                 # file paths
    re.compile(r"^\d+(\.\d+)?$"),                      # bare numbers
    # Monitor category labels — NEVER a real entity
    re.compile(r"^domain study[:\s]", re.IGNORECASE),
    re.compile(r"^monitor(ing)?\s+system\b", re.IGNORECASE),
    # Underscored/lowercase pseudo-entities generated from monitor names:
    # "energy_and_climate_intelligence", "trading_and_positioning_intelligence", etc.
    re.compile(r"^[a-z][a-z0-9_]*_intelligence$"),
    # Title-cased "X Architecture" as an Intel/CPU thing when conflated with Nova.
    # Nova's "nova architecture" question caused "Nova Architecture is_a Intel Cpu Platform".
    re.compile(r"^Nova\s+Architecture$", re.IGNORECASE),
]

_GARBAGE_VALUES = frozenset({
    "testuser", "test", "foo", "bar", "baz", "example",
    "null", "none", "undefined", "n/a", "na",
})


_SHORT_ENTITY_ALLOWLIST = frozenset({
    "c", "r", "go", "us", "uk", "eu", "ai", "ml",
    "os", "js", "ts", "py", "c#", "c++", "f#", "qt", "vi",
})

# Predicate-direction sanity checks. Catches the most common LLM reversals.
# Subject and object are checked against small known-entity lists; if the
# triple has the WRONG entity on the wrong side, reject (the LLM can re-emit
# it correctly next time). False negatives are fine — these are heuristics
# meant to filter the obvious garbage we observed in production.

# Known countries/regions that should never be the SUBJECT of capital_of
_KNOWN_COUNTRIES = frozenset({
    "us", "usa", "u.s.", "u.s.a.", "united states", "united states of america",
    "uk", "u.k.", "united kingdom", "great britain", "britain", "england",
    "russia", "russian federation", "soviet union", "ussr",
    "china", "people's republic of china", "prc",
    "india", "japan", "germany", "france", "italy", "spain", "canada", "mexico",
    "brazil", "argentina", "australia", "south korea", "north korea", "vietnam",
    "thailand", "indonesia", "philippines", "malaysia", "singapore", "ukraine",
    "poland", "turkey", "egypt", "israel", "iran", "iraq", "saudi arabia", "uae",
    "south africa", "nigeria", "kenya", "ethiopia", "morocco", "algeria",
    "pakistan", "bangladesh", "afghanistan", "switzerland", "sweden", "norway",
    "finland", "denmark", "netherlands", "belgium", "austria", "greece",
    "portugal", "ireland", "new zealand", "chile", "peru", "colombia", "venezuela",
    "european union", "eu", "scotland", "wales", "northern ireland",
    "taiwan", "hong kong", "korea", "czech republic", "hungary", "romania",
    "bulgaria", "serbia", "croatia", "slovenia", "slovakia", "kazakhstan",
    "uzbekistan", "syria", "lebanon", "jordan", "palestine", "qatar", "kuwait",
    "bahrain", "oman", "yemen", "libya", "tunisia", "ghana", "tanzania",
    "uganda", "zimbabwe", "angola", "mozambique", "cuba", "panama", "costa rica",
    "ecuador", "bolivia", "uruguay", "paraguay",
})

# Known orgs (companies, government agencies, sports teams) — never the SUBJECT
# of works_at / leads (those should have a person as subject).
_KNOWN_ORGS = frozenset({
    "tesla", "spacex", "apple", "google", "alphabet", "microsoft", "amazon",
    "meta", "facebook", "instagram", "twitter", "x", "openai", "anthropic",
    "nvidia", "amd", "intel", "tsmc", "samsung", "sony", "ibm", "oracle",
    "salesforce", "adobe", "uber", "lyft", "airbnb", "netflix", "disney",
    "spotify", "shopify", "stripe", "paypal", "visa", "mastercard",
    "berkshire hathaway", "jpmorgan", "goldman sachs", "morgan stanley",
    "blackrock", "bank of america", "wells fargo", "citigroup",
    "boeing", "lockheed martin", "northrop grumman", "raytheon",
    "ford", "general motors", "toyota", "honda", "bmw", "mercedes-benz",
    "exxon", "chevron", "shell", "bp", "saudi aramco",
    "deepseek", "alibaba", "tencent", "baidu", "huawei", "xiaomi", "byd",
    "sec", "fbi", "cia", "doj", "fda", "epa", "irs", "fed", "federal reserve",
    "ecb", "european central bank", "imf", "world bank",
    "un", "united nations", "nato", "who", "world health organization",
    "office of the us trade representative",
    "premier league", "nfl", "nba", "mlb", "nhl", "fifa",
    "arsenal", "chelsea", "manchester united", "manchester city", "liverpool",
    "real madrid", "barcelona", "los angeles dodgers", "new york yankees",
})


def _is_country(name: str) -> bool:
    return name.strip().lower() in _KNOWN_COUNTRIES


def _is_org(name: str) -> bool:
    return name.strip().lower() in _KNOWN_ORGS


# News-extraction noise guards (2026-06-21). Monitors extract triples from
# formatted news digests, which leaked two junk classes into the KG:
#   (a) a bare news-source domain as an entity ("X related_to ft.com")
#   (b) a headline clause/fragment as an entity ("US related_to iran claim ... shut")
# A real entity is a short noun phrase; a domain or a clause is not a durable fact.
# Match a bare news/source domain ONLY on news-style TLDs. Critically this must
# NOT catch dotted TECH entities (node.js, next.js, socket.io, asp.net, x.ai,
# character.ai) — those are real entities our AI/dev/semiconductor monitors
# extract, and the gate also runs in daily curation which would RETROACTIVELY
# delete them. So .js/.io/.ai/.net are deliberately excluded from the TLD set.
_BARE_DOMAIN_RE = re.compile(
    r"^[a-z0-9][a-z0-9-]*(\.[a-z]{2,})*\.(com|org|co|news|info|gov|edu|press|uk)(/|$)"
)
# Genuine finite verbs that signal a sentence fragment, not an entity name.
# Deliberately NARROW: conjunctions/prepositions (after, amid, while, that, which)
# were rejecting legitimate multi-word entities ("The Day After Tomorrow"), and
# ambiguous noun-verbs (will, plans, seeks, wants) reject names ("Last Will and
# Testament"). Keep only words that are almost always finite verbs in context.
_FRAGMENT_MARKERS = re.compile(
    r"\b(is|are|was|were|been|has|have|had|claims?|claimed|said|says|"
    r"announced|reported|criticized|shut|closed)\b"
)


def _looks_like_fragment(x: str) -> bool:
    """A subject/object that reads as a sentence fragment, not an entity name.

    Entity names are short noun phrases (<=5 words, no finite verb). Headline
    clauses ('iran claim waterway is shut') are not durable facts.
    """
    words = x.split()
    if len(words) > 5:
        return True
    if len(words) >= 3 and _FRAGMENT_MARKERS.search(x):
        return True
    return False


# Bare dates, months, years, and durations — meaningless as a related_to
# subject/object (the actual fact got lost in extraction).
_DATE_FRAGMENT_RE = re.compile(
    r"^(?:(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)(?:\s+\d{1,2})?(?:,?\s+\d{4})?"
    r"|\d{4}"
    r"|(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:second|minute|hour|day|week|month|year|decade)s?)$",
    re.IGNORECASE,
)

# Objects that describe Nova's OWN pipeline output rather than the world. The
# extractor occasionally banks "<topic> is_a domain overview" while writing a
# digest; that is bookkeeping, not knowledge. Measured 2026-08-29: 16 such
# triples were 0.3% of live facts but 9.6% of ALL retrievals.
_PIPELINE_ARTIFACT_OBJECTS = frozenset({
    "domain overview", "domain_overview", "domain study", "domain_study",
    "domain intelligence overview", "domain_intelligence_overview",
    "researched briefing", "research briefing", "briefing", "digest",
    "date of facts learned", "facts learned", "monitor output",
    "intelligence overview", "overview", "summary", "report",
})

# A BARE date as the subject of a fact — "July 08, 2026 has_status …". A date is
# not an entity. Deliberately anchored and strict so a titled document that
# merely contains a date ("Contracts for Aug. 3, 2026") still qualifies as a
# legitimate subject.
_MONTH_ALT = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
    r"|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_BARE_DATE_SUBJECT_RE = re.compile(
    # Abbreviated forms included: the live KG carries "Aug. 14, 2026" as well as
    # "August 14, 2026", and a full-name-only pattern silently missed the former
    # (caught by test_bare_date_cannot_be_a_subject).
    rf"^(?:{_MONTH_ALT})\.?\s+\d{{1,2}},?\s+\d{{4}}$"
    rf"|^\d{{1,2}}\s+(?:{_MONTH_ALT})\.?,?\s+\d{{4}}$"
    r"|^\d{4}-\d{2}-\d{2}$",
    re.IGNORECASE,
)
# A related_to endpoint that is really a QUANTITY / TIME / MEASUREMENT with no
# named-entity anchor — the extraction lost the actual fact and kept the number.
# Broader than _DATE_FRAGMENT_RE (which only anchors bare dates/simple durations):
# catches clock times ("1:00 AM UTC"), compound/suffixed durations ("1 hour 10
# minutes", "20 hours daily"), and quantity+unit heads ("10,000 repetitions per
# skill", "140 npm packages"). Anchored at START only (\b, not $) so trailing
# qualifiers don't let it escape. related_to-only, so specific predicates keep
# their legit numeric objects (price_of $95k, founded_in 1998). These were the
# dominant surviving junk class in the live KG (full-system exploration 2026-07-09:
# top-retrieved related_to facts were "Match ~ 1:00 AM UTC", "GLM-5.2 ~ 1 hour 10
# minutes", "training centers ~ 10,000 repetitions per skill").
_UNDERSCORE_NUMERIC_RE = re.compile(r"^[<>~≈]?\s*\d[\d,.]*_")
_QUANTITY_ENDPOINT_RE = re.compile(
    r"^(?:>|~|<|≈|over|under|more\s+than|at\s+least|nearly|about|around|roughly|"
    r"approximately)?\s*(?:"
    # money / counts / percentages (trailing (?:\b|$) so symbol units like %
    # and end-of-string terminate correctly — the plain \b never fired after %)
    r"\$?[€£]?\d[\d,.]*\s*(?:billion|million|trillion|thousand|bn|mn|k|%|percent|"
    r"units?|repetitions?|packages?|points?|times|teams?|people|users?)(?:\b|$)"
    # physical measurements — a bare number+unit is an attribute value that lost
    # its predicate ("Swift Observatory ~ 363 miles"), junk AS a related_to edge
    r"|\d[\d,.]*\s*(?:miles?|kilometers?|km|meters?|metres?|feet|foot|ft|inch(?:es)?|"
    r"yards?|pounds?|lbs?|kg|kilograms?|grams?|tons?|tonnes?|mph|kph|acres?|hectares?|"
    r"barrels?|bpd|watts?|kilowatts?|megawatts?|gigawatts?|[kmg]w|volts?|"
    r"[kmgt]b|gigabytes?|terabytes?|megabytes?|[kmg]hz|degrees?|°[cf]?)(?:\b|$)"
    r"|\d{1,2}:\d{2}\s*(?:[ap]\.?m\.?)?\s*(?:utc|gmt|est|pst|cet|cst|edt|pdt|bst)?\b"
    r"|\d[\d,.]*\s*(?:second|minute|hour|day|week|month|year|decade)s?\b"
    r")", re.IGNORECASE,
)
# An endpoint that is really an EVENT reference pinned to a date ("Russia attack
# on June 22", "hackers on June 19, 2026", "voltage fluctuation in July 2024") —
# not a durable entity. The durable fact would use a specific predicate + real
# object; as a related_to endpoint it's a lost event fragment (the two top-
# retrieved live junk facts, 780×/438×, full-system exploration 2026-07-09).
# Requires a PREPOSITION before the date so named events survive ("Iraq 2026
# tournament", "July 2026 trading app" — no leading on/in/by → kept).
_EMBEDDED_EVENT_DATE_RE = re.compile(
    r"\b(?:on|in|by|since|during|after|before)\s+(?:"
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|\d{1,2}[/-]\d{1,2})", re.IGNORECASE,
)
# Single generic CATEGORY word as a related_to endpoint — carries no identity, so
# the association is meaningless ("developer ~ coding tools daily" was retrieved
# 1081×). A real entity is NAMED; these are the bucket the news extractor reaches
# for when it can't name the actor. Exact single-token match only → high precision.
_GENERIC_ENTITY = frozenset({
    "developer", "developers", "match", "duration", "user", "users", "customer",
    "customers", "company", "companies", "researcher", "researchers", "study",
    "report", "market", "markets", "system", "systems", "project", "projects",
    "team", "teams", "event", "events", "product", "products", "service",
    "services", "technology", "data", "platform", "feature", "features", "update",
    "updates", "price", "rate", "level", "value", "growth", "increase", "decline",
})


def is_garbage_triple(subject: str, predicate: str, object_: str) -> bool:
    """Return True if a triple is obvious garbage that should not be stored."""
    s, o = subject.strip().lower(), object_.strip().lower()

    # Too short (unless in the allowlist of legitimate short entities)
    if len(s) < 2 and s not in _SHORT_ENTITY_ALLOWLIST:
        return True
    if len(o) < 2 and o not in _SHORT_ENTITY_ALLOWLIST:
        return True

    # Self-referential
    if s == o:
        return True

    # Known garbage values
    if s in _GARBAGE_VALUES or o in _GARBAGE_VALUES:
        return True

    # Pattern-based rejection
    for pat in _GARBAGE_PATTERNS:
        if pat.match(s) or pat.match(o):
            return True

    # News-extraction noise (source domains, headline fragments, synthesis labels)
    if _BARE_DOMAIN_RE.match(s) or _BARE_DOMAIN_RE.match(o):
        return True
    if _looks_like_fragment(s) or _looks_like_fragment(o):
        return True
    if ":" in s and re.match(r"^[a-z0-9_]+:", s):  # "cross_pattern:..." synthesis artifact
        return True

    # Pipeline self-description (2026-08-29). The extractor sometimes banks a
    # triple ABOUT Nova's own output instead of about the world:
    # "semiconductors is_a domain overview", "geopolitics is_a researched
    # briefing", "July 08, 2026 has_status date of facts learned". These are
    # bookkeeping, not knowledge, and they are catastrophically over-retrieved:
    # measured 2026-08-29, 16 such triples were 0.3% of live facts but 9.6% of
    # ALL retrievals (a 32x over-representation), crowding real knowledge out of
    # every prompt while `acquired` facts sat 83% never-retrieved.
    if o in _PIPELINE_ARTIFACT_OBJECTS:
        return True
    # A bare date is not an entity, so it cannot be the SUBJECT of a fact.
    # ("July 08, 2026 has_status date of facts learned" — 2321 retrievals.)
    # Only bare dates: a titled document that contains a date ("Contracts for
    # Aug. 3, 2026") is a legitimate subject and must survive.
    if _BARE_DATE_SUBJECT_RE.match(subject.strip()):
        return True

    # Predicate-direction sanity (rejects obvious reversals)
    p = predicate.strip().lower()
    if p == "capital_of" and _is_country(s):
        # "Russia capital_of Moscow" — backwards
        return True
    if p in ("works_at", "leads") and _is_org(s) and not _is_org(o):
        # "Tesla works_at Elon Musk" or "Federal Reserve leads Jerome Powell"
        return True
    if p in ("created_by", "invented_by", "founded_by") and _is_org(o) and _is_org(s):
        # Two orgs in created_by is almost always wrong (acquisitions/parents
        # use different predicates — part_of, owned_by, contains)
        return True

    # Degraded-predicate + date-fragment pairing: '(FIFA Committee, related_to,
    # July 5)' or '(X, related_to, eight years)' carries no retrievable meaning —
    # the sentence's actual fact was lost in extraction (audit 2026-07-06, the
    # dominant junk class in newly-banked research facts). Specific predicates
    # keep their date objects (founded_in 1998 is a real fact).
    if p == "related_to" and (_DATE_FRAGMENT_RE.match(s) or _DATE_FRAGMENT_RE.match(o)):
        return True

    # related_to with a QUANTITY/TIME/MEASUREMENT endpoint, or a single generic
    # category word for either endpoint — the extraction kept a number or a
    # bucket-noun instead of the real relationship (the dominant surviving junk
    # class, full-system exploration 2026-07-09). Also runs in daily curation, so
    # this retroactively purges the ~900 such facts already banked. related_to
    # ONLY — specific predicates keep legit numeric/quantity objects.
    if p == "related_to":
        if _QUANTITY_ENDPOINT_RE.match(s) or _QUANTITY_ENDPOINT_RE.match(o):
            return True
        if s in _GENERIC_ENTITY or o in _GENERIC_ENTITY:
            return True
        if _EMBEDDED_EVENT_DATE_RE.search(s) or _EMBEDDED_EVENT_DATE_RE.search(o):
            return True
        # Underscore-mangled numeric artifacts ("20_percent",
        # "2.6_million_barrels_per_day") — a digit immediately joined by an
        # underscore is a machine-mangled attribute value, never a real
        # entity (2026-08-14). High-precision: no legitimate name has this.
        if _UNDERSCORE_NUMERIC_RE.match(s) or _UNDERSCORE_NUMERIC_RE.match(o):
            return True

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time as ISO string (SQLite CURRENT_TIMESTAMP format)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _is_recent(ts: str | None, days: int = 7) -> bool:
    """Return True if a timestamp string is within the last N days."""
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        # If naive, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return dt >= cutoff
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

from app.config import config as _config


class KnowledgeGraph:
    """Structured fact store with 1-hop graph queries and temporal tracking."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS kg_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        object TEXT NOT NULL,
        confidence REAL DEFAULT 0.8,
        source TEXT DEFAULT 'extracted',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        valid_from TIMESTAMP,
        valid_to TIMESTAMP,
        provenance TEXT DEFAULT '',
        superseded_by INTEGER,
        UNIQUE(subject, predicate, object)
    );
    CREATE INDEX IF NOT EXISTS idx_kg_subject ON kg_facts(subject);
    CREATE INDEX IF NOT EXISTS idx_kg_object ON kg_facts(object);
    CREATE TABLE IF NOT EXISTS kg_entity_aliases (
        alias_lower TEXT PRIMARY KEY,
        canonical TEXT NOT NULL
    );
    """

    def __init__(self, db):
        self._db = db
        # Schema-ensure memo (audit 2026-08-23): this DDL block re-ran on every
        # construction — and KnowledgeGraph is constructed repeatedly in async
        # contexts, putting write-lock DDL on the event-loop thread (live: x8/2h).
        # First construction per SafeDB runs it; later ones skip to _finish_init.
        if getattr(db, "schema_ensured", lambda _t: False)("kg"):
            self._finish_init()
            return
        # Create table if not exists (safe to call multiple times)
        for stmt in self._SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._db.execute(stmt)

        # Migration: add temporal columns if missing (for existing databases)
        for col, typedef in [
            ("valid_from", "TIMESTAMP"),
            ("valid_to", "TIMESTAMP"),
            ("provenance", "TEXT DEFAULT ''"),
            ("superseded_by", "INTEGER"),
            ("times_retrieved", "INTEGER DEFAULT 0"),
            # Bitemporal: transaction time of supersession (added 2026-05-16, task #29).
            # Distinct from valid_to: valid_to = when the fact stopped being true in
            # the world; superseded_at = when WE recorded the supersession. For
            # current ingest they coincide, but the column lets us answer "what did
            # Nova believe about X on date Y" by filtering out facts that were
            # logically deleted by Y. created_at is the partner column (transaction
            # time of insertion).
            ("superseded_at", "TIMESTAMP"),
            # Memory-poisoning defense (2026-07-08, OWASP ASI06): facts banked
            # from untrusted web content carry a trust weight, and weakly-
            # trusted single-source facts are QUARANTINED -- stored and usable
            # for corroboration, but excluded from prompt injection until an
            # independent observation promotes them (AgentPoison-class attacks
            # reach >=80% success at <0.1% poison rate on unguarded stores).
            ("trust", "REAL DEFAULT 0.5"),
            ("quarantined", "INTEGER DEFAULT 0"),
        ]:
            try:
                self._db.execute(f"ALTER TABLE kg_facts ADD COLUMN {col} {typedef}")
            except Exception:
                pass  # Column already exists

        # Create index on valid_to (must come after migration adds the column)
        try:
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_valid_to ON kg_facts(valid_to)"
            )
        except Exception:
            pass
        try:
            # Index for bitemporal as-of queries on transaction time
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_superseded_at ON kg_facts(superseded_at)"
            )
        except Exception:
            pass

        # Backfill valid_from from created_at for existing rows
        self._db.execute(
            "UPDATE kg_facts SET valid_from = created_at WHERE valid_from IS NULL"
        )
        # Backfill superseded_at from valid_to for rows that were logically
        # deleted before this migration ran. valid_to was conflating world-validity
        # with transaction-time supersession; for historical rows the two coincide.
        self._db.execute(
            "UPDATE kg_facts SET superseded_at = valid_to "
            "WHERE superseded_at IS NULL AND valid_to IS NOT NULL"
        )
        getattr(db, "mark_schema_ensured", lambda _t: None)("kg")
        self._finish_init()

    def _finish_init(self) -> None:
        """Non-DDL instance state — runs on every construction."""
        # Insert counter for batched pruning (only prune every 50 inserts)
        self._inserts_since_prune = 0
        # Lock for concurrent supersession safety
        self._write_lock = asyncio.Lock()
        # ChromaDB collection for semantic search (lazy init)
        self._collection = None

    # --- ChromaDB vector collection for semantic KG search ---

    async def prune_stale_vectors(self) -> int:
        """Remove vector entries for facts that are no longer live
        (superseded / expired / quarantined). The index had NO lifecycle
        hygiene: deletes and supersessions never touched their vectors, and it
        grew to 3× the live set in two months (15,018 vectors vs 5,023 live
        facts, 2026-08-14) — every top-k neighborhood was ~2/3 dead rows,
        silently diluting paraphrase recall. Runs from daily maintenance."""
        col = self._get_collection()
        if col is None:
            return 0

        def _sync() -> int:
            # Snapshot the collection BEFORE reading the live set: a fact banked
            # between the two reads is then absent from `got` (never a delete
            # candidate) rather than absent from `live` (wrongly deleted while
            # live). The inverted order merely leaves a stale vector one more day.
            got = col.get(include=[])
            live = {str(r["id"]) for r in self._db.fetchall(
                "SELECT id FROM kg_facts WHERE superseded_at IS NULL "
                "AND valid_to IS NULL AND quarantined = 0")}
            stale = [i for i in (got.get("ids") or []) if i not in live]
            for i in range(0, len(stale), 500):
                col.delete(ids=stale[i:i + 500])
            return len(stale)

        return await asyncio.to_thread(_sync)

    def _get_collection(self):
        """Lazy-init ChromaDB collection for semantic KG fact search."""
        if self._collection is None:
            try:
                import chromadb
                from ..config import config
                from .embedding import open_collection
                client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
                self._collection = open_collection(
                    client, "kg_facts", reindex=self._backfill_collection,
                )
            except Exception as e:
                logger.warning("Failed to init kg_facts ChromaDB collection: %s", e)
                return None
        return self._collection

    def _backfill_collection(self, collection) -> int:
        """Populate `collection` from current KG facts, unconditionally.
        Shared by reindex_kg_facts (guarded) and the embedder-rebuild path.
        Strict live-set filter (2026-08-14): indexing merely valid_to-IS-NULL
        rows let superseded/quarantined facts into the index — combined with
        no lifecycle hygiene the index grew to 3× the live set."""
        all_rows = self._db.fetchall(
            "SELECT id, subject, predicate, object FROM kg_facts "
            "WHERE superseded_at IS NULL AND valid_to IS NULL AND quarantined = 0"
        )
        if not all_rows:
            return 0
        ids, documents, metadatas = [], [], []
        for row in all_rows:
            searchable = f"{row['subject']} {row['predicate'].replace('_', ' ')} {row['object']}"
            ids.append(str(row["id"]))
            documents.append(searchable)
            metadatas.append({"subject": row["subject"], "predicate": row["predicate"]})
        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info("Reindexed %d KG facts into ChromaDB", len(ids))
        return len(ids)

    def reindex_kg_facts(self) -> int:
        """One-time backfill of existing KG facts into ChromaDB. Returns count indexed."""
        collection = self._get_collection()
        if collection is None:
            return 0
        if collection.count() > 0:
            logger.info("KG facts collection already has %d entries, skipping reindex", collection.count())
            return 0
        return self._backfill_collection(collection)

    def rebuild_vectors(self, reason: str = "manual") -> int:
        """Drop and rebuild the kg_facts collection from the live SQL set.

        prune_stale_vectors keeps the CHROMA view clean, but every one of
        its deletes is only a tombstone to the underlying HNSW graph —
        hnswlib never compacts, so a churny index eventually fails all
        k>=8 queries (the lessons collection died exactly this way on
        2026-08-22 while holding 9× tombstones). Runs off-GPU on the
        dedicated CPU embedder; ~5k live facts re-embed in minutes.
        Records the churn watermark so historical supersessions stop
        counting toward the next rot assessment.
        """
        from . import vector_health

        try:
            import chromadb
            from ..config import config as _cfg

            client = chromadb.PersistentClient(path=_cfg.CHROMADB_PATH)
            try:
                client.delete_collection("kg_facts")
            except Exception:
                pass  # absent collection — nothing to drop
            self._collection = None
            collection = self._get_collection()
            if collection is None:
                return 0
            n = self._backfill_collection(collection)
            row = self._db.fetchone(
                "SELECT COALESCE(MAX(id), 0) AS m,"
                " (SELECT COUNT(*) FROM kg_facts WHERE superseded_at IS NULL"
                "  AND valid_to IS NULL AND quarantined = 0) AS c"
                " FROM kg_facts"
            )
            vector_health.set_watermark(
                self._db, "kg_facts", ever=row["m"], live=row["c"]
            )
            logger.warning(
                "KG vector index REBUILT (%s): %d facts re-embedded", reason, n
            )
            return n
        except Exception as e:
            logger.error("KG vector rebuild failed: %s", e)
            return 0

    def _add_to_vector(self, fact_id: int, subject: str, predicate: str, object_: str) -> None:
        """Add a single fact to the vector collection."""
        collection = self._get_collection()
        if collection is None:
            return
        try:
            searchable = f"{subject} {predicate.replace('_', ' ')} {object_}"
            collection.upsert(
                ids=[str(fact_id)],
                documents=[searchable],
                metadatas=[{"subject": subject, "predicate": predicate}],
            )
        except Exception as e:
            logger.warning("Failed to add KG fact %d to vector store — paraphrase retrieval degrades silently: %s", fact_id, e)

    def _remove_from_vector(self, fact_id: int) -> None:
        """Remove a fact from the vector collection (on supersession/deletion)."""
        collection = self._get_collection()
        if collection is None:
            return
        try:
            collection.delete(ids=[str(fact_id)])
        except Exception:
            pass

    @staticmethod
    def _rrf_fuse(
        keyword_ids: list[int],
        vector_ids: list[int],
        ppr_ids: list[int] | None = None,
        k: int = 60,
    ) -> list[int]:
        """Reciprocal Rank Fusion of up to three ranked ID lists.

        Each list contributes 1/(k + rank + 1) to a fact's score; final order
        is by descending sum. The PPR list is the HippoRAG 2 graph-walk signal
        — facts reachable from query entities via the kg_facts graph rank up
        even when their literal text doesn't match the query.
        """
        scores: dict[int, float] = {}
        for rank, rid in enumerate(keyword_ids):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        for rank, rid in enumerate(vector_ids):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        if ppr_ids:
            for rank, rid in enumerate(ppr_ids):
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    # --- Entity canonicalization ---

    def _canonical_entity(self, name: str, register: bool = False) -> str:
        """Map an entity to its one canonical casing via the kg_entity_aliases
        registry, so casing variants ("AMD"/"amd", "BlackRock"/"Blackrock") never
        fragment into separate graph nodes or duplicate facts.

        Lookup is case-insensitive (keyed on lower(name)). On a write path pass
        register=True: an unseen entity registers its (whitespace-cleaned) form as
        the canonical; a richer-cased form upgrades the canonical AND rewrites the
        entity's existing facts so storage stays consistent. Query paths pass
        register=False (lookup only — never pollute the registry with query casing).
        """
        clean = normalize_entity(name)
        if not clean:
            return clean
        low = clean.lower()
        row = self._db.fetchone(
            "SELECT canonical FROM kg_entity_aliases WHERE alias_lower = ?", (low,)
        )
        if row is not None:
            current = row["canonical"]
            if register and _casing_score(clean) > _casing_score(current):
                # A better-cased form arrived — upgrade canonical + rewrite facts.
                try:
                    self._db.execute(
                        "UPDATE kg_entity_aliases SET canonical = ? WHERE alias_lower = ?",
                        (clean, low),
                    )
                    self._db.execute(
                        "UPDATE kg_facts SET subject = ? WHERE LOWER(subject) = ?",
                        (clean, low),
                    )
                    self._db.execute(
                        "UPDATE kg_facts SET object = ? WHERE LOWER(object) = ?",
                        (clean, low),
                    )
                except Exception as e:
                    logger.warning("canonical upgrade failed for %r: %s", clean, e)
                return clean
            return current
        if register:
            try:
                self._db.execute(
                    "INSERT OR IGNORE INTO kg_entity_aliases (alias_lower, canonical) "
                    "VALUES (?, ?)", (low, clean),
                )
            except Exception as e:
                logger.warning("alias register failed for %r: %s", clean, e)
        return clean

    # --- Core operations ---

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 0.8,
        source: str = "extracted",
        valid_from: str | None = None,
        valid_to: str | None = None,
        provenance: str = "",
        trust: float | None = None,
    ) -> bool:
        """Add or update a fact. Returns True if added/updated.

        When a fact contradicts an existing one (same subject+predicate,
        different object), the old fact is superseded rather than deleted,
        creating a temporal trail.
        """
        subject = normalize_entity(subject)
        predicate = normalize_predicate(predicate)
        object_ = normalize_entity(object_)

        if not subject or not object_ or len(subject) > 200 or len(object_) > 200:
            return False

        # Sanitize confidence: NaN, Inf, negative → clamp to valid range
        if not isinstance(confidence, (int, float)) or math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.8  # default
        confidence = max(0.0, min(1.0, confidence))

        now = _now_iso()
        fact_valid_from = valid_from or now

        # All DB operations under the write lock to prevent TOCTOU races.
        # Sync DB work runs in a thread to avoid blocking the event loop.
        async with self._write_lock:
            result = await asyncio.to_thread(
                self._sync_add_fact, subject, predicate, object_,
                confidence, source, fact_valid_from, valid_to, provenance, now,
                trust,
            )
        return result

    # Source-default trust weights. Web-derived sources ("extracted" from
    # monitor digests, "researched" from deep-research banking) default low;
    # owner statements and internal derivations default high.
    _SOURCE_TRUST = {
        "user": 0.9, "correction": 0.9, "eval": 0.95,
        "principle": 0.75, "cross_synthesis": 0.7, "storyline": 0.65,
    }
    _WEB_SOURCES = frozenset({"extracted", "researched"})

    def _sync_add_fact(
        self, subject, predicate, object_, confidence, source,
        fact_valid_from, valid_to, provenance, now,
        trust=None,
    ) -> bool:
        """Sync helper for add_fact — all DB operations happen here (off event loop)."""
        if trust is None:
            trust = self._SOURCE_TRUST.get(source, 0.5)
        trust = max(0.0, min(1.0, float(trust)))
        # Quarantine gate: web-derived, weakly-trusted, pipeline-attributed
        # facts don't reach prompts until independently corroborated. Empty
        # provenance = manual/local add (tests, API) — never quarantined.
        quarantined = 1 if (
            trust < 0.7 and source in self._WEB_SOURCES and provenance
        ) else 0
        # Canonicalize entities (under the write lock) so casing variants collapse
        # to one form before storage — no fragmented graph nodes or dup facts.
        subject = self._canonical_entity(subject, register=True)
        object_ = self._canonical_entity(object_, register=True)
        # Check for exact duplicate
        existing = self._db.fetchone(
            "SELECT id, confidence, COALESCE(trust, 0.5) AS trust, "
            "COALESCE(quarantined, 0) AS quarantined, provenance FROM kg_facts "
            "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
            "AND valid_to IS NULL",
            (subject, predicate, object_),
        )

        if existing:
            # Corroboration promotion: the same triple observed again from a
            # DIFFERENT pipeline (provenance differs) is independent evidence —
            # lift trust and release any quarantine. This is the promotion
            # path recommended by the 2026 memory-poisoning literature:
            # corroborate-before-inject, never age-alone.
            corroborated = bool(
                provenance and existing["provenance"]
                and provenance != existing["provenance"]
            )
            if corroborated and (existing["quarantined"] or existing["trust"] < 0.9):
                new_trust = max(float(existing["trust"]), min(0.9, float(existing["trust"]) + 0.2))
                was_quarantined = bool(existing["quarantined"])
                self._db.execute(
                    "UPDATE kg_facts SET quarantined = 0, trust = ? WHERE id = ?",
                    (new_trust, existing["id"]),
                )
                # Re-index on release: prune_stale_vectors deletes a jailed fact's
                # vector, so a fact corroborated >1 day after banking would be
                # invisible to paraphrase retrieval without this (add_to_vector is
                # an idempotent upsert).
                if was_quarantined:
                    self._add_to_vector(existing["id"], subject, predicate, object_)
            if confidence > existing["confidence"]:
                self._db.execute(
                    "UPDATE kg_facts SET confidence = ?, source = ?, "
                    "provenance = CASE WHEN ? != '' THEN ? ELSE provenance END "
                    "WHERE id = ?",
                    (confidence, source, provenance, provenance, existing["id"]),
                )
                return True
            return corroborated

        # Forward contradiction: a DIFFERENT object for the same subject+predicate.
        # Only supersede when the predicate is single-valued (functional) — a
        # person lives in one place, a coin has one current price. Multi-valued
        # predicates (contains/member_of/has_property/...) coexist instead, so
        # multi-valued knowledge no longer silently collapses. Genuine
        # contradictions on multi-valued predicates are resolved upstream by the
        # LLM resolver, which sets valid_to before we get here.
        if predicate in MULTI_VALUED_PREDICATES:
            conflicts: list = []
        else:
            conflicts = list(self._db.fetchall(
                "SELECT id, object, confidence FROM kg_facts "
                "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) != LOWER(?) "
                "AND valid_to IS NULL",
                (subject, predicate, object_),
            ))

        # Inverse contradiction: a DIFFERENT subject for the same object on an
        # inverse-functional predicate (one leader per org → supersede the prior).
        if predicate in INVERSE_FUNCTIONAL_PREDICATES:
            inverse_conflicts = self._db.fetchall(
                "SELECT id, object, confidence FROM kg_facts "
                "WHERE LOWER(subject) != LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
                "AND valid_to IS NULL",
                (subject, predicate, object_),
            )
            conflicts = conflicts + list(inverse_conflicts)

        # Supersede conflicting facts + insert new fact atomically
        with self._db.transaction() as tx:
            for conflict in conflicts:
                tx.execute(
                    "UPDATE kg_facts SET valid_to = ? WHERE id = ?",
                    (now, conflict["id"]),
                )

            old_superseded = tx.fetchone(
                "SELECT id FROM kg_facts "
                "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
                "AND valid_to IS NOT NULL",
                (subject, predicate, object_),
            )

            if old_superseded:
                # Revival of a previously-superseded triple (the UNIQUE(s,p,o)
                # constraint forbids a second row, so we re-assert this one).
                # superseded_at and created_at MUST reset too: a revived row is
                # current as of NOW. Leaving a stale superseded_at made the fact
                # vanish from every bitemporal query — `query_as_of()` with no
                # args filters `superseded_at IS NULL`, and the recorded_at
                # belief query filters `superseded_at > recorded_at`, so a live
                # fact with an old supersession stamp was silently invisible.
                tx.execute(
                    "UPDATE kg_facts SET valid_from = ?, valid_to = NULL, "
                    "superseded_by = NULL, superseded_at = NULL, created_at = ?, "
                    "confidence = ?, source = ?, provenance = ?, "
                    "trust = ?, quarantined = ? WHERE id = ?",
                    (fact_valid_from, now, confidence, source, provenance,
                     trust, quarantined, old_superseded["id"]),
                )
                new_id = old_superseded["id"]
            else:
                # Explicitly set created_at = now (the bitemporal transaction
                # time) rather than letting SQLite's CURRENT_TIMESTAMP default
                # take it — keeps add_fact's own time control in one place and
                # makes the column reliable as a "when we recorded" filter.
                tx.execute(
                    "INSERT INTO kg_facts "
                    "(subject, predicate, object, confidence, source, "
                    " created_at, valid_from, valid_to, provenance, trust, quarantined) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (subject, predicate, object_, confidence, source,
                     now, fact_valid_from, valid_to, provenance, trust, quarantined),
                )
                new_row = tx.fetchone(
                    "SELECT id FROM kg_facts "
                    "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) "
                    "AND valid_to IS NULL ORDER BY id DESC LIMIT 1",
                    (subject, predicate, object_),
                )
                new_id = new_row["id"] if new_row else None

            if conflicts and new_id is not None:
                for conflict in conflicts:
                    # Bitemporal: superseded_at records the transaction-time
                    # of the supersession (when we LEARNED the new fact),
                    # parallel to valid_to which records world-time.
                    tx.execute(
                        "UPDATE kg_facts SET superseded_by = ?, superseded_at = ? WHERE id = ?",
                        (new_id, now, conflict["id"]),
                    )
                logger.info(
                    "KG: superseded %d fact(s) for %s/%s -> %s",
                    len(conflicts), subject, predicate, object_,
                )

        # Add new fact to vector store for semantic search
        if new_id is not None:
            self._add_to_vector(new_id, subject, predicate, object_)
        # Remove superseded facts from vector store
        for conflict in conflicts:
            self._remove_from_vector(conflict["id"])

        self._inserts_since_prune += 1
        if self._inserts_since_prune >= _PRUNE_BATCH_SIZE:
            self._prune()
            self._inserts_since_prune = 0

        # Emit event for event-driven triggers
        try:
            from app.monitors.event_trigger import emit_event
            emit_event("internal:kg_fact_added", {"subject": subject, "predicate": predicate, "object": object_})
        except Exception:
            pass

        # PPR adjacency cache invalidation — only invalidate when a contradicting
        # supersede happened (graph topology changed) or when this is a high-confidence
        # user/correction-sourced fact. Routine extraction flows (confidence < 0.85,
        # no supersede) coast on the 5 min TTL to avoid cache thrash.
        try:
            if conflicts or (confidence >= 0.85 and source in ("user", "correction", "user_stated")):
                from app.core import ppr as _ppr
                _ppr.invalidate_cache()
        except Exception:
            pass

        return True

    def _retire_fact(self, fact_id: int) -> bool:
        """Retire a fact by setting valid_to instead of deleting.

        This preserves temporal history. Works for single fact retirement.
        Returns True if a row was updated.
        """
        cursor = self._db.execute(
            "UPDATE kg_facts SET valid_to = CURRENT_TIMESTAMP WHERE id = ? AND valid_to IS NULL",
            (fact_id,),
        )
        return cursor.rowcount > 0

    def _retire_facts_batch(self, fact_ids: list[int]) -> int:
        """Retire multiple facts by setting valid_to. Returns count retired."""
        if not fact_ids:
            return 0
        placeholders = ",".join("?" for _ in fact_ids)
        cursor = self._db.execute(
            f"UPDATE kg_facts SET valid_to = CURRENT_TIMESTAMP "
            f"WHERE id IN ({placeholders}) AND valid_to IS NULL",
            tuple(fact_ids),
        )
        return cursor.rowcount

    async def delete_fact(self, subject: str, predicate: str, object_: str) -> bool:
        """Retire a specific fact triple (temporal retirement, not hard delete)."""
        async with self._write_lock:
            return await asyncio.to_thread(
                self._sync_delete_fact,
                subject.strip(), normalize_predicate(predicate), object_.strip(),
            )

    def _sync_delete_fact(self, subject: str, predicate: str, object_: str) -> bool:
        """Sync helper for delete_fact."""
        row = self._db.fetchone(
            "SELECT id FROM kg_facts WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) = LOWER(?) AND valid_to IS NULL",
            (subject, predicate, object_),
        )
        if row:
            ok = self._retire_fact(row["id"])
            if ok:
                try:
                    from app.core import ppr as _ppr
                    _ppr.invalidate_cache()
                except Exception:
                    pass
            return ok
        return False

    async def check_and_resolve_contradictions(
        self,
        subject: str,
        predicate: str,
        new_object: str,
        new_confidence: float = 0.8,
    ) -> bool:
        """Check for contradicting facts and resolve via LLM. Returns True if safe to add.

        Uses read-under-lock -> LLM call (no lock) -> re-read-and-write-under-lock
        pattern to avoid holding the lock during slow LLM calls while still
        preventing stale-data races.
        """
        subject = subject.strip()
        predicate = normalize_predicate(predicate)
        new_object = new_object.strip()

        # Phase 1: Read under lock — snapshot the conflicts
        async with self._write_lock:
            conflicts = await asyncio.to_thread(
                self._db.fetchall,
                "SELECT id, object, confidence FROM kg_facts "
                "WHERE LOWER(subject) = LOWER(?) AND predicate = ? AND LOWER(object) != LOWER(?) "
                "AND valid_to IS NULL",
                (subject, predicate, new_object),
            )
        if not conflicts:
            return True  # no contradiction

        # Phase 2: LLM calls outside the lock (slow I/O, no DB mutation)
        from app.core import llm

        decisions: list[tuple[dict, str]] = []  # (conflict_row, keep_verdict)
        for conflict in conflicts:
            old_object = conflict["object"]

            prompt = (
                f"Two facts conflict. Which is correct?\n"
                f"A: {subject} {predicate.replace('_', ' ')} {old_object}\n"
                f"B: {subject} {predicate.replace('_', ' ')} {new_object}\n\n"
                'Reply with JSON: {"keep": "A"} or {"keep": "B"} or {"keep": "both"} '
                'if they are not actually contradictory.'
            )
            try:
                raw = await llm.invoke_nothink(
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    json_prefix="{",
                    # 80 (was 50), 2026-08-14: the truncation tripwire caught
                    # the 9B rambling past 50 tokens — the JSON verdict was cut
                    # mid-object, extraction failed, and the silent `continue`
                    # below left BOTH conflicting facts live with no trace.
                    max_tokens=80,
                    # Schema-constrained (2026-08-29). Raising the cap treated the
                    # symptom; the cause is an UNREQUESTED field. Live repro of the
                    # four pairs that fail-opened in 24h: the 9B volunteers a
                    # "reasoning" string — 374 chars for a verdict needing 16 —
                    # which overruns the cap, cuts the JSON mid-string, and
                    # fail-opens (7/32 judgments = 22%, both facts left live).
                    # An enum schema makes the extra field structurally impossible,
                    # so the verdict cannot outgrow its budget no matter how
                    # chatty the model feels.
                    json_schema={
                        "type": "object",
                        "properties": {
                            "keep": {"type": "string", "enum": ["A", "B", "both"]},
                        },
                        "required": ["keep"],
                    },
                    temperature=0.1,
                )
                obj = llm.extract_json_object(raw)
                if not obj:
                    logger.warning(
                        "KG contradiction judge returned unparseable verdict "
                        "(%r vs %r) — both facts stay live", old_object[:60], new_object[:60])
                    continue
                keep = str(obj.get("keep", "both")).upper()
                decisions.append((conflict, keep))
            except Exception as e:
                # Unresolved → both values stay live. For a FUNCTIONAL predicate
                # that's a silent current-state contradiction, so surface it (was
                # DEBUG-only, invisible to the operator).
                logger.warning("KG contradiction check failed (allowing both): %s", e)

        # Keyed judge identity (TOKI, arXiv:2606.06240, adopted 2026-08-13):
        # replay consistency of a memory store REQUIRES logging which judge
        # adjudicated each contradiction — an unlabeled belief revision cannot
        # be audited or replayed. The losing row carries the annotation in its
        # provenance; a keep=A rejection has no stored row to annotate, so the
        # judge is named in the log line instead.
        from app.config import config as _cfg
        _judge = (getattr(_cfg, "LLM_MODEL", "") or "llm").strip()

        # Phase 3: Re-read and write under lock — verify data hasn't gone stale
        def _sync_resolve() -> bool | None:
            """Returns False to reject new fact, None to continue (allow)."""
            for conflict, keep in decisions:
                if keep == "B":
                    still_current = self._db.fetchone(
                        "SELECT id FROM kg_facts WHERE id = ? AND valid_to IS NULL",
                        (conflict["id"],),
                    )
                    if not still_current:
                        logger.debug("KG contradiction: conflict id=%d already retired, skipping", conflict["id"])
                        continue
                    now = _now_iso()
                    self._db.execute(
                        "UPDATE kg_facts SET valid_to = ?, "
                        "provenance = COALESCE(provenance, '') || ? "
                        "WHERE id = ? AND valid_to IS NULL",
                        (now, f" | adjudicated:{_judge}@{now[:10]} verdict:superseded",
                         conflict["id"]),
                    )
                    logger.info("KG contradiction resolved by %s: superseded old '%s' for new '%s'",
                                _judge, conflict["object"], new_object)
                elif keep == "A":
                    logger.info("KG contradiction resolved by %s: kept old '%s', rejected new '%s'",
                                _judge, conflict["object"], new_object)
                    return False
            return None

        async with self._write_lock:
            result = await asyncio.to_thread(_sync_resolve)
            if result is False:
                return False

        return True

    async def curate(self, sample_size: int = 20, *, heuristic: bool = True) -> dict:
        """Run curation: heuristic cleanup + LLM validation of low-confidence facts.

        Only curates current facts (valid_to IS NULL). Superseded facts are
        preserved as historical records.

        Args:
            sample_size: Number of low-confidence facts to validate via LLM (0 to skip).
            heuristic: Whether to run the heuristic filter pass.

        Returns dict with counts of deleted facts.
        """
        deleted_heuristic = 0
        deleted_llm = 0

        # Pass 1: Heuristic filters (only current facts)
        if heuristic:
            all_facts = await asyncio.to_thread(
                self._db.fetchall,
                "SELECT id, subject, predicate, object FROM kg_facts "
                "WHERE valid_to IS NULL",
            )
            ids_to_delete = []
            for row in all_facts:
                if is_garbage_triple(row["subject"], row["predicate"], row["object"]):
                    ids_to_delete.append(row["id"])

            if ids_to_delete:
                async with self._write_lock:
                    deleted_heuristic = await asyncio.to_thread(
                        self._retire_facts_batch, ids_to_delete
                    )
                logger.info("KG curation: retired %d garbage facts (heuristic)", deleted_heuristic)

        if sample_size <= 0:
            return {"heuristic": deleted_heuristic, "llm": 0}

        # Pass 2: LLM validation of lowest-confidence current facts
        low_facts = await asyncio.to_thread(
            self._db.fetchall,
            "SELECT id, subject, predicate, object, confidence FROM kg_facts "
            "WHERE valid_to IS NULL "
            "ORDER BY confidence ASC LIMIT ?",
            (sample_size,),
        )
        if not low_facts:
            return {"heuristic": deleted_heuristic, "llm": 0}

        # Batch into a single LLM call
        lines = []
        for i, f in enumerate(low_facts):
            lines.append(f"{i+1}. {f['subject']} {f['predicate'].replace('_', ' ')} {f['object']}")
        batch_text = "\n".join(lines)

        from app.core import llm as llm_mod

        prompt = (
            f"Rate each fact as 'keep' or 'garbage'. Garbage = obviously wrong, "
            f"nonsensical, test data, or trivially useless.\n\n{batch_text}\n\n"
            f'Return JSON: {{"results": [{{"id": 1, "verdict": "keep"}}, ...]}}'
        )
        try:
            raw = await llm_mod.invoke_nothink(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                json_prefix="{",
                max_tokens=500,
                temperature=0.1,
                # Schema-pinned (2026-08-29): the 9B returned "id" as a STRING,
                # so `1 <= idx` raised TypeError and the except below aborted the
                # WHOLE batch — 1 failure / 0 successes in 48h, i.e. LLM garbage
                # retirement never ran and only the heuristic pass did. Same
                # unconstrained-output class as the contradiction judge.
                json_schema={
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "verdict": {"type": "string",
                                                "enum": ["keep", "garbage"]},
                                },
                                "required": ["id", "verdict"],
                            },
                        },
                    },
                    "required": ["results"],
                },
            )
            obj = llm_mod.extract_json_object(raw)
            if obj and "results" in obj:
                garbage_ids = []
                for r in obj["results"]:
                    # Coerce anyway: the schema is the fix, this is the seatbelt.
                    # A single bad element must not abort the batch (that is what
                    # the string "id" did), so skip it and keep going.
                    try:
                        idx = int(r.get("id", 0))
                    except (TypeError, ValueError):
                        continue
                    if 1 <= idx <= len(low_facts) and r.get("verdict") == "garbage":
                        garbage_ids.append(low_facts[idx - 1]["id"])
                if garbage_ids:
                    async with self._write_lock:
                        # to_thread (2026-08-29), matching the heuristic pass ~80
                        # lines up: this is a sync UPDATE on the event-loop thread,
                        # and it holds the write lock while doing it — the
                        # 54h-freeze bug class. It was rare only because the LLM
                        # pass was itself broken (string "id" aborted every batch);
                        # now that the schema fix makes curation actually retire
                        # facts, this write would fire on every run.
                        deleted_llm = await asyncio.to_thread(
                            self._retire_facts_batch, garbage_ids)
                    logger.info("KG curation: retired %d garbage facts (LLM)", deleted_llm)
        except Exception as e:
            logger.warning("KG LLM curation failed (heuristic pass still ran): %s", e)

        return {"heuristic": deleted_heuristic, "llm": deleted_llm}

    # --- Querying ---

    def query(
        self,
        entity: str,
        hops: int = 1,
        max_results: int = 200,
        include_superseded: bool = False,
    ) -> list[dict]:
        """Get facts within N hops of an entity.

        Uses iterative BFS (1 query per hop) instead of recursive CTE
        to avoid SQLite limitations with multiple self-references.

        Args:
            entity: The entity to start from.
            hops: Number of hops to traverse.
            max_results: Maximum number of results.
            include_superseded: If False (default), only return current facts.
        """
        entity = self._canonical_entity(entity)
        if not entity:
            return []

        validity_filter = "" if include_superseded else "AND valid_to IS NULL"

        seen_ids: set[int] = set()
        visited: set[str] = set()
        results: list[dict] = []
        frontier: set[str] = {entity.lower()}

        for depth in range(hops + 1):
            if not frontier or len(results) >= max_results:
                break

            placeholders = ",".join("?" for _ in frontier)
            params = tuple(frontier) + tuple(frontier)
            rows = self._db.fetchall(
                f"SELECT id, subject, predicate, object, confidence, source, "
                f"COALESCE(quarantined, 0) AS quarantined "
                f"FROM kg_facts "
                f"WHERE (LOWER(subject) IN ({placeholders}) OR LOWER(object) IN ({placeholders})) "
                f"{validity_filter}",
                params,
            )

            next_entities: set[str] = set()
            for r in rows:
                if r["id"] in seen_ids:
                    continue
                seen_ids.add(r["id"])
                results.append({
                    "id": r["id"],
                    "subject": r["subject"],
                    "predicate": r["predicate"],
                    "object": r["object"],
                    "confidence": r["confidence"],
                    "source": r["source"],
                    "quarantined": r["quarantined"],
                    "depth": depth,
                })
                next_entities.add(r["subject"].lower())
                next_entities.add(r["object"].lower())

            visited.update(frontier)
            frontier = next_entities - visited  # only truly new entities
            # Cap frontier size to prevent query explosion on highly-connected graphs
            if len(frontier) > _config.KG_GRAPH_MAX_FRONTIER:
                frontier = set(list(frontier)[:_config.KG_GRAPH_MAX_FRONTIER])

        results.sort(key=lambda x: (x["depth"], -(x["confidence"] or 0)))
        final = results[:max_results]

        # Batch-update times_retrieved for all returned facts
        if final:
            ret_ids = [r["id"] for r in final if r.get("id") is not None]
            if ret_ids:
                placeholders = ",".join("?" for _ in ret_ids)
                try:
                    self._db.execute(
                        f"UPDATE kg_facts SET times_retrieved = times_retrieved + 1 "
                        f"WHERE id IN ({placeholders})",
                        tuple(ret_ids),
                    )
                except Exception:
                    pass  # backward compat if column missing

        return final

    def search(
        self,
        text: str,
        limit: int = 10,
        include_history: bool = False,
    ) -> list[dict]:
        """Search facts by text in subject or object.

        Args:
            text: Search term.
            limit: Maximum results.
            include_history: If True, include superseded facts.
        """
        text = text.strip().lower()
        if not text:
            return []

        # Escape LIKE wildcards
        escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        validity_filter = "" if include_history else "AND valid_to IS NULL"

        rows = self._db.fetchall(
            f"SELECT id, subject, predicate, object, confidence, source "
            f"FROM kg_facts "
            f"WHERE (subject LIKE ? ESCAPE '\\' OR object LIKE ? ESCAPE '\\') "
            f"{validity_filter} "
            f"ORDER BY confidence DESC LIMIT ?",
            (f"%{escaped}%", f"%{escaped}%", limit),
        )
        results = [dict(r) for r in rows]

        # Batch-update times_retrieved for all returned facts
        if results:
            ret_ids = [r["id"] for r in results if r.get("id") is not None]
            if ret_ids:
                placeholders = ",".join("?" for _ in ret_ids)
                try:
                    self._db.execute(
                        f"UPDATE kg_facts SET times_retrieved = times_retrieved + 1 "
                        f"WHERE id IN ({placeholders})",
                        tuple(ret_ids),
                    )
                except Exception:
                    pass  # backward compat if column missing

        return results

    def entity_subgraph(self, entity: str, limit: int = 40) -> list[Fact]:
        """All current facts touching an entity (subject OR object) plus a
        1-hop expansion — the query-time answer to "what do you know about X".

        LazyGraphRAG-style (2026-07-08, task #63): domain-level questions are
        served poorly by top-8 per-fact retrieval, but PREcomputed community
        summaries are the wrong fix (indexing cost, and the GraphRAG-family
        win-rate evidence largely collapsed under judge-bias correction —
        arXiv 2506.06331). Pull the live subgraph at query time and let the
        generation synthesize over it. Quarantined facts stay excluded.
        """
        ent = (entity or "").strip()
        if len(ent) < 2:
            return []
        # Reuse the existing BFS (query handles canonicalization, hop
        # traversal, frontier caps); it predates the quarantine column, so
        # re-check eligibility on the collected ids here.
        hits = self.query(ent, hops=1, max_results=limit * 3)
        if not hits:
            return []
        hits.sort(key=lambda r: (r.get("depth", 0), -(r.get("confidence") or 0)))
        ids = [r["id"] for r in hits[: limit * 2]]
        placeholders = ",".join("?" for _ in ids)
        rows = self._db.fetchall(
            f"SELECT * FROM kg_facts WHERE id IN ({placeholders}) "
            "AND COALESCE(quarantined, 0) = 0 "
            "ORDER BY confidence DESC LIMIT ?",
            tuple(ids) + (limit,),
        )
        return [
            Fact(
                id=r["id"], subject=r["subject"], predicate=r["predicate"],
                object=r["object"], confidence=r["confidence"], source=r["source"],
                created_at=r["created_at"],
                valid_from=r["valid_from"] if "valid_from" in r.keys() else None,
                valid_to=r["valid_to"] if "valid_to" in r.keys() else None,
                provenance=r["provenance"] if "provenance" in r.keys() else "",
                superseded_by=r["superseded_by"] if "superseded_by" in r.keys() else None,
            )
            for r in rows
        ]

    def get_relevant_facts(self, query: str, limit: int = 8) -> list[Fact]:
        """Get facts relevant to a query by hybrid keyword + semantic search.

        Uses RRF fusion of keyword overlap and ChromaDB vector similarity.
        Only returns current facts (valid_to IS NULL).
        """
        # Candidate set for relevance scoring. This MUST cover all valid facts:
        # a hard "LIMIT 500 ORDER BY confidence" silently made the majority of
        # facts unretrievable once the KG grew past 500 (found 2026-05-30 — 2734
        # valid facts, 1863 at conf>=0.95, so a freshly-relevant fact fell
        # outside the confidence window and get_relevant_facts returned []).
        # Use the KG cap so every valid fact is a candidate; keyword/vector/PPR
        # then rank by actual query relevance, not by a confidence pre-truncation.
        _cand_limit = int(getattr(_config, "MAX_KG_FACTS", 5000))
        # Quarantine gate (memory-poisoning defense, 2026-07-08): facts banked
        # from untrusted web content stay OUT of prompt injection until
        # independently corroborated — this retrieval path feeds the system
        # prompt, so it is exactly the surface a poisoned fact targets.
        all_facts = self._db.fetchall(
            "SELECT * FROM kg_facts WHERE valid_to IS NULL "
            "AND COALESCE(quarantined, 0) = 0 "
            "ORDER BY confidence DESC LIMIT ?",
            (_cand_limit,),
        )
        if not all_facts:
            return []

        rows_by_id = {row["id"]: row for row in all_facts}

        # --- Keyword search (existing approach) ---
        query_words = _normalize_words(query)
        keyword_ids: list[int] = []
        if query_words:
            scored: list[tuple[int, int]] = []
            for row in all_facts:
                fact_words = (
                    _normalize_words(row["subject"])
                    | _normalize_words(row["predicate"].replace("_", " "))
                    | _normalize_words(row["object"])
                )
                overlap = len(query_words & fact_words)
                if overlap >= 2:
                    scored.append((overlap, row["id"]))
            scored.sort(key=lambda x: -x[0])
            keyword_ids = [rid for _, rid in scored[:limit * 3]]

        # --- Vector search (semantic similarity via ChromaDB) ---
        vector_ids: list[int] = []
        vector_strong: set[int] = set()  # paraphrase-grade hits (may stand alone)
        collection = self._get_collection()
        if collection is not None and collection.count() > 0:
            try:
                from . import vector_health

                _k = min(limit * 3, collection.count())
                try:
                    results = collection.query(
                        query_texts=[query],
                        n_results=_k,
                        include=["distances"],
                    )
                except Exception as e:
                    if not vector_health.is_tombstone_error(e):
                        raise
                    # Tombstone-saturated HNSW index (the failure that killed
                    # the lessons collection 2026-08-22): retry below the
                    # observed k>=8 failure floor so the vector arm degrades
                    # instead of dying; the maintenance sweep rebuilds.
                    vector_health.record_failure("kg_facts")
                    logger.error(
                        "KG vector index tombstone-saturated (k=%d failed) — "
                        "degrading to k=%d until the maintenance rebuild",
                        _k, vector_health.DEGRADE_K,
                    )
                    results = collection.query(
                        query_texts=[query],
                        n_results=min(vector_health.DEGRADE_K, collection.count()),
                        include=["distances"],
                    )
                if results and results["ids"] and results["ids"][0]:
                    # Filter by cosine distance threshold (0 = identical, 2 = opposite).
                    # Default 0.8 (sim > 0.6) suits MiniLM; configurable because
                    # modern embedders (bge-m3) place relevant pairs at different
                    # distances. RRF fusion is the real ranker; this is a coarse gate.
                    _MAX_DISTANCE = float(getattr(_config, "KG_VECTOR_MAX_DISTANCE", 0.8))
                    # Strong tier: a tight gate (derived from the max, no new
                    # config knob) for matches confident enough to drive retrieval
                    # with no keyword/PPR support — a paraphrase has zero keyword
                    # overlap by definition. bge-m3 places correct matches at
                    # cosine 0.08-0.42, so 0.5 admits genuine paraphrases only.
                    _STRONG_DISTANCE = min(_MAX_DISTANCE, 0.5)
                    distances = results.get("distances", [[]])[0]
                    for rid_str, dist in zip(results["ids"][0], distances):
                        rid = int(rid_str)
                        if rid in rows_by_id and dist < _MAX_DISTANCE:
                            vector_ids.append(rid)
                            if dist <= _STRONG_DISTANCE:
                                vector_strong.add(rid)
            except Exception as e:
                logger.warning("KG vector search failed — retrieval degraded to keyword/PPR only: %s", e)

        # --- PPR (HippoRAG 2 style graph walk) ---
        # Adds a third signal: facts whose endpoints are reachable from query
        # entities via the kg_facts graph score high even when their literal
        # text doesn't match the query keywords. Captures multi-hop reasoning
        # like "tell me about Apple" surfacing facts about "Tim Cook" and
        # "iOS" without those entities being in the query.
        ppr_ids: list[int] = []
        if getattr(_config, "ENABLE_PPR_RETRIEVAL", False):
            try:
                from app.core import ppr as ppr_mod
                seeds = ppr_mod.extract_entities(query, max_seeds=6)
                if seeds:
                    ranked = ppr_mod.rank_facts_by_ppr(all_facts, seeds, top_k=limit * 3)
                    ppr_ids = [rid for rid, _ in ranked]
                    if ppr_ids:
                        logger.debug(
                            "[KG/PPR] %d facts ranked by graph walk (seeds=%s)",
                            len(ppr_ids), seeds[:3],
                        )
            except Exception as e:
                logger.warning("[KG/PPR] graph walk failed: %s", e)

        # --- RRF fusion ---
        # Fuse when there is a keyword match, a PPR (graph-reachability) signal,
        # OR a STRONG vector hit. Weak vector matches still only re-rank — they
        # are too noisy in small KGs where ChromaDB returns whatever it has. But
        # a strong (tightly distance-gated) vector hit is a real paraphrase match
        # and must be allowed to stand alone, or pure-paraphrase queries with no
        # keyword overlap and no graph seed retrieve nothing (the documented
        # remaining KG paraphrase-recall miss). HippoRAG-2 PPR likewise stands alone.
        if keyword_ids or ppr_ids or vector_strong:
            fused_ids = self._rrf_fuse(keyword_ids, vector_ids, ppr_ids)
            # When ONLY the vector arm fired, _rrf_fuse still ranks vector_ids;
            # restrict to strong hits so a lone weak match can't leak through.
            if not keyword_ids and not ppr_ids:
                fused_ids = [rid for rid in fused_ids if rid in vector_strong]
            # Filter to valid IDs and take top limit
            top_ids = [rid for rid in fused_ids if rid in rows_by_id][:limit]
        elif query_words:
            # No keyword matches with overlap >= 2, no PPR seeds in graph,
            # and no strong vector hit. Do NOT fall back to overlap >= 1 or weak
            # vector — single-word / low-confidence matches are too noisy.
            top_ids = []
        else:
            return []

        # --- Graph-neighbor enrichment (1-hop) on the FUSED result ---
        # If the fused (keyword + vector + PPR) result is short of `limit`, add
        # high-confidence neighbors of the matched entities so related facts
        # cluster together. Fixed 2026-05-30: this previously rebuilt a
        # keyword-overlap-only list and OVERWROTE `top_ids`, silently discarding
        # the vector/PPR ranking from the RRF fusion above. The fused result is
        # now authoritative; keyword/vector/PPR all contribute to ranking.
        if top_ids and len(top_ids) < limit:
            seen_ids = set(top_ids)
            entities: set[str] = set()
            for rid in top_ids:
                row = rows_by_id.get(rid)
                if row:
                    entities.add(row["subject"].lower())
                    entities.add(row["object"].lower())
            neighbor_budget = limit - len(top_ids)
            if entities and neighbor_budget > 0:
                placeholders = ",".join("?" for _ in entities)
                neighbors = self._db.fetchall(
                    f"SELECT id FROM kg_facts WHERE valid_to IS NULL "
                    f"AND (LOWER(subject) IN ({placeholders}) OR LOWER(object) IN ({placeholders})) "
                    f"ORDER BY confidence DESC LIMIT ?",
                    tuple(entities) + tuple(entities) + (neighbor_budget * 3,),
                )
                for nrow in neighbors:
                    nid = nrow["id"]
                    if nid not in seen_ids and nid in rows_by_id:
                        seen_ids.add(nid)
                        top_ids.append(nid)
                        if len(top_ids) >= limit:
                            break

        # Batch increment retrieval counts and update last_retrieved_at
        if top_ids:
            placeholders = ",".join("?" for _ in top_ids)
            try:
                self._db.execute(
                    f"UPDATE kg_facts SET times_retrieved = times_retrieved + 1, "
                    f"last_retrieved_at = datetime('now') WHERE id IN ({placeholders})",
                    tuple(top_ids),
                )
            except Exception:
                pass

        return [
            Fact(
                id=rows_by_id[rid]["id"],
                subject=rows_by_id[rid]["subject"],
                predicate=rows_by_id[rid]["predicate"],
                object=rows_by_id[rid]["object"],
                confidence=rows_by_id[rid]["confidence"],
                source=rows_by_id[rid]["source"],
                created_at=rows_by_id[rid]["created_at"],
                valid_from=rows_by_id[rid]["valid_from"] if "valid_from" in rows_by_id[rid].keys() else None,
                valid_to=rows_by_id[rid]["valid_to"] if "valid_to" in rows_by_id[rid].keys() else None,
                provenance=rows_by_id[rid]["provenance"] if "provenance" in rows_by_id[rid].keys() else "",
                superseded_by=rows_by_id[rid]["superseded_by"] if "superseded_by" in rows_by_id[rid].keys() else None,
            )
            for rid in top_ids
            if rid in rows_by_id
        ]

    # --- Temporal query methods ---

    def query_at(self, entity: str, at_time: str | None = None) -> list[dict]:
        """Query facts that were valid at a specific point in time.

        Args:
            entity: The entity to query (matched against subject or object).
            at_time: ISO timestamp string. If None, returns current facts
                     (where valid_to IS NULL).

        Returns:
            List of fact dicts valid at the given time.
        """
        entity = self._canonical_entity(entity)
        if not entity:
            return []

        if at_time is None:
            # Return current facts
            rows = self._db.fetchall(
                "SELECT id, subject, predicate, object, confidence, source, "
                "created_at, valid_from, valid_to, provenance "
                "FROM kg_facts "
                "WHERE (subject = ? OR object = ?) AND valid_to IS NULL "
                "ORDER BY confidence DESC",
                (entity, entity),
            )
        else:
            rows = self._db.fetchall(
                "SELECT id, subject, predicate, object, confidence, source, "
                "created_at, valid_from, valid_to, provenance "
                "FROM kg_facts "
                "WHERE (subject = ? OR object = ?) "
                "AND COALESCE(valid_from, created_at) <= ? "
                "AND (valid_to IS NULL OR valid_to > ?) "
                "ORDER BY confidence DESC",
                (entity, entity, at_time, at_time),
            )

        return [dict(r) for r in rows]

    def query_as_of(
        self,
        entity: str,
        *,
        valid_at: str | None = None,
        recorded_at: str | None = None,
    ) -> list[dict]:
        """Bitemporal query: what was the KG's belief about `entity` at a
        specific (valid_time, transaction_time) point?

        Two timelines, in Memento-style bitemporal logic:
          - valid_time: when the fact was/is true in the world
              (kept in `valid_from`/`valid_to`)
          - transaction_time: when the system recorded / superseded the fact
              (kept in `created_at`/`superseded_at`; superseded_at added
               2026-05-16, task #29)

        Args:
            entity: subject or object to look for (normalized).
            valid_at: ISO timestamp. If given, return only facts whose
                world-validity window contains this instant. If `None`,
                no valid-time filter (treat all rows as world-valid).
            recorded_at: ISO timestamp. If given, return only facts that
                the KG had recorded by this instant AND had not yet
                superseded by then. If `None`, no transaction-time filter
                (treat all rows as "currently recorded").

        Returns: list of fact dicts ordered by confidence DESC. Empty if
        the entity is unknown.

        Example — "What did we believe about Alice's job on 2026-04-01?":
            kg.query_as_of("Alice", recorded_at="2026-04-01T00:00:00")
        Returns rows we had in the DB on that date and hadn't superseded yet,
        regardless of whether those records were later overturned.
        """
        entity = self._canonical_entity(entity)
        if not entity:
            return []

        clauses = ["(subject = ? OR object = ?)"]
        params: list[Any] = [entity, entity]

        # Bitemporal filter cases — split on which arguments were given so each
        # has its own clean semantic (no double-filter conflation):
        if valid_at is not None and recorded_at is not None:
            # Audit query: "Which rows EXISTED in the DB by recorded_at AND
            # were world-valid at valid_at?" Row presence at recorded_at is
            # just `created_at <= RT`; supersession status at RT is irrelevant —
            # we want to reconstruct what records we *had* about that VT period.
            clauses.append("created_at <= ?")
            clauses.append("COALESCE(valid_from, created_at) <= ?")
            clauses.append("(valid_to IS NULL OR valid_to > ?)")
            params.extend([recorded_at, valid_at, valid_at])
        elif recorded_at is not None:
            # Belief query: "What did we believe at recorded_at?" — rows that
            # existed by RT and had not yet been superseded by RT.
            clauses.append("created_at <= ?")
            clauses.append("(superseded_at IS NULL OR superseded_at > ?)")
            params.extend([recorded_at, recorded_at])
        elif valid_at is not None:
            # Historical world-time query: any row world-valid at VT, including
            # ones currently superseded (we still HAVE the historical record).
            clauses.append("COALESCE(valid_from, created_at) <= ?")
            clauses.append("(valid_to IS NULL OR valid_to > ?)")
            params.extend([valid_at, valid_at])
        else:
            # No filters — currently believed facts (mirrors query_at(None)).
            clauses.append("superseded_at IS NULL")

        sql = (
            "SELECT id, subject, predicate, object, confidence, source, "
            "created_at, valid_from, valid_to, provenance, "
            "superseded_by, superseded_at "
            "FROM kg_facts WHERE " + " AND ".join(clauses) + " "
            "ORDER BY confidence DESC"
        )
        rows = self._db.fetchall(sql, tuple(params))
        return [dict(r) for r in rows]

    def get_fact_history(self, subject: str, predicate: str) -> list[dict]:
        """Return all versions of a fact over time (current + superseded).

        Args:
            subject: The subject entity.
            predicate: The predicate (will be normalized).

        Returns:
            List of fact dicts ordered by valid_from DESC (most recent first).
        """
        subject = self._canonical_entity(subject)
        predicate = normalize_predicate(predicate)

        rows = self._db.fetchall(
            "SELECT id, subject, predicate, object, confidence, source, "
            "created_at, valid_from, valid_to, provenance, superseded_by "
            "FROM kg_facts "
            "WHERE subject = ? AND predicate = ? "
            "ORDER BY valid_from DESC",
            (subject, predicate),
        )
        return [dict(r) for r in rows]

    def get_changes_since(self, since: str, limit: int = 50) -> list[dict]:
        """Return facts created or superseded since a given timestamp.

        Useful for "what changed in the last week?" queries.

        Args:
            since: ISO timestamp string.
            limit: Maximum results.

        Returns:
            List of fact dicts that were created or had their valid_to set
            since the given timestamp.
        """
        rows = self._db.fetchall(
            "SELECT id, subject, predicate, object, confidence, source, "
            "created_at, valid_from, valid_to, provenance, superseded_by "
            "FROM kg_facts "
            "WHERE valid_from >= ? OR (valid_to IS NOT NULL AND valid_to >= ?) "
            "ORDER BY COALESCE(valid_to, valid_from) DESC "
            "LIMIT ?",
            (since, since, limit),
        )
        return [dict(r) for r in rows]

    # --- Formatting ---

    @staticmethod
    def format_for_prompt(facts: list[Fact]) -> str:
        """Format facts as a prompt-ready string with confidence and temporal labels.

        Facts with valid_from within the last 7 days get a [NEW] label.
        Superseded facts are excluded. Source provenance is appended in parens
        when non-default so the model can weight evidence by where it came from
        (monitor vs chat extraction vs user-provided correction).
        """
        if not facts:
            return ""
        lines = []
        for f in facts:
            # Skip superseded facts
            if f.superseded_by is not None or f.valid_to is not None:
                continue

            pred = f.predicate.replace("_", " ")
            conf = f.confidence if f.confidence is not None else 0
            label = "[HIGH]" if conf >= 0.8 else ("[MED]" if conf >= 0.5 else "[LOW]")

            # Add [NEW] for recently-added facts
            new_tag = ""
            if _is_recent(f.valid_from, days=7):
                new_tag = "[NEW] "

            # Surface source for grounding — skip plain "extracted" since that's the
            # default for chat answers and adds no signal. Real provenance (monitor
            # name, "user", "correction") is informative for the model.
            src = f.provenance or f.source or ""
            src_tag = ""
            if src and src not in ("extracted", "inferred"):
                src_tag = f", src: {src[:40]}"

            # Natural-language rendering (2026-07-08, task #63): paraphrased
            # evidence measurably increases small-model receptiveness vs
            # bare templated triples (ACL 2025, arXiv 2409.10955). CONTRACT
            # with brain._kg_answers_query: every sentence starts with the
            # SUBJECT immediately followed by its verb phrase — the gate
            # parses "SUBJECT <verb>" to fire tool-less generation, so no
            # leading adverbs or inverted forms here.
            phrase = _PRED_PHRASES.get(f.predicate, pred)
            lines.append(
                f"- {new_tag}{label} {f.subject} {phrase} {f.object} "
                f"[confidence: {conf:.2f}{src_tag}]"
            )
        return "\n".join(lines)

    @staticmethod
    def format_summary_for_prompt(facts: list[Fact]) -> str:
        """Format facts as compact one-line summaries with IDs for lazy retrieval.

        Each line includes the fact ID so the LLM can call
        context_detail(category='kg_fact', item_id=N) for full details.
        """
        if not facts:
            return ""
        lines = []
        for f in facts:
            if f.superseded_by is not None or f.valid_to is not None:
                continue
            pred = f.predicate.replace("_", " ")
            conf = f.confidence if f.confidence is not None else 0
            lines.append(f"- [K{f.id}] {f.subject} —{pred}→ {f.object} ({conf:.1f})")
        return "\n".join(lines)

    # --- Management ---

    def get_all_facts(self, limit: int = 100, offset: int = 0) -> list[Fact]:
        """Paginated fact listing."""
        rows = self._db.fetchall(
            "SELECT * FROM kg_facts ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [
            Fact(
                id=r["id"], subject=r["subject"], predicate=r["predicate"],
                object=r["object"], confidence=r["confidence"],
                source=r["source"], created_at=r["created_at"],
                valid_from=r["valid_from"] if "valid_from" in r.keys() else None,
                valid_to=r["valid_to"] if "valid_to" in r.keys() else None,
                provenance=r["provenance"] if "provenance" in r.keys() else "",
                superseded_by=r["superseded_by"] if "superseded_by" in r.keys() else None,
            )
            for r in rows
        ]

    def get_top_entities(self, limit: int = 10) -> list[dict]:
        """Return the top entities by fact count (current facts only).

        Returns list of dicts with 'subject' and 'cnt' keys, ordered by count descending.
        """
        rows = self._db.fetchall(
            "SELECT subject, COUNT(*) as cnt FROM kg_facts "
            "WHERE valid_to IS NULL GROUP BY subject ORDER BY cnt DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Return KG statistics."""
        total = self._db.fetchone("SELECT COUNT(*) AS c FROM kg_facts")["c"]
        current = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM kg_facts WHERE valid_to IS NULL"
        )["c"]
        superseded = total - current
        entities_row = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM ("
            "SELECT subject AS e FROM kg_facts WHERE valid_to IS NULL "
            "UNION SELECT object FROM kg_facts WHERE valid_to IS NULL)"
        )
        predicates = self._db.fetchone(
            "SELECT COUNT(DISTINCT predicate) AS c FROM kg_facts WHERE valid_to IS NULL"
        )["c"]
        return {
            "total_facts": total,
            "current_facts": current,
            "superseded_facts": superseded,
            "unique_entities": entities_row["c"] if entities_row else 0,
            "unique_predicates": predicates,
        }

    def _prune(self) -> None:
        """If current kg_facts exceed _config.MAX_KG_FACTS, delete oldest low-confidence ones.

        Only prunes current facts. Superseded facts are historical and not counted.
        """
        count_row = self._db.fetchone(
            "SELECT COUNT(*) AS c FROM kg_facts WHERE valid_to IS NULL"
        )
        count = count_row["c"] if count_row else 0
        if count <= _config.MAX_KG_FACTS:
            return
        excess = count - _config.MAX_KG_FACTS
        # Retire (set valid_to) instead of hard-deleting to preserve temporal history
        prune_rows = self._db.fetchall(
            "SELECT id FROM kg_facts "
            "WHERE valid_to IS NULL "
            "ORDER BY times_retrieved ASC, confidence ASC, created_at ASC "
            "LIMIT ?",
            (excess,),
        )
        prune_ids = [r["id"] for r in prune_rows]
        retired = self._retire_facts_batch(prune_ids)
        logger.info("Pruned (retired) %d KG facts (over %d limit)", retired, _config.MAX_KG_FACTS)

    async def decay_stale(self, days: int = 60, decay_amount: float = 0.05) -> int:
        """Lower confidence on old current facts. Returns count affected."""
        cutoff = f"-{days} days"
        async with self._write_lock:
            def _do_decay():
                cursor = self._db.execute(
                    "UPDATE kg_facts SET confidence = MAX(0.1, confidence - ?) "
                    "WHERE created_at < datetime('now', ?) AND valid_to IS NULL "
                    "AND (last_retrieved_at IS NULL OR last_retrieved_at < datetime('now', ?))",
                    (decay_amount, cutoff, cutoff),
                )
                return cursor.rowcount
            return await asyncio.to_thread(_do_decay)

    async def hard_prune_dead_facts(self, days: int = 120, max_count: int = 1000) -> int:
        """Soft-retire facts that have NEVER been retrieved and are older
        than `days`.

        RECALIBRATED 2026-07-08: the original 60-day policy was justified by a
        2026-05-06 audit stat ("92% of KG facts never retrieved") measured
        while the retrieval stack was BROKEN (LIMIT-500 window + discarded RRF
        fusion, both fixed 2026-05-30). Re-measured on the working stack:
        51% of live facts HAVE been retrieved, and 40% of facts get their
        first retrieval within 14 days of banking. Never-retrieved-at-120d is
        real evidence of dead weight; never-retrieved-at-60d was mostly
        evidence the product's payoff horizon (storylines, forecasts) is
        longer than the window. Soft retire only (valid_to set), capped per
        cycle.
        """
        async with self._write_lock:
            def _do_prune():
                cursor = self._db.execute(
                    "UPDATE kg_facts SET valid_to = datetime('now') "
                    "WHERE id IN ("
                    "  SELECT id FROM kg_facts "
                    "  WHERE valid_to IS NULL "
                    "  AND last_retrieved_at IS NULL "
                    "  AND created_at < datetime('now', ?) "
                    "  ORDER BY created_at ASC LIMIT ?"
                    ")",
                    (f"-{days} days", max_count),
                )
                return cursor.rowcount
            return await asyncio.to_thread(_do_prune)

    async def prune_related_to_junk(self, days: int = 45, max_count: int = 1000) -> int:
        """Retire low-value `related_to` facts — the KG's biggest quality drag.

        `related_to` is the DEGRADE TARGET for any relation the extractor
        couldn't normalize to a specific predicate, so it accumulates vague
        associations ("Trump related_to Zelensky") that dilute retrieval. Manual
        audit 2026-07-09 found it was 61% of the live store. Two targeted rules,
        both safe (soft-retire, never touches a specific predicate):
          1. SUPERSEDED-BY-SPECIFIC: a `related_to` (s,o) where a specific
             predicate already links the SAME s→o (or o→s) is pure redundancy —
             the specific fact carries the real relation. Retire the vague one.
          2. STALE + UNUSED: a `related_to` never retrieved and older than
             `days` (shorter than the 120d generic prune — related_to earns its
             keep faster or not at all).
        Specific-predicate facts are never affected.
        """
        async with self._write_lock:
            def _do():
                # Rule 1: related_to redundant with a specific predicate on the same pair
                c1 = self._db.execute(
                    "UPDATE kg_facts SET valid_to = datetime('now') "
                    "WHERE predicate = 'related_to' AND valid_to IS NULL AND id IN ("
                    "  SELECT r.id FROM kg_facts r JOIN kg_facts s "
                    "    ON s.valid_to IS NULL AND s.predicate != 'related_to' "
                    "    AND ((LOWER(s.subject)=LOWER(r.subject) AND LOWER(s.object)=LOWER(r.object)) "
                    "      OR (LOWER(s.subject)=LOWER(r.object) AND LOWER(s.object)=LOWER(r.subject))) "
                    "  WHERE r.predicate='related_to' AND r.valid_to IS NULL "
                    "  LIMIT ?)", (max_count,)).rowcount
                # Rule 2: stale + never-retrieved related_to
                c2 = self._db.execute(
                    "UPDATE kg_facts SET valid_to = datetime('now') "
                    "WHERE predicate = 'related_to' AND valid_to IS NULL "
                    "AND last_retrieved_at IS NULL AND created_at < datetime('now', ?) "
                    "AND id IN (SELECT id FROM kg_facts WHERE predicate='related_to' "
                    "  AND valid_to IS NULL AND last_retrieved_at IS NULL "
                    "  AND created_at < datetime('now', ?) ORDER BY created_at ASC LIMIT ?)",
                    (f"-{days} days", f"-{days} days", max_count)).rowcount
                return c1 + c2
            return await asyncio.to_thread(_do)

    async def promote_aged_quarantine(self, days: int = 21, max_count: int = 500) -> int:
        """Age-release quarantined facts — to a NON-AUTHORITATIVE surfaced state.

        Quarantine (memory-poisoning defense) holds uncorroborated web-derived
        facts out of prompts UNTIL corroborated — but a fact never re-observed
        would stay a permanent grave, starving chat of real single-source intel
        (the #49 regression, 2026-07-08).

        HARDENED 2026-07-09 (full-system exploration): trust is keyed on source
        CREDIBILITY at banking time, so a credible single-source fact is NEVER
        quarantined in the first place (add_fact / deep_research._learn_facts).
        Everything still in quarantine is therefore, by construction, a LOW-
        CREDIBILITY single-source claim — exactly the patient-poisoner surface
        (bank one uncontradicted fabrication, wait for auto-release). The old
        7-day → trust 0.7 release handed that attacker an injected fact with the
        SAME authority as corroborated intel. Two changes close the hole without
        re-graving legit intel:
          • window 7 → 21 days (3× the cost of a patient poison; chat can still
            web-search in the meantime, so intel is delayed, not lost);
          • release to trust 0.6 (NOT 0.7) — it surfaces but renders sub-
            authoritative ([MED]/[LOW]), never stated as established fact.
        Corroboration still promotes to full trust immediately (add_fact). This
        is only the time-based backstop for the never-re-observed tail.
        Returns count released.
        """
        async with self._write_lock:
            def _do():
                rows = self._db.fetchall(
                    "SELECT id, subject, predicate, object FROM kg_facts "
                    "WHERE COALESCE(quarantined,0)=1 AND valid_to IS NULL "
                    "AND superseded_at IS NULL AND created_at < datetime('now', ?) LIMIT ?",
                    (f"-{days} days", max_count),
                )
                if not rows:
                    return 0
                ids = [r["id"] for r in rows]
                placeholders = ",".join("?" for _ in ids)
                self._db.execute(
                    "UPDATE kg_facts SET quarantined = 0, "
                    "trust = MIN(COALESCE(trust,0.5), 0.6) "
                    f"WHERE id IN ({placeholders})",
                    ids,
                )
                # Re-index released facts — prune_stale_vectors deleted their
                # vectors while jailed; without this the released fact is
                # invisible to paraphrase retrieval (memory-poisoning literature's
                # corroborate-before-inject only helps if the released fact is
                # actually findable again).
                for r in rows:
                    self._add_to_vector(r["id"], r["subject"], r["predicate"], r["object"])
                return len(ids)
            return await asyncio.to_thread(_do)

    async def prune_dead_aliases(self) -> int:
        """Delete entity aliases whose canonical no longer has any LIVE fact.
        kg_entity_aliases was insert-only (INSERT OR IGNORE) and grew to ~64%
        dead (2026-08-14 audit), so alias expansion resolved into a mostly-dead
        namespace. An alias is dead when its canonical is neither subject nor
        object of a live (unsuperseded, unexpired, unquarantined) fact. Runs from
        daily maintenance. Returns count pruned."""
        async with self._write_lock:
            def _do():
                cur = self._db.execute(
                    "DELETE FROM kg_entity_aliases WHERE canonical NOT IN ("
                    "  SELECT subject FROM kg_facts WHERE subject IS NOT NULL "
                    "    AND valid_to IS NULL AND superseded_at IS NULL "
                    "    AND COALESCE(quarantined,0)=0 "
                    "  UNION "
                    "  SELECT object FROM kg_facts WHERE object IS NOT NULL "
                    "    AND valid_to IS NULL AND superseded_at IS NULL "
                    "    AND COALESCE(quarantined,0)=0)"
                )
                return cur.rowcount
            return await asyncio.to_thread(_do)

    async def retire_stale_snapshots(self, days: int = 7, max_count: int = 500) -> int:
        """Soft-retire POINT-IN-TIME research facts whose truth has an expiry:
        prices, percentages, counts, and other magnitude objects banked from
        monitors ('Bitcoin price_of $97,500'). Without this they stay marked
        current forever and the KG silently fills with stale "truths" (audit
        2026-07-06; the fact-banking fix accelerates the inflow). Timeless facts
        (is_a, located_in, non-numeric objects) are untouched; retirement sets
        valid_to (recoverable), matching hard_prune_dead_facts.
        """
        volatile_predicates = (
            "price_of", "trading_at", "worth", "valued_at", "costs", "priced_at",
            "market_cap", "holds", "rate_of", "yield_of",
        )
        pred_sql = " OR ".join("predicate = ?" for _ in volatile_predicates)
        async with self._write_lock:
            def _do_retire():
                cursor = self._db.execute(
                    "UPDATE kg_facts SET valid_to = datetime('now') "
                    "WHERE id IN ("
                    "  SELECT id FROM kg_facts "
                    "  WHERE valid_to IS NULL "
                    "  AND provenance LIKE 'deep_research%' "
                    "  AND created_at < datetime('now', ?) "
                    "  AND (" + pred_sql + " "
                    "       OR object GLOB '*$[0-9]*' OR object GLOB '*[0-9]%*' "
                    "       OR object GLOB '[0-9][0-9,.]*[0-9]') "
                    "  ORDER BY created_at ASC LIMIT ?"
                    ")",
                    (f"-{days} days", *volatile_predicates, max_count),
                )
                return cursor.rowcount
            return await asyncio.to_thread(_do_retire)

    def get_provenance_usage_stats(self, provenance: str) -> dict:
        """Return usage stats for facts with a given provenance.

        Used to validate whether speculative provenance like 'cross_synthesis'
        actually produces useful facts (high times_retrieved) or just noise
        (high count, zero retrieval).

        Matches by prefix — e.g. provenance='cross_synthesis' will match
        'cross_synthesis:3_monitors:24h' (cross_monitor writes suffixed
        provenance for breadth metadata).
        """
        try:
            like_pattern = f"{provenance}%"
            row = self._db.fetchone(
                "SELECT "
                "  COUNT(*) AS total, "
                "  SUM(CASE WHEN times_retrieved > 0 THEN 1 ELSE 0 END) AS used, "
                "  AVG(times_retrieved) AS avg_retrievals, "
                "  MAX(times_retrieved) AS max_retrievals "
                "FROM kg_facts WHERE provenance LIKE ? AND valid_to IS NULL",
                (like_pattern,),
            )
            if not row:
                return {"total": 0, "used": 0, "avg_retrievals": 0.0, "max_retrievals": 0}
            return {
                "total": int(row["total"] or 0),
                "used": int(row["used"] or 0),
                "avg_retrievals": float(row["avg_retrievals"] or 0.0),
                "max_retrievals": int(row["max_retrievals"] or 0),
            }
        except Exception as e:
            logger.warning("Provenance usage stats failed: %s", e)
            return {"total": 0, "used": 0, "avg_retrievals": 0.0, "max_retrievals": 0}

    async def decay_unused_speculative(self, provenance: str = "cross_synthesis",
                                        days: int = 14, decay_amount: float = 0.15) -> int:
        """Aggressively decay speculative facts (e.g. cross_synthesis) that
        weren't retrieved within `days`. Closes the loop on synthesis quality —
        useful synthesis gets retrieved; useless synthesis decays out fast.

        Matches by prefix so 'cross_synthesis' covers 'cross_synthesis:3_monitors:24h' etc.
        """
        cutoff = f"-{days} days"
        like_pattern = f"{provenance}%"
        async with self._write_lock:
            def _do():
                cursor = self._db.execute(
                    "UPDATE kg_facts SET confidence = MAX(0.05, confidence - ?) "
                    "WHERE provenance LIKE ? AND valid_to IS NULL "
                    "AND created_at < datetime('now', ?) "
                    "AND COALESCE(times_retrieved, 0) = 0",
                    (decay_amount, like_pattern, cutoff),
                )
                return cursor.rowcount
            return await asyncio.to_thread(_do)
