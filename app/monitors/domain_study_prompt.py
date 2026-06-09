"""Strict Domain Study prompt template.

Replaces the per-monitor scattered prompt strings with one rigid template
the LLM must follow. Reduces format drift across the 36 Domain Study
monitors and enforces:

  - Time window: only past 48h, current year
  - Visual structure: numbered headlines with emojis
  - Source citations under each item
  - Date format: "Month DD, YYYY"
  - 3-5 items, no fluff between them

Used by `_build_enriched_query` in heartbeat_loop.py for any monitor
whose check_type=='query' and name starts with "Domain Study:".
"""

from __future__ import annotations

from datetime import datetime, timezone


# Per-domain emoji + focus areas. Drives the headline style and the
# "what to look for" hint the LLM gets.
_DOMAIN_PROFILES: dict[str, tuple[str, str]] = {
    "ai and ml": ("🤖", "model releases, research breakthroughs, benchmark results, major company announcements"),
    "space and astronomy": ("🚀", "rocket launches, satellite deployments, exoplanet discoveries, NASA/ESA/SpaceX missions"),
    "health and medicine": ("💊", "drug approvals, clinical trial results, disease outbreaks, public health policy"),
    "energy and climate": ("⚡", "renewable energy milestones, climate policy, emissions data, battery technology, nuclear"),
    "cybersecurity": ("🔒", "major breaches, new CVEs, ransomware, security tool releases, policy changes"),
    "geopolitics": ("🌍", "international conflicts, diplomatic negotiations, sanctions, military movements, trade disputes"),
    "crypto and web3": ("₿", "price movements, protocol upgrades, DeFi events, regulatory actions, ETF developments"),
    "quantum computing": ("⚛️", "qubit milestones, error correction, new processors, IBM/Google/IonQ announcements"),
    "robotics and autonomy": ("🦾", "humanoid robots, self-driving vehicles, industrial automation, drones, embodied AI"),
    "us policy and regulation": ("🏛️", "tech regulation, AI governance, trade policy, court rulings, executive orders"),
    "startups and vc": ("💰", "funding rounds, IPOs, acquisitions, unicorn valuations"),
    "physics and mathematics": ("🔬", "theoretical results, experimental confirmations, major papers, breakthrough proofs"),
    "biotech and genetics": ("🧬", "CRISPR advances, gene therapy trials, synthetic biology, longevity research"),
    "economics and markets": ("📊", "GDP, unemployment, inflation, central bank decisions, housing"),
    "whale watch": ("🐋", "large on-chain transactions, wallet movements, accumulation patterns"),
    "top trades and positioning": ("📈", "fund positioning, large trades, options activity"),
    "china tech and economy": ("🇨🇳", "Chinese tech companies, AI models (DeepSeek/Qwen), economic data, US-China competition"),
    "russia and eastern europe": ("🇷🇺", "Russia-Ukraine, Russian economy, Eastern European politics, NATO"),
    "middle east": ("🕌", "regional conflicts, OPEC, Gulf states, Iran, Israel-Palestine"),
    "india": ("🇮🇳", "Indian tech sector, startups, economic data, digital policy, semiconductors"),
    "europe and eu": ("🇪🇺", "EU regulation, ECB, European tech, defense policy"),
    "semiconductors": ("🧪", "NVIDIA/AMD/Intel/TSMC announcements, AI chips, fab construction, export controls"),
    "commodities and forex": ("🛢️", "oil, gold/silver, FX pairs, agricultural commodities, industrial metals"),
    "earnings and corporate events": ("📈", "earnings reports, M&A, IPOs, CEO changes, layoffs, product launches"),
    "open source and github": ("🐙", "trending repos, tool releases, version releases, license changes"),
    "defense and military tech": ("⚔️", "weapons systems, drones, autonomous platforms, AI in defense, defense contracts"),
    "defi and protocols": ("💰", "protocol upgrades, governance, TVL, bridge hacks, airdrops, L2/rollups"),
    "developer ecosystem": ("💻", "language updates, framework releases, package manager changes, IDE updates"),
    "latin america": ("🇲🇽", "Brazilian/Mexican economy, regional tech, lithium/resources, Argentine reforms"),
    "africa and emerging markets": ("🌍", "African fintech, EM currencies, natural resources, startup ecosystems"),
    "supply chain and trade": ("🚚", "shipping disruptions, tariff changes, reshoring, container rates, critical minerals"),
    "research frontiers": ("🧠", "trending arxiv papers, notable research, AI/biology/physics breakthroughs"),
    "current events": ("📰", "politics, environment, health, culture (NOT technology/AI)"),
    "world awareness": ("🌎", "global headlines worth knowing, cross-domain"),
    "finance": ("💵", "macro finance, equities, bonds, rates"),
    "technology": ("💻", "programming tools, frameworks, platforms, software releases"),
}


def _profile_for(monitor_name: str) -> tuple[str, str]:
    """Return (emoji, focus_areas) for a Domain Study monitor."""
    label = monitor_name.replace("Domain Study:", "").strip().lower()
    return _DOMAIN_PROFILES.get(label, ("📰", "developments worth tracking"))


def build_domain_study_prompt(monitor_name: str) -> str:
    """Build the strict prompt for a Domain Study monitor.

    The format is rigid because Discord rendering depends on it. Every
    item must use the **exact** template — the body formatter (and the
    reject-on-stale-date guard) assume this shape.
    """
    emoji, focus = _profile_for(monitor_name)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_human = datetime.now(timezone.utc).strftime("%B %d, %Y")
    label = monitor_name.replace("Domain Study:", "").strip()

    # Compute the lower bound of acceptable dates (last 48h) so the model
    # has the explicit cutoff in front of it instead of having to do date
    # math.
    from datetime import timedelta
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=48)
    cutoff_human = cutoff_dt.strftime("%B %d, %Y")

    return f"""TODAY IS {today_human} ({today}). YEAR={cutoff_dt.year}.

This is REAL — not hypothetical, not a future scenario. The system clock is authoritative.
Acceptable date range for this report: **{cutoff_human} through {today_human} (last 48 hours)**.

Use the `deep_research` tool (preferred — it fetches actual source pages and verifies dates) OR `web_search` (mode='news') to find 3-5 {label} developments inside that window.

Focus: {focus}.

═══ HARD RULES (a violation invalidates the entire report) ═══

1. EVERY item MUST have:
   - A real outlet name (e.g. "Reuters", "TechCrunch", "Wikipedia") — never "tech-insider.org" or fabricated
   - A real, fully-spelled date in the form **{today_human[:5]} DD, {cutoff_dt.year}** (Month DD, YYYY)
   - A real URL of the form https://domain.tld/path — never "stackoverflowcom" or "techinsideorg" (these are fabrications)

2. NO HEDGING in dates. The following are BANNED and will trigger a re-roll:
   - "Apr/May", "April–May", "April-May" (you must pick one specific date)
   - "~Apr", "approximately", "around", "early/mid/late April"
   - "X days ago", "this week", "recently" (use the calendar date)
   - Date ranges like "April 7–8" (pick the publish date)

3. EVERY date must fall on or after **{cutoff_human}**. Anything before that is OUT OF SCOPE — drop it from the report.

4. NO HEDGING in numbers. Use exact figures from the source ("51% of devs use Copilot", not "around half"). If you don't have a number, omit the claim — don't invent one.

5. If after 2 tool calls you cannot find 3 items meeting all rules, output EXACTLY:
   "No significant {label} developments in the past 48 hours."
   Padding with stale or hedged items will fail the quality gate and re-roll.

═══ OUTPUT FORMAT (rigid — Discord depends on it) ═══

## {emoji} {label} — {today_human}

**1. [Headline as a noun phrase, ≤80 chars, no hedging]**
*{emoji} Source: [Outlet Name] · Date: {today_human[:5]} DD, {cutoff_dt.year} · https://[real-url]*
[2-3 sentences: what happened + why it matters + one named entity or specific number]

**2. [Headline]**
*{emoji} Source: [Outlet Name] · Date: {today_human[:5]} DD, {cutoff_dt.year} · https://[real-url]*
[2-3 sentences]

**3. [Headline]**
*{emoji} Source: [Outlet Name] · Date: {today_human[:5]} DD, {cutoff_dt.year} · https://[real-url]*
[2-3 sentences]

(Items 4-5 only if all meet every rule above.)

Start the response with the `##` header. No preamble. No "Sources:" block at the bottom.
"""
