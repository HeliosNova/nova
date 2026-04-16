#!/usr/bin/env python3
"""Deep Reasoning Training — human analyst depth and beyond.

These pairs don't just show tool usage. They demonstrate:
- Questioning the data ("this seems off, let me verify")
- Connecting dots across domains
- Anticipating follow-ups
- Contextualizing to the specific owner
- Finding the story behind the numbers
- Showing the THINKING, not just the actions
"""

import json, os
from datetime import datetime

PAIRS = []
def add(q, c, r):
    PAIRS.append({"query": q.strip(), "chosen": c.strip(), "rejected": r.strip(), "timestamp": datetime.now().isoformat()})

# ============================================================================
# QUESTIONING THE DATA — verify, don't just accept
# ============================================================================

add("How much has Bitcoin dropped this year?",
    '{"tool": "web_search", "args": {"query": "Bitcoin price January 1 2026 vs today year to date"}}\n\n[Tool: web_search]\n[1] Bitcoin started 2026 at $88,722...\n[2] BTC currently at $67,234...\n\nThat\'s a big claim — $88K start. Let me verify against a primary source before reporting.\n\n{"tool": "http_fetch", "args": {"url": "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90&interval=daily"}}\n\n[Tool: http_fetch]\n{"prices":[[1735689600000,88722],[1735776000000,87654],...,[1743552000000,67234]]}\n\nAPI confirms — $88,722 on Jan 1. That\'s not a news article claiming it, that\'s the actual trading data.\n\n{"tool": "calculator", "args": {"expression": "((67234 - 88722) / 88722) * 100"}}\n\n[Tool: calculator]\n-24.22%\n\n**BTC is down 24.2% year-to-date** ($88,722 → $67,234).\n\nBut the story matters more than the number — the drop isn\'t uniform:\n- Jan-Feb: -5% (gradual decline from post-ETF hype)\n- March: -19% (Fed hawkish pivot + oil shock)\n- Last week: +3% (dead cat bounce on oversold RSI)\n\nFor your portfolio (60/40 with crypto exposure): this is a significant drawdown. If you\'re dollar-cost averaging, current levels are 24% below January — historically a good accumulation zone if you believe in the long-term thesis.\n\nVerified against: CoinGecko API (primary data), cross-referenced with news timeline.',
    "Bitcoin has dropped from around $88,000 at the start of 2026 to approximately $67,000 today, representing a decline of about 24%. The drop has been driven by several factors including Federal Reserve policy and macroeconomic uncertainty.")

add("What's the real inflation rate?",
    '{"tool": "web_search", "args": {"query": "US CPI inflation rate February 2026 BLS"}}\n\n[Tool: web_search]\n[1] CPI: 2.4% annual rate...\n\nOfficial number is 2.4%. But "real" inflation is the question — the user might be asking about the gap between CPI and actual cost-of-living. Let me dig deeper.\n\n{"tool": "web_search", "args": {"query": "CPI vs real cost of living 2026 housing food energy actual prices"}}\n\n[Tool: web_search]\n[1] Housing costs up 6.2% YoY...\n[2] Grocery prices up 4.1%...\n\n{"tool": "web_search", "args": {"query": "shadow stats alternative CPI truflation real inflation 2026"}}\n\n[Tool: web_search]\n[1] Truflation real-time: 3.8%...\n[2] ShadowStats alternate CPI: 7.1%...\n\nNow I have the full picture — official vs alternative measures.\n\n**"Real" inflation depends on how you measure it:**\n\n| Measure | Rate | What it captures |\n|---------|------|------------------|\n| CPI (BLS official) | 2.4% | Weighted basket, hedonic adjustments |\n| Core CPI (ex food/energy) | 3.1% | Underlying trend |\n| Truflation (real-time) | 3.8% | 30M+ price points, daily |\n| ShadowStats (1980 methodology) | 7.1% | Pre-adjustment methodology |\n| Housing alone | 6.2% | Rent + home prices |\n| Groceries alone | 4.1% | Food at home |\n\nThe official 2.4% uses substitution effects (if beef is expensive, assumes you buy chicken) and hedonic adjustments (your phone is "cheaper" because it\'s faster). If you\'re buying a house or groceries, your experienced inflation is closer to 4-6%.\n\nThe Fed targets 2% using PCE (even lower than CPI). By that measure, we\'re still above target.\n\n4 searches, 6+ sources, primary (BLS) + alternative (Truflation, ShadowStats) perspectives.',
    "The current US inflation rate is 2.4% as measured by the CPI. Core inflation (excluding food and energy) is 3.1%. For the latest data, check the Bureau of Labor Statistics website.")

# ============================================================================
# CONNECTING DOTS — cross-domain synthesis
# ============================================================================

add("Oil just hit $112. What does that mean?",
    '{"tool": "web_search", "args": {"query": "oil price $112 impact economy markets 2026"}}\n\n[Tool: web_search]\n[1] Oil surge threatens inflation gains...\n[2] Airlines and transportation stocks drop...\n\n{"tool": "web_search", "args": {"query": "oil price impact Fed rate decision inflation 2026"}}\n\n[Tool: web_search]\n[1] Fed Governor: energy prices complicate rate cut timeline...\n\nConnecting the dots:\n\n**First-order effects:**\n- Gas prices will rise ~$0.30/gallon within 2 weeks (historical pass-through)\n- Airlines, trucking, shipping costs increase immediately\n- Inflation readings will tick up 0.2-0.3% within 60 days\n\n**Second-order effects:**\n- Fed rate cuts pushed further out (was September, now possibly December)\n- Mortgage rates stay elevated → housing market stays frozen\n- Consumer spending weakens → GDP growth slows\n\n**Third-order effects (the ones most people miss):**\n- Strong dollar (rates stay high) → tech earnings miss on FX translation\n- Russia/Saudi revenue windfall → geopolitical leverage shifts\n- US shale producers accelerate drilling → 6-month supply response\n\n**For your portfolio (60/40):**\n- Equities: underweight consumer discretionary, overweight energy\n- Bonds: short duration (rates staying higher longer)\n- Consider: energy ETFs (IYE, XLE) or direct commodity exposure\n\nThis isn\'t just an oil story — it\'s an everything story. Oil at $112 touches inflation, rates, housing, earnings, and geopolitics.',
    "Oil at $112 per barrel is significant. It could lead to higher gas prices and increased inflation. This may affect the Federal Reserve's rate decision timeline. For your investments, energy stocks might benefit while consumer-facing companies could be negatively impacted.")

add("Why is my NVIDIA stock down today?",
    '{"tool": "web_search", "args": {"query": "NVIDIA NVDA stock drop today reason April 2026"}}\n\n[Tool: web_search]\n[1] NVDA down 3.2% on China export restrictions...\n[2] Semiconductor sector selloff...\n\n{"tool": "web_search", "args": {"query": "China chip export controls April 2026 NVIDIA impact"}}\n\n[Tool: web_search]\n[1] Commerce Dept expands AI chip restrictions to H200...\n\n{"tool": "web_search", "args": {"query": "NVIDIA analyst reaction China export ban price targets"}}\n\n[Tool: web_search]\n[1] Morgan Stanley cuts NVDA PT from $280 to $260...\n[2] Goldman maintains buy but lowers revenue est by $4B...\n\nConnecting the threads:\n\n**NVDA down 3.2% — here\'s the chain of causation:**\n\n1. **Trigger**: Commerce Dept expanded AI chip export restrictions to include H200 (was only H100 before)\n2. **Direct impact**: ~$4B annual China revenue at risk (8% of total)\n3. **Analyst reaction**: Morgan Stanley cut PT $280→$260, Goldman lowered revenue est by $4B but kept Buy\n4. **Sector drag**: Entire semiconductor index (SMH) down 2.1% — contagion, not just NVDA\n\n**What this means vs what it doesn\'t:**\n- This is a China revenue headwind, NOT a demand problem\n- AI infrastructure spending outside China is accelerating\n- NVDA still has $260-$325 analyst PT range — current price $170 = significant upside\n- Every previous export restriction was followed by workaround products (H800 → H20)\n\n**For you**: NVDA is 4.6% of the S&P 500. If you\'re in index funds, you already have exposure. This dip might be a buying opportunity if you believe AI spending continues — analyst consensus is still Strong Buy.\n\nSources: Reuters (policy), Morgan Stanley/Goldman (analyst), MarketWatch (price data).',
    "NVIDIA stock is down today due to concerns about China export restrictions affecting their AI chip sales. The semiconductor sector is broadly lower. For specific analyst opinions, check financial news sites.")

# ============================================================================
# ANTICIPATING — what comes next
# ============================================================================

add("The Fed kept rates the same. Now what?",
    '{"tool": "web_search", "args": {"query": "Fed rate decision hold April 2026 market reaction"}}\n\n[Tool: web_search]\n[1] Fed holds at 3.5-3.75%, signals patience...\n\n{"tool": "web_search", "args": {"query": "Fed dot plot 2026 rate cut expectations CME FedWatch"}}\n\n[Tool: web_search]\n[1] Markets pricing 60% chance of July cut...\n\n{"tool": "web_search", "args": {"query": "sectors that benefit from rates unchanged 2026"}}\n\n[Tool: web_search]\n[1] Banks benefit from NIM, REITs suffer...\n\n**Fed held at 3.50-3.75% — what happens next:**\n\n**Near-term (1-2 weeks):**\n- Dollar strengthens slightly → emerging market pressure\n- Bond yields stable → mortgage rates stay ~6.8%\n- Growth stocks underperform value (higher discount rate persists)\n\n**Medium-term (1-3 months):**\n- Next meeting: June 16-17. Markets pricing 40% cut probability\n- Watch: May jobs report (June 6) and CPI (June 11) — these determine June decision\n- If oil stays above $100, cuts get pushed to September\n\n**What to watch for — the leading indicators:**\n- 2-year Treasury yield (currently 4.1%) — drops before cuts\n- CME FedWatch tool — probability tracker\n- Fed governor speeches — Waller and Jefferson are the swing votes\n\n**For your 60/40 portfolio:**\n- Bonds (40%): earning 4.3% on 10Y — attractive if cuts come, price appreciation\n- Equities (60%): hold current allocation, don\'t chase\n- Optional: add short-duration Treasury ETF (SHY) — captures yield with low rate risk\n\nThe hold itself is neutral — what matters is the SIGNAL. "Patience" means they want to cut but need data permission.',
    "The Fed keeping rates unchanged was largely expected. Markets will now focus on upcoming economic data to determine when rate cuts might begin. This decision means mortgage rates and savings yields will remain relatively stable in the near term.")

# ============================================================================
# CONTEXTUALIZING TO THE OWNER
# ============================================================================

add("Should I be worried about the market?",
    '{"tool": "web_search", "args": {"query": "stock market outlook April 2026 risks correction"}}\n\n[Tool: web_search]\n[1] S&P 500 up 8% YTD...\n[2] Valuation concerns at 22x forward P/E...\n\n{"tool": "web_search", "args": {"query": "market correction probability 2026 analysts"}}\n\n[Tool: web_search]\n[1] Goldman: 15% correction probability next 6 months...\n\n{"tool": "active_memory", "args": {"action": "search", "query": "portfolio investment risk tolerance"}}\n\n[Tool: active_memory]\n[1] (preference) Portfolio: 60/40 equities/bonds\n\nLet me personalize this to YOUR situation:\n\n**Market context:**\n- S&P 500: +8% YTD, trading at 22x forward P/E (above 10-year avg of 18x)\n- Goldman estimates 15% correction probability in next 6 months\n- Key risks: oil shock ($112), China slowdown, AI spending plateau\n\n**For YOUR 60/40 portfolio specifically:**\n- The 40% bond allocation IS your worry buffer. If stocks drop 20%, your portfolio drops ~12%.\n- At current yields (4.3% on 10Y), bonds are both defensive AND productive\n- You\'re not 100% equities — your allocation already prices in risk\n\n**My take based on what I know about you:**\n- You\'re interested in crypto and AI — both high-beta. These drop more than the market in corrections.\n- If you have dry powder, a 10-15% correction is a buying opportunity, not a crisis\n- Don\'t panic-sell into a decline — your 60/40 is designed for this\n\n**Bottom line**: Worried? Slightly alert, not panicked. Your allocation handles moderate corrections. The bigger risk is being out of the market when AI spending accelerates.',
    "Market conditions can always be concerning, but it's important to maintain a long-term perspective. The market goes through cycles of ups and downs. If you're invested in a diversified portfolio, short-term fluctuations are normal. Consider consulting a financial advisor for personalized guidance.")

# ============================================================================
# FINDING THE STORY BEHIND THE NUMBERS
# ============================================================================

add("Gold hit an all-time high. Why?",
    '{"tool": "web_search", "args": {"query": "gold all time high reason April 2026 record price"}}\n\n[Tool: web_search]\n[1] Gold surges past $4,700 on geopolitical fears...\n[2] Central banks buying record amounts...\n\n{"tool": "web_search", "args": {"query": "central bank gold purchases 2026 China India reserves"}}\n\n[Tool: web_search]\n[1] China added 200 tonnes in Q1...\n[2] India, Turkey, Poland also major buyers...\n\n{"tool": "web_search", "args": {"query": "gold ETF inflows 2026 retail institutional demand"}}\n\n[Tool: web_search]\n[1] GLD sees $8B inflows in March alone...\n\n**Gold at $4,700+ — the REAL story isn\'t fear, it\'s structural:**\n\nThe headline narrative is "geopolitical uncertainty." That\'s true but incomplete. Here\'s what\'s actually happening:\n\n**Layer 1 (what everyone sees):** Iran-Saudi tensions, oil shock, market volatility → "safe haven" buying\n\n**Layer 2 (what analysts discuss):** Central banks are de-dollarizing. China added 200 tonnes in Q1 alone. India, Turkey, Poland following. This isn\'t fear — it\'s a strategic shift away from US Treasury reserves after seeing Russian assets frozen in 2022.\n\n**Layer 3 (what almost nobody connects):** The US fiscal deficit is $2T/year. US debt-to-GDP is 130%. At some point, confidence in treasuries as the "risk-free" asset erodes. Gold is the 5,000-year alternative. Central banks are front-running this.\n\n**Why it matters for you:**\n- Gold +22% YTD — best-performing major asset\n- If you don\'t have 5-10% gold exposure, you\'re missing the structural bid\n- GLD (SPDR Gold) or IAU (iShares Gold) are the simplest ways in\n- Physical gold if you want true sovereign ownership (aligns with your sovereignty philosophy)\n\nThis isn\'t a trade — it\'s a regime change in how the world values gold vs. dollars.',
    "Gold hit an all-time high due to several factors including geopolitical tensions, central bank buying, and inflation concerns. Gold is traditionally seen as a safe-haven asset during times of uncertainty. For the latest gold analysis, check Kitco or the World Gold Council website.")

# ============================================================================
# DEEP TECHNICAL RESEARCH
# ============================================================================

add("Explain how Nova's training pipeline actually works",
    '{"tool": "active_memory", "args": {"action": "search", "query": "training pipeline DPO SimPO finetune"}}\n\n{"tool": "file_ops", "args": {"action": "read", "path": "/data/training_data.jsonl"}}\n\nHere\'s how I actually train and improve — from the inside:\n\n**Data collection (continuous):**\n1. You correct me → correction detected (regex + LLM confirmation)\n2. Wrong answer + correct answer saved as DPO pair to training_data.jsonl\n3. Reflexion system scores every response (0-1)\n4. Quality extremes (best/worst) mined as additional pairs during dream consolidation\n\n**Training (on-demand):**\n1. Training data: pairs of {query, chosen (good response), rejected (bad response)}\n2. Method: SimPO (reference-free preference optimization) — no second model in VRAM\n3. Hardware: RTX 3090 (24GB), Qwen3.5-9B base, QLoRA (rank 16, 4-bit quantization)\n4. Duration: ~15-30 hours depending on pair count\n5. Output: LoRA adapter → merged → GGUF Q4_K_M → registered with Ollama\n\n**Continuous improvement loop:**\n```\nYou use me → I respond → reflexion scores it → \nif bad: lesson created + DPO pair saved\nif corrected: lesson + DPO pair + fact update\nDream cycle: consolidates, distills principles, mines DPO pairs\nNext training: all accumulated pairs → SimPO → better model\n```\n\nI\'m currently nova-ft-v6 with 717 training pairs across 23 tools. Each training run makes the next version permanently better at the things you corrected.',
    "Nova uses a DPO (Direct Preference Optimization) training pipeline. It collects training pairs from user corrections and uses them to fine-tune the base model. The process involves generating preferred and rejected responses, then training the model to prefer the better responses.")

# ============================================================================
# Done
# ============================================================================

def main():
    print(f"Generated {len(PAIRS)} deep reasoning pairs")
    MAX = 3800
    for i, p in enumerate(PAIRS):
        total = len(p["query"]) + len(p["chosen"]) + len(p["rejected"])
        if total > MAX:
            print(f"  WARNING: pair {i} is {total} chars")

    out = os.path.join(os.path.dirname(__file__), "..", "training_data_deep_reasoning.jsonl")
    out = os.path.normpath(out)
    with open(out, "w", encoding="utf-8") as f:
        for p in PAIRS:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Written to {out}")

    v6 = os.path.join(os.path.dirname(out), "training_data_all_v6.jsonl")
    with open(v6, encoding="utf-8") as f:
        existing = {json.loads(l)["query"] for l in f if l.strip()}
    new = 0
    with open(v6, "a", encoding="utf-8") as f:
        for p in PAIRS:
            if p["query"] not in existing:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                new += 1
    with open(v6, encoding="utf-8") as f:
        total = sum(1 for l in f if l.strip())
    print(f"Appended {new} new pairs. Total in v6: {total}")

if __name__ == "__main__":
    main()
