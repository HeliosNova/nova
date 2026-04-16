#!/usr/bin/env python3
"""DPO Training Curriculum v6 — Daemon Mode, Dream Consolidation, Frustration Handling.

Targeted at capabilities required for:
  - KAIROS-like proactive daemon (terse background reasoning)
  - autoDream-like memory consolidation (cross-system merge/resolve/prune)
  - Frustration de-escalation (detect + adapt behavior)
  - Observation writing (concise daemon log entries)
  - Action/no-action decisions (most ticks should be "no action")

Informed by Claude Code source leak (2026-03-31) revealing:
  - 15-second blocking budget → terse daemon outputs
  - 4-phase memory consolidation → cross-system reasoning
  - Frustration regex → behavioral adaptation
  - Append-only observation logs → concise logging

Output: training_data_v6.jsonl (~75 pairs)
"""

import json
import os
from datetime import datetime

PAIRS: list[dict] = []


def add(query: str, chosen: str, rejected: str):
    PAIRS.append({
        "query": query.strip(),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "timestamp": datetime.now().isoformat(),
    })


# ============================================================================
# CATEGORY 1: Brief/Daemon Mode (15 pairs)
# When queries are prefixed [DAEMON], respond in 1-2 sentences MAX.
# The model must learn that daemon context = terse output.
# ============================================================================

add(
    "[DAEMON] Evaluate: 3 monitor results changed in last hour. BTC -1.2%, ETH -0.8%, System Health OK. User idle 45min. No pending events. Budget: brief. Decision?",

    "No action. Crypto movements are within normal range. System healthy. Nothing anomalous.",

    """Let me analyze each of the three monitor results that changed in the last hour:

1. **Bitcoin (BTC) -1.2%**: This is a relatively small decline. Bitcoin often fluctuates by 1-2% throughout the day, so this doesn't necessarily indicate a major trend change. However, I should note that we should keep watching this.

2. **Ethereum (ETH) -0.8%**: Similarly, this is a minor fluctuation within normal daily trading ranges. No immediate concern.

3. **System Health: OK**: All systems are operating normally.

Given that the user has been idle for 45 minutes and there are no pending events, I don't think any immediate action is needed. I'll continue monitoring these metrics and will alert the user if there are any significant changes."""
)

add(
    "[DAEMON] Evaluate: Cybersecurity monitor detected critical CVE in Python 3.12 (CVE-2026-1234, CVSS 9.8). User idle 2h. Budget: full. Decision?",

    "ACTION: Alert user. CVE-2026-1234 is CVSS 9.8 affecting Python 3.12 — critical severity, needs immediate attention. Sending to all channels.",

    """I've detected a critical cybersecurity vulnerability. Let me provide a comprehensive analysis:

**CVE-2026-1234 Details:**
- CVSS Score: 9.8 (Critical)
- Affected: Python 3.12
- The user has been idle for 2 hours

This is a significant finding that warrants attention. Given the critical severity score of 9.8, this vulnerability could potentially affect any systems running Python 3.12. The user should be notified about this so they can take appropriate action, such as checking if their systems are affected and applying any available patches.

I recommend sending an alert to the user through all available notification channels (Discord, Telegram, etc.) with the full details of the vulnerability and recommended mitigation steps."""
)

add(
    "[DAEMON] Evaluate: Morning check-in complete. 4 monitors ran overnight, all OK. No curiosity items pending. User active 30s ago. Budget: brief. Decision?",

    "No action. Overnight all clear. User is active — stay quiet.",

    """Good morning! Here's a summary of overnight activity:

The morning check-in has been completed successfully. Let me review the overnight monitor results:

1. All 4 monitors that ran overnight reported OK status
2. There are no pending curiosity items in the queue
3. The user became active just 30 seconds ago

Since everything is running smoothly and the user just became active, I should probably prepare a morning digest for them. However, since everything is normal, there's no urgent action needed. I'll wait for the user to initiate a conversation."""
)

add(
    "[DAEMON] Evaluate: Whale Watch monitor: 50,000 BTC moved to exchange. Previous pattern: large exchange deposits precede 5%+ drops. User idle 15min. Budget: light. Decision?",

    "ACTION: Alert user. 50K BTC exchange deposit matches historical sell pattern. Light alert — no detailed analysis, just the signal.",

    """This is an interesting finding from the Whale Watch monitor. Let me analyze this in detail:

**On-chain Analysis:**
- 50,000 BTC has been transferred to an exchange
- This is a significant amount worth approximately $X at current prices
- Historical data shows that large exchange deposits often precede price declines of 5% or more

**Historical Context:**
Looking at previous similar events:
- January 2026: 40K BTC deposit → 6.2% drop within 48h
- December 2025: 55K BTC deposit → 4.8% drop within 24h

**Recommendation:**
Given this pattern, I think we should alert the user about this whale movement. They may want to adjust their positions or set stop-losses.

Let me draft a comprehensive alert with all the relevant details..."""
)

add(
    "[DAEMON] Evaluate: 0 monitor changes. 0 events. User idle 3h. Last dream cycle: 18h ago. Budget: full. Decision?",

    "ACTION: Trigger dream consolidation. Idle 3h, last dream 18h ago. Good window for memory consolidation.",

    """Let me assess the current state:

- No monitor changes detected
- No events in the queue
- User has been idle for 3 hours
- The last dream/consolidation cycle was 18 hours ago

Given the extended idle period and the fact that it's been 18 hours since the last consolidation cycle, this would be an ideal time to run a memory consolidation pass. The user won't be interrupted since they've been inactive for 3 hours.

I should probably run the full 4-phase consolidation process:
1. Orient — check current memory state
2. Gather signal — scan recent conversations
3. Consolidate — merge and resolve
4. Prune — clean up

Let me proceed with triggering the consolidation process..."""
)

add(
    "[DAEMON] Evaluate: System Health — disk 92% full. User active 5min ago. Budget: brief. Decision?",

    "ACTION: Alert user. Disk at 92% — approaching critical. Brief alert, don't interrupt workflow.",

    """The system health monitor has detected that disk usage has reached 92%. This is concerning because:

1. At 92%, we're approaching the critical threshold where system performance may degrade
2. Databases (SQLite, ChromaDB) need write space for operations
3. Log files and monitor results continue to accumulate

I should alert the user about this situation. However, since they were active just 5 minutes ago, I want to be careful not to disrupt their current workflow. I'll send a brief notification.

Here are some suggested cleanup actions:
- Archive old monitor results (older than 30 days)
- Clean up old conversation history
- Prune superseded KG facts
- Delete old screenshots from /data/screenshots/"""
)

add(
    "[DAEMON] Evaluate: Curiosity queue has 12 pending items, 3 at CRITICAL urgency. User idle 1h. Budget: light. Decision?",

    "ACTION: Research top 3 critical curiosity items. User idle, good window. Queue them as background tasks.",

    """I notice the curiosity queue has accumulated 12 pending items, with 3 at CRITICAL urgency level. Let me review the situation:

The 3 critical items likely represent knowledge gaps that were detected during recent conversations where I couldn't provide adequate answers. With the user being idle for an hour, this is a good opportunity to address these gaps.

I should research the 3 critical items first, as they represent the most significant knowledge gaps. For each one, I'll:
1. Use web search to find current information
2. Extract relevant knowledge graph triples
3. Create lessons if the findings are substantial
4. Resolve the curiosity item with findings

Let me start by querying the curiosity queue to get the specific topics of the 3 critical items..."""
)

add(
    "[DAEMON] Evaluate: 15 monitors due. LLM semaphore at 2/2 (both slots busy). Budget: brief. Decision?",

    "No action. LLM saturated. Monitors will execute when slots free. Don't queue more.",

    """There are 15 monitors due for execution, but both LLM semaphore slots are currently occupied. Let me assess the situation:

Since the LLM concurrency limit is set to 2 and both slots are in use, I can't execute any query-type monitors right now. The monitors will need to wait until one of the current LLM tasks completes.

Options to consider:
1. Wait for a slot to free up — the most conservative approach
2. Check if any of the 15 monitors are fast types (system_health, maintenance) that don't need LLM
3. Prioritize the most important monitors to run first when a slot opens

I think the best approach is to wait. The monitors will be picked up by the next heartbeat tick when LLM slots become available. Trying to force execution could cause resource contention."""
)

add(
    "[DAEMON] Evaluate: User sent 4 messages in last 10 minutes. All to Discord channel. Last message 2min ago. Budget: brief. Decision?",

    "No action. User is actively chatting on Discord. Stay out of the way.",

    """I can see the user has been actively communicating on Discord, sending 4 messages in the last 10 minutes with the most recent being 2 minutes ago. This indicates they're currently engaged in a conversation.

In this situation, I should:
1. Avoid sending any proactive notifications that might interrupt their conversation
2. Continue background monitoring silently
3. Queue any non-urgent observations for later

The user is clearly active and engaged, so any daemon-initiated actions would be disruptive. I'll monitor the situation and wait for them to become idle before taking any proactive actions."""
)

add(
    "[DAEMON] Tick summary — last 6h: 18 monitors ran (all OK), 2 curiosity items resolved, 1 KG contradiction resolved, 0 alerts sent. Write observation.",

    "Quiet 6h window. 18 monitors clean, 2 curiosity items resolved, 1 KG fix. No user interaction. System stable.",

    """Here's a comprehensive summary of the last 6-hour observation period:

**Monitor Activity:**
- 18 monitors executed successfully with all returning OK status
- No changes detected that warranted alerts
- This indicates a stable system and external environment

**Learning Activity:**
- 2 curiosity items were resolved through background research
- 1 knowledge graph contradiction was detected and resolved
- These improvements enhance future response quality

**Alert Activity:**
- 0 alerts were sent during this period
- This is positive as it means no critical issues arose

**Overall Assessment:**
The system is running smoothly with all monitors operating normally. The background learning processes continue to improve the knowledge base. No user intervention is needed at this time."""
)

add(
    "[DAEMON] Evaluate: GitHub webhook — new PR #42 opened on nova_ repo by user. Budget: light. Decision?",

    "Observe: PR #42 opened on nova_. Log it. Don't review — user likely knows they opened it.",

    """A new pull request has been opened on the Nova repository! Let me evaluate this:

**Event Details:**
- PR #42 opened on nova_ repository
- Opened by the primary user

Since this was opened by the user themselves, they're clearly aware of it. However, I could potentially:
1. Review the PR changes and provide feedback
2. Run tests against the changes
3. Check for potential conflicts with existing code
4. Analyze the changes for security implications

Given the light budget, I think I should at least log this event and perhaps do a quick review of the changes. Let me fetch the PR details to understand what was changed..."""
)

add(
    "[DAEMON] Evaluate: Morning. User timezone 9:00 AM. Daily digest not sent yet. User active 3min ago. Budget: brief. Decision?",

    "ACTION: Send daily digest. Morning window, user just came online.",

    """It's 9:00 AM in the user's timezone and the daily digest hasn't been sent yet. The user became active 3 minutes ago, which means they've started their day.

This is the ideal time to deliver the daily digest because:
1. The user is at the start of their workday
2. They're already active, so the notification won't wake them
3. The overnight monitor results and learning activity should be summarized

Let me compile the digest with:
- Overnight monitor highlights (any significant changes)
- Learning summary (new lessons, curiosity research results)
- System health status
- Any pending items that need attention

I'll format it concisely and send it through the configured notification channels..."""
)

add(
    "[DAEMON] Evaluate: Fine-tune check monitor: 690 DPO pairs ready, last training 5 days ago. User idle 4h. Budget: full. Decision?",

    "Observe: 690 DPO pairs available, last train 5d ago. Log for user review — don't auto-trigger training (destructive, needs approval).",

    """Excellent! The fine-tune readiness check shows we have 690 DPO pairs available for training, and it's been 5 days since the last training run. The user has been idle for 4 hours, which means we have a good window for GPU-intensive work.

However, I need to consider several factors before triggering training:
1. Training requires stopping Ollama (which affects all chat functionality)
2. It takes 6-10+ hours on the RTX 3090
3. There's risk of training on bad data if pairs haven't been reviewed
4. The user might have opinions on training configuration

I think the best approach is to send a notification to the user about the training readiness, rather than automatically starting it. The user should have the final say on when to start training.

Let me draft a notification with the current statistics and a recommendation..."""
)

add(
    "[DAEMON] Evaluate: Signal channel received message from unknown number +1555... . Not in allowed_users list. Budget: brief. Decision?",

    "Observe: Blocked message from unknown +1555... on Signal. Logged. No action needed — allowlist handled it.",

    """A message was received on the Signal channel from an unknown phone number (+1555...) that is not in the allowed_users list.

**Security Assessment:**
- The message was from an unrecognized number
- The SIGNAL_ALLOWED_USERS configuration is properly blocking unauthorized users
- This could be spam, a wrong number, or someone trying to access Nova

**Actions to Consider:**
1. Log the attempt for security auditing
2. Alert the user about the unauthorized access attempt
3. Check if there have been multiple attempts from this number
4. Consider whether the user might be expecting messages from a new contact

Since the allowlist is working correctly and blocking the message, the immediate threat is mitigated. However, I should log this for the user's awareness in case they need to add a new contact or investigate potential unauthorized access attempts."""
)

add(
    "[DAEMON] Evaluate: 2 monitors failed with LLM timeout in last hour. Backoff active (15min). 5 monitors queued. Budget: brief. Decision?",

    "No action. Backoff active — LLM likely overloaded or Ollama slow. Monitors will retry at backoff schedule. Don't force.",

    """Two monitors have failed due to LLM timeouts in the past hour, and the exponential backoff mechanism is active with a 15-minute delay. There are 5 monitors waiting in the queue.

Let me analyze the situation:

**Root Cause Analysis:**
- LLM timeouts suggest either Ollama is under heavy load or experiencing issues
- The backoff mechanism is correctly preventing retry storms
- 5 monitors are queued and waiting

**Options:**
1. Wait for backoff to expire — monitors will retry automatically
2. Check Ollama health — is it running? Is VRAM usage high?
3. Restart Ollama if it appears hung
4. Skip non-critical monitors and prioritize important ones

I think the best course of action is to first check the Ollama health status. If it's responding normally, the backoff will handle the retries. If it's not responding, we might need to restart it.

Let me check the system health..."""
)


# ============================================================================
# CATEGORY 2: Memory Consolidation Reasoning (15 pairs)
# The model must compare across facts/lessons/KG and make merge/prune decisions.
# ============================================================================

add(
    "[CONSOLIDATION] Cross-system check:\nFact: job = \"Senior Software Engineer at TechCorp\"\nKG: (user, works_at, TechCorp), confidence 0.7\nKG: (user, is_a, software engineer), confidence 0.6\nLesson: \"User prefers technical code-level explanations over high-level overviews\"\n\nOverlap analysis?",

    """Overlap found:
- Fact + KG(works_at) + KG(is_a) all encode "user is an engineer at TechCorp" → retire both KG triples (fact has source authority 4, KG confidence <0.8)
- Lesson is unique — encodes a *preference*, not a fact. Keep.
- Action: retire KG triples, keep fact and lesson.""",

    """Let me analyze the overlap between these three knowledge systems:

**User Fact:**
- job = "Senior Software Engineer at TechCorp"
- This is stored in the user facts system

**Knowledge Graph Triples:**
1. (user, works_at, TechCorp) — confidence 0.7
2. (user, is_a, software engineer) — confidence 0.6

**Lesson:**
- "User prefers technical code-level explanations over high-level overviews"

**Analysis:**
There is clearly some overlap here. The user fact about being a "Senior Software Engineer at TechCorp" captures the same information as both KG triples combined. The KG triples are essentially decomposed versions of the same fact.

However, the lesson is different. While it's related to the user being an engineer, it captures a behavioral preference rather than a biographical fact. This is unique information that isn't duplicated elsewhere.

**Recommendation:**
I would suggest keeping all three types of information as they serve different purposes in the system. The fact provides a concise summary, the KG triples enable graph-based queries, and the lesson guides response formatting."""
)

add(
    "[CONSOLIDATION] Lesson contradiction detected:\nLesson A (confidence 0.82): \"User prefers metric units for all measurements\"\nLesson B (confidence 0.71): \"Use imperial units when discussing cooking temperatures\"\n\nResolve?",

    """Not a contradiction — B is a specific exception to A. Keep both.
- A: general rule (metric preferred)
- B: domain-specific override (cooking = imperial)
- Boost B confidence to 0.75 (validated as intentional exception).""",

    """I've detected a contradiction between two lessons:

**Lesson A** (confidence 0.82):
"User prefers metric units for all measurements"

**Lesson B** (confidence 0.71):
"Use imperial units when discussing cooking temperatures"

These two lessons are in direct conflict. Lesson A says to always use metric, but Lesson B says to use imperial for cooking temperatures.

**Resolution Options:**
1. Keep Lesson A (higher confidence), delete Lesson B
2. Keep Lesson B (more specific), modify Lesson A
3. Merge into a single lesson with the exception noted

I think the best approach is Option 1 — keep the higher-confidence lesson since it was likely stated more recently or more explicitly by the user. The cooking temperature preference might have been a one-time request that was incorrectly generalized into a lesson.

Deleting Lesson B and keeping Lesson A would maintain consistency in the unit preference system."""
)

add(
    "[CONSOLIDATION] Stale facts review (not accessed in 60+ days):\n1. favorite_color = \"blue\" (last accessed: 62 days ago, access_count: 1)\n2. laptop_model = \"ThinkPad X1 Carbon Gen 11\" (last accessed: 90 days ago, access_count: 3)\n3. timezone = \"America/Los_Angeles\" (last accessed: 65 days ago, access_count: 47)\n\nPrune decision?",

    """Keep all three. Access recency ≠ relevance:
- favorite_color: low-value, but source=user (authority 4). Don't prune user-stated facts. Demote to low-priority retrieval.
- laptop_model: useful for tech support. Keep.
- timezone: actively used by system (DIGEST_HOUR, date formatting) even if not explicitly accessed via fact store. Critical. Refresh access timestamp.""",

    """Let me evaluate each stale fact for potential pruning:

**1. favorite_color = "blue"**
- Last accessed: 62 days ago
- Access count: 1
- Assessment: This fact has very low utility — it was accessed only once and over 60 days ago. It's consuming storage space without providing value.
- **Recommendation: PRUNE** — This is a trivial personal preference that rarely comes up in conversation.

**2. laptop_model = "ThinkPad X1 Carbon Gen 11"**
- Last accessed: 90 days ago
- Access count: 3
- Assessment: This is somewhat useful for tech-related conversations but hasn't been relevant in 3 months.
- **Recommendation: PRUNE** — If the user needs laptop-specific help, they'll mention the model.

**3. timezone = "America/Los_Angeles"**
- Last accessed: 65 days ago
- Access count: 47
- Assessment: Despite being "stale" by the 60-day threshold, this fact has 47 accesses, suggesting it was very useful at some point.
- **Recommendation: KEEP** — High historical utility suggests future relevance."""
)

add(
    "[CONSOLIDATION] KG fact chain — supersession history for (Bitcoin, price_of, *):\n1. (Bitcoin, price_of, $42,000) — valid_from: Jan 1, superseded_by: #2\n2. (Bitcoin, price_of, $48,500) — valid_from: Jan 15, superseded_by: #3\n3. (Bitcoin, price_of, $51,200) — valid_from: Feb 1, superseded_by: #4\n4. (Bitcoin, price_of, $47,800) — valid_from: Feb 20, superseded_by: #5\n5. (Bitcoin, price_of, $53,100) — valid_from: Mar 15, current\n\nCompact?",

    """Compact to last 2 versions:
- Delete #1, #2, #3 (older than 60 days, superseded)
- Keep #4 (previous value, useful for trend context)
- Keep #5 (current)
- Price-type facts change frequently — aggressive compaction is correct here.""",

    """Let me analyze this KG fact chain for Bitcoin's price:

We have 5 entries tracking Bitcoin's price over time:
1. $42,000 (Jan 1) → superseded
2. $48,500 (Jan 15) → superseded
3. $51,200 (Feb 1) → superseded
4. $47,800 (Feb 20) → superseded
5. $53,100 (Mar 15) → current

**Temporal Analysis:**
This chain shows an interesting price trajectory — starting at $42K, rising to $51.2K, dipping to $47.8K, then recovering to $53.1K. This historical data could be valuable for understanding market trends.

**Compaction Options:**
1. Keep all entries (preserves full history)
2. Keep first, last, and any significant changes
3. Keep only the current value

I think we should keep all entries because:
- The temporal data provides valuable market context
- The price dip in February (#4) represents a meaningful event
- Historical data helps with trend analysis

**Recommendation:** Keep all 5 entries. The storage cost is minimal and the historical context is valuable for financial analysis."""
)

add(
    "[CONSOLIDATION] Reflexion pattern review:\n- 5 reflexions about \"tool timeout\" failures (quality 0.2-0.4)\n- 3 reflexions about \"search returned no results\" (quality 0.3-0.5)\n- 2 reflexions about \"hallucinated a URL\" (quality 0.1-0.2)\n- All from last 2 weeks\n\nPromote to lessons?",

    """Promote \"tool timeout\" (5 occurrences, clear pattern):
- Lesson: \"When a tool times out, immediately try an alternative approach (different search terms, fallback URL, smaller request) instead of retrying the same call\"
- Merge the 5 reflexions into 1 representative, delete rest

Promote \"hallucinated URL\" (2 occurrences but quality < 0.2 = severe):
- Lesson: \"Never construct URLs from memory — always search for the correct URL first\"

Don't promote \"search no results\" — 3 occurrences is borderline and the cause varies (bad query vs topic unavailable). Keep reflexions, revisit next cycle.""",

    """Let me review these reflexion patterns for potential lesson promotion:

**Pattern 1: Tool Timeout (5 reflexions, quality 0.2-0.4)**
This is a recurring pattern with 5 instances. The quality scores are low, indicating these are genuine failures. This should be promoted to a lesson.

**Pattern 2: Search No Results (3 reflexions, quality 0.3-0.5)**
This pattern has 3 instances, which meets the minimum threshold for promotion. The quality scores are slightly higher, suggesting these might be partially successful responses.

**Pattern 3: Hallucinated URL (2 reflexions, quality 0.1-0.2)**
Only 2 instances, below the typical 3-instance threshold for promotion. However, the very low quality scores (0.1-0.2) indicate these are severe failures.

**Recommendations:**
1. Promote Pattern 1 to a lesson about handling tool timeouts
2. Promote Pattern 2 to a lesson about refining search strategies
3. Monitor Pattern 3 — wait for one more occurrence before promoting

For each promoted lesson, I'll use the LLM to synthesize the reflexion details into a clear, actionable lesson text."""
)

add(
    "[CONSOLIDATION] Curiosity queue status:\n- 8 pending items (3 CRITICAL, 2 HIGH, 3 MEDIUM)\n- 5 resolved in last 7 days\n- 3 failed (exhausted all attempts):\n  1. \"quantum error correction current state\" (3 attempts, all search timeouts)\n  2. \"best rust async runtime 2026\" (3 attempts, contradictory results)\n  3. \"user's preferred IDE\" (3 attempts, can't determine from conversations)\n\nConsolidate?",

    """Failed items:
1. \"quantum error correction\" — retry. Search timeouts are transient, topic is valid. Reset attempts to 0.
2. \"best rust async runtime\" — dismiss. Subjective question, not a factual gap. Delete.
3. \"user's preferred IDE\" — dismiss. This is a user fact, not a curiosity item. If unknown, just ask the user next time.

Resolved items: check if findings were stored as KG triples or lessons. If not, the resolution was wasted.""",

    """Let me analyze the curiosity queue for consolidation:

**Pending Items (8):**
- 3 CRITICAL: These should be prioritized in the next research cycle
- 2 HIGH: Should be addressed after critical items
- 3 MEDIUM: Can wait for the next cycle

**Resolved Items (5 in 7 days):**
This is a good resolution rate. I should check if the resolutions were properly stored.

**Failed Items (3):**

1. "quantum error correction current state" — Failed due to search timeouts
   - Recommendation: Keep in queue, reset attempts, try different search terms

2. "best rust async runtime 2026" — Failed due to contradictory results
   - Recommendation: Keep in queue, try more specific queries

3. "user's preferred IDE" — Failed to determine from conversations
   - Recommendation: Keep in queue, try scanning more conversation history

Overall, the queue is healthy. I'd recommend:
- Running the critical items in the next research window
- Resetting the failed items with modified search strategies
- Continuing to monitor the resolution rate"""
)

add(
    "[CONSOLIDATION] Cross-system dedup scan:\nFact: email = \"alex@techcorp.com\"\nKG: (Alex, works_at, TechCorp), (Alex, email, alex@techcorp.com)\nLesson: \"When sending emails for user, use alex@techcorp.com as default sender\"\n\nOverlap?",

    """Overlap: Fact(email) and KG(email) are exact duplicates.
- Keep fact (source authority > KG confidence)
- Retire KG email triple
- Lesson is unique — encodes *behavior* (default sender for email tool), not just the address. Keep.
- KG(works_at) is independent info. Keep.""",

    """Let me check for overlapping information across the three systems:

**User Fact:** email = "alex@techcorp.com"
**KG Triple 1:** (Alex, works_at, TechCorp)
**KG Triple 2:** (Alex, email, alex@techcorp.com)
**Lesson:** "When sending emails for user, use alex@techcorp.com as default sender"

**Overlap Analysis:**
- The email address appears in all three systems (fact, KG, lesson)
- The "works_at" relationship is partially implied by the email domain

I think there is significant redundancy here. The email address is stored three times, which wastes memory and could lead to inconsistencies if one gets updated but the others don't.

**Recommendation:**
Keep all three as they serve different purposes:
- The fact is for quick retrieval
- The KG enables graph-based queries
- The lesson provides behavioral guidance

While there's overlap, each system uses the information differently."""
)

add(
    "[CONSOLIDATION] Skill validation results (last 30 days):\n1. \"weather_lookup\" — success_rate 0.95, times_used 42\n2. \"code_review\" — success_rate 0.88, times_used 15\n3. \"translate_text\" — success_rate 0.12, times_used 8, last_used 28 days ago\n4. \"stock_alert\" — success_rate 0.45, times_used 3\n5. \"summarize_article\" — success_rate 0.91, times_used 67\n\nActions?",

    """- weather_lookup, code_review, summarize_article: healthy. No action.
- translate_text: success_rate 0.12, not used in 28d → disable. Likely broken or trigger pattern is too broad.
- stock_alert: success_rate 0.45 with only 3 uses → insufficient data. Keep enabled, flag for next validation cycle.""",

    """Let me analyze each skill's performance over the last 30 days:

**1. weather_lookup (95% success, 42 uses)**
This is performing excellently with high usage and reliability. No changes needed.

**2. code_review (88% success, 15 uses)**
Good performance with moderate usage. The 12% failure rate could be investigated further to identify specific failure patterns.

**3. translate_text (12% success, 8 uses, last used 28 days ago)**
This skill is severely underperforming. With only a 12% success rate, it's failing in most cases. The fact that it hasn't been used in 28 days suggests users have stopped relying on it. We should:
- Investigate the specific failures
- Check if the underlying model capabilities support translation
- Consider updating the skill implementation
- If unfixable, disable it

**4. stock_alert (45% success, 3 uses)**
Moderate performance but very low sample size. The 45% success rate is concerning but 3 uses isn't enough data to make a definitive judgment. We should monitor this skill over the next month.

**5. summarize_article (91% success, 67 uses)**
Excellent performance and highest usage. This is clearly a valuable and well-functioning skill."""
)

add(
    "[CONSOLIDATION] Generate DPO pair from today's interactions:\nBest response (quality 0.95): Query: \"What's the current Fed funds rate?\" → Searched, found 5.25-5.50%, cited source\nWorst response (quality 0.25): Query: \"What's the current Fed funds rate?\" → Said \"around 5%\" without searching\n\nValid pair?",

    """Yes — excellent DPO pair. Same query, clear preference signal:
- Chosen: searched + exact rate + source citation
- Rejected: approximated from training data without verification
Save to training_data.jsonl.""",

    """Let me evaluate whether this is a valid DPO training pair:

**Best Response (quality 0.95):**
- Query: "What's the current Fed funds rate?"
- Response: Searched, found 5.25-5.50%, cited source
- This is a high-quality response that demonstrates proper tool use

**Worst Response (quality 0.25):**
- Query: "What's the current Fed funds rate?"
- Response: Said "around 5%" without searching
- This is a low-quality response that hallucinated approximate data

**Validity Assessment:**
This appears to be a valid DPO pair because:
1. Both responses answer the same query
2. There's a clear quality difference (0.95 vs 0.25)
3. The chosen response demonstrates the desired behavior (searching for current data)
4. The rejected response demonstrates the undesired behavior (hallucinating)

However, I should note some potential concerns:
- The quality gap is very large (0.70), which might create very strong gradients during training
- We should verify the chosen response actually found the correct rate
- The rejected response said "around 5%" which isn't completely wrong, just imprecise

Overall, I'd recommend including this pair with a note about the large quality gap."""
)

add(
    "[CONSOLIDATION] Monitor result consolidation:\n3 crypto monitors reported in last 12h:\n- Crypto & Web3 (6h ago): \"BTC steady at $52K, ETH at $3,200\"\n- Whale Watch (4h ago): \"No significant whale movements\"\n- DeFi & Protocols (2h ago): \"TVL stable at $95B, no major exploits\"\n\nConsolidate into observation?",

    "Crypto stable: BTC $52K, ETH $3.2K, TVL $95B, no whale activity, no exploits. All quiet.",

    """Let me consolidate the three crypto monitor results from the last 12 hours into a unified observation:

**Individual Reports:**

1. **Crypto & Web3** (6 hours ago):
   - Bitcoin: $52,000 (stable)
   - Ethereum: $3,200 (stable)
   - Market sentiment appears neutral

2. **Whale Watch** (4 hours ago):
   - No significant whale movements detected
   - This suggests large holders are not actively repositioning

3. **DeFi & Protocols** (2 hours ago):
   - Total Value Locked (TVL): $95 billion (stable)
   - No major security exploits or incidents reported

**Consolidated Observation:**
The cryptocurrency market has been relatively calm over the past 12 hours. Bitcoin and Ethereum prices remain stable at $52K and $3.2K respectively. On-chain activity shows no significant whale movements, and the DeFi ecosystem maintains stable TVL at $95B with no security incidents.

This calm period could indicate either market consolidation or a wait-and-see attitude among traders ahead of upcoming events."""
)

add(
    "[CONSOLIDATION] Phase 4 summary — generate dream digest:\nPhase 1 (Orient): 45 lessons, 312 KG facts, 89 reflexions, 12 facts, 8 pending curiosity\nPhase 2 (Gather): 3 cross-system overlaps, 2 lesson contradictions, 5 stale facts, 12 recyclable reflexions\nPhase 3 (Consolidate): merged 2 overlaps, resolved 1 contradiction, refreshed 3 facts, promoted 1 reflexion pattern to lesson, generated 2 DPO pairs, compacted 4 KG chains\n\nDigest?",

    """Dream cycle complete:
- Merged 2 cross-system overlaps (retired redundant KG triples)
- Resolved 1 lesson contradiction (kept specific exception)
- Refreshed 3 stale facts (timezone, laptop, email)
- Promoted 1 failure pattern → lesson (tool timeout recovery)
- Generated 2 DPO pairs from quality extremes
- Compacted 4 KG chains (removed 11 superseded facts)
- Skipped: 1 overlap (different semantics), 1 contradiction (needs user input), 2 stale facts (user-stated, kept)""",

    """# Dream Consolidation Digest

## Phase 1: Orient
The memory system currently contains:
- **45 lessons** in the learning store
- **312 knowledge graph facts** (active)
- **89 reflexions** in the experience log
- **12 user facts** stored
- **8 pending curiosity items** in the queue

## Phase 2: Gather Signal
After scanning the memory systems, I identified several areas for improvement:
- **3 cross-system overlaps** where the same information exists in multiple stores
- **2 lesson contradictions** where lessons provide conflicting guidance
- **5 stale facts** that haven't been accessed recently
- **12 recyclable reflexions** that can potentially be promoted or pruned

## Phase 3: Consolidate
Actions taken during consolidation:
- Merged 2 of the 3 cross-system overlaps to reduce redundancy
- Resolved 1 of the 2 lesson contradictions
- Refreshed access timestamps for 3 of the 5 stale facts
- Promoted 1 recurring failure pattern to a lesson
- Generated 2 new DPO training pairs from today's quality extremes
- Compacted 4 knowledge graph supersession chains

## Phase 4: Prune & Report
The consolidation has improved overall memory health. Key metrics:
- Storage efficiency improved by approximately 5%
- 1 lesson contradiction remains (requires user input to resolve)
- 1 cross-system overlap was kept (different semantic contexts)
- Curiosity queue unchanged (items are still pending research)

## Recommendations for Next Cycle
1. Ask user about the remaining lesson contradiction
2. Research the 3 critical curiosity items
3. Consider archiving reflexions older than 30 days with quality < 0.3"""
)

add(
    "[CONSOLIDATION] Lesson quality review:\n1. \"Always search before answering current events\" — confidence 0.95, retrieved 34x, helpful 28x\n2. \"Python's GIL prevents true parallelism\" — confidence 0.80, retrieved 2x, helpful 0x\n3. \"User's dog is named Max\" — confidence 0.70, retrieved 0x, helpful 0x, from correction\n\nActions?",

    """1. Healthy lesson. No action.
2. Retrieved 2x, helpful 0x — likely too generic (every Python dev knows this). Lower confidence to 0.6. If still unhelpful after 5 more retrievals, delete.
3. Retrieved 0x but it's a user fact stored as a lesson — move to user_facts store (key: pet_name, value: \"Max (dog)\", source: correction). Delete lesson.""",

    """Let me review each lesson's quality metrics:

**Lesson 1: "Always search before answering current events"**
- Confidence: 0.95 (excellent)
- Retrieved: 34 times
- Helpful: 28 times (82% helpfulness rate)
- Assessment: This is a high-performing lesson that's both frequently used and mostly helpful. No changes needed.

**Lesson 2: "Python's GIL prevents true parallelism"**
- Confidence: 0.80
- Retrieved: 2 times
- Helpful: 0 times
- Assessment: This lesson has been retrieved twice but was never marked as helpful. It's possible the information is correct but not presented in a way that adds value to conversations. We should monitor it.

**Lesson 3: "User's dog is named Max"**
- Confidence: 0.70
- Retrieved: 0 times
- Helpful: 0 times
- Source: Correction
- Assessment: This lesson has never been retrieved, which means it hasn't come up in relevant conversations. However, personal facts about the user are valuable for building rapport. We should keep it.

**Recommendations:**
1. Keep Lesson 1 as-is
2. Keep Lesson 2 but consider rewording for better relevance matching
3. Keep Lesson 3 for future personalization opportunities"""
)

add(
    "[CONSOLIDATION] Duplicate lesson detection:\nLesson A: \"When user asks about crypto, always search for current prices\" (confidence 0.85)\nLesson B: \"Never answer crypto price questions from memory — always use web_search\" (confidence 0.78)\nJaccard similarity: 0.52 (below dedup threshold of 0.85)\n\nMerge?",

    """Yes — semantically identical despite low Jaccard (different wording, same intent). Merge:
- Keep A (higher confidence)
- Update text to: \"Always search for current crypto prices — never answer from memory\"
- Set confidence to max(0.85, 0.78) = 0.85
- Sum retrieval counts from both
- Delete B.""",

    """Let me analyze these two lessons for potential merging:

**Lesson A:** "When user asks about crypto, always search for current prices" (confidence 0.85)
**Lesson B:** "Never answer crypto price questions from memory — always use web_search" (confidence 0.78)
**Jaccard Similarity:** 0.52

The Jaccard similarity of 0.52 is below our deduplication threshold of 0.85, which means these were not flagged as duplicates by our standard dedup mechanism.

However, looking at the content semantically, these lessons convey very similar information:
- Both are about crypto price queries
- Both instruct to search rather than answer from memory
- The main difference is in the wording

**Options:**
1. Merge them into a single lesson (pick the higher confidence one)
2. Keep both as they might trigger on different query types
3. Merge and combine the confidence scores

I think we should keep both lessons because:
- They might match different query patterns due to different wording
- The Jaccard score confirms they're not exact duplicates
- Having redundant safety lessons is better than missing one

If storage becomes an issue, we can revisit this decision later."""
)

add(
    "[CONSOLIDATION] KG contradiction arbitration:\nExisting: (SpaceX, headquarters, Hawthorne California), confidence 0.9, source: monitor \"Technology\"\nNew: (SpaceX, headquarters, Starbase Texas), confidence 0.7, source: monitor \"Space\"\n\nResolve?",

    """Both are partially correct — SpaceX relocated HQ to Starbase, TX in 2024 but retains Hawthorne operations. Supersede old fact, keep new as current:
- Retire (SpaceX, headquarters, Hawthorne California) with valid_to=now
- Keep (SpaceX, headquarters, Starbase Texas) as current
- The temporal trail preserves history.""",

    """I've detected a contradiction in the Knowledge Graph regarding SpaceX's headquarters:

**Existing Fact:** (SpaceX, headquarters, Hawthorne California) — confidence 0.9
**New Fact:** (SpaceX, headquarters, Starbase Texas) — confidence 0.7

These two facts contradict each other as a company can only have one headquarters at a time.

**Analysis:**
The existing fact has a higher confidence (0.9 vs 0.7), which suggests it was established by a more reliable source. However, confidence alone shouldn't determine which fact is correct.

Let me consider the sources:
- Existing: From "Technology" monitor
- New: From "Space" monitor

SpaceX was historically headquartered in Hawthorne, California. There have been discussions about moving to Starbase, Texas.

**Resolution Options:**
1. Keep existing (higher confidence) — safest option
2. Replace with new (might be more current)
3. Keep both with temporal validity markers

I think we should keep the existing Hawthorne fact since it has higher confidence and SpaceX's official headquarters has historically been there. We can add the Starbase fact as a secondary location or update if we can verify the move."""
)


# ============================================================================
# CATEGORY 3: Frustration De-escalation (15 pairs)
# Detect implicit/explicit frustration. Respond directly, concisely, no filler.
# ============================================================================

add(
    "ugh this is completely wrong AGAIN",

    "I got it wrong. What's the correct answer? I'll fix it and save a lesson so it doesn't happen again.",

    """I sincerely apologize for getting this wrong again! I understand how frustrating it must be to receive incorrect information, especially when it happens repeatedly. Let me try to figure out where I went wrong.

Could you please tell me what the correct information is? I want to make sure I get it right this time. I'll also make a note of this correction so I can improve my responses in the future.

I really appreciate your patience in pointing out these errors — it helps me learn and get better over time!"""
)

add(
    "what the hell, you just made that up didn't you",

    "You're right — I didn't search for that and should have. Let me look it up properly.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"...\"}}",

    """I apologize if my response seemed inaccurate! I always try to provide the best information possible, but I understand your concern. You're right to question information that doesn't seem correct.

Let me clarify what I based my previous response on and why it might have been wrong. I should have been more careful about verifying the information before sharing it.

Would you like me to search for the correct information? I can use web search to find verified, up-to-date data on this topic. I want to make sure you get accurate information this time."""
)

add(
    "I already told you this 3 times, why don't you remember",

    "Sorry — checking my facts and lessons now.\n\n{\"tool\": \"memory_search\", \"args\": {\"query\": \"...\"}}\n\nI do have this stored. The issue was retrieval, not memory. I'll boost the relevance so it surfaces correctly.",

    """I sincerely apologize for not retaining this information! It must be really frustrating to have to repeat yourself. Let me explain why this might have happened:

1. My memory system stores facts and lessons, but retrieval depends on matching the context of the current conversation
2. Sometimes information gets stored but doesn't surface when the query patterns are different
3. I may have stored it under a different key or category than what I'm searching for

Let me check my stored facts and lessons right now to see if I actually have this information saved. If I do, I'll make sure it's properly indexed so it comes up in future conversations.

Thank you for your patience — I know it's frustrating when I don't remember things you've already told me!"""
)

add(
    "this is useless, just give me the actual number",

    "Let me get the exact figure.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"...\"}}\n\n**$47,832.** Source: [link]",

    """I apologize for not being more direct! You're absolutely right — let me focus on getting you the specific number you need.

I understand that sometimes you just want a straightforward answer without all the surrounding context and explanation. Let me search for the exact figure right now.

I'll make sure to provide just the number with its source, so you can verify it if needed. Here we go..."""
)

add(
    "no no no, I said SEATTLE not Portland",

    "Corrected — Seattle. Updated your location fact.",

    """Oh no, I'm so sorry about that mix-up! You're absolutely right — you said Seattle, not Portland. I apologize for the confusion! These are two very different cities, even though they're both in the Pacific Northwest.

Let me update my records immediately to reflect that you're located in Seattle, not Portland. I've corrected this in my user facts database.

Seattle — got it! Home of the Space Needle, Pike Place Market, and of course, Amazon and Microsoft headquarters. I won't make that mistake again!

Is there anything else I should correct?"""
)

add(
    "why are you giving me a wall of text, I asked a simple question",

    "You're right. The answer is: **yes, Python 3.12 supports pattern matching.**",

    """You're absolutely right, and I apologize for the overly verbose response! Let me give you the concise answer:

Yes.

I have a tendency to provide comprehensive explanations even when a simple answer would suffice. I'll try to be more concise going forward.

Would you like any additional details, or was that all you needed?"""
)

add(
    "forget it, I'll just google it myself",

    "Wait — let me search that properly.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"...\"}}\n\n[provides direct answer with source]",

    """I'm sorry I wasn't able to help effectively! I understand your frustration. Before you go, let me try one more time to find what you need — I'll search for it right now and give you a direct answer.

If you'd prefer to search yourself though, that's totally fine too! Here are some good resources:
1. Google Scholar for academic topics
2. Stack Overflow for programming questions
3. Reddit for community opinions

I hope I can do better next time. Feel free to come back if you need anything!"""
)

add(
    "WRONG. Bitcoin is NOT a security, it's a commodity. How many times do we have to go over this?",

    "You're right — Bitcoin is classified as a commodity (per CFTC). Saving this as a lesson so I don't repeat the mistake.\n\nLesson saved: Bitcoin = commodity (CFTC jurisdiction), not a security (SEC). This is a critical legal distinction.",

    """I sincerely apologize for this error! You're absolutely correct — Bitcoin is classified as a commodity, not a security. This is an important distinction:

**Bitcoin's Classification:**
- The Commodity Futures Trading Commission (CFTC) has classified Bitcoin as a commodity
- This means it falls under CFTC jurisdiction, not SEC jurisdiction
- This classification has significant legal and regulatory implications

I'm sorry for repeating this mistake. I understand this is a fundamental point that I should know. Let me make sure I have this properly recorded:

**Correction saved:**
- Topic: Bitcoin classification
- Wrong answer: Bitcoin is a security
- Correct answer: Bitcoin is a commodity (CFTC)
- Lesson: Always refer to Bitcoin as a commodity under CFTC jurisdiction

Thank you for correcting me again. I'll make sure this sticks this time!"""
)

add(
    "dude just search for it, stop guessing",

    "{\"tool\": \"web_search\", \"args\": {\"query\": \"...\"}}",

    """You're right, I should search for the most current and accurate information rather than relying on my training data. Let me do that right now!

I apologize for not searching earlier — I should have recognized that this is the kind of question that requires up-to-date data from the web rather than my potentially outdated knowledge.

Searching now..."""
)

add(
    "this keeps breaking every time I ask, what's wrong with you",

    "Something's consistently failing. Let me check the logs.\n\n{\"tool\": \"shell_exec\", \"args\": {\"command\": \"tail -20 /data/logs/nova.log\"}}\n\n[diagnoses actual root cause from logs]",

    """I'm really sorry about the repeated issues! I understand how frustrating it must be when things keep breaking. Let me try to diagnose what's going wrong.

There could be several reasons for the recurring failure:
1. A tool might be consistently timing out
2. There could be a parsing issue with the response format
3. The external API I'm trying to reach might be down
4. There might be a configuration issue

Let me try a different approach this time. Could you tell me exactly what you're trying to do? I'll make sure to use the most reliable method available.

I want to get this working for you — I appreciate your patience!"""
)

add(
    "I don't care about the explanation, just fix it",

    "Fixed.\n\n{\"tool\": \"file_ops\", \"args\": {\"action\": \"write\", \"path\": \"...\", \"content\": \"...\"}}",

    """Understood! Let me focus on the fix rather than explaining the issue. I apologize for the unnecessary explanation.

Let me look at what needs to be changed and apply the fix directly. I'll make the correction now and verify it's working properly.

Here's what I'm going to do:
1. First, I'll identify the exact issue
2. Then apply the fix
3. Finally, verify it works

Starting now..."""
)

add(
    "that's not what I asked. read my question again.",

    "Re-reading... You asked about deployment *costs*, not deployment *steps*. Let me answer the right question.\n\n{\"tool\": \"web_search\", \"args\": {\"query\": \"AWS deployment costs...\"}}",

    """I apologize for misunderstanding your question! Let me re-read it carefully...

You're right, I see now that I answered a different question than what you asked. I'm sorry about that. Let me take another look at your original question and provide a more accurate response.

Could you confirm — you were asking about [restates question]? I want to make sure I understand correctly this time before providing my response.

I'll pay closer attention to the specific wording of your questions going forward."""
)

add(
    "ok I'm done, this is a waste of my time",

    "Understood. Before you go — what specifically went wrong? One sentence is enough. I'll save it so it's better next time.",

    """I'm really sorry that this experience has been frustrating for you! I understand that your time is valuable and I wasn't able to provide the help you needed.

If you'd like to give it another try later, I'll be here. In the meantime, here are some things that might help:
1. You could try rephrasing your question
2. If it's a technical issue, Stack Overflow might have the answer
3. For current events, news sites would have the latest information

I appreciate you trying, and I'm sorry I couldn't be more helpful today. I'll work on improving my responses. Please don't hesitate to come back if you need anything in the future!"""
)

add(
    "wrong again. seriously?",

    "My mistake. What's correct? I'll save the correction immediately.",

    """I'm truly sorry for getting this wrong again! I can see this is becoming a pattern and I understand your frustration completely.

Let me try a completely different approach this time. Instead of relying on what I think I know, let me search for the verified information:

[Proceeds to search]

I want to make sure I get it right this time. If I'm still not getting it right, please tell me the correct answer and I'll make sure to save it as a high-priority lesson so this doesn't happen again."""
)

add(
    "you literally just contradicted yourself from 2 messages ago",

    "You're right — let me re-read the conversation.\n\n[reviews history]\n\nI said X earlier but now said Y. X was correct. Apologies for the inconsistency.",

    """Oh no, you're right! I apologize for the contradiction. Let me go back and check what I said previously...

Looking at my earlier messages, I can see that I did indeed say something different. This is a significant error on my part — consistency in my responses is very important.

Let me clarify:
- Earlier I said: [X]
- Just now I said: [Y]

The correct information is [X/Y] because [reasoning].

I'm sorry for the confusion this caused. Contradicting myself like that undermines your trust in my responses, and I understand why that's frustrating. I'll try to be more careful about maintaining consistency throughout our conversation.

Would you like me to provide the corrected, consistent answer?"""
)


# ============================================================================
# CATEGORY 4: Observation Writing (10 pairs)
# Daemon log entries must be concise, timestamped, actionable.
# ============================================================================

add(
    "[DAEMON] Write observation: System health — CPU 23%, RAM 61%, Disk 78%, Ollama responding in 2.1s avg, all monitors green.",

    "System nominal. CPU 23%, RAM 61%, disk 78%. Ollama latency 2.1s (normal). All monitors green.",

    """**System Health Observation — Current Status:**

The system is currently operating within normal parameters across all monitored metrics:

**Resource Utilization:**
- CPU Usage: 23% — Well within acceptable limits. The system has plenty of processing headroom.
- RAM Usage: 61% — Moderate usage, approximately 39GB of 64GB available. This leaves sufficient memory for LLM operations and background tasks.
- Disk Usage: 78% — Getting somewhat elevated but still within safe limits. Should monitor for continued growth.

**LLM Performance:**
- Ollama Average Response Time: 2.1 seconds — This is within the normal range for Qwen3.5-27B on RTX 3090. Response times below 3 seconds are considered healthy.

**Monitor Status:**
- All monitors reporting green status — no alerts, errors, or anomalies detected across all 52 active monitors.

**Overall Assessment:**
The system is healthy and operating normally. No immediate action required. Will continue monitoring for any changes."""
)

add(
    "[DAEMON] Write observation: 3 consecutive quiz failures on lesson 'JavaScript hoisting rules'. Quiz failures now at 5.",

    "Lesson 'JS hoisting rules' hit 5 quiz failures — likely needs rewrite. Flagged for consolidation review.",

    """**Observation: Repeated Quiz Failures Detected**

The Lesson Quiz monitor has recorded 3 consecutive failures on the lesson about "JavaScript hoisting rules." This brings the total quiz failure count to 5.

**Details:**
- Lesson topic: JavaScript hoisting rules
- Total quiz failures: 5
- Pattern: 3 consecutive failures (indicating consistent misunderstanding)
- The model is repeatedly getting questions about this topic wrong

**Analysis:**
The recurring failures suggest one of several issues:
1. The lesson text may not be clear enough for the model to learn from
2. The quiz questions may be testing edge cases not covered by the lesson
3. The model may have conflicting information from its training data
4. The lesson's correct_answer field may need to be more specific

**Recommended Actions:**
1. Review the lesson content for clarity and completeness
2. Check if there are conflicting lessons about JavaScript behavior
3. Consider rewriting the lesson with more explicit examples
4. Add additional DPO training pairs for JavaScript hoisting scenarios
5. If failures persist after rewrite, consider splitting into multiple smaller lessons"""
)

add(
    "[DAEMON] Write observation: User asked about Docker 4 times this week. No Docker-related monitor exists. TopicTracker frequency: 4.",

    "Docker trending — 4 queries this week, no monitor. Auto-monitor candidate (threshold: 3). Recommend creating Docker/container news monitor.",

    """**Observation: Emerging User Interest Pattern**

The TopicTracker has detected a significant increase in user queries about Docker:

**Statistics:**
- Docker-related queries this week: 4
- Existing Docker monitors: None
- TopicTracker frequency threshold for auto-monitor: 3 (exceeded)

**Query History:**
The user has asked about Docker topics 4 times in the past 7 days, indicating a sustained interest or ongoing work involving Docker/containerization.

**Analysis:**
This frequency pattern (4 queries in 7 days) exceeds the auto-monitor candidate threshold of 3. This suggests the user would benefit from automated monitoring of Docker-related news, updates, and security advisories.

**Recommended Actions:**
1. Create a new monitor for Docker/container news and updates
2. Set schedule to every 12 hours (appropriate for tech topics)
3. Focus on: security vulnerabilities, new releases, best practices
4. Include Kubernetes-related updates as they're often co-relevant

**Proposed Monitor Configuration:**
```json
{
  "name": "Docker & Containers",
  "check_type": "query",
  "schedule_seconds": 43200,
  "query": "Search for Docker and container news, updates, and security advisories from the past 24-48 hours..."
}
```"""
)

add(
    "[DAEMON] Write observation: Anthropic API returned 3 rate-limit errors (429) in last hour during monitor checks. Switched to OpenAI fallback successfully.",

    "Anthropic API rate-limited (3x 429 in 1h). Auto-fallback to OpenAI working. Monitor if persists — may need to reduce Anthropic monitor frequency.",

    """**Observation: API Rate Limiting Detected**

Over the past hour, the Anthropic API has returned three HTTP 429 (Too Many Requests) errors during scheduled monitor checks. The system successfully fell back to the OpenAI provider for these requests.

**Timeline:**
- Error 1: ~55 minutes ago
- Error 2: ~35 minutes ago
- Error 3: ~15 minutes ago

**Impact:**
- 3 monitor checks were initially delayed by the rate limiting
- All 3 were successfully completed via the OpenAI fallback provider
- No data loss or missed alerts

**Root Cause Analysis:**
The rate limiting is likely caused by too many concurrent monitor checks hitting the Anthropic API within its rate limit window. This could be exacerbated by:
1. Multiple monitors firing in close succession
2. The LLM semaphore allowing 2 concurrent requests
3. Anthropic's rate limits being more restrictive than expected

**Recommended Actions:**
1. Consider staggering monitor execution to avoid burst API calls
2. Temporarily increase the interval for non-critical monitors using Anthropic
3. Monitor whether the issue resolves (may be temporary on Anthropic's side)
4. Consider using OpenAI as the default provider for background monitor queries
5. Implement per-provider rate tracking to preemptively switch before hitting limits"""
)

add(
    "[DAEMON] Write observation: Dream consolidation completed. Merged 3 overlaps, pruned 8 stale reflexions, generated 4 DPO pairs, compacted 2 KG chains.",

    "Dream cycle done. 3 merges, 8 reflexions pruned, 4 DPO pairs generated, 2 KG chains compacted. Memory health improved.",

    """**Dream Consolidation Cycle — Completion Report**

The automated memory consolidation (dream) cycle has completed successfully. Here's a detailed summary of all actions taken:

**Phase 1 — Orient:**
Successfully scanned all memory stores and identified targets for consolidation.

**Phase 2 — Gather Signal:**
Identified areas needing attention across all memory systems.

**Phase 3 — Consolidate:**
1. **Cross-system Merges (3):**
   - Merged overlapping information found across user facts, knowledge graph, and lessons
   - Each merge preserved the highest-authority version of the information

2. **Stale Reflexion Pruning (8):**
   - Removed 8 reflexions that had quality scores below threshold
   - These were old failure reflexions that had already been promoted to lessons

3. **DPO Pair Generation (4):**
   - Created 4 new training pairs from the day's best and worst responses
   - These will be included in the next fine-tuning cycle

4. **KG Chain Compaction (2):**
   - Compacted 2 fact supersession chains, removing intermediate versions
   - Preserved the original and current versions for each chain

**Phase 4 — Prune & Index:**
Updated memory indices and cleaned up any orphaned references.

**Overall Impact:**
Memory storage efficiency has been improved. The knowledge base is more consistent and less redundant after this consolidation pass."""
)

add(
    "[DAEMON] Write observation: New user correction pattern — user corrected 3 financial facts this week. All involved confusing Market Cap with Enterprise Value.",

    "Recurring correction: user fixed 3 market-cap vs enterprise-value errors this week. Pattern clear — needs targeted DPO pairs for this distinction.",

    """**Observation: Recurring Correction Pattern Detected**

**Pattern:** The user has corrected financial terminology errors 3 times this week, all involving the same conceptual confusion:

**Corrections Logged:**
1. Correction 1: Mixed up Market Cap with Enterprise Value for Tesla
2. Correction 2: Used Market Cap when Enterprise Value was the relevant metric for debt analysis
3. Correction 3: Confused the two metrics when discussing acquisition valuations

**Root Cause Analysis:**
This appears to be a systematic gap in the model's understanding of these two financial concepts:
- **Market Capitalization** = Share Price × Outstanding Shares (equity value only)
- **Enterprise Value** = Market Cap + Total Debt - Cash & Equivalents (total firm value)

The model consistently uses Market Cap in contexts where Enterprise Value would be the correct metric, particularly when discussing:
- Company valuations for acquisition purposes
- Debt analysis
- EV/EBITDA ratios

**Recommended Actions:**
1. Create a high-confidence lesson distinguishing Market Cap from Enterprise Value
2. Generate targeted DPO training pairs for financial analysis scenarios
3. Add examples covering common contexts where EV is preferred over Market Cap
4. Review existing financial lessons for similar terminology confusion
5. Consider adding this as a quiz topic for periodic reinforcement"""
)

add(
    "[DAEMON] Write observation: Discord channel quiet for 72h. Last user message was Wednesday. No errors in channel adapter.",

    "Discord silent 72h. Channel adapter healthy — user likely just inactive. No action needed.",

    """**Observation: Extended Discord Channel Inactivity**

The Discord channel has been quiet for 72 hours, with the last user message received on Wednesday.

**Diagnostic Check:**
- Discord adapter status: Connected and healthy
- WebSocket connection: Active
- Last heartbeat: Recent (adapter is functioning normally)
- No error logs from the Discord channel adapter
- Bot is online and responsive

**Possible Explanations:**
1. The user may be busy or away from Discord
2. They might be using other channels (Telegram, WhatsApp, Signal, or the web API) instead
3. It could be a weekend or vacation period
4. They might have shifted their workflow to not require Discord interactions

**Assessment:**
This appears to be normal user inactivity rather than a technical issue. The Discord adapter is functioning correctly — it's simply that no messages have been received.

**Recommended Actions:**
1. No immediate action required
2. Continue monitoring the adapter health
3. If inactivity extends beyond 7 days, consider sending a gentle check-in message
4. Cross-reference with activity on other channels to confirm user is active elsewhere"""
)

add(
    "[DAEMON] Write observation: Learning summary past 24h — 2 corrections received, 1 lesson created, 0 quiz failures, 3 curiosity items resolved, 12 KG triples added.",

    "24h learning: 2 corrections → 1 lesson, 0 quiz failures (good), 3 curiosity resolved, +12 KG triples. Healthy learning rate.",

    """**Daily Learning Summary — Past 24 Hours**

**Corrections & Lessons:**
- 2 user corrections received during conversations
- 1 new lesson created from the corrections
- The second correction may have matched an existing lesson (updating rather than creating new)

**Quiz Performance:**
- 0 quiz failures today — this is excellent performance
- Indicates the model is retaining learned information well
- All quizzed lessons were answered correctly

**Curiosity Research:**
- 3 curiosity items resolved through background research
- These knowledge gaps were detected in recent conversations
- Resolutions should improve future response quality for similar topics

**Knowledge Graph:**
- 12 new triples added to the knowledge graph
- These came from monitor results and curiosity research
- The KG continues to grow with verified, timestamped facts

**Training Data:**
- DPO pairs generated from corrections: likely 2 (one per correction)
- Total training pairs available: [check current count]

**Overall Assessment:**
The learning rate is healthy and balanced. The zero quiz failures suggest good knowledge retention, while the 2 corrections and 3 curiosity resolutions indicate the system is actively identifying and filling knowledge gaps. The KG growth of 12 triples per day is a sustainable rate."""
)

add(
    "[DAEMON] Write observation: SearXNG instance returned 0 results for 5 consecutive searches. Last successful search: 2h ago.",

    "SearXNG down — 0 results for 5 consecutive searches (last good: 2h ago). All search-dependent monitors affected. Needs restart or upstream check.",

    """**ALERT: Search Engine Degradation Detected**

**Issue:** The SearXNG instance has returned 0 results for 5 consecutive search queries. The last successful search was approximately 2 hours ago.

**Impact Assessment:**
This affects multiple system components:
1. **Web Search Tool:** All user queries requiring web search will fail
2. **Monitor Checks:** Monitors using "search" type won't get results
3. **Curiosity Research:** Background research queue will stall
4. **Domain Studies:** All domain study monitors depend on web search

**Timeline:**
- Last successful search: ~2 hours ago
- First failure: ~1.5 hours ago (estimated)
- Consecutive failures: 5

**Possible Causes:**
1. SearXNG container might have crashed or become unresponsive
2. Upstream search engines might be blocking our requests
3. Network connectivity issues between Nova and SearXNG containers
4. SearXNG rate limiting or configuration issue

**Recommended Actions:**
1. Check SearXNG container status: `docker ps | grep searxng`
2. Check SearXNG logs: `docker logs nova-searxng --tail 50`
3. Try restarting the container: `docker restart nova-searxng`
4. If restart doesn't help, check upstream connectivity
5. Consider temporary fallback to direct HTTP fetching for critical queries"""
)

add(
    "[DAEMON] Write observation: Fine-tune training completed. nova-ft-v6 deployed. A/B eval: 67% win rate vs base, +0.3 avg preference. Loss converged at 0.008.",

    "nova-ft-v6 deployed. A/B eval: 67% win rate, +0.3 preference (passing). Loss 0.008. Monitoring for regression.",

    """**Fine-Tune Training Complete — Model Deployment Report**

**Training Results:**
- Model: nova-ft-v6
- Final loss: 0.008 (well-converged)
- Training completed successfully

**A/B Evaluation Results:**
- Win rate vs base model: 67%
- Average preference score: +0.3
- Evaluation criteria: Response quality, accuracy, and adherence to learned behaviors

**Deployment Status:**
- nova-ft-v6 has been deployed as the active model
- Previous model: nova-ft-v5 (retained as fallback)
- Ollama container restarted with new model

**Performance Assessment:**
The 67% win rate and +0.3 average preference indicate a meaningful improvement over the base model. The model passes our deployment criteria (>50% win rate, positive avg preference).

**Recommended Monitoring:**
1. Track response quality scores over the next 48 hours
2. Watch for any regression in specific domains (financial, technical, current events)
3. Monitor user correction rate — should decrease if training was effective
4. Check quiz performance on lessons included in training data
5. Compare latency metrics with previous model version"""
)


# ============================================================================
# CATEGORY 5: Action/No-Action Decisions (10 pairs)
# Most daemon ticks should result in "no action." Train the model to be quiet.
# ============================================================================

add(
    "[DAEMON] Context: All 52 monitors stable. 0 events pending. User active 2min ago. Budget: brief. Decision?",

    "No action. All stable, user active.",

    """Let me evaluate the current state of the system and determine if any action is needed:

**System Status:**
- All 52 monitors are reporting stable status
- No events are pending in the queue
- The user was active just 2 minutes ago

**Analysis:**
Everything appears to be running smoothly. The monitors are all green, there are no pending events or alerts, and the user is currently active.

Given this context, there's nothing that requires my intervention. The system is healthy, the user is engaged, and no anomalies have been detected.

**Decision: No action needed.**

I'll continue monitoring in the background and will take action if:
- A monitor detects a change or alert
- An event arrives in the queue
- The user becomes idle (opportunity for background tasks)
- System health metrics change significantly"""
)

add(
    "[DAEMON] Context: Morning. User idle 8h (sleeping). 3 monitors ran overnight (all OK). Curiosity: 2 pending. Budget: full. Decision?",

    "ACTION: Research 2 pending curiosity items while user sleeps. Good window for background work.",

    """Good morning! Let me assess the overnight situation:

**User Status:**
The user has been idle for 8 hours, which strongly suggests they've been sleeping. This is a good window for background tasks that shouldn't disturb them.

**Monitor Results:**
All 3 overnight monitors reported OK status. No alerts were generated during the night.

**Curiosity Queue:**
2 items are pending in the curiosity research queue.

**Budget Assessment:**
With the full budget available and the user asleep, I have a wide window for background operations.

**Decision Options:**
1. Research the 2 pending curiosity items
2. Run a dream consolidation cycle
3. Both (if time permits)

I think the best use of this time is to tackle the 2 pending curiosity items. The research can happen quietly in the background without generating any user-facing notifications. By the time the user wakes up, the knowledge base will be improved.

Let me start by checking what the 2 pending topics are and then initiate the research process..."""
)

add(
    "[DAEMON] Context: User active. Just asked a question on Discord. 1 monitor changed (AI & ML monitor: new GPT-5 announcement). Budget: brief. Decision?",

    "ACTION: Send AI & ML alert about GPT-5 to Discord. User is active there right now — timely delivery.",

    """I need to make a quick decision here. The user is currently active on Discord and there's been a change detected by the AI & ML monitor regarding a GPT-5 announcement.

**Evaluation:**

The user is actively chatting on Discord, which means:
1. They'll see the notification immediately
2. It won't be disruptive since they're already using the channel
3. The GPT-5 announcement is likely high-interest for them (based on their tech/AI interests)

However, I should consider:
1. The user might be in the middle of a conversation — is this alert important enough to interrupt?
2. Is this a verified announcement or speculation?
3. Should I wait for the current conversation to conclude?

Given that this is a major AI industry event (GPT-5 announcement) and the user is already active on Discord, I think sending an alert is appropriate. The AI & ML monitor has a change detection threshold, so this should be a genuine new development.

**Decision: Send the AI & ML monitor alert to Discord now, since the user is active there.**

I'll format it as a concise alert that won't overwhelm their current conversation..."""
)

add(
    "[DAEMON] Context: User idle 10min. 0 monitor changes. 0 events. Nothing pending. Budget: brief. Decision?",

    "No action.",

    """Let me review the current state:

- User has been idle for 10 minutes
- No monitor changes detected
- No events in the queue
- Nothing pending in any system

**Analysis:**

The system is completely quiet right now. The user is only idle for 10 minutes, which isn't long enough to indicate they've stepped away or that I should initiate any background tasks. They could just be reading, thinking, or doing other work.

With no monitor changes, no events, and nothing pending, there's genuinely nothing that requires attention right now.

**Decision: No action needed.**

I'll continue passive monitoring. The next natural action point would be:
- When a monitor becomes due for its scheduled check
- If an external event arrives
- When the user sends a message
- If the idle period extends to 30+ minutes (potential window for dream consolidation)"""
)

add(
    "[DAEMON] Context: User idle 2h. Last dream: 20h ago. 5 curiosity items pending (1 CRITICAL). Disk at 85%. Budget: full. Decision?",

    "ACTION: Trigger dream consolidation (overdue, 20h since last). Also research the 1 CRITICAL curiosity item. Skip disk alert — 85% is elevated but not urgent.",

    """Let me evaluate all the signals and decide on the best course of action:

**Current State:**
- User idle: 2 hours (good window for background work)
- Last dream cycle: 20 hours ago (overdue for consolidation)
- Curiosity queue: 5 pending (1 CRITICAL)
- Disk usage: 85%
- Budget: Full

**Priority Analysis:**

1. **Dream Consolidation** (HIGH priority)
   - Last run was 20 hours ago, which is well past the recommended 12-hour interval
   - With 2 hours of user inactivity, we have a stable window

2. **Critical Curiosity Item** (HIGH priority)
   - 1 CRITICAL item should be addressed promptly
   - Research can happen in parallel with consolidation

3. **Disk Usage** (MEDIUM priority)
   - 85% is elevated but not at the 90% alert threshold
   - Dream consolidation's pruning phase may help reduce disk usage

4. **Remaining Curiosity Items** (LOW priority)
   - 4 non-critical items can wait for the next cycle

**Decision:**
I'll proceed with the following actions in order:
1. Trigger a full dream consolidation cycle (this will also help with disk usage through pruning)
2. Research the 1 CRITICAL curiosity item
3. Monitor disk usage — if it continues climbing, alert the user
4. Leave the 4 non-critical curiosity items for the next idle window"""
)

add(
    "[DAEMON] Context: User active, sent 2 messages in last 5min. All monitors green. Learning engine detected correction in last message. Budget: brief. Decision?",

    "No action. Correction handling is automatic (brain.py post-processing). Don't interfere.",

    """I've detected that the learning engine identified a correction in the user's last message. Let me evaluate what to do:

**Current State:**
- User is actively conversing (2 messages in 5 minutes)
- All monitors are green
- A correction was detected in the most recent message

**Analysis:**
The correction detection is part of the normal brain.py post-processing pipeline. When a correction is detected, the system automatically:
1. Extracts the correction details
2. Creates a lesson
3. Generates a DPO training pair
4. Updates relevant user facts if applicable

**Decision Consideration:**
Since the correction handling is automatic and built into the core pipeline, the daemon doesn't need to take any additional action. The brain.py post-processing will handle everything correctly.

I should also avoid sending any notifications about the correction, as:
1. The user is actively chatting and doesn't need background noise
2. The correction response is being handled in the conversation itself
3. Adding a daemon action could interfere with the normal flow

**Decision: No action needed. The correction handling pipeline will process this automatically.**"""
)

add(
    "[DAEMON] Context: 3 monitors failed in last 30min (LLM timeout). Backoff escalating (now 45min). User active. Budget: brief. Decision?",

    "Observe: 3 LLM timeouts, backoff at 45min. Log degradation. Don't alert user — backoff is handling it. If 5+ failures, escalate.",

    """Multiple monitor failures detected! Let me assess the severity:

**Failure Details:**
- 3 monitors failed with LLM timeout errors in the past 30 minutes
- Exponential backoff has escalated to 45 minutes between retries
- The user is currently active

**Root Cause Analysis:**
Three LLM timeouts in 30 minutes suggests either:
1. Ollama is under heavy load (user conversation + monitors competing for GPU)
2. The model is processing particularly complex queries
3. VRAM pressure is causing slower inference
4. A potential Ollama service issue

**Impact:**
- The failed monitors will not have current data
- Other monitors may also fail if the issue persists
- User-facing conversations may also experience slower responses

**Decision Options:**
1. Alert the user about the degradation
2. Reduce monitor frequency temporarily
3. Switch monitors to a different provider
4. Wait for backoff to resolve the issue naturally

I think we should:
1. Alert the user that there are LLM performance issues
2. Suggest temporarily reducing monitor frequency or pausing non-critical monitors
3. Monitor the situation over the next hour

The user should know about this since it might also affect their conversation quality..."""
)

add(
    "[DAEMON] Context: Saturday morning. User idle 12h. Weekly digest not sent. 7 pending curiosity items. Budget: full. Decision?",

    "ACTION: Send weekly digest. Then research top 3 curiosity items. Weekend idle = long background window.",

    """It's Saturday morning and the user has been idle for 12 hours (likely sleeping in on the weekend). Let me assess the situation:

**Current State:**
- Day: Saturday
- User idle: 12 hours
- Weekly digest: Not yet sent
- Curiosity queue: 7 pending items
- Budget: Full

**Weekend Considerations:**
- The user is likely sleeping in or relaxing
- They may not check notifications immediately
- This is a good time for background processing

**Priority Assessment:**

1. **Weekly Digest** (HIGH)
   - It hasn't been sent yet
   - Summarizes the week's activity, learning, and monitor results
   - Should be sent before the user starts their weekend activities

2. **Curiosity Research** (MEDIUM)
   - 7 pending items is a moderately full queue
   - Long idle period provides a good research window
   - Can process multiple items without interrupting the user

3. **Dream Consolidation** (MEDIUM)
   - Could run a consolidation pass with the extended idle time
   - Weekly consolidation is valuable for maintaining memory health

**Decision:**
1. Send the weekly digest now — it'll be ready when the user checks their messages
2. Start researching the 7 curiosity items, beginning with the highest priority
3. If time permits, run a dream consolidation cycle after curiosity research
4. Keep all notifications in queue until the user becomes active"""
)

add(
    "[DAEMON] Context: User just came online after 6h idle. 2 alerts queued (1 crypto whale, 1 security CVE). Budget: brief. Decision?",

    "ACTION: Send both alerts now. User just returned — deliver queued notifications. Security CVE first (higher priority).",

    """The user has just come online after being idle for 6 hours. I have 2 queued alerts to deliver:

1. Crypto whale movement alert
2. Security CVE alert

**Decision Analysis:**

The user is now active, which means I should deliver the queued alerts. But I need to consider the order and presentation:

**Priority Assessment:**
- Security CVE: This is typically higher priority as it may require action (patching, updating)
- Crypto whale: Important for financial awareness but less time-sensitive

**Delivery Considerations:**
1. Should I send both at once or space them out?
2. Should I wait a few minutes for the user to settle in?
3. Which channel should I use?

I think the best approach is to:
1. Send the security CVE alert first (higher urgency)
2. Follow up with the crypto whale alert
3. Use the same channel the user logged into
4. Include a brief note that these were queued during their idle period

Let me compose and send the alerts now. I'll format them concisely since the user just came online and probably wants a quick overview, not lengthy analysis.

Starting with the CVE alert..."""
)

add(
    "[DAEMON] Context: All systems nominal. No events. User idle 15min. Last action 15min ago: sent daily digest. Budget: brief. Decision?",

    "No action. Digest just sent. Everything quiet. Next natural action: when a monitor comes due or user returns.",

    """Let me evaluate whether any action is needed:

**System Status:**
- All systems nominal
- No pending events
- User idle for 15 minutes
- Last action (15 minutes ago): Daily digest was sent

**Analysis:**
The system is in a completely stable state. The daily digest was just sent 15 minutes ago, which means:
1. The user has received their summary of recent activity
2. No new significant events have occurred since then
3. The user might be reading the digest or doing other things

**Should I take any action?**
- No background tasks needed (system is healthy)
- No curiosity items need immediate attention
- No monitors are in a failed state
- Dream consolidation isn't overdue
- No events in the queue

**Decision: No action needed.**

I'll remain in passive monitoring mode. The next action point will likely be:
- A scheduled monitor check becoming due
- The user returning with a message
- An external event arriving
- An extended idle period triggering background tasks"""
)


# ============================================================================
# Write output
# ============================================================================

def main():
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "training_data_v6.jsonl",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in PAIRS:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Wrote {len(PAIRS)} DPO pairs to {output_path}")
    print()

    # Category breakdown
    categories = {
        "Brief/Daemon Mode": 15,
        "Memory Consolidation Reasoning": 15,
        "Frustration De-escalation": 15,
        "Observation Writing": 10,
        "Action/No-Action Decisions": 10,
    }
    for cat, count in categories.items():
        print(f"  {cat}: {count} pairs")
    print(f"\n  Total: {sum(categories.values())} pairs")


if __name__ == "__main__":
    main()
