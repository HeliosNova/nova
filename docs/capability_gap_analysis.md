# Nova — Complete Capability Gap Analysis (March 2026)

## Methodology
Cross-referenced Nova's 21 tools against OpenClaw (26 tools + 53 skills + 13K community skills),
Open Assistant (self-hosted, OAuth2-based integrations), CoPaw, and 2026 user expectations.

---

## CATEGORY 1: COMMUNICATION

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Send email (SMTP) | YES (action_email) | YES (gog skill) | YES (Gmail/Outlook) | No |
| Read email (IMAP/API) | NO | YES (Gmail via gog) | YES (Gmail/Outlook) | **YES** |
| Email triage/summarize | NO | YES | YES | **YES** |
| Draft email replies | NO | YES | YES | **YES** |
| Send SMS/text | NO | YES (Twilio skill) | NO | **YES** |
| iMessage | NO | YES (imessage skill) | NO | **YES** (macOS only) |
| Slack (read/send/react) | NO | YES (slack skill) | NO | **YES** |
| Push notifications to phone | NO | YES (ntfy/pushover) | NO | **YES** |
| Discord (as tool, not just channel) | Alert only | YES (discord skill) | NO | Partial |
| Telegram (as tool, not just channel) | Alert only | YES | NO | Partial |
| WhatsApp (as tool, not just channel) | Alert only | YES | YES | Partial |

**Nova's gap:** Can send emails and push alerts to 4 channels, but cannot READ inbound messages, triage them, or interact with Slack/SMS. No push notifications to phone.

---

## CATEGORY 2: CALENDAR & SCHEDULING

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Local calendar (.ics) | YES (action_calendar) | NO (uses Google/Apple) | NO | — |
| Google Calendar sync | NO | YES (gog skill) | YES (OAuth2) | **YES** |
| Outlook Calendar sync | NO | YES | YES (OAuth2) | **YES** |
| Apple Calendar | NO | YES (apple-calendar skill) | NO | **YES** (macOS only) |
| Find open time slots | NO | YES | YES | **YES** |
| Schedule meetings | NO | YES | YES | **YES** |
| Calendar event reminders | Partial (reminder tool) | YES | YES | Partial |
| Recurring event management | NO | YES | YES | **YES** |

**Nova's gap:** Calendar is local-only .ics file. No sync with any real calendar service. Can't find open slots, schedule meetings, or manage recurring events.

---

## CATEGORY 3: TASK & PROJECT MANAGEMENT

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| To-do list / tasks | NO | YES (Apple Reminders, Todoist) | YES (Notion) | **YES** |
| Trello/Jira boards | NO | YES (trello skill) | NO | **YES** |
| Notion integration | NO | YES (notion skill) | YES (OAuth2) | **YES** |
| Obsidian notes | NO | YES (obsidian skill) | NO | **YES** |
| Apple Notes | NO | YES (apple-notes skill) | NO | **YES** (macOS) |
| GitHub issues/PRs | NO (only via shell) | YES (github skill + OAuth) | NO | **YES** |
| Linear / project trackers | NO | Community skills | NO | **YES** |

**Nova's gap:** No native task management, note-taking, or project tracker integration. Could do some via shell_exec + gh CLI but no structured integration.

---

## CATEGORY 4: FILE & DOCUMENT MANAGEMENT

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Read/write local files | YES (file_ops) | YES (read/write tools) | NO | No |
| List/search files | YES (file_ops) | YES (list/search tools) | NO | No |
| Google Drive | NO | YES (gog skill) | NO | **YES** |
| OneDrive | NO | NO | YES (OAuth2) | **YES** |
| Dropbox | NO | Community skill | NO | **YES** |
| PDF create/read | NO | YES (pdf tool) | NO | **YES** |
| ZIP compress/extract | NO | YES (zip tool) | NO | **YES** |
| Document ingestion (RAG) | YES (retriever) | NO | NO | No (Nova wins) |

**Nova's gap:** No cloud storage integration. No PDF/ZIP tools. But Nova's RAG retriever is a strength — no competitor has hybrid vector+BM25+RRF.

---

## CATEGORY 5: WEB & BROWSER

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Web search | YES (web_search) | YES (Brave/Perplexity/etc.) | YES (Brave) | No |
| HTTP fetch | YES (http_fetch) | YES (web_fetch) | NO | No |
| Browser automation | YES (browser) | YES (browser tool) | YES (Playwright) | No |
| Screenshot capture | YES (screenshot) | YES | NO | No |
| URL summarization | Partial (via think) | YES (dedicated skill) | NO | Partial |

**Nova's gap:** Roughly at parity here. Nova is solid on web capabilities.

---

## CATEGORY 6: SYSTEM & DEVELOPMENT

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Shell commands | YES (shell_exec) | YES (exec) | NO | No |
| Code execution (sandbox) | YES (code_exec) | YES | NO | No |
| Desktop GUI automation | YES (desktop, Linux) | YES (macOS/Linux) | NO | No |
| Database queries | NO | YES (database tool) | NO | **YES** |
| Git operations | Via shell_exec | YES (github skill) | NO | Partial |
| Docker management | Via shell_exec | Community skill | NO | Partial |
| SSH remote commands | NO | Community skill | NO | **YES** |
| Cron/scheduled scripts | Via monitors | YES (schedule/heartbeat) | NO | Partial |
| Image generation | NO | YES (image_gen: DALL-E/SD) | NO | **YES** |

**Nova's gap:** No direct database tool, no SSH, no image generation. Git/Docker possible but not structured. Monitors are actually stronger than OpenClaw's heartbeat.

---

## CATEGORY 7: SMART HOME & IoT

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Home Assistant | Via webhook only | YES (home-assistant skill) | NO | **YES** |
| Philips Hue | NO | Community skill | NO | **YES** |
| Smart plugs/switches | NO | Community skill | NO | **YES** |
| Temperature/sensors | NO | Community skill | NO | **YES** |

**Nova's gap:** No native smart home. Could use webhook tool to hit Home Assistant API, but no structured integration or device discovery.

---

## CATEGORY 8: MEDIA & ENTERTAINMENT

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Spotify control | NO | YES (spotify skill) | NO | **YES** |
| YouTube summary | Via web_search | Community skill | NO | Partial |
| Podcast management | NO | Community skill | NO | **YES** |
| Music playback | NO | Community skill | NO | **YES** |

**Nova's gap:** No media control whatsoever.

---

## CATEGORY 9: FINANCE & DATA

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Stock/crypto prices | Via web_search | YES (finance skills) | NO | Partial |
| Portfolio tracking | NO | Community skill | NO | **YES** |
| Weather | Via web_search | YES (weather skill) | NO | Partial |
| News aggregation | YES (domain study monitors) | Community skill | NO | No (Nova wins) |

**Nova's gap:** Domain study monitors are actually better than OpenClaw for news/research. Financial tools are weak.

---

## CATEGORY 10: VOICE & MULTIMODAL

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Speech-to-text (STT) | YES (Whisper) | NO (via skills) | NO | No (Nova wins) |
| Text-to-speech (TTS) | NO | Community skill | NO | **YES** |
| Image understanding | NO (model can but not tool) | Via LLM | NO | Partial |
| Image generation | NO | YES (DALL-E/SD) | NO | **YES** |

**Nova's gap:** Has STT but no TTS. No image generation.

---

## CATEGORY 11: AUTOMATION & WORKFLOWS

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Scheduled checks/monitors | YES (51 monitors, 29 domains) | YES (heartbeat/schedule) | NO | No (Nova wins) |
| Conditional actions (if X then Y) | NO (alert only) | Via skill logic | NO | **YES** |
| Multi-step workflows | Partial (delegate) | YES (skill chains) | YES (via plugins) | **YES** |
| Webhook triggers | YES (action_webhook) | YES | NO | No |
| OAuth2 flows | NO | YES (per-skill) | YES (built-in) | **YES** |
| Community skill/plugin system | NO | YES (ClawHub: 13K+) | YES (plugin system) | **YES** |
| Reminders | YES (action_reminder) | YES | YES | No |

**Nova's gap:** The biggest structural gap. No conditional actions, no OAuth2, no community skills ecosystem.

---

## CATEGORY 12: LEARNING & INTELLIGENCE (Nova's Moat)

| Capability | Nova | OpenClaw | Open Assistant | Gap? |
|---|---|---|---|---|
| Learn from corrections | YES (full pipeline) | NO | NO | Nova wins |
| DPO training data gen | YES | NO | NO | Nova wins |
| Auto fine-tuning + A/B eval | YES | NO | NO | Nova wins |
| Temporal knowledge graph | YES | NO | NO | Nova wins |
| Hybrid retrieval (Vec+BM25+RRF) | YES | NO | NO | Nova wins |
| Curiosity engine | YES | NO | NO | Nova wins |
| Reflexion (failure learning) | YES | NO | NO | Nova wins |
| Domain studies (proactive research) | YES | NO | NO | Nova wins |
| Skill auto-creation | YES | NO | NO | Nova wins |
| Prompt injection detection | YES (4-category) | NO (CVE'd) | NO | Nova wins |
| MCP server (expose intelligence) | YES | NO | NO | Nova wins |

**No gap here.** This is Nova's moat. No competitor has any of this.

---

## SUMMARY: THE COMPLETE GAP LIST

### Must Build (high impact, users expect these daily)
1. **OAuth2 connector framework** — unlocks everything below
2. **Gmail/Outlook read** (IMAP or API) — "summarize my unread emails"
3. **Google Calendar sync** — "what's on my schedule today"
4. **Conditional monitor actions** — "if X happens, do Y" (not just alert)
5. **Push notifications** (ntfy/Pushover/Gotify) — alert to phone, not just chat
6. **TTS** (edge-tts or Piper) — voice responses
7. **GitHub integration** (structured, not just shell) — issues, PRs, reviews

### Should Build (strong differentiators)
8. **Notion/Obsidian integration** — note-taking
9. **Todoist/task management** — to-do lists
10. **Home Assistant** — smart home control
11. **Slack integration** — read/send messages
12. **Image generation** (SD/DALL-E API) — creative tasks
13. **PDF read/create** — document handling
14. **Database tool** — query SQLite/Postgres directly

### Nice to Have (community can build via MCP)
15. **Spotify/media control**
16. **SMS (Twilio)**
17. **Cloud storage** (GDrive, OneDrive, Dropbox)
18. **SSH remote commands**
19. **ZIP/compression**
20. **Weather widget** (dedicated, not web search)
21. **Portfolio/finance tracking**
22. **Skill sharing format** (Nova's version of ClawHub)

---

## WHAT NOVA ALREADY WINS

Don't lose sight of this — Nova has things NO competitor has:
- Full learning loop (correction → lesson → DPO → fine-tune → A/B eval)
- Temporal knowledge graph with contradiction resolution
- Hybrid retrieval (vector + BM25 + RRF fusion)
- Curiosity engine (auto-detects gaps, researches autonomously)
- Reflexion store (learns from silent failures)
- 51 autonomous monitors across 29 domains (finance, geopolitics, crypto, science, sports, etc.)
- MCP server (exposes intelligence to other agents)
- Prompt injection detection (4-category heuristic)
- True sovereignty (zero cloud dependency)

The brain is world-class. It just needs hands.
