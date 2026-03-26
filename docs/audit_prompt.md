# Nova Deep Audit Prompt

Paste this into a fresh Claude Code context window.

---

## The Prompt

You are auditing Nova, a sovereign self-improving personal AI assistant. Your job is to find everything that's broken, wrong, or degraded — in the actual running system, not just in code review.

### What Nova Is

Nova is NOT a chatbot. It is a closed-loop learning system that:
- Runs a local LLM (Qwen3.5:27b fine-tuned via DPO) on an RTX 3090
- Learns permanently from user corrections (correction → lesson → DPO training pair → fine-tune → deploy)
- Builds a temporal knowledge graph from autonomous web research
- Monitors 15+ domains on schedule, detects changes, sends alerts via Discord/Telegram
- Quizzes itself on learned lessons, validates its own skills, critiques its own responses
- Detects knowledge gaps in conversations and queues them for autonomous research
- Promotes recurring failures into lessons automatically
- Generates an evening digest summarizing what it learned and observed

The entire point is the learning loop. If corrections don't become lessons, if lessons don't become training data, if training data doesn't improve the model — Nova is broken regardless of what unit tests say.

### How to Audit

**Start with runtime behavior, not code.** A passing test suite means nothing if the live system produces garbage. Check the database. Check the logs. Check what the monitors actually output. Ask Nova questions and evaluate the answers.

**Think like a user.** Ask casual questions ("yo whats happening with crypto"). Ask factual questions ("who leads the Premier League"). Ask questions that need tools ("what's the current price of gold"). Ask questions the model should search for but might not ("whats the latest iphone"). Watch what it does — does it use tools? Does it answer from stale data? Does it expose internal debugging? Does it hallucinate?

**Trace every learning loop end to end:**
1. Tell Nova a fact about yourself → does it extract and store it?
2. Give Nova a wrong answer, then correct it → does it save a lesson AND a DPO pair? Does it NOT save lessons when it correctly pushes back on bad corrections?
3. Check if the lesson quiz actually tests real knowledge or generates garbage questions from empty context
4. Check if skill validation actually matches test queries to trigger patterns or just skips
5. Check if curiosity research produces real findings or "[No pending items]"
6. Check if reflexion quality scores are meaningful or all 0.00 (false hallucination flags)

**Check what the monitors actually produce.** Read the last 20 monitor_results from the database. Are domain studies producing real content about real topics? Is the morning check-in coherent? Is system health reporting accurate numbers? Or is it all garbage/errors/timeouts?

**Check every channel.** Is Discord connected? Is Telegram delivering alerts (not "Chat not found")? Are alert messages complete or truncated?

**Then read the code.** Once you know what's broken from runtime, trace the cause in code. The audit_prompt.md in older versions told you which files and line numbers to look at — ignore that. Follow the actual bug from symptom to root cause.

### Rules

- **Runtime evidence or it didn't happen.** Don't say "this looks correct" from reading code. Prove it works by checking the database, API response, or logs.
- **Clean up after yourself.** Every test fact, conversation, lesson, or curiosity item you create during testing must be deleted. Nova's database is production.
- **Fix root causes.** If something crashes, find WHY — don't wrap it in try/except.
- **Search the web when you're stuck.** Dependency conflicts, Docker issues, library breaking changes — look up the current solution instead of guessing for hours.
- **Don't add features or refactor working code.** Audit means fix what's broken, not redesign what works.
- **Don't add security layers.** This is a personal AI on a home network. Focus on: does it work? Does it produce good output? Does it learn?
- **The user is in Los Angeles (Pacific Time).** All user-facing times should reflect this.

### What to Deliver

A graded report with evidence for each dimension:

| Dimension | Question |
|-----------|----------|
| Chat Quality | Does it answer well? Use tools when needed? Natural tone? |
| Learning Loop | Corrections → lessons → DPO → fine-tune → deployed? End to end? |
| Knowledge Systems | KG growing with real facts? Retrieval finding relevant docs? |
| Monitor Quality | All monitors producing real content? Not garbage/errors? |
| Self-Improvement | Quizzes meaningful? Skills validated? Curiosity researching real gaps? |
| Channels | Discord/Telegram connected and delivering complete alerts? |
| Code Quality | Bugs? Dead code? Logic errors? Edge cases? |
| Runtime Stability | Errors in logs? Crashes? Silent failures? |

Grade each 1-10 with specific evidence (database rows, API responses, log lines — not code snippets). Overall grade is the average.

For every bug found: trace it to the root cause, fix it properly, deploy, and prove the fix works by checking production output.

### Start Here

Read `nova_/CLAUDE.md` to understand the architecture. Then check if the system is running:

```bash
curl -s http://localhost:8000/api/health
docker exec nova-ollama ollama list
```

Then start testing. Don't plan — just start asking Nova questions and checking what happens.
