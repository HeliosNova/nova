"""One REAL heartbeat tick, end to end, on a scripted fake LLM (2026-09-02).

Every unit test in this suite mocks the layer below it, so the plumbing
between layers — dispatch key → executor → result row → KG extraction →
alert journal, and the knowing tier's storyline → forecast → dossier →
revision → curiosity chain — had no test that could fail when a key was
misspelled or a stage silently stopped being reached (the 2026-08-17
duplicate dispatch key, the five-week storyline outage).

This runs `HeartbeatLoop._loop` for exactly one tick against the real
MonitorStore, the real KnowledgeGraph, the real storyline/dossier/forecast
modules and the real delivery journal. Only the model, the web and the chat
brain are scripted. The negative control deletes one dispatch key and shows
the same assertions catch it.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from app.monitors import heartbeat_loop as hb
from app.monitors.heartbeat_loop import HeartbeatLoop
from app.monitors.monitor_store import MonitorStore

# --------------------------------------------------------------- fixtures

DIGEST_A = (
    "## AI and ML — daily digest\n"
    "Nvidia announced the Rubin GPU platform at its Santa Clara event on 2026-09-01 (reuters.com).\n"
    "OpenAI released GPT-6 to enterprise customers this week, according to The Information (theinformation.com).\n"
    "Anthropic opened a Tokyo office to serve Japanese enterprises, the company said on 2026-08-31 (bloomberg.com).\n"
    "Google DeepMind published AlphaFold 4 results in Nature covering RNA structures (nature.com).\n"
    "Microsoft and Nvidia signed a multi-year supply agreement for Rubin accelerators (wsj.com).\n"
    "Bottom line: the Rubin platform is now the reference accelerator for every hyperscaler roadmap.\n"
)
DIGEST_B = (
    "## Semiconductors — daily digest\n"
    "TSMC began volume production of its 2nm node in Hsinchu on 2026-09-01, Nikkei Asia reported (asia.nikkei.com).\n"
    "Samsung Electronics won a Rubin packaging contract from Nvidia worth several billion dollars (reuters.com).\n"
    "ASML shipped its first High NA EUV tool to Intel Foundry in Oregon this week (asml.com).\n"
    "Micron Technology raised guidance on HBM4 demand tied to the Rubin platform (micron.com).\n"
    "Bottom line: the Rubin supply chain is now the largest single driver of advanced-node capacity.\n"
)

TRIPLES = [
    {"subject": "Nvidia", "predicate": "located_in", "object": "Santa Clara", "confidence": 0.9},
    {"subject": "Jensen Huang", "predicate": "leads", "object": "Nvidia", "confidence": 0.9},
]

STORY_UPDATE = (
    "Nvidia has moved the Rubin platform from announcement to first shipments, with Microsoft "
    "signing a multi-year supply agreement and the supply chain from TSMC to Micron aligning behind it.\n"
    "CHANGED: Rubin moved from announcement to first shipments and supply-chain lock-in.\n"
    "STATE: Nvidia Rubin | shipping to hyperscalers\n"
    "FORECAST: Microsoft deploys Rubin clusters in at least three Azure regions by 2026-12-31 "
    "| resolves 2026-12-31 | 0.6"
)

DOSSIER_BODY = (
    "## Current understanding\n"
    "Nvidia's Rubin platform, announced in Santa Clara on 2026-09-01 (reuters.com), is now the "
    "reference accelerator for hyperscaler roadmaps. Microsoft signed a multi-year supply "
    "agreement (wsj.com) and the supply chain — TSMC 2nm volume production (asia.nikkei.com), "
    "Samsung packaging (reuters.com), Micron HBM4 (micron.com) — is aligning behind it.\n"
    "## How we got here\n"
    "- 2026-08-31: Anthropic opened a Tokyo office (bloomberg.com)\n"
    "- 2026-09-01: Rubin announced; Microsoft supply agreement signed\n"
    "## Key facts & figures\n"
    "- Rubin announced 2026-09-01 (reuters.com)\n"
    "- TSMC 2nm volume production began 2026-09-01 (asia.nikkei.com)\n"
    "## Open questions\n"
    "- What is Nvidia's current Rubin production capacity per quarter?\n"
    "- How much of TSMC's 2nm capacity is committed to Nvidia today?\n"
    "Watch for: Rubin revenue disclosure, Azure region deployments\n"
    "CHANGED: Rubin moved from announcement to a locked-in supply chain.\n"
    "FORECAST: Nvidia reports Rubin platform revenue above $10B in its Q3 FY2027 results "
    "| resolves 2026-11-20 | 0.6"
)


def _instance(schema: dict):
    """Smallest valid instance of a JSON schema (fallback for unscripted calls)."""
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object" or "properties" in schema:
        return {k: _instance(v) for k, v in schema.get("properties", {}).items()
                if k in schema.get("required", schema.get("properties", {}))}
    if t == "array":
        return []
    if t == "string":
        return ""
    if t in ("number", "integer"):
        return 0
    if t == "boolean":
        return False
    return None


class FakeProvider:
    """Scripted stand-in for OllamaProvider: answers by prompt shape."""

    def __init__(self):
        self.calls: list[str] = []

    def capabilities(self):
        from app.core.llm import ProviderCapabilities
        return ProviderCapabilities()

    async def invoke_nothink(self, messages, *, json_mode=False, json_prefix="{",
                             json_schema=None, max_tokens=8000, temperature=0.1,
                             model=None, num_ctx=None):
        prompt = messages[-1]["content"] if messages else ""
        self.calls.append(prompt[:160])
        if json_schema is not None:
            return json.dumps(self._for_schema(json_schema, prompt))
        if "PRIOR DOSSIER:" in prompt:
            return DOSSIER_BODY
        if "PRIOR STATE:" in prompt and "NEW DEVELOPMENTS" in prompt:
            return STORY_UPDATE
        if json_mode:
            return "{}"
        return "Nothing further to report."

    @staticmethod
    def _for_schema(schema: dict, prompt: str):
        props = (schema.get("items") or {}).get("properties", {}) if schema.get("type") == "array" \
            else schema.get("properties", {})
        if "subject" in props and "predicate" in props:
            return TRIPLES
        if "title" in props and "items" in props:
            n = len(re.findall(r"(?m)^\d+\. \[", prompt))
            return [{"title": "Nvidia Rubin platform rollout", "items": list(range(n))}]
        if "verdict" in props:
            return {"verdict": "miss", "evidence_date": "2026-08-30",
                    "reason": "shipments slipped past the stated deadline"}
        return _instance(schema)

    async def generate_with_tools(self, *a, **k):
        raise AssertionError("chat generation must not run inside a monitor tick")

    async def stream_with_thinking(self, *a, **k):
        raise AssertionError("streaming must not run inside a monitor tick")
        yield  # pragma: no cover

    async def close(self):
        return None


class FakeBot:
    """A channel that accepts every digest (the ledger must then be cleaned)."""

    default_chat_id = "chat-1"

    def __init__(self):
        self.sent: list[str] = []

    async def send_alert(self, text: str) -> bool:
        self.sent.append(text)
        return True


class _AsyncioShim:
    """The loop's own sleeps, trimmed to one tick: the 10s startup wait returns
    at once; the end-of-tick HEARTBEAT_INTERVAL sleep stops the loop."""

    def __init__(self, real, stop):
        self._real, self._stop = real, stop

    def __getattr__(self, name):
        return getattr(self._real, name)

    async def sleep(self, delay, *a, **k):
        if delay == 10:
            return
        if delay >= 60:
            self._stop()
            return
        await self._real.sleep(min(delay, 0.01))


async def _fake_think(query, **kwargs):
    """brain.think stand-in: one TOKEN event carrying a digest."""
    from app.schema import EventType, StreamEvent
    yield StreamEvent(type=EventType.TOKEN, data={"text": DIGEST_A})
    yield StreamEvent(type=EventType.DONE, data={})


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def world(monkeypatch):
    """Real DB + KG + store with the tick's monitors and the knowing tier's inputs."""
    from app.core import llm
    from app.core.brain import Services, set_services
    from app.core.kg import KnowledgeGraph
    from app.database import get_db

    db = get_db()
    db.init_schema()
    kg = KnowledgeGraph(db)
    set_services(Services(kg=kg))
    fake = FakeProvider()
    llm.set_provider(fake)
    # llm._last_interactive_monotonic is PROCESS-global: any earlier test in
    # the run that called brain.think() leaves the "owner is chatting" latch
    # set, and the tick then defers all but two badly-overdue LLM monitors
    # (consolidation never ran, full-suite only — 2026-09-02). This test is
    # about the tick reaching every layer; chat deferral has its own tests.
    monkeypatch.setattr(llm, "_last_interactive_monotonic", -1e9)
    monkeypatch.setattr("app.core.brain.think", _fake_think)

    async def _evidence(claim, *, created_at=None, max_results=8):
        return "- (2026-08-30) Nvidia said Rubin shipments would begin in October, later than planned (reuters.com)"
    monkeypatch.setattr("app.core.forecasts._gather_evidence", _evidence)

    store = MonitorStore(db)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    store.create("System Health", "system_health", {"threshold_pct": 10}, 7200, 60, "on_change")
    store.create("Pathway Liveness", "pathway_liveness", {}, 21600, 300, "on_change")
    store.create("Tick Probe", "query", {"query": "What changed in the lab today?"}, 3600, 60, "always")
    store.create("Storyline Tracker", "storyline", {}, 28800, 60, "on_change")
    store.create("Knowledge Consolidation", "consolidation", {}, 14400, 60, "on_change")
    store.create("Forecast Resolution", "forecast_resolve", {}, 21600, 60, "on_change")
    # The digest producers themselves stay disabled (their runner is the web
    # chain); their OUTPUT is what the knowing tier consumes.
    a = store.create("Domain Study: Alpha Lab", "query", {"query": "x"}, 28800, 60, "always", enabled=False)
    b = store.create("Domain Study: Beta Lab", "query", {"query": "x"}, 28800, 60, "always", enabled=False)
    for mid, digest in ((a, DIGEST_A), (b, DIGEST_B)):
        db.execute("INSERT INTO monitor_results (monitor_id, status, value, created_at) VALUES (?, 'ok', ?, ?)",
                   (mid, digest, _ts(now - timedelta(minutes=30))))
    # A prior dossier two days old, so consolidation must append a revision.
    db.execute("INSERT INTO dossiers (kind, dkey, title, body, changed_note, update_count, created_at, updated_at) "
               "VALUES ('domain', 'alpha-lab', 'Alpha Lab', ?, 'initial dossier', 1, ?, ?)",
               ("## Current understanding\nRubin was only a rumour.\n## Open questions\n- none",
                _ts(now - timedelta(days=2)), _ts(now - timedelta(days=2))))
    # One forecast past its horizon, so resolution has work.
    db.execute("INSERT INTO forecasts (claim, storyline_key, confidence, resolves_at, status, source_monitor, created_at) "
               "VALUES (?, 'nvidia-rubin', 0.7, ?, 'open', 'seed', ?)",
               ("Nvidia ships Rubin GPUs to a hyperscaler before 2026-09-01",
                _ts(now - timedelta(days=1)), _ts(now - timedelta(days=10))))
    return {"db": db, "store": store, "kg": kg, "fake": fake, "now": now}


async def _run_one_tick(world, monkeypatch) -> tuple[HeartbeatLoop, FakeBot, list[int]]:
    bot = FakeBot()
    loop = HeartbeatLoop(world["store"], telegram_bot=bot)
    # Spy on the ledger cleanup: every confirmed send must delete its journal row.
    cleaned: list[int] = []
    orig_delete = loop._delete_journal_rows

    async def _spy_delete(row_ids):
        cleaned.extend(i for i in row_ids if i is not None)
        await orig_delete(row_ids)
    loop._delete_journal_rows = _spy_delete
    monkeypatch.setattr(hb, "asyncio", _AsyncioShim(asyncio, lambda: setattr(loop, "_running", False)))
    loop._running = True
    await asyncio.wait_for(loop._loop(), timeout=240)
    if loop._kg_bg_tasks:
        await asyncio.gather(*list(loop._kg_bg_tasks), return_exceptions=True)
    return loop, bot, cleaned


def _results(db) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in db.fetchall("SELECT m.name AS name, mr.status AS status, mr.value AS value, mr.message AS message "
                         "FROM monitor_results mr JOIN monitors m ON m.id = mr.monitor_id ORDER BY mr.id"):
        out.setdefault(r["name"], []).append(dict(r))
    return out


# ------------------------------------------------------------------ tests

@pytest.mark.asyncio
async def test_one_tick_reaches_every_layer(world, monkeypatch):
    db = world["db"]
    loop, bot, cleaned = await _run_one_tick(world, monkeypatch)

    res = _results(db)
    for name in ("System Health", "Pathway Liveness", "Tick Probe", "Storyline Tracker",
                 "Knowledge Consolidation", "Forecast Resolution"):
        rows = [r for r in res.get(name, []) if r["status"] != "error"]
        assert rows, f"{name}: no successful result row after the tick ({res.get(name)})"
    unknown = [(n, r["value"]) for n, rs in res.items() for r in rs
               if (r["value"] or "").startswith("[Unknown check_type")]
    assert not unknown, f"undispatched monitors: {unknown}"

    # Fast lane: the liveness registry ran and judged a fresh install healthy.
    assert "all pathways alive" in res["Pathway Liveness"][-1]["value"]

    # Query monitor → brain → result → KG extraction (background task) → facts.
    assert DIGEST_A.splitlines()[1] in res["Tick Probe"][-1]["value"]
    facts = db.fetchall("SELECT subject, predicate, object FROM kg_facts WHERE provenance = 'Tick Probe'")
    assert {(f["subject"], f["predicate"], f["object"]) for f in facts} >= {("nvidia", "located_in", "santa clara")} \
        or len(facts) >= 1, "KG extraction after a query monitor banked nothing"

    # Alert → delivery journal → digest → channel → journal row cleaned.
    assert any("Tick Probe" in t and DIGEST_A.splitlines()[1] in t for t in bot.sent), \
        "the query monitor's digest never reached the channel"
    assert cleaned, "no journal row was written and cleaned — the delivery ledger was bypassed"
    assert not db.fetchall("SELECT 1 FROM pending_deliveries"), "confirmed sends must clear the ledger"
    assert db.fetchone("SELECT last_alert_at FROM monitors WHERE name = 'Tick Probe'")["last_alert_at"]
    assert res["Tick Probe"][-1]["status"] == "alert"

    # Storyline tracker: cluster → update → events + a minted forecast.
    events = db.fetchall("SELECT summary FROM storyline_events")
    assert events, "storyline tracker recorded no events"
    stories = db.fetchall("SELECT title, update_count FROM storylines")
    assert stories and stories[0]["title"] == "Nvidia Rubin platform rollout"
    assert res["Storyline Tracker"][-1]["value"].startswith("## 🧵 STORYLINE UPDATES")

    # Consolidation: revision trail + updated dossier + curiosity fed from an open question.
    revs = db.fetchall("SELECT dossier_id, body FROM dossier_revisions")
    assert revs and "only a rumour" in revs[0]["body"], "prior dossier body not preserved as a revision"
    doss = db.fetchone("SELECT update_count, body FROM dossiers WHERE dkey = 'alpha-lab'")
    assert doss["update_count"] == 2 and "reference accelerator" in doss["body"]
    curiosity = db.fetchall("SELECT topic, source FROM curiosity_queue WHERE source = 'dossier_open_question'")
    assert curiosity and "production capacity" in curiosity[0]["topic"]
    # Open-questions ledger (Phase 4.4): the body's questions were reconciled and
    # the one handed to curiosity is marked queued.
    ledger = {r["question"]: r["status"] for r in db.fetchall(
        "SELECT question, status FROM dossier_questions WHERE dkey = 'alpha-lab'")}
    assert ledger, "consolidation did not reconcile the open-questions ledger"
    assert ledger.get("What is Nvidia's current Rubin production capacity per quarter?") == "queued", ledger

    # Forecasts: the due one was graded from (fake) web evidence; new ones were minted.
    seeded = db.fetchone("SELECT status, resolved_at, resolution FROM forecasts WHERE source_monitor = 'seed'")
    assert seeded["status"] == "miss" and seeded["resolved_at"], seeded
    minted = db.fetchall("SELECT source_monitor, status, resolves_at FROM forecasts WHERE source_monitor != 'seed'")
    assert {m["source_monitor"] for m in minted} >= {"Storyline Tracker", "Knowledge Consolidation"}, minted
    assert all(m["status"] == "open" for m in minted)


@pytest.mark.asyncio
async def test_a_broken_dispatch_key_is_caught_by_the_same_tick(world, monkeypatch):
    """Negative control: drop one dispatch key and the tick must visibly fail."""
    monkeypatch.delitem(HeartbeatLoop._CHECK_DISPATCH, "storyline")
    db = world["db"]
    await _run_one_tick(world, monkeypatch)  # noqa: RUF — the verdict is in the DB
    res = _results(db)
    values = [r["value"] for r in res.get("Storyline Tracker", [])]
    assert any((v or "").startswith("[Unknown check_type: storyline]") for v in values), values
    assert not db.fetchall("SELECT 1 FROM storyline_events"), "events written with no dispatch — impossible"
    # Everything else still ran: the failure is isolated, not a crash.
    assert res["Knowledge Consolidation"] and res["Tick Probe"]
