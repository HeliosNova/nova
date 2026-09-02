"""Deterministic research-quality probes for the eval harness (2026-09-01).

The nightly suite graded chat answers only (9 of 11 categories at 100% for
weeks) and nothing about the product — digests, dossiers, forecasts, the KG.
These checks exercise the code paths that make Nova KNOWING, with fixtures,
no LLM and no GPU, so a regression in any of them fails the suite the next
night instead of surfacing weeks later in a live audit.

Each check returns (passed: bool, detail: str). Register new ones in CHECKS.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone


def _ok(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)


def check_resolver_window() -> tuple[bool, str]:
    """Forecast grading: pre-window evidence never confirms a hit; guidance is
    not an outcome (prompt rule); dates and junk hosts are filtered."""
    from types import SimpleNamespace
    from unittest.mock import patch
    from app.core import forecasts as fc
    fails: list[str] = []
    created = "2026-08-15 10:00:00"
    ok, why = fc._hit_is_grounded("hit", "2026-06-01", created,
                                  "- x (2026-06-01) [reuters.com]: old")
    _ok(not ok and "predates" in why, "pre-window dated hit was accepted", fails)
    ok, _ = fc._hit_is_grounded("hit", "2026-08-28", created, "- x (2026-08-28) [reuters.com]")
    _ok(ok, "in-window dated hit was rejected", fails)
    ok, _ = fc._hit_is_grounded("hit", None, created, "- x (2026-08-28) [reuters.com]")
    _ok(not ok, "undated hit accepted although dated evidence existed", fails)
    _ok("are NOT outcomes" in fc._RESOLVE_PROMPT and "it resolves on" in fc._RESOLVE_PROMPT,
        "resolver prompt lost the window/outcome rules", fails)
    results = [SimpleNamespace(url="https://reuters.com/a", title="after", published_date="2026-08-20", snippet=""),
               SimpleNamespace(url="https://reuters.com/b", title="before", published_date="2025-12-09", snippet="")]
    with patch("app.core.source_authority.authority", return_value=0.9):
        kept, old, junk = fc._filter_evidence(results, created)
    _ok(len(kept) == 1 and old == 1, "evidence date filter regressed", fails)
    _ok(fc._MAX_DAYS >= 365, f"horizon clamp regressed to {fc._MAX_DAYS} days", fails)
    dl = fc.claim_deadline("approval by Q4 2026", now=datetime(2026, 9, 1))
    _ok(dl == datetime(2026, 12, 31), "claim deadline parser regressed", fails)
    return (not fails, "; ".join(fails) or "resolver window, outcome rule, date filter, horizon, deadline parser OK")


def check_priming_key() -> tuple[bool, str]:
    """Every Domain Study profile label resolves to its consolidation key and the
    priming excerpt carries the Open questions."""
    from app.core.dossiers import _slug, priming_excerpt, resolve_domain_dkey
    from app.monitors.domain_study_runner import _DOMAIN_PROFILES
    fails: list[str] = []
    misses = [key for key, prof in _DOMAIN_PROFILES.items()
              if resolve_domain_dkey(prof[1]) != _slug(key)]
    _ok(not misses, f"labels that miss their dossier key: {misses[:5]}", fails)
    body = "## Current understanding\nA.\n\n## How we got here\nB.\n\n## Open questions\n- Q1?\n"
    ex = priming_excerpt(body)
    _ok("## Open questions" in ex and "Q1?" in ex and "How we got here" not in ex,
        "priming excerpt lost the Open questions section", fails)
    return (not fails, "; ".join(fails) or f"{len(_DOMAIN_PROFILES)} profile labels resolve; excerpt carries open questions")


_DEFECTIVE_DIGEST = (
    "## 🌐 open source — domain overview\n_read 14 sources: cnbc.com · September 01, 2026_\n"
    "**Lead Development: Real news**\nThe consortium shipped its charter on August 30, 2026 (cnbc.com).\n"
    "* **Apache httpd share:** Apache httpd held 24% of the server market in April 2020 (wikipedia.org).\n"
    "* The launched a **$10 million initiative** (cnbc.com).\n"
    "* One-sixth of the population was displaced (deep analysis: Historical Context).\n"
)
_CLEAN_DIGEST = (
    "## 🌐 open source — domain overview\n_read 14 sources: cnbc.com · September 01, 2026_\n"
    "**Lead Development: Real news**\nThe consortium shipped its charter on August 30, 2026 (cnbc.com).\n"
    "* Membership rose to 412 organisations in August 2026 (bbc.com).\n"
)


def check_digest_canary() -> tuple[bool, str]:
    """The judge's deterministic canaries flag seeded defects and stay quiet on
    a clean digest."""
    from app.core.output_eval import _canaries
    fails: list[str] = []
    flags = set(_canaries(_DEFECTIVE_DIGEST, year=2026))
    _ok({"stale-year", "artifact", "pseudo-citation"} <= flags, f"canaries missed defects: got {sorted(flags)}", fails)
    _ok(_canaries(_CLEAN_DIGEST, year=2026) == [], "canaries fired on a clean digest", fails)
    return (not fails, "; ".join(fails) or "stale-year, artifact, pseudo-citation detected; clean digest quiet")


def check_fact_analysis() -> tuple[bool, str]:
    """Analysis is recognised, de-cited and never auto-cited; artifacts drop."""
    from app.monitors import deep_research as dr
    fails: list[str] = []
    _ok(dr._is_analytical("This incident forces an urgent reassessment of NATO's defensive doctrine."),
        "analytical detector lost the 'forces a reassessment' shape", fails)
    _ok(not dr._is_analytical("DFDV's stock jumped 8.03% to close at $5.38 on August 31."),
        "a figure sentence was classed as analysis", fails)
    out, n = dr._decite_analysis("Brent rose 4% to $92 (reuters.com). This move signals a broader repricing (reuters.com).")
    _ok(n == 1 and "repricing." in out and "$92 (reuters.com)" in out, "de-cite analysis regressed", fails)
    out, n = dr._drop_artifact_sentences("Fine (cnbc.com). The launched a **$10 million** plan (cnbc.com).")
    _ok(n == 1 and "launched a" not in out, "artifact canary regressed", fails)
    _ok(dr._strip_pseudo_citations("x (deep analysis: y).")[1] == 1, "pseudo-citation strip regressed", fails)
    return (not fails, "; ".join(fails) or "analysis detection, de-cite, artifact drop, pseudo-citation strip OK")


def check_kg_eviction() -> tuple[bool, str]:
    """A young fact is never cap-evicted; the FTS keyword arm finds a fact past
    any candidate window."""
    from app.config import config as _cfg
    from app.core.kg import KnowledgeGraph
    from app.database import SafeDB
    fails: list[str] = []
    d = tempfile.mkdtemp()
    db = SafeDB(os.path.join(d, "kg.db"))
    db.init_schema()
    kg = KnowledgeGraph(db)
    old_cap = _cfg.MAX_KG_FACTS
    try:
        _cfg.update(MAX_KG_FACTS=10)
        for i in range(8):
            db.execute("INSERT INTO kg_facts (subject, predicate, object, confidence, created_at, valid_from, valid_to) "
                       "VALUES (?, 'is_a', 'old', 0.8, datetime('now','-40 days'), datetime('now','-40 days'), NULL)",
                       (f"OldFact{i}",))
        for i in range(4):
            db.execute("INSERT INTO kg_facts (subject, predicate, object, confidence, created_at, valid_from, valid_to) "
                       "VALUES (?, 'is_a', 'new', 0.8, datetime('now','-1 days'), datetime('now','-1 days'), NULL)",
                       (f"YoungFact{i}",))
        kg._prune()
        live = {r["subject"] for r in db.fetchall("SELECT subject FROM kg_facts WHERE valid_to IS NULL")}
        _ok(all(f"YoungFact{i}" in live for i in range(4)), "a young fact was cap-evicted", fails)
        _ok(len(live) == 10, f"live count after prune {len(live)} != cap 10", fails)
        db.execute("INSERT INTO kg_facts (subject, predicate, object, confidence, valid_to) "
                   "VALUES ('Zephyrian Quasar Drive', 'produces', 'tachyon pulses', 0.3, NULL)")
        kg2 = KnowledgeGraph(db)
        kg2._get_collection = lambda: None
        got = {f.subject for f in kg2.get_relevant_facts("what does the Zephyrian Quasar Drive produce", limit=8)}
        _ok("Zephyrian Quasar Drive" in got, "FTS keyword arm missed a low-confidence fact", fails)
    finally:
        _cfg.update(MAX_KG_FACTS=old_cap)
        try:
            db.close()
        except Exception:
            pass
    return (not fails, "; ".join(fails) or "young facts protected; FTS keyword arm reaches past the cap")


def check_curiosity_gate() -> tuple[bool, str]:
    """Operator probes never become research; status strings are never banked."""
    from app.core.curiosity import _looks_like_operator_probe
    from app.monitors.heartbeat_loop import _provisional_acceptable
    fails: list[str] = []
    _ok(_looks_like_operator_probe("in one word, are you operational"), "probe detector regressed", fails)
    _ok(not _looks_like_operator_probe("How does the EU AI Act classify general-purpose models?"),
        "probe detector rejects a real question", fails)
    _ok(not _provisional_acceptable("Can X?", "[provisional] no change | last: 2026-08-18T15:59:00Z"),
        "status string accepted as a provisional answer", fails)
    _ok(_provisional_acceptable("Can CV-QKD keep its key-rate edge?",
                                "On August 28, 2026 nature.com reported CV-QKD keeps its key-rate edge on metro links."),
        "dated, sourced provisional answer rejected", fails)
    return (not fails, "; ".join(fails) or "probe gate and provisional gate OK")


def check_tool_scoping() -> tuple[bool, str]:
    """Research generations never see side-effect tools; the taint sets hold."""
    from app.core import access_tiers
    from app.core.brain import _TAINT_STRIPPED_TOOLS, _WEB_INGEST_TOOLS
    fails: list[str] = []
    wl = access_tiers.RESEARCH_TOOLS
    _ok(not (wl & {"shell_exec", "file_ops", "desktop", "tool_create"}), "research whitelist leaks side-effect tools", fails)
    access_tiers.set_tool_whitelist(None)
    with access_tiers.research_scope():
        _ok(not access_tiers.is_tool_allowed("shell_exec"), "research scope allows shell_exec", fails)
    _ok(access_tiers.get_tool_whitelist() is None, "research scope did not restore the whitelist", fails)
    _ok({"web_search", "http_fetch", "browser"} <= _WEB_INGEST_TOOLS and "shell_exec" in _TAINT_STRIPPED_TOOLS,
        "taint sets regressed", fails)
    return (not fails, "; ".join(fails) or "research whitelist and taint sets OK")


CHECKS = {
    "resolver_window": check_resolver_window,
    "priming_key": check_priming_key,
    "digest_canary": check_digest_canary,
    "fact_analysis": check_fact_analysis,
    "kg_eviction": check_kg_eviction,
    "curiosity_gate": check_curiosity_gate,
    "tool_scoping": check_tool_scoping,
}
