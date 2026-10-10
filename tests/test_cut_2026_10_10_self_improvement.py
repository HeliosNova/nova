"""Guard for the 2026-10-10 self-improvement cut (plan task K2).

Removed: the skill layer (SkillStore, external AgentSkills loader, auto/procedure
skill induction, skill export signing), the Lesson Quiz monitor, the persisted
ReflexionStore (success patterns, recurring-failure promotion, reflexion
decay/prune/reindex, the reflexion API + UI), the Skill Quality Monitor, the
conformal-abstention calibration that read the reflexions table, and the
write-only capability_gaps table. Kept: the quality SCORE (assess_quality,
should_use_llm_critique, critique_response) that gates refinement and footers.

These tests pin the removal so nothing re-imports or re-wires the layer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

GONE_MODULES = [
    "app/core/skills.py",
    "app/core/skill_loader.py",
    "app/core/auto_skills.py",
    "app/core/skill_export.py",
    "app/core/calibration.py",
    "scripts/skill_export.py",
    "scripts/migrate_skills.py",
    "scripts/migrate_skills_v2.py",
    "scripts/verify_skills_v2.py",
    "frontend/src/pages/learning/SkillsSection.tsx",
    "frontend/src/pages/learning/ReflexionsSection.tsx",
]

GONE_FLAGS = [
    "ENABLE_AUTO_SKILL_CREATION", "REQUIRE_SIGNED_SKILLS", "SKILLS_DIR",
    "MAX_SKILLS_CHECK", "ENABLE_SEMANTIC_SKILL_MATCHING", "SKILL_SEMANTIC_THRESHOLD",
    "SKILL_EMA_ALPHA", "SKILL_STALE_DAYS", "SKILL_SUCCESS_QUALITY",
    "MAX_REFLEXIONS", "MAX_REFLEXIONS_IN_PROMPT", "MAX_SUCCESS_PATTERNS_IN_PROMPT",
    "REFLEXION_DECAY_DAYS", "REFLEXION_DECAY_AMOUNT",
    "REFLEXION_FAILURE_THRESHOLD", "REFLEXION_SUCCESS_THRESHOLD",
    "ENABLE_CONFORMAL_ABSTENTION",
]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("rel", GONE_MODULES)
def test_module_is_gone(rel):
    assert not (ROOT / rel).exists(), f"{rel} should have been removed"


@pytest.mark.parametrize("flag", GONE_FLAGS)
def test_flag_is_gone(flag):
    for rel in ("app/config.py", ".env.example", "app/api/system.py"):
        assert not re.search(rf"\b{flag}\b", _read(rel)), f"{flag} still in {rel}"


def test_no_module_imports_the_layer():
    pat = re.compile(r"app\.core\.(skills|skill_loader|auto_skills|skill_export|calibration)\b")
    offenders = []
    for py in list((ROOT / "app").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        if pat.search(py.read_text(encoding="utf-8", errors="replace")):
            offenders.append(str(py.relative_to(ROOT)))
    assert offenders == [], offenders


def test_reflexion_module_keeps_scoring_and_drops_the_store():
    import app.core.reflexion as r
    assert hasattr(r, "assess_quality")
    assert hasattr(r, "should_use_llm_critique")
    assert hasattr(r, "critique_response")
    for gone in ("ReflexionStore", "Reflexion", "check_recurring_failures",
                 "sweep_recurring_failures", "_normalize_words"):
        assert not hasattr(r, gone), gone


def test_services_has_no_skill_or_reflexion_slots():
    from app.core.brain import Services
    svc = Services()
    for attr in ("skills", "reflexions", "external_skills"):
        assert not hasattr(svc, attr), attr


def test_brain_has_no_skill_or_reflexion_store_wiring():
    src = _read("app/core/brain.py")
    for needle in ("matched_skill", "svc.skills", "svc.reflexions", "skills_text",
                   "reflexions_text", "success_pattern", "external_skills",
                   "_detect_capability_gap", "capability_gaps", "calibration import",
                   "format_skills_for_prompt"):
        assert needle not in src, needle


def test_prompt_builder_has_no_skill_or_reflexion_blocks():
    import inspect
    from app.core.prompt import build_system_prompt
    params = inspect.signature(build_system_prompt).parameters
    for gone in ("skills_text", "reflexions", "success_patterns",
                 "external_skills_text", "matched_external_skill_text"):
        assert gone not in params, gone
    src = _read("app/core/prompt.py")
    assert "format_skills_for_prompt" not in src
    assert "Learned Skills" not in src
    assert "[R15]" not in src


def test_heartbeat_has_no_quiz_or_skill_quality():
    from app.monitors.heartbeat_loop import HeartbeatLoop
    assert "quiz" not in HeartbeatLoop._CHECK_DISPATCH
    assert "skill_quality" not in HeartbeatLoop._CHECK_DISPATCH
    assert not hasattr(HeartbeatLoop, "_execute_quiz")
    assert not hasattr(HeartbeatLoop, "_execute_skill_quality_check")
    src = _read("app/monitors/heartbeat_loop.py")
    assert "_UNQUIZZABLE_ANSWER_RE" not in src
    assert '"skill_test"' not in src


def test_monitor_catalog_has_no_quiz_or_skill_monitors():
    import app.monitors.monitor_store as ms
    for name in ("Lesson Quiz", "Skill Quality Monitor", "Skill Validation"):
        assert name not in ms._SYSTEM_CATEGORY_NAMES, name
    for ct in ("quiz", "skill_quality"):
        assert ct not in ms._SYSTEM_CATEGORY_CHECK_TYPES, ct
    src = _read("app/monitors/monitor_store.py")
    assert '"name": "Lesson Quiz"' not in src
    assert '"Skill Quality Monitor", "check_type"' not in src
    from app.api.monitors import _ALL_CHECK_TYPES
    assert "quiz" not in _ALL_CHECK_TYPES
    assert "skill_validation" not in _ALL_CHECK_TYPES


def test_schema_has_no_skills_reflexions_or_capability_gaps_tables():
    src = _read("app/database.py")
    for ddl in ("CREATE TABLE IF NOT EXISTS skills", "CREATE TABLE IF NOT EXISTS reflexions",
                "CREATE TABLE IF NOT EXISTS capability_gaps", "ALTER TABLE skills",
                "ALTER TABLE reflexions", "idx_reflexions", "idx_skills"):
        assert ddl not in src, ddl
    # Migration 39 retires the monitors with a dump, so the live rows are kept
    # on disk before they go.
    assert "if 39 not in applied" in src
    assert "retired_monitors" in src


def test_pathway_registry_dropped_skills_and_reflexions():
    from app.monitors.pathways import PATHWAYS
    names = {p.name for p in PATHWAYS}
    assert "skills" not in names
    assert "reflexions" not in names


def test_api_schema_dropped_skill_fields():
    from app import schema
    assert not hasattr(schema, "SkillInfo")
    assert "total_skills" not in schema.LearningMetrics.model_fields
    assert "skills" not in schema.StatusResponse.model_fields
    assert "reflexions" not in schema.StatusResponse.model_fields
    assert "skill_used" not in schema.ChatResponse.model_fields
    assert "reflexions_used" not in schema.ChatResponse.model_fields


def test_learning_api_has_no_skill_or_reflexion_routes():
    from app.api.learning import router
    paths = {r.path for r in router.routes}
    assert not any("skills" in p or "reflexions" in p for p in paths), paths


def test_suite_yaml_has_no_skill_tasks():
    src = _read("evals/suite.yaml")
    for needle in ("seed_skill", "skill_matched", "skill_not_matched",
                   "category: skill-match", "category: semantic-match"):
        assert needle not in src, needle


def test_eval_harness_dropped_skill_assertions():
    from app.monitors.eval_harness import check_assertion, CategoryMetrics
    assert check_assertion({"type": "skill_matched"}, "x", [], None) is False
    assert not hasattr(CategoryMetrics(category="c", total=0, passed=0, pass_rate=0.0,
                                       latency_p50=0.0, latency_p95=0.0), "hit_rate")
    src = _read("app/monitors/eval_harness.py")
    assert "skill_used" not in src
    assert "_seed_skills" not in src


def test_prompt_optimizer_dropped_skill_module():
    from app.core.prompt_optimizer import _SELF_MOD_ALLOWED_MODULES, _DRIFT_RULES
    from app.core.prompt_optimizer_baselines import MODULE_BASELINES
    assert "skill_extraction_prompt" not in _SELF_MOD_ALLOWED_MODULES
    assert "skill_extraction_prompt" not in MODULE_BASELINES
    assert not any(r[0] in ("skill-match", "semantic-match") for r in _DRIFT_RULES)


def test_dream_dropped_reflexion_and_skill_substeps():
    from app.core.dream import DreamConsolidator, GatherSignals, ConsolidationResult, DreamInventory
    for gone in ("_prune_reflexions", "_promote_reflexions", "_disable_weak_skills",
                 "_mine_dpo_pairs"):
        assert not hasattr(DreamConsolidator, gone), gone
    for gone in ("low_quality_reflexions", "high_quality_unpromoted",
                 "recurring_failure_clusters", "weak_skills", "quality_extremes"):
        assert gone not in GatherSignals.__dataclass_fields__, gone
    for gone in ("reflexions_promoted", "reflexions_pruned", "skills_disabled",
                 "dpo_pairs_generated"):
        assert gone not in ConsolidationResult.__dataclass_fields__, gone
    for gone in ("reflexion_count", "reflexions_near_limit", "skill_count"):
        assert gone not in DreamInventory.__dataclass_fields__, gone


def test_context_detail_dropped_reflexion_category():
    from app.tools.context_detail import ContextDetailTool
    assert "reflexion" not in ContextDetailTool.input_schema["properties"]["category"]["enum"]


def test_frontend_dropped_skills_and_reflexions():
    api = _read("frontend/src/lib/api.ts")
    for fn in ("getSkills", "toggleSkill", "deleteSkill", "getReflexions"):
        assert fn not in api, fn
    types = _read("frontend/src/lib/types.ts")
    for t in ("SkillInfo", "ReflexionInfo", "total_skills", "skill_used", "ENABLE_AUTO_SKILL_CREATION"):
        assert t not in types, t
    store = _read("frontend/src/lib/store.ts")
    assert "skills" not in store
    idx = _read("frontend/src/pages/learning/index.ts")
    assert "SkillsSection" not in idx and "ReflexionsSection" not in idx


def test_docs_mark_the_removal():
    doc = _read("CLAUDE.md")
    assert "Removed 2026-10-10: the self-improvement layer" in doc
    for rel in GONE_MODULES[:5]:
        assert rel in doc, f"{rel} should be listed as removed in CLAUDE.md"
