"""Regressions for a liveness sweep: code that exists but can never run.

This repo's own history says this is its highest-yield defect class:
  * decay_stale_skills   -- defined, never called (2026-08-17)
  * ENABLE_AUTO_FINETUNE -- flag read nowhere, inert for months
  * ENABLE_MULTI_AGENT   -- one shared flag silently killed deliberation 10 days
  * _CHECK_DISPATCH      -- a duplicate dict key killed Dream Consolidation

An AST sweep over app/ (dead functions / inert config / duplicate literal keys /
unreachable statements) turned up three things worth locking down.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
APP = REPO / "app"


class TestWebSearchTimeoutIsWired:
    """WEB_SEARCH_TIMEOUT was declared, env-allow-listed, and read NOWHERE.

    The 2026-07-08 audit found it and only annotated it
    "INERT ... setting this is a no-op", leaving it dead for ~7 weeks. Anyone
    tuning it in .env got silence. Now native_search sources SEARCH_TIMEOUT
    from it.
    """

    def test_search_timeout_reads_config(self):
        from app.config import config
        from app.tools import native_search

        assert native_search.SEARCH_TIMEOUT == config.WEB_SEARCH_TIMEOUT, (
            "SEARCH_TIMEOUT must come from config.WEB_SEARCH_TIMEOUT — a "
            "hardcoded constant makes the documented env var a silent no-op"
        )

    def test_default_preserves_the_fail_fast_value(self):
        """Wiring it must not change behaviour.

        The constant it replaced was 12.0 and that value is load-bearing:
        search fails fast so one hung aggregator engine cannot stall a whole
        concurrent gather (searxng/settings.yml.example documents the 12s
        contract). The old config default was 35.0, so wiring without moving
        the default would have near-tripled every search timeout.
        """
        from app.config import config

        assert config.WEB_SEARCH_TIMEOUT == pytest.approx(12.0), (
            "default must stay 12.0 so wiring the knob is behaviour-neutral"
        )

    def test_body_fetches_keep_the_longer_timeout(self):
        """Only SEARCH queries fail fast; article-body reads may be slow."""
        from app.tools import native_search

        assert native_search.DEFAULT_TIMEOUT == pytest.approx(30.0)
        assert native_search.DEFAULT_TIMEOUT > native_search.SEARCH_TIMEOUT


class TestSkillEmbedUsesUpsert:
    """_embed_skill did delete-then-add, so every NEW skill issued a delete for
    an id that was never indexed. Chroma records the no-op delete in its log and
    REPLAYS it on every PersistentClient open, emitting bursts of
    "Delete of nonexisting embedding ID: skill_N" for long-dead ids (observed
    skill_2..skill_45 on a store whose live ids are 68-99)."""

    def test_embed_skill_has_no_blind_delete(self):
        src = (APP / "core" / "skills.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "_embed_skill"),
            None,
        )
        assert fn is not None, "_embed_skill disappeared"
        body = ast.unparse(fn)
        assert "collection.upsert(" in body, (
            "_embed_skill must upsert; delete-then-add pollutes the chroma log"
        )
        assert "collection.delete(" not in body, (
            "the unconditional delete is what logged a no-op for every new "
            "skill — upsert already handles the replace case"
        )


class TestDeadCitationGateStaysDeleted:
    """_domain_study_passes_citation_gate demanded >=2 'Source:' citations.

    git log -S showed exactly ONE commit ever touched it: the one that created
    it in v1.6.0. Measured against 120 real digests it passed 0%, with 97.5%
    failing on that first check because the literal string "Source:" appears
    ZERO times in all 120 — real digests cite via a
    "_read 20 sources: apnews.com, ..._" header. Its hedging regex also
    hardcoded apr/may/jun, so it could only fire in one season.

    Same class as the freshness rubric's never-emitted date lines and the
    format rubric's never-emitted numbered items: output judged against a
    structure it never had. Wiring it as-written would suppress every digest.
    """

    def test_gate_and_its_helpers_are_gone(self):
        from app.monitors import heartbeat_loop

        for name in ("_domain_study_passes_citation_gate", "_CITATION_RE",
                     "_HEDGE_RE", "_DATE_RE_GATE"):
            assert not hasattr(heartbeat_loop, name), (
                f"{name} is back. It passed 0/120 real digests; re-wiring it "
                f"as written would drop 100% of output. If a citation gate is "
                f"wanted, calibrate it against the real '_read N sources:' "
                f"format first."
            )

    def test_no_caller_was_left_dangling(self):
        hits = []
        for p in APP.rglob("*.py"):
            if re.search(r"_domain_study_passes_citation_gate",
                         p.read_text(encoding="utf-8")):
                hits.append(str(p.relative_to(REPO)))
        assert not hits, f"dangling reference to the deleted gate in {hits}"


class TestNoDuplicateDispatchKeys:
    """A duplicate key in _CHECK_DISPATCH silently killed scheduled Dream
    Consolidation (it ran the dossier cycle instead) — Python keeps the LAST
    binding, so the earlier handler becomes unreachable with no error."""

    @pytest.mark.parametrize("relpath", [
        "monitors/heartbeat_loop.py",
        "monitors/heartbeat.py",
    ])
    def test_dispatch_dicts_have_unique_keys(self, relpath):
        path = APP / relpath
        if not path.exists():
            pytest.skip(f"{relpath} not present")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        problems = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            seen = {}
            for k in node.keys:
                if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                    continue
                if k.value in seen:
                    problems.append(
                        f"{relpath}:{k.lineno} duplicate {k.value!r} "
                        f"(first at {seen[k.value]})")
                else:
                    seen[k.value] = k.lineno
        assert not problems, "duplicate dispatch keys: " + "; ".join(problems)
