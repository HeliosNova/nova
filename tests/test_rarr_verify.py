"""RARR revise-instead-of-delete (Phase 2, gated by ENABLE_RARR) — deterministic
coverage of the verify-pass prompt builder + the sentence counter that measures
what the verify pass deletes. The live A/B (RACE/FACT harness) covers quality."""
from __future__ import annotations

from app.monitors import deep_research as dr


def test_legacy_overview_prompt_unchanged_delete_semantics():
    p = dr._verify_prompt("EVIDENCE-X", "DRAFT-Y", overview=True, rarr=False)
    assert "DELETE any sentence or bullet that lacks an inline outlet citation" in p
    assert "MINIMALLY EDIT" not in p
    assert "EVIDENCE-X" in p and "DRAFT-Y" in p


def test_rarr_overview_prompt_revises_instead_of_deleting():
    p = dr._verify_prompt("EVIDENCE-X", "DRAFT-Y", overview=True, rarr=True)
    assert "MINIMALLY EDIT" in p
    assert "ONLY if the findings contain nothing supporting its core claim" in p
    # the hard delete-uncited rule is replaced by re-cite-or-delete
    assert "an uncited claim is where fabrication hides" not in p
    assert "append the correct outlet citation" in p
    # structure-preservation rule stays in both modes
    assert "preserve every sourced detail" in p
    assert "EVIDENCE-X" in p and "DRAFT-Y" in p


def test_briefing_prompt_both_modes():
    legacy = dr._verify_prompt("EV", "DR", overview=False, rarr=False)
    rarr = dr._verify_prompt("EV", "DR", overview=False, rarr=True)
    assert "Keep only supported claims" in legacy and "MINIMALLY EDIT" not in legacy
    assert "MINIMALLY EDIT" in rarr and "core claim" in rarr


def test_count_content_sentences_skips_headers_and_metadata():
    text = (
        "## 🌐 finance — domain overview\n"
        "_read 6 sources: apnews.com, cnbc.com · 6 facts · July 05_\n"
        "**Secondary Developments**\n"
        "* The Federal Reserve held rates at 3.5% to 3.75% citing sticky services inflation (reuters.com). "
        "Markets priced in one further cut by December after the statement (cnbc.com).\n"
        "* Oracle shares fell 11 percent after cloud revenue guidance missed expectations (cnbc.com).\n"
    )
    # 3 content sentences; header/metadata/bold-label lines contribute 0, and the
    # 3.5% decimal must not split a sentence.
    assert dr._count_content_sentences(text) == 3
    assert dr._count_content_sentences("") == 0


def test_count_content_sentences_measures_verify_delta():
    draft = ("* Alpha Corp won a $2 billion contract for satellite launch services (reuters.com). "
             "Analysts called the award a turning point for the smallsat sector (cnbc.com).\n")
    final = "* Alpha Corp won a $2 billion contract for satellite launch services (reuters.com).\n"
    assert dr._count_content_sentences(draft) - dr._count_content_sentences(final) == 1


def test_enable_rarr_flag_default_off_and_mutable():
    from app.config import config, _MUTABLE_FIELDS
    assert config.ENABLE_RARR is False          # default off pending the measured trial
    assert "ENABLE_RARR" in _MUTABLE_FIELDS     # flippable via /data/config_overrides.json
