"""The dossier never reaches the prompt the briefing is written from (2026-09-03).

A paired A/B over 16 frozen topics measured the PRIOR UNDERSTANDING injection
as no gain and a real cost: overall 0.838 primed vs 0.844 unprimed (primed won
7/16, p=0.61) and deterministic fact support 0.642 vs 0.690 (primed won 4/16,
p=0.099, down on 12 of 16 topics). Prior context was being restated as though
today's sources carried it. The dossier is still LOADED, because the
out-of-band KNOWN-VS-NEW count runs on the finished briefing and cannot change
it — that separation is the whole reason retiring the injection was cheap.
"""
from __future__ import annotations

import inspect

from app.monitors import deep_research as dr


def test_shared_prefix_takes_no_dossier():
    """The evidence pack every chain stage shares is evidence only."""
    params = inspect.signature(dr._common_context).parameters
    assert "prior_block" not in params
    pack = dr._common_context("September 03, 2026", "finance", "ANALYSES\n", "FINDINGS")
    assert "PRIOR UNDERSTANDING" not in pack
    assert "FINDINGS" in pack and "ANALYSES" in pack


def test_no_priming_paths_survive_in_the_synthesis_source():
    src = inspect.getsource(dr._synthesize_from_evidence)
    assert "prior_block" not in src
    # the novelty instruction that rode with the injection is gone too
    assert "LEAD with what is genuinely" not in src
    assert "CONTRADICTS PRIOR UNDERSTANDING" not in src
    # the dossier is still loaded, and only for the measurement
    assert "priming_excerpt" in src and "prior_text" in src
    assert "_count_known_vs_new(prior_text" in src


def test_the_measurement_still_receives_the_dossier_and_labels_it():
    src = inspect.getsource(dr._count_known_vs_new)
    assert "PRIOR UNDERSTANDING" in src, "the count must label the text it is given"
    assert "prior_text" in src
    assert "final[:9000]" in src, "it must judge the FINISHED briefing"


def test_measurement_cannot_reach_the_draft():
    """Ordering guard: the count is called after the briefing is final."""
    src = inspect.getsource(dr._synthesize_from_evidence)
    assert src.index("_best_synthesis(") < src.index("_count_known_vs_new("), \
        "KNOWN-VS-NEW must run after synthesis, never before it"


def test_dossier_lookup_helpers_are_still_used():
    """resolve_domain_dkey / priming_excerpt stay: the measurement needs them."""
    from app.core.dossiers import priming_excerpt, resolve_domain_dkey
    assert resolve_domain_dkey("AI/ML", "Domain Study: AI and ML") == "ai-and-ml"
    body = ("## Current understanding\nRubin ships.\n"
            "## How we got here\nlong history " * 5 + "\n## Open questions\n- What next?\n")
    out = priming_excerpt(body)
    assert out.startswith("## Current understanding")
