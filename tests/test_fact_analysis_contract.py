"""Fact/analysis contract for digests (audit 2026-09-01).

Measured: the synthesis prompt demanded a citation on EVERY sentence, so the
27B cited its own analysis; the deterministic auto-citer then glued a host onto
any factual-looking sentence; MiniCheck judged 78% of the first 24 cited
sentences unsupported and dropped half — 52% of the drops were analysis
("This incident forces an urgent reassessment…"); sentences after the 24th and
every '(deep analysis: …)' pseudo-citation shipped unchecked; regex excision
left "the launched a $10 million…" and "solidifying 's position" in shipped
digests; the 27B verify pass changed 0 sentences in 17 of 20 runs.
"""
from __future__ import annotations

import inspect

from app.monitors import deep_research as dr


def test_synthesis_prompt_separates_facts_from_analysis():
    src = inspect.getsource(dr._synthesize_from_evidence)
    assert "EVERY sentence and bullet MUST cite" not in src
    assert "Every FACTUAL sentence" in src and "carries NO citation" in src


def test_analytical_detector_catches_the_live_drop_shapes():
    assert dr._is_analytical("This incident forces an urgent reassessment of NATO's defensive doctrine.")
    assert dr._is_analytical("The G20 incident in Asheville is not merely a diplomatic snub but a stress test for the alliance.")
    assert dr._is_analytical("The broader implication is a decoupling of enterprise strategy from subscription APIs.")
    assert dr._is_analytical("This escalation raises the prospect of a wider regional conflict.")
    # figures are facts, never analysis
    assert not dr._is_analytical("DFDV's stock jumped 8.03% to close at $5.38 on August 31, having risen 110%.")
    assert not dr._is_analytical("Airbus delivered 63 aircraft in August (reuters.com).")


def test_pseudo_citations_are_stripped():
    text = ("More than one-sixth of the population has been displaced (deep analysis: Historical Context). "
            "A 20% levy applies to cargo (deep analysis:). Real fact (reuters.com).")
    out, n = dr._strip_pseudo_citations(text)
    assert "deep analysis" not in out and n == 2
    assert "(reuters.com)" in out


def test_analysis_is_decited_not_checked():
    text = ("Brent rose 4% to $92 on Monday (reuters.com). "
            "This move signals a broader repricing of Gulf risk (reuters.com).")
    out, n = dr._decite_analysis(text)
    assert n == 1
    assert "Brent rose 4% to $92 on Monday (reuters.com)." in out
    assert "This move signals a broader repricing of Gulf risk." in out


def test_auto_citer_never_cites_analysis():
    arts = [("Brent risk", "https://reuters.com/a",
             "This move signals a broader repricing of Gulf risk according to traders in London and Dubai.")]
    text = "This move signals a broader repricing of Gulf risk in London and Dubai."
    out, n = dr._cite_uncited_sentences(text, arts)
    assert n == 0 and "(reuters.com)" not in out


def test_artifact_sentences_are_dropped_whole():
    text = ("Real sentence one (cnbc.com). The launched a **$10 million Responsible AI Initiative** (cnbc.com). "
            "Another real sentence (bbc.com). The foundation is solidifying 's position in the market (cnbc.com). "
            "A debate the is attempting to mediate continues (cnbc.com).")
    out, n = dr._drop_artifact_sentences(text)
    assert n == 3
    assert "Real sentence one (cnbc.com)." in out and "Another real sentence (bbc.com)." in out
    assert "launched a" not in out and "'s position" not in out and "the is attempting" not in out
    head, n2 = dr._drop_artifact_sentences("**Strategic Decoupling of the **\nBody (deep analysis:) text.")
    assert n2 >= 1 and "(deep analysis:)" not in head


def test_entailment_gate_covers_the_whole_digest():
    sig = inspect.signature(dr._entailment_gate)
    assert sig.parameters["max_checks"].default >= 48


def test_overview_verify_pass_is_retired():
    src = inspect.getsource(dr._synthesize_from_evidence)
    assert "_verify_prompt(" not in src, "the inert 27B verify pass must not run in the overview chain"
    assert "verify pass retired" in src
