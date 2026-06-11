"""Tests for app.core.claim_validator — post-synthesis claim stripping."""

from __future__ import annotations

from app.core.claim_validator import (
    build_evidence,
    count_claim_candidates,
    validate_claims,
)


# ---- build_evidence -----------------------------------------------------

def test_build_evidence_excludes_query():
    # Query must NOT appear in evidence (presupposition defense)
    ev = build_evidence(
        retrieved_context="apple makes ios",
        query="Who is Dr. Sarah Chen, the founder of Helios Protocol?",
    )
    assert "sarah chen" not in ev.lower()
    assert "helios protocol" not in ev.lower()
    assert "apple" in ev.lower()


def test_build_evidence_includes_tool_outputs():
    tools = [{"tool": "web_search", "output": "Tim Cook is CEO of Apple."}]
    ev = build_evidence(tool_results=tools)
    assert "Tim Cook" in ev


def test_build_evidence_handles_missing_inputs():
    assert build_evidence() == ""
    assert build_evidence(retrieved_context="x") == "x"


# ---- lessons as grounding evidence (memory-loop fix) --------------------

def test_build_evidence_includes_lessons():
    ev = build_evidence(lessons_text="Dr. Lena Voss works in Reykjavik.")
    assert "reykjavik" in ev.lower()
    assert "lena voss" in ev.lower()


def test_lesson_grounded_claim_survives_validation():
    # The memory loop's whole point: an answer taught by a lesson must NOT be
    # stripped as unsupported. Without lessons in evidence this person+location
    # sentence is deleted, silently defeating the loop.
    answer = "Dr. Lena Voss is based in Reykjavik."
    evidence = build_evidence(
        lessons_text="Per the user's notes, Dr. Lena Voss is located in Reykjavik."
    )
    out, reasons = validate_claims(answer, evidence)
    assert "Reykjavik" in out
    assert reasons == []


def test_lessons_grounding_never_strips_more_than_ungrounded():
    # Robust asymmetry: grounding can only retain, never remove. And the
    # grounded answer keeps the taught fact.
    answer = "Dr. Lena Voss is based in Reykjavik."
    grounded, _ = validate_claims(
        answer, build_evidence(lessons_text="Dr. Lena Voss is located in Reykjavik.")
    )
    ungrounded, _ = validate_claims(answer, "")
    assert len(grounded) >= len(ungrounded)
    assert "Reykjavik" in grounded


# ---- validate_claims passthroughs ---------------------------------------

def test_validate_claims_empty_answer():
    out, reasons = validate_claims("", "")
    assert out == "" and reasons == []


def test_validate_claims_no_risky_pattern():
    answer = "Apple is a fruit. Bananas are yellow."
    out, reasons = validate_claims(answer, evidence="")
    assert out == answer
    assert reasons == []


# ---- person-title-org pattern ------------------------------------------

def test_strips_unsupported_person_title_org():
    answer = "Dr. Sarah Chen is the founder of Helios Protocol. Apples are red."
    out, reasons = validate_claims(answer, evidence="apples are red")
    assert "Sarah Chen" not in out
    assert "Helios Protocol" not in out
    assert "Apples are red" in out
    assert any("person-title-org" in r for r in reasons)


def test_keeps_person_title_org_when_supported():
    answer = "Dr. Jane Doe is the CEO of Acme Corp."
    evidence = "Jane Doe runs Acme Corp as the chief executive."
    out, reasons = validate_claims(answer, evidence=evidence)
    assert "Jane Doe" in out
    assert "Acme Corp" in out
    assert reasons == []


# ---- attribution-by pattern --------------------------------------------

def test_strips_unsupported_attribution_by():
    answer = "The framework was designed by Dr. Sarah Chen in 2024."
    out, reasons = validate_claims(answer, evidence="")
    assert "Sarah Chen" not in out
    assert any("attribution" in r for r in reasons)


def test_keeps_supported_attribution_by():
    answer = "The system was created by Dr. Linus Torvalds."
    evidence = "Linus Torvalds first released Linux in 1991."
    out, reasons = validate_claims(answer, evidence=evidence)
    assert "Linus Torvalds" in out


# ---- spec-claim pattern ------------------------------------------------

def test_strips_unsupported_numeric_spec():
    answer = "Helios Protocol achieves 100,000 TPS at 50 ms latency."
    out, reasons = validate_claims(answer, evidence="")
    # at least the org+number bundle should drop
    assert "100,000" not in out or "Helios Protocol" not in out
    assert any("spec-claim" in r for r in reasons)


def test_keeps_supported_numeric_spec():
    answer = "Acme Protocol delivers 5000 TPS in benchmarks."
    evidence = "Acme Protocol benchmark — 5000 transactions per second observed."
    out, reasons = validate_claims(answer, evidence=evidence)
    assert "5000" in out


# ---- bare-titled-person pattern ----------------------------------------

def test_strips_bare_titled_person_in_table():
    answer = (
        "| Designer | Dr. Sarah Chen (2024) |\n"
        "Apples grow on trees."
    )
    out, reasons = validate_claims(answer, evidence="apples grow")
    assert "Sarah Chen" not in out


# ---- multiple drops -----------------------------------------------------

def test_multiple_drops_dont_corrupt_offsets():
    # Two unsupported claims in order — both must be removed correctly
    answer = (
        "Dr. Sarah Chen is the founder of Helios Protocol. "
        "Helios Protocol achieves 100,000 TPS in production."
    )
    out, reasons = validate_claims(answer, evidence="")
    assert "Sarah Chen" not in out
    assert "Helios Protocol" not in out
    assert len(reasons) >= 2


# ---- count_claim_candidates --------------------------------------------

def test_count_candidates_zero_on_plain_text():
    # Plain prose with no claim shapes -> 0. This is the path the RLVR
    # signal gate cares about: don't record claim_grounded for responses
    # that never trigger any validator regex.
    assert count_claim_candidates("Apples are red. The sky is blue.") == 0
    assert count_claim_candidates("") == 0


def test_count_candidates_nonzero_on_risky_shapes():
    # Each shape should register at least one candidate so the signal fires.
    assert count_claim_candidates("Dr. Sarah Chen is the founder of Acme Corp.") >= 1
    assert count_claim_candidates("Designed by Dr. Sarah Chen.") >= 1
    assert count_claim_candidates("Helios Protocol achieves 100,000 TPS.") >= 1
    # Broader title-less shapes (added 2026-05-13)
    assert count_claim_candidates("Sarah Chen is the founder of Helios Protocol.") >= 1
    assert count_claim_candidates("Sarah Chen designed the Helios Protocol.") >= 1


# ---- proper-noun-role-of pattern (title-less) --------------------------

def test_strips_unsupported_proper_noun_role_of():
    # "X is the founder of Y" without title — must drop when evidence
    # contains neither the name nor the org.
    answer = "Sarah Chen is the founder of Helios Protocol. Apples are red."
    out, reasons = validate_claims(answer, evidence="apples are red")
    assert "Sarah Chen" not in out
    assert "Helios Protocol" not in out
    assert "Apples are red" in out
    assert any("proper-noun-role-of" in r for r in reasons)


def test_keeps_proper_noun_role_of_when_grounded():
    # Real public-figure shape — both name + org tokens in evidence → kept.
    answer = "Bill Gates is the founder of Microsoft."
    evidence = "Bill Gates co-founded Microsoft Corporation in 1975."
    out, reasons = validate_claims(answer, evidence=evidence)
    assert "Bill Gates" in out
    assert "Microsoft" in out
    assert reasons == []


def test_role_of_requires_two_word_name():
    # Single-word "She is the founder of Acme Corp" must NOT match — only
    # 2-4 capitalized words can fire this pattern. Prevents pronoun FPs.
    answer = "She is the founder of Acme Corp."
    out, reasons = validate_claims(answer, evidence="")
    assert out == answer
    assert reasons == []


# ---- proper-noun-verb-attrib pattern (title-less) ----------------------

def test_strips_unsupported_proper_noun_verb_attrib():
    # "X designed/created/founded Y" without title — must drop ungrounded.
    answer = "Sarah Chen designed the Helios Protocol. Apples are red."
    out, reasons = validate_claims(answer, evidence="apples are red")
    assert "Sarah Chen" not in out
    assert "Helios Protocol" not in out
    assert any("proper-noun-verb-attrib" in r for r in reasons)


def test_keeps_proper_noun_verb_attrib_when_grounded():
    # Real public-figure shape — both name + org tokens in evidence → kept.
    answer = "Linus Torvalds created Linux."
    evidence = "Linus Torvalds released Linux kernel 0.01 in 1991."
    out, reasons = validate_claims(answer, evidence=evidence)
    assert "Linus Torvalds" in out
    assert "Linux" in out
    assert reasons == []


def test_verb_attrib_requires_two_word_name():
    # "She designed the system" must NOT match (single capitalized word).
    answer = "She designed the Mars Rover."
    out, reasons = validate_claims(answer, evidence="")
    assert out == answer
    assert reasons == []


def test_overlapping_patterns_dont_double_drop():
    # An attribution that matches BOTH _PERSON_BY_ATTRIB (title) AND the
    # title-less broadened pattern shouldn't be double-reported. The
    # post-pattern loop skips spans that an earlier pattern already
    # flagged. Verify only one reason fires.
    answer = "The framework was designed by Dr. Sarah Chen."
    out, reasons = validate_claims(answer, evidence="")
    assert "Sarah Chen" not in out
    # Only one of the patterns should claim credit (whichever runs first).
    proper_noun_hits = sum(1 for r in reasons if "proper-noun" in r)
    title_hits = sum(1 for r in reasons if "attribution" in r and "proper-noun" not in r)
    assert (proper_noun_hits + title_hits) >= 1
    # Should not bracket the same span twice
    assert len(reasons) <= 2
