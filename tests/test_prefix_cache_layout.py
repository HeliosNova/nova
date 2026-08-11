"""Prefix-cache prompt layout (#34): every synthesis-chain stage (synthesis →
judge → enrich → verify) must START with the byte-identical `_common_context`
evidence pack so llama.cpp reuses its KV cache across the chain — only the
per-stage instructions + draft after it are new tokens. These tests pin the
byte-identity contract; the speed win is measured live via Ollama's
prompt_eval_count on the chain calls."""
from __future__ import annotations

import app.monitors.deep_research as dr


def test_common_context_byte_identical_and_capped():
    a = dr._common_context("July 06, 2026", "finance", "A" * 20000, "E" * 12000)
    b = dr._common_context("July 06, 2026", "finance", "A" * 20000, "E" * 12000)
    assert a == b, "same inputs must produce identical bytes — that's the cache contract"
    assert "A" * dr._COMMON_ANALYSIS_CAP in a and "A" * (dr._COMMON_ANALYSIS_CAP + 1) not in a
    assert "E" * dr._COMMON_EVIDENCE_CAP in a and "E" * (dr._COMMON_EVIDENCE_CAP + 1) not in a


def test_verify_prompt_common_layout_prefix():
    common = dr._common_context("July 06, 2026", "finance", "ANALYSES\n", "FINDINGS")
    for rarr in (False, True):
        p = dr._verify_prompt("EVID-UNUSED", "DRAFT-X", overview=True, rarr=rarr, common=common)
        assert p.startswith(common)
        assert "DRAFT:\nDRAFT-X" in p
        assert "EVID-UNUSED" not in p   # common carries the findings; no duplicate block
    # legacy (no common) layout unchanged — the RARR A/B baseline depends on it
    legacy = dr._verify_prompt("EVID-Y", "DRAFT-X", overview=True, rarr=False)
    assert "SOURCE FINDINGS:\nEVID-Y" in legacy and not legacy.startswith(common)


async def test_enrich_prompt_starts_with_common(monkeypatch):
    captured = {}

    async def fake_invoke(messages, **kw):
        captured["content"] = messages[0]["content"]
        captured.update(kw)
        return ""

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake_invoke)
    common = dr._common_context("July 06, 2026", "tech", "DEEP ANALYSES\n", "FINDINGS")
    out = await dr._enrich_overview("THE-DRAFT", "DEEP ANALYSES\n", common, "tech", model=None)
    assert out == "THE-DRAFT"           # empty generation → draft kept (accept-guard)
    assert captured["content"].startswith(common)
    assert captured["content"].rstrip().endswith("DRAFT:\nTHE-DRAFT")
    # num_ctx raised 12288→16384 (2026-07-09) so the enrich pass can rewrite a
    # full digest without clipping its prompt (part of the truncation fix).
    assert captured["num_ctx"] == 16384


async def test_best_synthesis_merge_shares_context_prefix(monkeypatch):
    """Aggregation-over-selection (task #62): the MERGE pass replaces the
    judge-pick and must reuse the byte-identical evidence prefix (KV cache)."""
    calls = []

    async def fake_invoke(messages, **kw):
        calls.append((messages[0]["content"], kw))
        if "=== DRAFT 1 ===" in messages[0]["content"]:   # the merge call
            return "MERGED briefing aggregating both drafts with enough length"
        return f"CANDIDATE-{len(calls)} body with enough substance to survive strip"

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake_invoke)
    ctx = "CTX-COMMON-PREFIX\n\n"
    out = await dr._best_synthesis("PROMPT", "EVIDENCE", n=2, temps=(0.1, 0.5),
                                   model="m27", context_block=ctx)
    merges = [c for c in calls if "=== DRAFT 1 ===" in c[0]]
    assert len(merges) == 1
    content, kw = merges[0]
    assert content.startswith(ctx)
    assert "EVIDENCE" not in content    # context_block replaces the legacy evidence slice
    # merge must run on the SAME model + ctx size as the gens or the cache is useless
    assert kw.get("num_ctx") == 12288 and kw.get("model") == "m27"
    assert out.startswith("MERGED")


async def test_best_synthesis_without_context_block_uses_evidence_prefix(monkeypatch):
    calls = []

    async def fake_invoke(messages, **kw):
        calls.append((messages[0]["content"], kw))
        if "=== DRAFT 1 ===" in messages[0]["content"]:
            return "MERGED briefing aggregating both drafts with enough length"
        return f"CANDIDATE-{len(calls)} body with enough substance to survive strip"

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake_invoke)
    await dr._best_synthesis("PROMPT", "THE-EVIDENCE", n=2, temps=(0.1, 0.5))
    merges = [c for c in calls if "=== DRAFT 1 ===" in c[0]]
    assert len(merges) == 1 and "SOURCE FINDINGS:\nTHE-EVIDENCE" in merges[0][0]


async def test_best_synthesis_short_merge_falls_back_to_judge(monkeypatch):
    """A refusal/fragment from the merge (<50% of the longest draft) must not
    ship — the legacy judge-pick fallback selects a full candidate."""
    calls = []

    async def fake_invoke(messages, **kw):
        calls.append((messages[0]["content"], kw))
        if "=== DRAFT 1 ===" in messages[0]["content"]:
            return "nope"                       # degenerate merge
        if kw.get("max_tokens") == 6:           # judge fallback
            return "2"
        return f"CANDIDATE-{len(calls)} body with enough substance to survive strip"

    monkeypatch.setattr(dr.llm, "invoke_nothink", fake_invoke)
    out = await dr._best_synthesis("PROMPT", "THE-EVIDENCE", n=2, temps=(0.1, 0.5))
    judge = [c for c in calls if c[1].get("max_tokens") == 6]
    assert len(judge) == 1
    assert out.startswith("CANDIDATE-")
