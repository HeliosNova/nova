"""Context-integrity fixes (2026-07-08).

Three defects isolated by the post-harness-fix kg-retrieval eval run:

1. The facts-first gate could never fire for single-token proper-noun
   subjects ("Nimbus") — the ≥2-distinctive-tokens rule structurally
   excluded them (kg_runs_paraphrase failed with its fact in-prompt).
2. Silent num_ctx truncation had no detector. Ollama reports the prompt
   tokens it actually processed, so prompt_tokens >= num_ctx is
   deterministic PROOF of truncation — now an ERROR-level tripwire.
3. (No test here) _CONTEXT_GATHER_TIMEOUT_S raised 5→15s: semantic arms
   embed through Ollama and queue behind 27B digest generation; 5s
   silently zeroed knowledge injection under load.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.brain import _kg_answers_query


class TestFactsFirstGate:
    KG_MULTI = "[2026-07-07] Vertex Dynamics Labs makes the Halcyon Probe Engine"
    KG_SINGLE = "[2026-07-07] Nimbus is led by Petra Kovacs"

    def test_multi_token_subject_fires(self):
        assert _kg_answers_query(
            "Who makes the Vertex Dynamics Labs probe?", self.KG_MULTI
        ) is True

    def test_single_token_proper_noun_subject_fires(self):
        """'Nimbus' — one capitalized ≥5-char token in the query is enough."""
        assert _kg_answers_query("Who runs Nimbus?", self.KG_SINGLE) is True

    def test_single_token_subject_absent_from_query_does_not_fire(self):
        assert _kg_answers_query("Who runs Stratus?", self.KG_SINGLE) is False

    def test_short_single_token_subject_does_not_fire(self):
        # <5 chars is too weak a signal for the single-token path
        assert _kg_answers_query("What is Iris?", "[2026-01-01] Iris is a person") is False

    def test_action_verbs_veto_the_gate(self):
        assert _kg_answers_query("Search for Nimbus news", self.KG_SINGLE) is False
        assert _kg_answers_query("What is the latest on Nimbus?", self.KG_SINGLE) is False

    def test_no_facts_no_gate(self):
        assert _kg_answers_query("Who runs Nimbus?", "") is False


class TestNumCtxTripwire:
    def _provider(self):
        from app.core.providers.ollama import OllamaProvider
        provider = OllamaProvider.__new__(OllamaProvider)
        provider._llm_model = "test-model"
        provider._get_client = MagicMock(return_value=MagicMock())
        return provider

    def _resp(self, prompt_tokens: int) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "fine"},
            "eval_count": 50,
            "prompt_eval_count": prompt_tokens,
        }
        return mock_resp

    @pytest.mark.asyncio
    async def test_prompt_tokens_at_num_ctx_logs_error(self, caplog):
        from app.core.providers import ollama as ollama_mod
        provider = self._provider()
        with patch(
            "app.core.providers.ollama.retry_on_transient",
            new_callable=AsyncMock,
            return_value=self._resp(ollama_mod._CHAT_NUM_CTX),
        ):
            with caplog.at_level(logging.ERROR):
                result = await provider.generate_with_tools(
                    [{"role": "user", "content": "x"}], tools=[]
                )
        assert any("TRUNCATED" in r.message for r in caplog.records)
        assert result.usage["prompt_tokens"] == ollama_mod._CHAT_NUM_CTX

    @pytest.mark.asyncio
    async def test_prompt_below_num_ctx_is_silent(self, caplog):
        from app.core.providers import ollama as ollama_mod
        provider = self._provider()
        with patch(
            "app.core.providers.ollama.retry_on_transient",
            new_callable=AsyncMock,
            return_value=self._resp(ollama_mod._CHAT_NUM_CTX - 1000),
        ):
            with caplog.at_level(logging.ERROR):
                await provider.generate_with_tools(
                    [{"role": "user", "content": "x"}], tools=[]
                )
        assert not any("TRUNCATED" in r.message for r in caplog.records)

    def test_chat_num_ctx_covers_measured_prompt_plus_history(self):
        """The system prompt measures ~15.3k real tokens; num_ctx must leave
        real headroom for 10 history messages + generation on top of it."""
        from app.core.providers.ollama import _CHAT_NUM_CTX
        assert _CHAT_NUM_CTX >= 24576

    def test_system_budget_fits_inside_num_ctx_in_real_tokens(self, monkeypatch):
        """MAX_SYSTEM_TOKENS is enforced with len//4 estimation which
        undercounts real tokens ~35% — the REAL ceiling implied by the
        SHIPPED DEFAULT (not the conftest env pin) must stay inside num_ctx
        with room for history + generation."""
        from app.config import Config
        from app.core.providers.ollama import _CHAT_NUM_CTX
        monkeypatch.delenv("MAX_SYSTEM_TOKENS", raising=False)
        default = Config.__dataclass_fields__["MAX_SYSTEM_TOKENS"].default_factory()
        real_ceiling = default * 1.35
        assert real_ceiling + 6000 <= _CHAT_NUM_CTX + 1  # history+gen headroom
