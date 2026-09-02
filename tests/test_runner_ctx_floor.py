"""Runner context floor (audit follow-up 2026-09-02).

Measured after the WSL memory fix: the 27B was still restarted 92 times in
150 minutes (63 minutes of loading) while only 11 real model swaps happened.
The Ollama log showed the cause — the same weights started with `-c 8192`,
then `-c 16384` 36 seconds later: every stage of the digest chain asked for a
different context size, and Ollama restarts the runner for each. Every
background request now asks for at least the floor, so one runner per model
stays resident.
"""
from __future__ import annotations

import inspect

from app.core.providers import ollama as prov


def test_floor_applies_and_larger_requests_win():
    assert prov._RUNNER_CTX_FLOOR >= 24576
    assert prov._runner_ctx(8192) == prov._RUNNER_CTX_FLOOR
    assert prov._runner_ctx(None) == prov._RUNNER_CTX_FLOOR
    assert prov._runner_ctx(0) == prov._RUNNER_CTX_FLOOR
    assert prov._runner_ctx(32768) == 32768
    assert prov._runner_ctx("nonsense") == prov._RUNNER_CTX_FLOOR


def test_invoke_nothink_always_pins_the_context():
    src = inspect.getsource(prov.OllamaProvider.invoke_nothink) if hasattr(prov, "OllamaProvider") else inspect.getsource(prov)
    assert 'payload["options"]["num_ctx"] = _runner_ctx(num_ctx)' in src
    assert "_effective_ctx = _runner_ctx(num_ctx)" in src


def test_chat_context_is_not_below_the_floor():
    assert prov._CHAT_NUM_CTX >= prov._RUNNER_CTX_FLOOR
