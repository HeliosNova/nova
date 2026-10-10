"""Cut round two, part 1 (2026-10-10): the agent tool layer.

The record behind it (2026-10-09 maps, live DB): custom_tools held 2 rows, both
disabled, last created 2026-08-23; auto_tool_candidates last wrote 2026-08-31
and its only synthesiser was retired 2026-09-01; agent_workspace last wrote
2026-08-28 (58 rows over 98 runs); the exec sidecar served code_exec for two
eval probes and nothing else; no monitor of check_type 'command' ever existed;
delegate never reached eval traffic (stripped for ephemeral) and the chat that
could have used it was removed 2026-10-09. The deliberation route, the
workspace and the tool-creation loop were all interactive-agent machinery on
a system whose owner does not chat with it.

Gone: the exec sidecar (exec_runner, code_exec, code_verify, shell_exec and the
nova-exec service), custom/auto tools (custom_tools, auto_tools, tool_triggers,
self_improvement), and the agent loop (agent_loop, agent_workspace, delegate,
the /api/agent route). The three reasoning predicates brain used from
agent_loop live on in app/core/reasoning_cues.py: they gate math-tool routing,
the thinking auto-gate and best-of-N on the generic think() path.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import app as _app_pkg
from app import config

ROOT = Path(_app_pkg.__file__).resolve().parents[1]

GONE_MODULES = [
    "app/exec_runner.py", "app/tools/code_exec.py", "app/tools/code_verify.py", "app/tools/shell_exec.py",
    "app/core/custom_tools.py", "app/core/auto_tools.py", "app/core/tool_triggers.py",
    "app/core/self_improvement.py",
    "app/core/agent_loop.py", "app/core/agent_workspace.py", "app/tools/delegate.py", "app/api/agent.py",
]

GONE_FLAGS = [
    "ENABLE_SHELL_EXEC", "CODE_EXEC_TIMEOUT", "SHELL_EXEC_TIMEOUT",
    "ENABLE_CUSTOM_TOOLS", "ENABLE_AUTONOMOUS_TOOL_CREATION", "AUTO_TOOL_CREATION_THRESHOLD",
    "MAX_CUSTOM_TOOLS", "MAX_CUSTOM_TOOL_CODE_LENGTH",
    "ENABLE_DELEGATION", "MAX_DELEGATION_DEPTH", "ENABLE_DELIBERATION",
]

CUT_TOOL_NAMES = ('"code_exec"', '"shell_exec"', '"code_verify"', '"tool_create"', '"delegate"')


@pytest.mark.parametrize("rel", GONE_MODULES)
def test_the_module_is_gone(rel):
    assert not (ROOT / rel).exists(), rel


@pytest.mark.parametrize("flag", GONE_FLAGS)
def test_the_setting_is_gone(flag):
    assert not hasattr(config, flag), flag


@pytest.mark.skipif(not (ROOT / "docker-compose.yml").exists(), reason="not shipped in the image")
def test_compose_carries_none_of_it():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for needle in ("exec-runner", "nova-exec", "exec_queue", "ENABLE_SHELL_EXEC", "app.exec_runner"):
        assert needle not in text, needle


def test_no_tool_list_names_the_cut_tools():
    hits = []
    for rel in ("app/core/brain.py", "app/core/prompt.py", "app/core/trust.py",
                "app/core/curiosity.py", "app/core/reflexion.py", "app/tools/file_ops.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        for name in CUT_TOOL_NAMES:
            if name in src:
                hits.append(f"{rel}: {name}")
    assert not hits, hits


def test_the_prompt_no_longer_teaches_the_cut_tools():
    src = (ROOT / "app/core/prompt.py").read_text(encoding="utf-8")
    for needle in ("code_exec", "shell_exec", "code_verify", "tool_create", "delegate"):
        assert needle not in src, needle


def test_main_wires_none_of_it():
    src = (ROOT / "app/main.py").read_text(encoding="utf-8")
    for needle in ("CodeExecTool", "CodeVerifyTool", "ShellExecTool", "DelegateTool",
                   "CustomToolStore", "DynamicTool", "agent_router", "app.api.agent"):
        assert not re.search(needle, src), needle


def test_the_reasoning_predicates_moved_to_a_neutral_module():
    from app.core import reasoning_cues
    assert reasoning_cues._is_math_logic_word_problem(
        "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?")
    assert reasoning_cues._is_hard_reasoning_query("Compare Python and JavaScript for backend development")
    assert reasoning_cues._is_deliberation_query("Walk me through how you would design a rate limiter")
    assert not reasoning_cues._is_hard_reasoning_query("hi")
    src = (ROOT / "app/core/brain.py").read_text(encoding="utf-8")
    assert "app.core.agent_loop" not in src
    assert "app.core.agent_workspace" not in src
    assert "from app.core.reasoning_cues import" in src


def test_services_has_no_custom_tools_field():
    from app.core.brain import Services
    assert "custom_tools" not in Services.__dataclass_fields__


def test_dispatch_has_no_command_type():
    from app.monitors.heartbeat_loop import HeartbeatLoop
    assert "command" not in HeartbeatLoop._CHECK_DISPATCH
    from app.api.monitors import _USER_CHECK_TYPES
    assert "command" not in _USER_CHECK_TYPES


def test_the_suite_asserts_no_cut_tool():
    text = (ROOT / "evals/suite.yaml").read_text(encoding="utf-8")
    assert "value: code_exec" not in text
    assert "deliberation-route" not in text
