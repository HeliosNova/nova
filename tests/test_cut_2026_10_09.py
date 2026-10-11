"""Cut list, 2026-10-09 (owner: "would it be better to make v2 strictly with
what works" → cut v1 down to what works instead).

The record over the last healthy week (09-22..09-26): 48 query digests
delivered 364 briefings, 68 dossiers revised, 592 storyline events, 25
longform consolidations — the knowing tier is the product. Against that:
40 user chat messages EVER (all August), zero voice calls, zero desktop
actions, zero MCP clients, two custom tools, one goal, and WhatsApp/Signal
never configured. Every one of these is a surface the residency gate, the
startup path and the test suite have to carry for nothing.

This round removes the cleanly separable layer — nothing in the
monitor/digest/dossier/storyline/forecast/KG path imports any of it:
voice, desktop automation (and Xvfb), MCP (client and server), the
background task manager, goals, the WhatsApp and Signal adapters, and the
inbound chat handlers on Discord and Telegram. Delivery on Discord and
Telegram is untouched. The chat API stays for the frontend, and the brain
stays because every query-type monitor and the eval harness run through it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import app as _app_pkg
from app import config

ROOT = Path(_app_pkg.__file__).resolve().parents[1]

GONE_MODULES = [
    "app/core/voice.py", "app/api/voice.py",
    "app/tools/desktop.py",
    "app/mcp_server.py", "app/tools/mcp.py", "scripts/mcp_server_runner.py", "scripts/test_mcp_server.py",
    "app/core/task_manager.py", "app/tools/background_task.py",
    "app/core/goals.py", "app/core/goal_deriver.py",
    "app/channels/whatsapp.py", "app/channels/signal.py",
]

GONE_FLAGS = [
    "ENABLE_VOICE", "ENABLE_TTS", "WHISPER_MODEL_SIZE", "VOICE_MAX_DURATION", "TTS_MODEL_PATH",
    "ENABLE_DESKTOP_AUTOMATION", "DESKTOP_CLICK_DELAY",
    "ENABLE_MCP", "MCP_CONFIG_DIR",
    "MAX_BACKGROUND_TASKS", "BACKGROUND_TASK_TIMEOUT",
]

GONE_DEPS = ["openai-whisper", "piper-tts", "pyautogui", "mcp>=", "mcp=="]


@pytest.mark.parametrize("rel", GONE_MODULES)
def test_the_module_is_gone(rel):
    assert not (ROOT / rel).exists(), rel


@pytest.mark.parametrize("flag", GONE_FLAGS)
def test_the_setting_is_gone(flag):
    assert not hasattr(config, flag), flag


@pytest.mark.skipif(not (ROOT / "requirements.txt").exists(), reason="not shipped in the image")
@pytest.mark.parametrize("dep", GONE_DEPS)
def test_the_dependency_is_gone(dep):
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert not any(line.strip().startswith(dep) for line in text.splitlines()), dep


@pytest.mark.skipif(not (ROOT / "entrypoint.sh").exists(), reason="not shipped in the image")
def test_no_virtual_display_is_started():
    assert "Xvfb" not in (ROOT / "entrypoint.sh").read_text(encoding="utf-8")


@pytest.mark.skipif(not (ROOT / "docker-compose.yml").exists(), reason="not shipped in the image")
def test_compose_carries_none_of_it():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for needle in ("ENABLE_VOICE", "ENABLE_TTS", "ENABLE_DESKTOP_AUTOMATION", "DISPLAY:", "XAUTHORITY",
                   "ENABLE_MCP", "MCP_CONFIG_DIR", "./mcp:"):
        assert needle not in text, needle


def test_no_tool_list_names_the_cut_tools():
    """brain/prompt/trust keep name lists of registered tools; a name
    for a tool that no longer registers is a phantom example in the prompt."""
    hits = []
    for rel in ("app/core/brain.py", "app/core/prompt.py", "app/core/trust.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        for name in ('"desktop"', '"background_task"'):
            if name in src:
                hits.append(f"{rel}: {name}")
    assert not hits, hits


def test_channels_deliver_but_do_not_answer():
    for rel in ("app/channels/discord.py", "app/channels/telegram.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "def send_alert" in src, rel
        assert "_handle_query" not in src, rel
        assert "from app.core.brain import think" not in src, rel


def test_main_wires_none_of_it():
    src = (ROOT / "app/main.py").read_text(encoding="utf-8")
    for needle in ("voice", "DesktopTool", "MCPManager", "TaskManager", "BackgroundTaskTool",
                   "whatsapp", "signal_bot", "SignalBot"):
        assert not re.search(needle, src), needle


def test_services_has_no_task_manager_field():
    from app.core.brain import Services
    assert "task_manager" not in Services.__dataclass_fields__
