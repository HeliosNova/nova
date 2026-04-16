"""Desktop automation tool — click, type, move mouse, capture screen."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import config
from app.core.access_tiers import _tier
from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)

# Check for display and PyAutoGUI availability at import time
_HAS_DISPLAY = bool(os.environ.get("DISPLAY"))
_HAS_PYAUTOGUI = False
pyautogui = None  # type: ignore[assignment]
try:
    import pyautogui as _pyautogui  # noqa: N811

    _pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    _pyautogui.PAUSE = 0.5  # Delay between actions
    pyautogui = _pyautogui
    _HAS_PYAUTOGUI = True
except (ImportError, KeyError):
    # KeyError: pyautogui/mouseinfo reads DISPLAY env var on import
    pass


class DesktopTool(BaseTool):
    name = "desktop"
    description = (
        "Control the desktop GUI for automation tasks. Actions: screenshot (capture screen), "
        "click (click at x,y coordinates), type (type text at cursor), move (move mouse to x,y), "
        "hotkey (press key combinations like ctrl+c), scroll (scroll up/down), "
        "smart_click (find a UI element by description using vision and click it), "
        "smart_type (find an input field by description and type text into it), "
        "autonomous_workflow (complete a multi-step desktop task using vision). "
        "The smart_ actions use the vision model to understand the screen — no coordinates needed."
    )
    parameters = (
        "action: str (screenshot|click|type|move|hotkey|scroll|smart_click|smart_type|autonomous_workflow), "
        "x: int, y: int, text: str, keys: str, target: str (description for smart actions), "
        "goal: str (for autonomous_workflow), "
        "button: str (left|right|middle), clicks: int, amount: int"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["screenshot", "click", "type", "move", "hotkey", "scroll", "smart_click", "smart_type", "autonomous_workflow"],
                "description": "Desktop action to perform.",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate for click, move, or scroll.",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate for click, move, or scroll.",
            },
            "text": {
                "type": "string",
                "description": "Text to type (for type action). Max 1000 chars.",
            },
            "keys": {
                "type": "string",
                "description": "Key combination for hotkey (e.g., 'ctrl+c', 'alt+tab').",
            },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button for click. Defaults to 'left'.",
            },
            "clicks": {
                "type": "integer",
                "description": "Number of clicks (1-3). Defaults to 1.",
            },
            "amount": {
                "type": "integer",
                "description": "Scroll amount. Positive=up, negative=down. Max absolute value 50.",
            },
            "target": {
                "type": "string",
                "description": "UI element description for smart_click/smart_type (e.g. 'the Send button', 'email input field').",
            },
            "goal": {
                "type": "string",
                "description": "High-level goal for autonomous_workflow (e.g. 'open Chrome and navigate to gmail.com').",
            },
        },
        "required": ["action"],
    }

    def __init__(self):
        self._last_action_time = 0.0
        self._screenshot_dir = Path(config.SCREENSHOT_DIR)

    async def execute(self, **kwargs) -> ToolResult:
        if not config.ENABLE_DESKTOP_AUTOMATION:
            return ToolResult(
                output="",
                success=False,
                error="Desktop automation is disabled. Set ENABLE_DESKTOP_AUTOMATION=true",
                error_category=ErrorCategory.PERMISSION,
            )

        if not _HAS_PYAUTOGUI:
            return ToolResult(
                output="",
                success=False,
                error="PyAutoGUI is not installed. Install with: pip install pyautogui",
                error_category=ErrorCategory.PERMISSION,
            )

        if not _HAS_DISPLAY:
            return ToolResult(
                output="",
                success=False,
                error="No display server available. Set DISPLAY env var or run with X11 forwarding",
                error_category=ErrorCategory.PERMISSION,
            )

        # Access tier check
        tier = _tier()
        if tier not in ("full", "none"):
            return ToolResult(
                output="",
                success=False,
                error=f"Desktop automation requires 'full' or 'none' access tier (current: {tier})",
                error_category=ErrorCategory.PERMISSION,
            )

        # Rate limit rapid actions
        now = time.time()
        min_delay = config.DESKTOP_CLICK_DELAY
        if now - self._last_action_time < min_delay:
            await asyncio.sleep(min_delay - (now - self._last_action_time))
        self._last_action_time = time.time()

        action = kwargs.get("action", "").lower()

        try:
            if action == "screenshot":
                return await self._screenshot()
            elif action == "click":
                return await self._click(kwargs)
            elif action == "type":
                return await self._type(kwargs)
            elif action == "move":
                return await self._move(kwargs)
            elif action == "hotkey":
                return await self._hotkey(kwargs)
            elif action == "scroll":
                return await self._scroll(kwargs)
            elif action == "smart_click":
                return await self._smart_click(kwargs)
            elif action == "smart_type":
                return await self._smart_type(kwargs)
            elif action == "autonomous_workflow":
                return await self._autonomous_workflow(kwargs)
            else:
                return ToolResult(
                    output="",
                    success=False,
                    error=f"Unknown action: {action}. Use: screenshot, click, type, move, hotkey, scroll",
                    error_category=ErrorCategory.VALIDATION,
                )
        except Exception as e:
            logger.error("[Desktop] Action '%s' failed: %s", action, e)
            return ToolResult(output="", success=False, error=f"Desktop action failed: {e}", retriable=True, error_category=ErrorCategory.INTERNAL)

    async def _screenshot(self) -> ToolResult:
        def _capture():
            import subprocess
            try:
                self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Cannot create screenshot directory {self._screenshot_dir}: {e}")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self._screenshot_dir / f"desktop_{ts}.png"
            # Use scrot directly (more reliable than pyautogui.screenshot in headless)
            result = subprocess.run(["scrot", str(path)], capture_output=True, timeout=10)
            if result.returncode != 0:
                # Fallback to pyautogui
                img = pyautogui.screenshot()
                img.save(str(path))
            size = pyautogui.size()
            return str(path), size

        path, size = await asyncio.to_thread(_capture)
        return ToolResult(
            success=True,
            output=f"Screenshot saved to {path} (screen: {size.width}x{size.height})",
        )

    async def _click(self, kwargs) -> ToolResult:
        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            return ToolResult(output="", success=False, error="click requires x and y coordinates", error_category=ErrorCategory.VALIDATION)
        button = kwargs.get("button", "left")
        clicks = kwargs.get("clicks", 1)
        if button not in ("left", "right", "middle"):
            return ToolResult(output="", success=False, error="button must be left, right, or middle", error_category=ErrorCategory.VALIDATION)
        if not (1 <= int(clicks) <= 3):
            return ToolResult(output="", success=False, error="clicks must be 1-3", error_category=ErrorCategory.VALIDATION)

        await asyncio.to_thread(
            lambda: pyautogui.click(x=int(x), y=int(y), button=button, clicks=int(clicks)),
        )
        return ToolResult(success=True, output=f"Clicked ({x}, {y}) with {button} button ({clicks}x)")

    async def _type(self, kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        if not text:
            return ToolResult(output="", success=False, error="type requires text parameter", error_category=ErrorCategory.VALIDATION)
        if len(text) > 1000:
            return ToolResult(output="", success=False, error="Text too long (max 1000 chars)", error_category=ErrorCategory.VALIDATION)

        if text.isascii():
            await asyncio.to_thread(lambda: pyautogui.typewrite(text, interval=0.02))
        else:
            try:
                import pyperclip
                await asyncio.to_thread(lambda: (pyperclip.copy(text), pyautogui.hotkey('ctrl', 'v')))
            except (ImportError, Exception) as e:
                return ToolResult(
                    output="",
                    success=False,
                    error=f"Cannot type non-ASCII text: pyperclip unavailable ({e})",
                    error_category=ErrorCategory.PERMISSION,
                )
        return ToolResult(success=True, output=f"Typed {len(text)} character(s)")

    async def _move(self, kwargs) -> ToolResult:
        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            return ToolResult(output="", success=False, error="move requires x and y coordinates", error_category=ErrorCategory.VALIDATION)

        await asyncio.to_thread(
            lambda: pyautogui.moveTo(int(x), int(y), duration=0.3)
        )
        return ToolResult(success=True, output=f"Moved mouse to ({x}, {y})")

    async def _hotkey(self, kwargs) -> ToolResult:
        keys = kwargs.get("keys", "")
        if not keys:
            return ToolResult(output="", success=False, error="hotkey requires keys parameter (e.g. 'ctrl+c')", error_category=ErrorCategory.VALIDATION)

        parts = [k.strip() for k in keys.split("+")]
        # Block dangerous combos
        dangerous = {
            "alt+f4", "ctrl+alt+delete", "ctrl+alt+del", "ctrl+alt+t",
            "super+r", "super+e", "ctrl+shift+esc", "super+l",
        }
        if keys.lower().replace(" ", "") in dangerous:
            return ToolResult(
                output="",
                success=False,
                error=f"Blocked dangerous key combination: {keys}",
                error_category=ErrorCategory.PERMISSION,
            )

        await asyncio.to_thread(
            lambda: pyautogui.hotkey(*parts)
        )
        return ToolResult(success=True, output=f"Pressed hotkey: {keys}")

    async def _scroll(self, kwargs) -> ToolResult:
        amount = kwargs.get("amount", 0)
        if not amount:
            return ToolResult(
                output="",
                success=False,
                error="scroll requires amount (positive=up, negative=down)",
                error_category=ErrorCategory.VALIDATION,
            )
        if abs(int(amount)) > 50:
            return ToolResult(output="", success=False, error="Scroll amount too large (max 50)", error_category=ErrorCategory.VALIDATION)

        x = kwargs.get("x")
        y = kwargs.get("y")

        await asyncio.to_thread(
            lambda: pyautogui.scroll(
                int(amount),
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
            ),
        )
        direction = "up" if int(amount) > 0 else "down"
        return ToolResult(success=True, output=f"Scrolled {direction} by {abs(int(amount))}")

    async def _vision_find_element(self, target: str) -> dict | None:
        """Use Ollama vision model to find a UI element by description. Returns {x, y} or None."""
        import base64
        import json as _json
        import subprocess
        import httpx

        # Take screenshot via scrot (more reliable in headless)
        tmp_path = "/tmp/_nova_vision_screenshot.png"
        await asyncio.to_thread(lambda: subprocess.run(["scrot", tmp_path], capture_output=True, timeout=10))
        with open(tmp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        screen = pyautogui.size()

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{config.OLLAMA_URL}/api/chat",
                    json={
                        "model": config.VISION_MODEL or config.LLM_MODEL,
                        "messages": [{
                            "role": "user",
                            "content": (
                                f"You are a GUI automation agent. This screenshot is {screen.width}x{screen.height} pixels. "
                                f"The origin (0,0) is the top-left corner. X increases right, Y increases down. "
                                f"Find the UI element matching: \"{target}\". "
                                f"Return the x,y pixel coordinates of its center."
                            ),
                            "images": [img_b64],
                        }],
                        "format": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "element": {"type": "string"}}, "required": ["x", "y"]},
                        "stream": False,
                    },
                    timeout=120.0,
                )
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                coords = _json.loads(content)
                if coords.get("x", -1) > 0 and coords.get("y", -1) > 0:
                    return coords
                return None
            except Exception as e:
                logger.warning("[Desktop] Vision element search failed: %s", e)
                return None

    async def _smart_click(self, kwargs) -> ToolResult:
        """Find a UI element by description and click it."""
        target = kwargs.get("target", "")
        if not target:
            return ToolResult(output="", success=False, error="smart_click requires 'target' — describe the UI element to click", error_category=ErrorCategory.VALIDATION)

        coords = await self._vision_find_element(target)
        if not coords:
            return ToolResult(output="", success=False, error=f"Could not find UI element: \"{target}\". Try a more specific description.", error_category=ErrorCategory.NOT_FOUND)

        x, y = coords["x"], coords["y"]
        element = coords.get("element", target)
        button = kwargs.get("button", "left")
        await asyncio.to_thread(lambda: pyautogui.click(x=x, y=y, button=button))
        return ToolResult(success=True, output=f"Found \"{element}\" at ({x}, {y}) and clicked it")

    async def _smart_type(self, kwargs) -> ToolResult:
        """Find an input field by description, click it, and type text."""
        target = kwargs.get("target", "")
        text = kwargs.get("text", "")
        if not target:
            return ToolResult(output="", success=False, error="smart_type requires 'target' — describe the input field", error_category=ErrorCategory.VALIDATION)
        if not text:
            return ToolResult(output="", success=False, error="smart_type requires 'text' — the text to type", error_category=ErrorCategory.VALIDATION)
        if len(text) > 1000:
            return ToolResult(output="", success=False, error="Text too long (max 1000 chars)", error_category=ErrorCategory.VALIDATION)

        coords = await self._vision_find_element(target)
        if not coords:
            return ToolResult(output="", success=False, error=f"Could not find input field: \"{target}\"", error_category=ErrorCategory.NOT_FOUND)

        x, y = coords["x"], coords["y"]
        element = coords.get("element", target)
        await asyncio.to_thread(lambda: pyautogui.click(x=x, y=y))
        await asyncio.sleep(0.2)

        if text.isascii():
            await asyncio.to_thread(lambda: pyautogui.typewrite(text, interval=0.02))
        else:
            try:
                import pyperclip
                await asyncio.to_thread(lambda: (pyperclip.copy(text), pyautogui.hotkey("ctrl", "v")))
            except Exception:
                await asyncio.to_thread(lambda: pyautogui.typewrite(text, interval=0.02))

        return ToolResult(success=True, output=f"Found \"{element}\" at ({x}, {y}), clicked, and typed {len(text)} chars")

    async def _autonomous_workflow(self, kwargs) -> ToolResult:
        """Complete a multi-step desktop task using vision-guided actions."""
        import base64
        import json as _json
        import io
        import httpx

        goal = kwargs.get("goal", "")
        if not goal:
            return ToolResult(output="", success=False, error="autonomous_workflow requires 'goal' — describe what to accomplish", error_category=ErrorCategory.VALIDATION)

        max_steps = min(int(kwargs.get("max_steps", 15)), 25)
        actions_taken = []
        recent_actions: list[str] = []

        for step in range(max_steps):
            # Screenshot current state via scrot
            import subprocess
            tmp_path = "/tmp/_nova_workflow_screenshot.png"
            await asyncio.to_thread(lambda: subprocess.run(["scrot", tmp_path], capture_output=True, timeout=10))
            with open(tmp_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            screen = pyautogui.size()

            # Ask vision model for next action
            history = "\n".join(f"  Step {i+1}: {a}" for i, a in enumerate(actions_taken[-5:]))
            prompt = (
                f"You are a desktop automation agent. Screen: {screen.width}x{screen.height}px. "
                f"Origin (0,0) is top-left. Goal: \"{goal}\"\n"
                + (f"Actions taken so far:\n{history}\n" if history else "")
                + "What is the next action? Options: click (click at coordinates), "
                "type (type text), hotkey (key combo), scroll (scroll direction), done (goal achieved). "
                "Return the action as JSON."
            )

            action_schema = {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["click", "type", "hotkey", "scroll", "done"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "text": {"type": "string"},
                    "keys": {"type": "string"},
                    "amount": {"type": "integer"},
                    "reasoning": {"type": "string"},
                },
                "required": ["action", "reasoning"],
            }

            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{config.OLLAMA_URL}/api/chat",
                        json={
                            "model": config.VISION_MODEL or config.LLM_MODEL,
                            "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                            "format": action_schema,
                            "stream": False,
                        },
                        timeout=120.0,
                    )
                    data = resp.json()
                    content = data.get("message", {}).get("content", "")
                    next_action = _json.loads(content)
            except Exception as e:
                actions_taken.append(f"Vision failed: {e}")
                break

            act = next_action.get("action", "done")
            reasoning = next_action.get("reasoning", "")

            # Guard against action repetition
            action_sig = f"{act}:{next_action.get('x','')}:{next_action.get('y','')}:{next_action.get('text','')}"
            recent_actions.append(action_sig)
            if len(recent_actions) >= 3 and len(set(recent_actions[-3:])) == 1:
                actions_taken.append(f"Aborted: repeated same action 3x ({act})")
                break

            if act == "done":
                actions_taken.append(f"Done: {reasoning}")
                break
            elif act == "click":
                x, y = next_action.get("x", 0), next_action.get("y", 0)
                if x > 0 and y > 0:
                    await asyncio.to_thread(lambda _x=x, _y=y: pyautogui.click(x=_x, y=_y))
                    actions_taken.append(f"Clicked ({x}, {y}) — {reasoning}")
                else:
                    actions_taken.append(f"Invalid click coords ({x}, {y})")
            elif act == "type":
                text = next_action.get("text", "")
                if text:
                    await asyncio.to_thread(lambda: pyautogui.typewrite(text, interval=0.02) if text.isascii() else None)
                    actions_taken.append(f"Typed \"{text[:30]}...\" — {reasoning}")
            elif act == "hotkey":
                keys = next_action.get("keys", "")
                if keys:
                    parts = [k.strip() for k in keys.split("+")]
                    await asyncio.to_thread(lambda: pyautogui.hotkey(*parts))
                    actions_taken.append(f"Hotkey {keys} — {reasoning}")
            elif act == "scroll":
                amt = next_action.get("amount", -3)
                await asyncio.to_thread(lambda: pyautogui.scroll(amt))
                actions_taken.append(f"Scrolled {amt} — {reasoning}")

            await asyncio.sleep(0.5)  # Brief pause between actions

        summary = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(actions_taken))
        return ToolResult(success=True, output=f"Autonomous workflow completed ({len(actions_taken)} steps):\n{summary}")
