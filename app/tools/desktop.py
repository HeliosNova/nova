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
from app.tools.base import BaseTool, ToolResult

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
        "Control the desktop GUI. Actions: screenshot (capture screen), "
        "click (click at x,y), type (type text), move (move mouse), "
        "hotkey (press key combo like ctrl+c), scroll (scroll up/down). "
        "Requires ENABLE_DESKTOP_AUTOMATION=true and a display server."
    )
    parameters = (
        "action: str (screenshot|click|type|move|hotkey|scroll), "
        "x: int, y: int, text: str, keys: str, "
        "button: str (left|right|middle), clicks: int, amount: int"
    )

    def __init__(self):
        self._last_action_time = 0.0
        self._screenshot_dir = Path("/data/screenshots")

    async def execute(self, **kwargs) -> ToolResult:
        if not config.ENABLE_DESKTOP_AUTOMATION:
            return ToolResult(
                output="",
                success=False,
                error="Desktop automation is disabled. Set ENABLE_DESKTOP_AUTOMATION=true",
            )

        if not _HAS_PYAUTOGUI:
            return ToolResult(
                output="",
                success=False,
                error="PyAutoGUI is not installed. Install with: pip install pyautogui",
            )

        if not _HAS_DISPLAY:
            return ToolResult(
                output="",
                success=False,
                error="No display server available. Set DISPLAY env var or run with X11 forwarding",
            )

        # Access tier check
        tier = _tier()
        if tier not in ("full", "none"):
            return ToolResult(
                output="",
                success=False,
                error=f"Desktop automation requires 'full' or 'none' access tier (current: {tier})",
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
            else:
                return ToolResult(
                    output="",
                    success=False,
                    error=f"Unknown action: {action}. Use: screenshot, click, type, move, hotkey, scroll",
                )
        except Exception as e:
            logger.error("[Desktop] Action '%s' failed: %s", action, e)
            return ToolResult(output="", success=False, error=f"Desktop action failed: {e}")

    async def _screenshot(self) -> ToolResult:
        def _capture():
            try:
                self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Cannot create screenshot directory {self._screenshot_dir}: {e}")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self._screenshot_dir / f"desktop_{ts}.png"
            img = pyautogui.screenshot()
            img.save(str(path))
            size = pyautogui.size()
            return str(path), size

        path, size = await asyncio.get_event_loop().run_in_executor(None, _capture)
        return ToolResult(
            success=True,
            output=f"Screenshot saved to {path} (screen: {size.width}x{size.height})",
        )

    async def _click(self, kwargs) -> ToolResult:
        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            return ToolResult(output="", success=False, error="click requires x and y coordinates")
        button = kwargs.get("button", "left")
        clicks = kwargs.get("clicks", 1)
        if button not in ("left", "right", "middle"):
            return ToolResult(output="", success=False, error="button must be left, right, or middle")
        if not (1 <= int(clicks) <= 3):
            return ToolResult(output="", success=False, error="clicks must be 1-3")

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pyautogui.click(x=int(x), y=int(y), button=button, clicks=int(clicks)),
        )
        return ToolResult(success=True, output=f"Clicked ({x}, {y}) with {button} button ({clicks}x)")

    async def _type(self, kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        if not text:
            return ToolResult(output="", success=False, error="type requires text parameter")
        if len(text) > 1000:
            return ToolResult(output="", success=False, error="Text too long (max 1000 chars)")

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pyautogui.typewrite(text, interval=0.02) if text.isascii() else pyautogui.write(text),
        )
        return ToolResult(success=True, output=f"Typed {len(text)} character(s)")

    async def _move(self, kwargs) -> ToolResult:
        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            return ToolResult(output="", success=False, error="move requires x and y coordinates")

        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.moveTo(int(x), int(y), duration=0.3)
        )
        return ToolResult(success=True, output=f"Moved mouse to ({x}, {y})")

    async def _hotkey(self, kwargs) -> ToolResult:
        keys = kwargs.get("keys", "")
        if not keys:
            return ToolResult(output="", success=False, error="hotkey requires keys parameter (e.g. 'ctrl+c')")

        parts = [k.strip() for k in keys.split("+")]
        # Block dangerous combos
        dangerous = {"alt+f4", "ctrl+alt+delete", "ctrl+alt+del"}
        if keys.lower().replace(" ", "") in dangerous:
            return ToolResult(
                output="",
                success=False,
                error=f"Blocked dangerous key combination: {keys}",
            )

        await asyncio.get_event_loop().run_in_executor(
            None, lambda: pyautogui.hotkey(*parts)
        )
        return ToolResult(success=True, output=f"Pressed hotkey: {keys}")

    async def _scroll(self, kwargs) -> ToolResult:
        amount = kwargs.get("amount", 0)
        if not amount:
            return ToolResult(
                output="",
                success=False,
                error="scroll requires amount (positive=up, negative=down)",
            )
        if abs(int(amount)) > 50:
            return ToolResult(output="", success=False, error="Scroll amount too large (max 50)")

        x = kwargs.get("x")
        y = kwargs.get("y")

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pyautogui.scroll(
                int(amount),
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
            ),
        )
        direction = "up" if int(amount) > 0 else "down"
        return ToolResult(success=True, output=f"Scrolled {direction} by {abs(int(amount))}")
