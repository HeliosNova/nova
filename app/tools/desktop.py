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
        "Control the desktop GUI via PyAutoGUI for automation tasks. Actions: screenshot (capture screen), "
        "click (click at x,y coordinates), type (type text at cursor), move (move mouse to x,y), "
        "hotkey (press key combinations like ctrl+c), scroll (scroll up/down). Requires "
        "ENABLE_DESKTOP_AUTOMATION=true, PyAutoGUI installed, X11 display, and 'full' or 'none' access "
        "tier. Rate-limited to prevent rapid fire. Dangerous key combinations (alt+f4, ctrl+alt+delete) "
        "are blocked."
    )
    parameters = (
        "action: str (screenshot|click|type|move|hotkey|scroll), "
        "x: int, y: int, text: str, keys: str, "
        "button: str (left|right|middle), clicks: int, amount: int"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["screenshot", "click", "type", "move", "hotkey", "scroll"],
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

        path, size = await asyncio.to_thread( _capture)
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
