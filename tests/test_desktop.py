"""Tests for the DesktopTool (GUI automation)."""

from __future__ import annotations

from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# DesktopTool
# ===========================================================================

class TestDesktopTool:
    """Test desktop automation with PyAutoGUI mocked out."""

    def _make_tool(self):
        """Create a DesktopTool with screenshot dir pointing to tmp."""
        from app.tools.desktop import DesktopTool
        tool = DesktopTool()
        return tool

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        """ENABLE_DESKTOP_AUTOMATION defaults to false."""
        tool = self._make_tool()
        result = await tool.execute(action="screenshot")
        assert not result.success
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_pyautogui(self):
        """Error when PyAutoGUI is not installed."""
        tool = self._make_tool()
        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", False):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            result = await tool.execute(action="screenshot")
            assert not result.success
            assert "pyautogui" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_display(self):
        """Error when no display server is available."""
        tool = self._make_tool()
        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", False):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            result = await tool.execute(action="screenshot")
            assert not result.success
            assert "display" in result.error.lower()

    @pytest.mark.asyncio
    async def test_access_tier_restriction(self):
        """Desktop requires 'full' or 'none' access tier."""
        tool = self._make_tool()
        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="sandboxed"):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="screenshot")
            assert not result.success
            assert "access tier" in result.error.lower()

    @pytest.mark.asyncio
    async def test_screenshot_action(self, tmp_path):
        """Screenshot captures and saves to file."""
        tool = self._make_tool()
        tool._screenshot_dir = tmp_path

        Size = namedtuple("Size", ["width", "height"])
        mock_img = MagicMock()
        mock_pyautogui = MagicMock()
        mock_pyautogui.screenshot.return_value = mock_img
        mock_pyautogui.size.return_value = Size(1920, 1080)

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"), \
             patch("app.tools.desktop.pyautogui", mock_pyautogui):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="screenshot")
            assert result.success
            assert "screenshot saved" in result.output.lower()
            assert "1920x1080" in result.output
            mock_pyautogui.screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_missing_coordinates(self):
        """Click without x/y returns error."""
        tool = self._make_tool()
        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="click")
            assert not result.success
            assert "x and y" in result.error.lower()

    @pytest.mark.asyncio
    async def test_click_valid(self):
        """Click at valid coordinates succeeds."""
        tool = self._make_tool()
        mock_pyautogui = MagicMock()

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"), \
             patch("app.tools.desktop.pyautogui", mock_pyautogui):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="click", x=100, y=200)
            assert result.success
            assert "100" in result.output
            assert "200" in result.output
            mock_pyautogui.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_type_with_text(self):
        """Type action types the given text."""
        tool = self._make_tool()
        mock_pyautogui = MagicMock()

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"), \
             patch("app.tools.desktop.pyautogui", mock_pyautogui):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="type", text="hello world")
            assert result.success
            assert "11" in result.output  # len("hello world") == 11

    @pytest.mark.asyncio
    async def test_type_empty_text(self):
        """Type with no text returns error."""
        tool = self._make_tool()
        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="type", text="")
            assert not result.success
            assert "text" in result.error.lower()

    @pytest.mark.asyncio
    async def test_hotkey_dangerous_blocked(self):
        """Dangerous key combos are blocked."""
        tool = self._make_tool()
        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="hotkey", keys="alt+f4")
            assert not result.success
            assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        """Unknown action returns error."""
        tool = self._make_tool()
        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="dance")
            assert not result.success
            assert "unknown action" in result.error.lower()

    @pytest.mark.asyncio
    async def test_scroll(self):
        """Scroll action works with amount."""
        tool = self._make_tool()
        mock_pyautogui = MagicMock()

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"), \
             patch("app.tools.desktop.pyautogui", mock_pyautogui):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="scroll", amount=-5)
            assert result.success
            assert "down" in result.output.lower()
            assert "5" in result.output
            mock_pyautogui.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_move_valid(self):
        """Move action with valid coordinates calls pyautogui.moveTo."""
        tool = self._make_tool()
        mock_pyautogui = MagicMock()

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"), \
             patch("app.tools.desktop.pyautogui", mock_pyautogui):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="move", x=500, y=300)
            assert result.success
            assert "500" in result.output
            assert "300" in result.output
            mock_pyautogui.moveTo.assert_called_once()

    @pytest.mark.asyncio
    async def test_hotkey_valid_safe(self):
        """Safe hotkey combo like ctrl+c should succeed."""
        tool = self._make_tool()
        mock_pyautogui = MagicMock()

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"), \
             patch("app.tools.desktop.pyautogui", mock_pyautogui):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="hotkey", keys="ctrl+c")
            assert result.success
            assert "ctrl+c" in result.output.lower()
            mock_pyautogui.hotkey.assert_called_once_with("ctrl", "c")

    @pytest.mark.asyncio
    async def test_scroll_default_amount(self):
        """Scroll without amount (default 0) returns error requiring amount."""
        tool = self._make_tool()

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="scroll")
            assert not result.success
            assert "amount" in result.error.lower()

    @pytest.mark.asyncio
    async def test_type_text_too_long(self):
        """Type with text exceeding 1000 chars returns max length error."""
        tool = self._make_tool()

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            long_text = "a" * 1001
            result = await tool.execute(action="type", text=long_text)
            assert not result.success
            assert "max" in result.error.lower() or "1000" in result.error

    @pytest.mark.asyncio
    async def test_click_invalid_button(self):
        """Click with invalid button value returns error."""
        tool = self._make_tool()

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="click", x=100, y=200, button="invalid")
            assert not result.success
            assert "button" in result.error.lower()

    @pytest.mark.asyncio
    async def test_action_pyautogui_exception(self):
        """PyAutoGUI exception during action returns graceful ToolResult error."""
        tool = self._make_tool()
        mock_pyautogui = MagicMock()
        mock_pyautogui.click.side_effect = Exception("X11 connection refused")

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"), \
             patch("app.tools.desktop.pyautogui", mock_pyautogui):
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 0.0
            result = await tool.execute(action="click", x=100, y=200)
            assert not result.success
            assert "failed" in result.error.lower()
            assert "X11 connection refused" in result.error

    @pytest.mark.asyncio
    async def test_rate_limiting_delay(self):
        """DESKTOP_CLICK_DELAY causes asyncio.sleep between rapid actions."""
        tool = self._make_tool()
        mock_pyautogui = MagicMock()

        with patch("app.tools.desktop.config") as mock_config, \
             patch("app.tools.desktop._HAS_PYAUTOGUI", True), \
             patch("app.tools.desktop._HAS_DISPLAY", True), \
             patch("app.tools.desktop._tier", return_value="full"), \
             patch("app.tools.desktop.pyautogui", mock_pyautogui), \
             patch("app.tools.desktop.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_config.ENABLE_DESKTOP_AUTOMATION = True
            mock_config.DESKTOP_CLICK_DELAY = 1.0

            # First action — sets _last_action_time
            result1 = await tool.execute(action="move", x=10, y=10)
            assert result1.success

            # Second action immediately — should trigger sleep
            result2 = await tool.execute(action="move", x=20, y=20)
            assert result2.success
            mock_sleep.assert_called()
            # The sleep delay should be > 0 and <= DESKTOP_CLICK_DELAY
            delay_arg = mock_sleep.call_args[0][0]
            assert 0 < delay_arg <= 1.0
