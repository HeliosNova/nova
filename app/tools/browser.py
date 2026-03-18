"""Browser automation tool — Playwright-based web interaction with persistent sessions.

Sessions persist across tool calls so Nova can log in, navigate, click, and
read across multiple steps — just like a human using a browser.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.config import config
from app.tools.base import BaseTool, ToolResult
from app.tools.http_fetch import _is_safe_url

logger = logging.getLogger(__name__)

_MAX_CONTENT = 8000
_SESSION_TIMEOUT = 600  # 10 min idle timeout


class BrowserTool(BaseTool):
    name = "browser"
    description = (
        "Control a persistent web browser. Sessions survive across calls so you can "
        "log in, navigate multi-step workflows, and interact with web apps. "
        "Actions: navigate (go to URL), click (click element), type (type into field), "
        "press_key (press Enter/Tab/etc.), get_text (read page/element text), get_links (list links), "
        "fill_form (fill multiple fields), screenshot (capture current page), back (go back), "
        "wait (wait for element), close_session (end browser session)."
    )
    parameters = (
        "action: str (navigate|click|type|press_key|get_text|get_links|fill_form|screenshot|back|wait|close_session), "
        "url: str (only needed for navigate or first visit), "
        "selector: str (CSS selector for click/type/get_text/wait), "
        "text: str (text to type), form_data: dict (selector:value pairs), "
        "script: str (JS for evaluate_js)"
    )

    _playwright = None
    _browser = None
    _context = None       # Persistent browser context (holds cookies/state)
    _page = None          # Persistent page (holds navigation state)
    _last_used = 0.0      # Timestamp for session timeout

    async def _get_page(self):
        """Return the persistent page, creating browser/context/page if needed."""
        now = time.time()

        # Session timeout — close stale sessions
        if (BrowserTool._page and BrowserTool._last_used
                and now - BrowserTool._last_used > _SESSION_TIMEOUT):
            logger.info("[Browser] Session timed out after %ds idle", _SESSION_TIMEOUT)
            await self._close_session()

        BrowserTool._last_used = now

        # Reuse existing page if browser is still connected and page is usable
        if (BrowserTool._page and BrowserTool._browser
                and BrowserTool._browser.is_connected()):
            try:
                await BrowserTool._page.evaluate("1")  # Quick liveness check
                return BrowserTool._page
            except Exception:
                logger.info("[Browser] Stale session detected, reconnecting")

        # Clean up anything stale
        await self._close_session()

        from playwright.async_api import async_playwright
        BrowserTool._playwright = await async_playwright().start()

        # Connect to host browser (visible on desktop) or launch headless
        cdp_url = config.BROWSER_CDP_URL
        if cdp_url:
            # Chrome rejects non-localhost Host headers on CDP, so we fetch
            # the WebSocket URL manually with a spoofed Host header
            import httpx
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{cdp_url}/json/version",
                        headers={"Host": "localhost"},
                        timeout=5,
                    )
                    ws_url = resp.json()["webSocketDebuggerUrl"]
                    # Rebuild ws URL with the CDP host:port so Docker can reach it
                    from urllib.parse import urlparse
                    cdp_parsed = urlparse(cdp_url)
                    ws_parsed = urlparse(ws_url)
                    host_port = f"{cdp_parsed.hostname}:{cdp_parsed.port or 9222}"
                    ws_url = f"ws://{host_port}{ws_parsed.path}"
            except Exception as e:
                raise RuntimeError(f"Cannot reach host browser at {cdp_url}: {e}")
            BrowserTool._browser = await BrowserTool._playwright.chromium.connect_over_cdp(ws_url)
            logger.info("[Browser] Connected to host browser at %s", cdp_url)
        else:
            BrowserTool._browser = await BrowserTool._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            )
            logger.info("[Browser] Launched headless browser")

        BrowserTool._context = await BrowserTool._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        BrowserTool._context.set_default_timeout(config.BROWSER_TIMEOUT * 1000)
        BrowserTool._page = await BrowserTool._context.new_page()
        logger.info("[Browser] New session started")
        return BrowserTool._page

    @classmethod
    async def _close_session(cls):
        """Close the persistent session (context + browser)."""
        if cls._page:
            try:
                await cls._page.close()
            except Exception:
                pass
            cls._page = None
        if cls._context:
            try:
                await cls._context.close()
            except Exception:
                pass
            cls._context = None
        if cls._browser:
            try:
                await cls._browser.close()
            except Exception:
                pass
            cls._browser = None
        if cls._playwright:
            try:
                await cls._playwright.stop()
            except Exception:
                pass
            cls._playwright = None

    # Keep the old name for shutdown hooks
    _close_browser = _close_session

    async def execute(
        self,
        *,
        action: str = "",
        url: str = "",
        selector: str = "",
        text: str = "",
        form_data: dict | None = None,
        script: str = "",
        **kwargs,
    ) -> ToolResult:
        if not action:
            return ToolResult(
                output="",
                success=False,
                error=(
                    "No action specified. Use: navigate, click, type, get_text, "
                    "get_links, fill_form, screenshot, back, wait, close_session."
                ),
            )

        try:
            from playwright.async_api import async_playwright  # noqa: F401
        except ImportError:
            return ToolResult(
                output="",
                success=False,
                error="Playwright not installed. Browser tool unavailable.",
            )

        action = action.lower()

        # close_session doesn't need a page
        if action == "close_session":
            await self._close_session()
            return ToolResult(output="Browser session closed.", success=True)

        try:
            page = await self._get_page()

            if action == "navigate":
                return await self._navigate(page, url)
            elif action == "click":
                return await self._click(page, url, selector)
            elif action == "type":
                return await self._type(page, url, selector, text)
            elif action == "press_key":
                return await self._press_key(page, text)
            elif action == "get_text":
                return await self._get_text(page, url, selector)
            elif action == "get_links":
                return await self._get_links(page, url)
            elif action == "fill_form":
                return await self._fill_form(page, url, form_data)
            elif action == "evaluate_js":
                return await self._evaluate_js(page, url, script)
            elif action == "screenshot":
                return await self._screenshot(page)
            elif action == "back":
                return await self._back(page)
            elif action == "wait":
                return await self._wait(page, selector)
            else:
                return ToolResult(
                    output="",
                    success=False,
                    error=f"Unknown action '{action}'.",
                )

        except Exception as e:
            logger.warning("[Browser] Action '%s' failed: %s", action, e)
            # Only kill session if browser itself is dead, not on action errors
            if BrowserTool._browser and not BrowserTool._browser.is_connected():
                await self._close_session()
            return ToolResult(
                output="", success=False,
                error=f"Browser action failed: {e}",
            )

    # ------------------------------------------------------------------
    # URL safety
    # ------------------------------------------------------------------

    @staticmethod
    def _check_url(url: str) -> ToolResult | None:
        """Return a ToolResult error if URL is unsafe, else None."""
        if not url:
            return ToolResult(output="", success=False, error="No URL provided")
        if not _is_safe_url(url):
            return ToolResult(
                output="", success=False,
                error="URL blocked: internal/private addresses not allowed",
            )
        return None

    async def _maybe_navigate(self, page, url: str) -> ToolResult | None:
        """Navigate to url if provided; skip if already on a page."""
        if not url:
            # No URL = use current page
            current = page.url
            if not current or current == "about:blank":
                return ToolResult(
                    output="", success=False,
                    error="No URL provided and no page loaded. Use navigate first.",
                )
            return None
        if err := self._check_url(url):
            return err
        await page.goto(url, wait_until="domcontentloaded")
        return None

    def _page_summary(self, page) -> str:
        """One-line summary of current page state."""
        return f"[{page.url}]"

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def _navigate(self, page, url: str) -> ToolResult:
        if err := self._check_url(url):
            return err
        await page.goto(url, wait_until="domcontentloaded")
        title = await page.title()
        text = await page.evaluate("() => document.body.innerText")
        if len(text) > _MAX_CONTENT:
            text = text[:_MAX_CONTENT] + "\n[... content truncated]"
        output = f"Page: {title}\nURL: {page.url}\n\n{text}"
        if config.ENABLE_INJECTION_DETECTION:
            from app.core.injection import sanitize_content
            output = sanitize_content(output, context="web page")
        return ToolResult(output=output, success=True)

    async def _click(self, page, url: str, selector: str) -> ToolResult:
        if not selector:
            return ToolResult(
                output="", success=False,
                error="'selector' is required for click.",
            )
        if err := await self._maybe_navigate(page, url):
            return err
        await page.click(selector)
        await page.wait_for_load_state("domcontentloaded")
        title = await page.title()
        # Return some page text so the LLM knows what happened
        text = await page.evaluate("() => document.body.innerText")
        if len(text) > 3000:
            text = text[:3000] + "\n[... truncated]"
        return ToolResult(
            output=f"Clicked '{selector}'. Page: {title} ({page.url})\n\n{text}",
            success=True,
        )

    async def _type(self, page, url: str, selector: str, text: str) -> ToolResult:
        if not selector or not text:
            return ToolResult(
                output="", success=False,
                error="'selector' and 'text' are required for type.",
            )
        if err := await self._maybe_navigate(page, url):
            return err
        await page.fill(selector, text)
        return ToolResult(
            output=f"Typed '{text[:80]}' into '{selector}' {self._page_summary(page)}",
            success=True,
        )

    async def _press_key(self, page, key: str) -> ToolResult:
        """Press a keyboard key (Enter, Tab, Escape, ArrowDown, etc.)."""
        if not key:
            return ToolResult(
                output="", success=False,
                error="'text' parameter required for press_key (e.g. 'Enter', 'Tab').",
            )
        allowed = {
            "enter", "tab", "escape", "backspace", "delete", "space",
            "arrowup", "arrowdown", "arrowleft", "arrowright",
            "home", "end", "pageup", "pagedown",
        }
        if key.lower() not in allowed:
            return ToolResult(
                output="", success=False,
                error=f"Key '{key}' not allowed. Use: {', '.join(sorted(allowed))}",
            )
        await page.keyboard.press(key)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass  # Key may not trigger navigation
        title = await page.title()
        try:
            text = await page.evaluate("() => document.body.innerText")
        except Exception:
            text = f"(page navigating after {key})"
        if len(text) > 3000:
            text = text[:3000] + "\n[... truncated]"
        return ToolResult(
            output=f"Pressed '{key}'. Page: {title} ({page.url})\n\n{text}",
            success=True,
        )

    async def _get_text(self, page, url: str, selector: str = "") -> ToolResult:
        if err := await self._maybe_navigate(page, url):
            return err
        if selector:
            element = await page.query_selector(selector)
            if not element:
                return ToolResult(
                    output="", success=False,
                    error=f"Selector '{selector}' not found on {page.url}",
                )
            text = await element.inner_text()
        else:
            text = await page.evaluate("() => document.body.innerText")
        if len(text) > _MAX_CONTENT:
            text = text[:_MAX_CONTENT] + "\n[... content truncated]"
        if config.ENABLE_INJECTION_DETECTION:
            from app.core.injection import sanitize_content
            text = sanitize_content(text, context="web page")
        return ToolResult(output=text, success=True)

    async def _get_links(self, page, url: str) -> ToolResult:
        if err := await self._maybe_navigate(page, url):
            return err
        links = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]'))
                .slice(0, 50)
                .map(a => ({text: a.innerText.trim().substring(0, 80), href: a.href}))
                .filter(l => l.text && l.href.startsWith('http'))
        """)
        if not links:
            return ToolResult(output="No links found on page.", success=True)
        lines = [f"[{i+1}] {l['text']} -> {l['href']}" for i, l in enumerate(links)]
        output = "\n".join(lines)
        if config.ENABLE_INJECTION_DETECTION:
            from app.core.injection import sanitize_content
            output = sanitize_content(output, context="web page links")
        return ToolResult(output=output, success=True)

    async def _fill_form(self, page, url: str, form_data: dict | None) -> ToolResult:
        if not form_data:
            return ToolResult(
                output="", success=False,
                error="'form_data' required (dict of selector: value).",
            )
        if err := await self._maybe_navigate(page, url):
            return err
        filled = []
        for sel, value in form_data.items():
            await page.fill(sel, str(value))
            filled.append(f"  {sel} = {str(value)[:50]}")
        return ToolResult(
            output=f"Filled {len(filled)} field(s) {self._page_summary(page)}:\n"
                   + "\n".join(filled),
            success=True,
        )

    async def _evaluate_js(self, page, url: str, script: str) -> ToolResult:
        if not script:
            return ToolResult(
                output="", success=False,
                error="'script' is required for evaluate_js.",
            )

        # Check blocked patterns BEFORE navigating (fail-fast)
        import re as _re

        _BLOCKED_PATTERNS = [
            (r"\bfetch\s*\(", "fetch()"),
            (r"\bXMLHttpRequest\b", "XMLHttpRequest"),
            (r"\bdocument\s*\.\s*cookie\b", "document.cookie"),
            (r"\blocalStorage\b", "localStorage"),
            (r"\bsessionStorage\b", "sessionStorage"),
            (r"\bnavigator\b", "navigator"),
            (r"\bWebSocket\b", "WebSocket"),
            (r"\beval\s*\(", "eval()"),
            (r"\bFunction\s*\(", "Function()"),
            (r"\bwindow\s*\.\s*open\b", "window.open"),
            (r"\bdocument\s*\.\s*write(?:ln)?\s*\(", "document.write"),
            (r"\bnew\s+Image\s*\(", "new Image()"),
            (r"\bimport\s*\(", "dynamic import"),
            (r"\brequire\s*\(", "require()"),
            (r"\b__proto__\b", "__proto__"),
            (r"\bconstructor\b", "constructor"),
            (r"\bprototype\b", "prototype"),
            (r"\bglobalThis\b", "globalThis"),
            (r"\bReflect\b", "Reflect"),
            (r"\bProxy\b", "Proxy"),
            (r"\bpostMessage\b", "postMessage"),
            (r"\bsetTimeout\b", "setTimeout"),
            (r"\bsetInterval\b", "setInterval"),
            (r"\baddEventListener\b", "addEventListener"),
            (r"\bonclick\b", "onclick handler"),
        ]
        for pattern, label in _BLOCKED_PATTERNS:
            if _re.search(pattern, script):
                return ToolResult(
                    output="", success=False,
                    error=f"Script contains blocked pattern: {label}",
                )

        if err := await self._maybe_navigate(page, url):
            return err

        result = await page.evaluate(script)
        output = str(result)
        if len(output) > _MAX_CONTENT:
            output = output[:_MAX_CONTENT] + "\n[... truncated]"
        return ToolResult(output=output, success=True)

    async def _screenshot(self, page) -> ToolResult:
        """Capture current page as PNG screenshot."""
        from pathlib import Path
        from datetime import datetime, timezone

        screenshot_dir = Path("/data/screenshots")
        try:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = screenshot_dir / f"browser_{ts}.png"
        await page.screenshot(path=str(path), full_page=False)
        title = await page.title()
        return ToolResult(
            output=f"Screenshot saved: {path}\nPage: {title} ({page.url})",
            success=True,
        )

    async def _back(self, page) -> ToolResult:
        """Go back to previous page."""
        await page.go_back(wait_until="domcontentloaded")
        title = await page.title()
        text = await page.evaluate("() => document.body.innerText")
        if len(text) > 3000:
            text = text[:3000] + "\n[... truncated]"
        return ToolResult(
            output=f"Went back. Page: {title} ({page.url})\n\n{text}",
            success=True,
        )

    async def _wait(self, page, selector: str) -> ToolResult:
        """Wait for an element to appear on the current page."""
        if not selector:
            return ToolResult(
                output="", success=False,
                error="'selector' is required for wait.",
            )
        try:
            await page.wait_for_selector(selector, timeout=10000)
        except Exception:
            return ToolResult(
                output="", success=False,
                error=f"Timed out waiting for '{selector}' (10s).",
            )
        return ToolResult(
            output=f"Element '{selector}' found on {page.url}",
            success=True,
        )
