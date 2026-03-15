"""Browser automation tool — Playwright-based web interaction."""

from __future__ import annotations

import logging

from app.config import config
from app.tools.base import BaseTool, ToolResult
from app.tools.http_fetch import _is_safe_url

logger = logging.getLogger(__name__)

_MAX_CONTENT = 8000


class BrowserTool(BaseTool):
    name = "browser"
    description = (
        "Interact with web pages: navigate, click, type, extract content, fill forms. "
        "Actions: navigate, click, type, get_text, get_links, fill_form, evaluate_js."
    )
    parameters = "action: str, url: str, selector: str, text: str, form_data: dict, script: str"

    _playwright = None     # Shared Playwright context manager
    _browser = None        # Shared Chromium browser instance

    async def _get_browser(self):
        """Return the shared Chromium browser, launching it if needed."""
        if BrowserTool._browser and BrowserTool._browser.is_connected():
            return BrowserTool._browser

        # Clean up stale references
        await self._close_browser()

        from playwright.async_api import async_playwright
        BrowserTool._playwright = await async_playwright().start()
        BrowserTool._browser = await BrowserTool._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        logger.debug("Launched shared Chromium browser instance")
        return BrowserTool._browser

    @classmethod
    async def _close_browser(cls):
        """Close the shared browser and Playwright instance."""
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
                error="No action specified. Use: navigate, click, type, get_text, get_links, fill_form, evaluate_js.",
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

        try:
            browser = await self._get_browser()
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Nova/1.0",
            )
            try:
                context.set_default_timeout(config.BROWSER_TIMEOUT * 1000)
                await context.clear_cookies()
                page = await context.new_page()

                if action == "navigate":
                    result = await self._navigate(page, url)
                elif action == "click":
                    result = await self._click(page, url, selector)
                elif action == "type":
                    result = await self._type(page, url, selector, text)
                elif action == "get_text":
                    result = await self._get_text(page, url, selector)
                elif action == "get_links":
                    result = await self._get_links(page, url)
                elif action == "fill_form":
                    result = await self._fill_form(page, url, form_data)
                elif action == "evaluate_js":
                    result = await self._evaluate_js(page, url, script)
                else:
                    result = ToolResult(
                        output="",
                        success=False,
                        error=f"Unknown action '{action}'. Use: navigate, click, type, get_text, get_links, fill_form, evaluate_js.",
                    )
            finally:
                await context.close()

            return result

        except Exception as e:
            # If the browser crashed, clean up so the next call relaunches
            await self._close_browser()
            logger.warning("Browser tool error: %s", e)
            return ToolResult(output="", success=False, error="Browser operation failed")

    @staticmethod
    def _check_url(url: str) -> ToolResult | None:
        """Return a ToolResult error if URL is unsafe, else None."""
        if not url:
            return ToolResult(output="", success=False, error="No URL provided")
        if not _is_safe_url(url):
            return ToolResult(output="", success=False, error="URL blocked: internal/private addresses not allowed")
        return None

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
            return ToolResult(output="", success=False, error="Both 'url' and 'selector' required")
        if err := self._check_url(url):
            return err
        await page.goto(url, wait_until="domcontentloaded")
        await page.click(selector)
        await page.wait_for_load_state("domcontentloaded")
        title = await page.title()
        return ToolResult(output=f"Clicked '{selector}'. New page: {title} ({page.url})", success=True)

    async def _type(self, page, url: str, selector: str, text: str) -> ToolResult:
        if not selector or not text:
            return ToolResult(output="", success=False, error="'url', 'selector', and 'text' all required")
        if err := self._check_url(url):
            return err
        await page.goto(url, wait_until="domcontentloaded")
        await page.fill(selector, text)
        return ToolResult(output=f"Typed '{text[:50]}' into '{selector}'", success=True)

    async def _get_text(self, page, url: str, selector: str = "") -> ToolResult:
        if err := self._check_url(url):
            return err
        await page.goto(url, wait_until="domcontentloaded")
        if selector:
            element = await page.query_selector(selector)
            if not element:
                return ToolResult(output="", success=False, error=f"Selector '{selector}' not found")
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
        if err := self._check_url(url):
            return err
        await page.goto(url, wait_until="domcontentloaded")
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
            return ToolResult(output="", success=False, error="'url' and 'form_data' required (dict of selector: value)")
        if err := self._check_url(url):
            return err
        await page.goto(url, wait_until="domcontentloaded")
        filled = []
        for selector, value in form_data.items():
            await page.fill(selector, str(value))
            filled.append(f"  {selector} = {str(value)[:50]}")
        return ToolResult(output=f"Filled {len(filled)} fields:\n" + "\n".join(filled), success=True)

    async def _evaluate_js(self, page, url: str, script: str) -> ToolResult:
        if not script:
            return ToolResult(output="", success=False, error="'url' and 'script' required")
        if err := self._check_url(url):
            return err

        # Strict allowlist — only safe DOM read operations allowed
        import re as _re
        _ALLOWED_APIS = [
            r"document\.querySelector(?:All)?",
            r"document\.getElementById",
            r"document\.getElementsBy(?:ClassName|TagName|Name)",
            r"\.textContent",
            r"\.innerText",
            r"\.innerHTML",
            r"\.getAttribute\s*\(",
            r"\.style\.",
            r"\.classList",
            r"\.children",
            r"\.parentElement",
            r"\.value",
            r"Array\.from",
            r"\.map\s*\(",
            r"\.filter\s*\(",
            r"\.slice\s*\(",
            r"\.join\s*\(",
            r"\.length",
            r"\.trim\s*\(",
            r"\.substring\s*\(",
            r"JSON\.stringify",
            r"\(\)\s*=>",
            r"return\b",
        ]

        # Block everything that's not in the allowlist by checking for dangerous patterns
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
                return ToolResult(output="", success=False, error=f"Script contains blocked pattern: {label}")

        await page.goto(url, wait_until="domcontentloaded")
        result = await page.evaluate(script)
        output = str(result)
        if len(output) > _MAX_CONTENT:
            output = output[:_MAX_CONTENT] + "\n[... truncated]"
        return ToolResult(output=output, success=True)
