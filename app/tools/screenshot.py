"""Screenshot tool — capture web page screenshots via Playwright."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from app.config import config
from app.core.access_tiers import requires_tier
from app.tools.base import BaseTool, ToolResult, ErrorCategory
from app.tools.http_fetch import _is_safe_url

logger = logging.getLogger(__name__)

_SCREENSHOT_DIR = Path(config.SCREENSHOT_DIR)


class ScreenshotTool(BaseTool):
    name = "screenshot"
    description = (
        "Take a screenshot of a web page using a headless Playwright browser. Saves PNG to "
        "/data/screenshots/ and returns the file path, page title, and dimensions. Supports full-page "
        "capture and element-specific screenshots via CSS selector. Use for visual analysis, archiving, "
        "or debugging web pages. Do NOT use for interactive browsing (use the browser tool) or fetching "
        "page content as text (use http_fetch)."
    )
    parameters = "url: str, full_page: bool, selector: str"
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the web page to screenshot.",
            },
            "full_page": {
                "type": "boolean",
                "description": "If true, capture the full scrollable page. Defaults to false (viewport only).",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector to screenshot a specific element instead of the full page.",
            },
        },
        "required": ["url"],
    }

    @requires_tier("standard", "full")
    async def execute(
        self,
        *,
        url: str = "",
        full_page: bool = False,
        selector: str = "",
        **kwargs,
    ) -> ToolResult:
        if not url:
            return ToolResult(output="", success=False, error="No URL provided", error_category=ErrorCategory.VALIDATION)

        if not _is_safe_url(url):
            return ToolResult(output="", success=False, error="URL blocked: internal/private addresses not allowed", error_category=ErrorCategory.PERMISSION)

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return ToolResult(
                output="",
                success=False,
                error="Playwright not installed. Screenshot tool unavailable.",
                error_category=ErrorCategory.INTERNAL,
            )

        try:
            _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
                )
                context = await browser.new_context(viewport={"width": 1280, "height": 720})
                context.set_default_timeout(config.BROWSER_TIMEOUT * 1000)
                page = await context.new_page()

                try:
                    await page.goto(url, wait_until="domcontentloaded")
                    await page.wait_for_timeout(2000)
                    title = await page.title()

                    parsed = urlparse(url)
                    safe_host = parsed.hostname.replace(".", "_") if parsed.hostname else "page"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{safe_host}_{timestamp}.png"
                    filepath = _SCREENSHOT_DIR / filename

                    screenshot_args = {"path": str(filepath), "type": "png"}

                    if selector:
                        element = await page.query_selector(selector)
                        if element:
                            await element.screenshot(**screenshot_args)
                        else:
                            return ToolResult(
                                output="",
                                success=False,
                                error=f"Selector '{selector}' not found on page",
                                error_category=ErrorCategory.NOT_FOUND,
                            )
                    else:
                        screenshot_args["full_page"] = full_page
                        await page.screenshot(**screenshot_args)

                    file_size = filepath.stat().st_size
                    return ToolResult(
                        output=(
                            f"Screenshot saved: /data/screenshots/{filename}\n"
                            f"Page: {title}\n"
                            f"Size: {file_size:,} bytes\n"
                            f"Dimensions: {'full page' if full_page else '1280x720'}"
                        ),
                        success=True,
                    )

                finally:
                    await browser.close()

        except Exception as e:
            return ToolResult(output="", success=False, error=f"Screenshot failed: {e}", error_category=ErrorCategory.TRANSIENT)
