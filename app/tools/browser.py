"""Browser automation tool — Playwright-based web interaction with persistent sessions.

Sessions persist across tool calls so Nova can log in, navigate, click, and
read across multiple steps — just like a human using a browser.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.config import config
from app.core.access_tiers import requires_tier
from app.tools.base import BaseTool, ToolResult, ErrorCategory
from app.tools.http_fetch import _is_safe_url

logger = logging.getLogger(__name__)

def _max_content_length() -> int:
    return config.TOOL_OUTPUT_MAX_CHARS
_SESSION_TIMEOUT = 600  # 10 min idle timeout

# ---------------------------------------------------------------------------
# JS snippet — extract visible interactive elements with reliable selectors
# ---------------------------------------------------------------------------

_INTERACTIVE_ELEMENTS_JS = """
() => {
    function isVisible(el) {
        if (!el.offsetParent && el.tagName !== 'BODY') return false;
        const s = getComputedStyle(el);
        return s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0';
    }
    function getSelector(el) {
        if (el.id) return '#' + CSS.escape(el.id);
        if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name.replace(/"/g, '\\\\"') + '"]';
        const classes = Array.from(el.classList).filter(c => c.length > 0 && c.length < 40);
        if (classes.length) {
            const sel = el.tagName.toLowerCase() + '.' + classes.map(c => CSS.escape(c)).join('.');
            if (document.querySelectorAll(sel).length === 1) return sel;
        }
        const text = (el.innerText || '').trim().substring(0, 30);
        if (text && el.tagName !== 'INPUT' && el.tagName !== 'SELECT') {
            const sel = el.tagName.toLowerCase() + ':has-text("' + text.replace(/"/g, '\\\\"') + '")';
            return sel;
        }
        const parent = el.parentElement;
        if (parent) {
            const siblings = Array.from(parent.children).filter(c => c.tagName === el.tagName);
            if (siblings.length > 1) {
                const idx = siblings.indexOf(el) + 1;
                return el.tagName.toLowerCase() + ':nth-of-type(' + idx + ')';
            }
        }
        return el.tagName.toLowerCase();
    }
    function getLabel(el) {
        if (el.labels && el.labels.length) return el.labels[0].innerText.trim().substring(0, 40);
        const aria = el.getAttribute('aria-label');
        if (aria) return aria.substring(0, 40);
        const ph = el.getAttribute('placeholder');
        if (ph) return ph.substring(0, 40);
        const title = el.getAttribute('title');
        if (title) return title.substring(0, 40);
        const text = (el.innerText || el.value || '').trim().substring(0, 40);
        return text;
    }
    const result = {buttons: [], inputs: [], links: [], selects: [], radioGroups: [], checkboxGroups: []};

    // Buttons
    document.querySelectorAll('button, input[type="button"], input[type="submit"], input[type="reset"], [role="button"]').forEach(el => {
        if (!isVisible(el)) return;
        result.buttons.push({label: getLabel(el), selector: getSelector(el), type: el.type || ''});
    });

    // Text-like inputs + textareas (NOT radio/checkbox — those go in groups)
    document.querySelectorAll('input:not([type="button"]):not([type="submit"]):not([type="reset"]):not([type="hidden"]):not([type="radio"]):not([type="checkbox"]), textarea').forEach(el => {
        if (!isVisible(el)) return;
        const val = (el.value || '').substring(0, 40);
        result.inputs.push({label: getLabel(el), selector: getSelector(el), type: el.type || 'text', value: val});
    });

    // Radio groups — group by name, show options + which is selected
    const radioNames = new Set();
    document.querySelectorAll('input[type="radio"]').forEach(el => {
        if (el.name) radioNames.add(el.name);
    });
    radioNames.forEach(name => {
        const radios = Array.from(document.querySelectorAll('input[type="radio"][name="' + name + '"]'));
        const options = radios.map(r => {
            const lbl = getLabel(r) || r.value;
            return r.checked ? lbl + ' [selected]' : lbl;
        });
        const selected = radios.find(r => r.checked);
        result.radioGroups.push({
            label: getLabel(radios[0]) || name,
            selector: 'input[name="' + name + '"]',
            type: 'radio',
            options: options.join(', '),
            selected: selected ? (selected.value || '') : ''
        });
    });

    // Checkbox groups — group by name, show options + which are checked
    const checkNames = new Set();
    document.querySelectorAll('input[type="checkbox"]').forEach(el => {
        if (el.name) checkNames.add(el.name);
    });
    checkNames.forEach(name => {
        const checks = Array.from(document.querySelectorAll('input[type="checkbox"][name="' + name + '"]'));
        const options = checks.map(c => {
            const lbl = getLabel(c) || c.value;
            return c.checked ? lbl + ' [checked]' : lbl;
        });
        const checked = checks.filter(c => c.checked).map(c => c.value);
        result.checkboxGroups.push({
            label: getLabel(checks[0]) || name,
            selector: 'input[name="' + name + '"]',
            type: 'checkbox',
            options: options.join(', '),
            checked: checked.join(', ')
        });
    });

    // Selects
    document.querySelectorAll('select').forEach(el => {
        if (!isVisible(el)) return;
        const opts = Array.from(el.options).slice(0, 8).map(o => o.text.trim()).join(', ');
        const selected = el.options[el.selectedIndex] ? el.options[el.selectedIndex].text.trim() : '';
        result.selects.push({label: getLabel(el), selector: getSelector(el), options: opts, selected: selected});
    });

    // Links
    document.querySelectorAll('a[href]').forEach(el => {
        if (!isVisible(el)) return;
        const text = (el.innerText || '').trim().substring(0, 60);
        if (!text) return;
        result.links.push({label: text, selector: getSelector(el), href: el.href});
    });

    // Cap each category
    result.buttons = result.buttons.slice(0, 20);
    result.inputs = result.inputs.slice(0, 20);
    result.radioGroups = result.radioGroups.slice(0, 10);
    result.checkboxGroups = result.checkboxGroups.slice(0, 10);
    result.selects = result.selects.slice(0, 10);
    result.links = result.links.slice(0, 30);
    return result;
}
"""


class BrowserTool(BaseTool):
    name = "browser"
    description = (
        "Control a persistent web browser for multi-step web interactions. "
        "Sessions survive across calls, enabling login flows, form filling, and multi-page navigation. "
        "Actions: navigate (load URL, returns page text and interactive elements), click (CSS selector), "
        "type (fill field), press_key, get_text, get_links, get_interactive_elements, "
        "fill_form (fill all fields at once), evaluate_js (run JavaScript), screenshot, back, wait, close_session. "
        "Always use CSS selectors from the interactive elements output — do NOT guess selectors. "
        "For forms, use a single fill_form call with all fields, then click submit. "
        "Do NOT use for simple URL fetches (use http_fetch) or screenshots of a URL (use screenshot tool)."
    )
    parameters = (
        "action: str (navigate|click|type|press_key|get_text|get_links|get_interactive_elements|fill_form|screenshot|back|wait|close_session), "
        "url: str (only needed for navigate or first visit), "
        "selector: str (CSS selector from interactive elements output — for click/type/get_text/wait), "
        "text: str (text to type), "
        "form_data: dict (selector:value pairs — works for text, radio, checkbox, select fields), "
        "script: str (JS for evaluate_js)"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate", "click", "type", "press_key", "get_text", "get_links",
                         "get_interactive_elements", "fill_form", "evaluate_js",
                         "screenshot", "back", "wait", "close_session"],
                "description": "Browser action to perform.",
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to. Required for navigate, optional for other actions (uses current page if omitted).",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for the target element. Use selectors from interactive elements output.",
            },
            "text": {
                "type": "string",
                "description": "Text to type, or key name for press_key (e.g., 'Enter', 'Tab').",
            },
            "form_data": {
                "type": "object",
                "description": "Dict of CSS selector to value pairs for fill_form. Handles text, radio, checkbox, and select inputs.",
            },
            "script": {
                "type": "string",
                "description": "JavaScript code for evaluate_js action.",
            },
        },
        "required": ["action"],
    }

    _playwright = None
    _browser = None
    _last_used = 0.0      # Timestamp for session timeout

    def trim_output(self, output: str) -> str:
        """Keep page header + text excerpt + interactive elements."""
        if len(output) <= 2000:
            return output
        # Try to preserve the interactive elements section at the end
        ie_marker = '--- Interactive Elements'
        ie_idx = output.rfind(ie_marker)
        if ie_idx > 0:
            ie_section = output[ie_idx:][:500]
            budget = 2000 - len(ie_section) - 20
            return output[:budget] + '\n[...truncated]\n' + ie_section
        return output[:2000] + '\n[...truncated]'

    async def _ensure_browser(self):
        """Ensure the shared browser is running; create playwright + browser if needed."""
        now = time.time()

        # Session timeout — close stale browser
        if (BrowserTool._browser and BrowserTool._last_used
                and now - BrowserTool._last_used > _SESSION_TIMEOUT):
            logger.info("[Browser] Session timed out after %ds idle", _SESSION_TIMEOUT)
            await self._close_session()

        # Reuse existing browser if still connected
        if BrowserTool._browser and BrowserTool._browser.is_connected():
            try:
                # Quick liveness check — create and close a throwaway context
                test_ctx = await BrowserTool._browser.new_context()
                await test_ctx.close()
                BrowserTool._last_used = now
                return
            except Exception:
                logger.info("[Browser] Stale browser detected, reconnecting")

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
                # CDP unavailable — fall back to headless launch
                logger.warning("[Browser] Cannot reach host browser at %s: %s — launching headless", cdp_url, e)
                cdp_url = None  # Fall through to headless launch below

        if cdp_url:
            BrowserTool._browser = await BrowserTool._playwright.chromium.connect_over_cdp(ws_url)
            logger.info("[Browser] Connected to host browser at %s", cdp_url)
        else:
            BrowserTool._browser = await BrowserTool._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            )
            logger.info("[Browser] Launched headless browser")

        BrowserTool._last_used = now

    async def _new_context_and_page(self):
        """Create a fresh isolated browser context + page for a single execute() call."""
        context = await BrowserTool._browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        context.set_default_timeout(config.BROWSER_TIMEOUT * 1000)
        page = await context.new_page()
        logger.info("[Browser] New isolated context created")
        return context, page

    @classmethod
    async def _close_session(cls):
        """Close the shared browser + playwright."""
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

    @requires_tier("standard", "full")
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
                error_category=ErrorCategory.VALIDATION,
            )

        try:
            from playwright.async_api import async_playwright  # noqa: F401
        except ImportError:
            return ToolResult(
                output="",
                success=False,
                error="Playwright not installed. Browser tool unavailable.",
                error_category=ErrorCategory.PERMISSION,
            )

        action = action.lower()

        # close_session doesn't need a page
        if action == "close_session":
            await self._close_session()
            return ToolResult(output="Browser session closed.", success=True)

        context = None
        try:
            await self._ensure_browser()
            context, page = await self._new_context_and_page()

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
            elif action == "get_interactive_elements":
                return await self._get_interactive_elements(page, url)
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
                    error_category=ErrorCategory.VALIDATION,
                )

        except Exception as e:
            logger.warning("[Browser] Action '%s' failed: %s", action, e)
            # Only kill session if browser itself is dead, not on action errors
            if BrowserTool._browser and not BrowserTool._browser.is_connected():
                await self._close_session()
            return ToolResult(
                output="", success=False,
                error=f"Browser action failed: {e}",
                retriable=True,
                error_category=ErrorCategory.TRANSIENT,
            )
        finally:
            if context:
                try:
                    await context.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Interactive element extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _format_interactive_elements(data: dict) -> str:
        """Format extracted interactive elements compactly for LLM consumption."""
        sections = []
        if data.get("buttons"):
            items = []
            for i, b in enumerate(data["buttons"], 1):
                type_hint = f", {b['type']}" if b.get("type") else ""
                items.append(f'[{i}] "{b["label"]}" ({b["selector"]}{type_hint})')
            sections.append("Buttons: " + "  ".join(items))
        if data.get("inputs"):
            items = []
            for i, inp in enumerate(data["inputs"], 1):
                type_hint = f", {inp['type']}" if inp.get("type") else ""
                val = inp.get("value", "")
                val_hint = f' = "{val}"' if val else " (empty)"
                items.append(f'[{i}] "{inp["label"]}" ({inp["selector"]}{type_hint}){val_hint}')
            sections.append("Inputs: " + "  ".join(items))
        if data.get("radioGroups"):
            items = []
            for i, rg in enumerate(data["radioGroups"], 1):
                sel = rg.get("selected", "")
                sel_hint = f', selected="{sel}"' if sel else ", none selected"
                items.append(f'[{i}] "{rg["label"]}" ({rg["selector"]}, radio, options: {rg["options"]}{sel_hint})')
            sections.append("Radio groups: " + "  ".join(items))
        if data.get("checkboxGroups"):
            items = []
            for i, cg in enumerate(data["checkboxGroups"], 1):
                chk = cg.get("checked", "")
                chk_hint = f', checked="{chk}"' if chk else ", none checked"
                items.append(f'[{i}] "{cg["label"]}" ({cg["selector"]}, checkbox, options: {cg["options"]}{chk_hint})')
            sections.append("Checkbox groups: " + "  ".join(items))
        if data.get("selects"):
            items = []
            for i, s in enumerate(data["selects"], 1):
                opts = f", options: {s['options']}" if s.get("options") else ""
                sel = s.get("selected", "")
                sel_hint = f', selected="{sel}"' if sel else ""
                items.append(f'[{i}] "{s["label"]}" ({s["selector"]}{opts}{sel_hint})')
            sections.append("Selects: " + "  ".join(items))
        if data.get("links"):
            items = []
            for i, l in enumerate(data["links"], 1):
                items.append(f'[{i}] "{l["label"]}" ({l["selector"]})')
            sections.append("Links: " + "  ".join(items))
        return "\n".join(sections) if sections else "(no interactive elements found)"

    async def _extract_interactive(self, page) -> str:
        """Run the interactive elements JS and return formatted string."""
        try:
            data = await page.evaluate(_INTERACTIVE_ELEMENTS_JS)
            return self._format_interactive_elements(data)
        except Exception as e:
            logger.debug("[Browser] Interactive element extraction failed: %s", e)
            return "(could not extract interactive elements)"

    # ------------------------------------------------------------------
    # URL safety
    # ------------------------------------------------------------------

    @staticmethod
    def _check_url(url: str) -> ToolResult | None:
        """Return a ToolResult error if URL is unsafe, else None."""
        if not url:
            return ToolResult(output="", success=False, error="No URL provided", error_category=ErrorCategory.VALIDATION)
        if not _is_safe_url(url):
            return ToolResult(
                output="", success=False,
                error="URL blocked: internal/private addresses not allowed",
                error_category=ErrorCategory.PERMISSION,
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
                    error_category=ErrorCategory.VALIDATION,
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
        if len(text) > _max_content_length():
            text = text[:_max_content_length()] + "\n[... content truncated]"
        interactive = await self._extract_interactive(page)
        output = f"Page: {title}\nURL: {page.url}\n\n{text}\n\n--- Interactive Elements ---\n{interactive}"
        if config.ENABLE_INJECTION_DETECTION:
            from app.core.injection import sanitize_content
            output = sanitize_content(output, context="web page")
        return ToolResult(output=output, success=True)

    async def _click(self, page, url: str, selector: str) -> ToolResult:
        if not selector:
            return ToolResult(
                output="", success=False,
                error="'selector' is required for click.",
                error_category=ErrorCategory.VALIDATION,
            )
        if err := await self._maybe_navigate(page, url):
            return err
        try:
            await page.click(selector)
        except Exception as click_err:
            interactive = await self._extract_interactive(page)
            return ToolResult(
                output="",
                success=False,
                error=(
                    f"Selector '{selector}' not found. "
                    f"Available elements:\n{interactive}"
                ),
                error_category=ErrorCategory.NOT_FOUND,
            )
        await page.wait_for_load_state("domcontentloaded")
        title = await page.title()
        text = await page.evaluate("() => document.body.innerText")
        if len(text) > 3000:
            text = text[:3000] + "\n[... truncated]"
        interactive = await self._extract_interactive(page)
        return ToolResult(
            output=(
                f"Clicked '{selector}'. Page: {title} ({page.url})\n\n{text}"
                f"\n\n--- Interactive Elements ---\n{interactive}"
            ),
            success=True,
        )

    async def _type(self, page, url: str, selector: str, text: str) -> ToolResult:
        if not selector or not text:
            return ToolResult(
                output="", success=False,
                error="'selector' and 'text' are required for type.",
                error_category=ErrorCategory.VALIDATION,
            )
        if err := await self._maybe_navigate(page, url):
            return err
        try:
            await page.fill(selector, text)
        except Exception:
            interactive = await self._extract_interactive(page)
            return ToolResult(
                output="",
                success=False,
                error=(
                    f"Selector '{selector}' not found. "
                    f"Available elements:\n{interactive}"
                ),
                error_category=ErrorCategory.NOT_FOUND,
            )
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
                error_category=ErrorCategory.VALIDATION,
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
                error_category=ErrorCategory.VALIDATION,
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
                    error_category=ErrorCategory.NOT_FOUND,
                )
            text = await element.inner_text()
        else:
            text = await page.evaluate("() => document.body.innerText")
        if len(text) > _max_content_length():
            text = text[:_max_content_length()] + "\n[... content truncated]"
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

    @staticmethod
    def _escape_css_value(s: str) -> str:
        """Escape characters that are special in CSS selector attribute values."""
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("]", "\\]")

    async def _fill_field(self, page, sel: str, value: str) -> None:
        """Fill a single form field, auto-detecting type for radio/checkbox/select."""
        el_info = await page.evaluate(
            """(sel) => {
                const el = document.querySelector(sel);
                if (!el) return null;
                const tag = el.tagName.toLowerCase();
                const type = (el.type || '').toLowerCase();
                // For radio/checkbox groups, collect all values in this name group
                let values = [];
                if ((type === 'radio' || type === 'checkbox') && el.name) {
                    document.querySelectorAll(
                        el.tagName + '[name="' + el.name + '"]'
                    ).forEach(r => values.push(r.value));
                }
                return {tag: tag, type: type, name: el.name || '', values: values};
            }""",
            sel,
        )
        if el_info is None:
            raise ValueError(f"Selector '{sel}' not found")

        tag = el_info["tag"]
        el_type = el_info["type"]
        group_values = el_info.get("values", [])

        if tag == "select":
            # Try exact, then case-insensitive label/value match
            try:
                await page.select_option(sel, value)
            except Exception:
                await page.select_option(sel, label=value)
        elif el_type == "radio":
            # Find matching value (case-insensitive)
            match = next(
                (v for v in group_values if v.lower() == value.lower()),
                None,
            )
            if match is None:
                raise ValueError(
                    f"Radio '{sel}' has no value matching '{value}' "
                    f"(available: {', '.join(group_values)})"
                )
            name = self._escape_css_value(el_info["name"])
            match_escaped = self._escape_css_value(match)
            await page.click(f'input[name="{name}"][value="{match_escaped}"]')
        elif el_type == "checkbox":
            name = el_info["name"]
            # If value looks like a checkbox value/label, check the right one
            bool_values = {"true", "false", "0", "1", "on", "off", "yes", "no", ""}
            if value.lower() in bool_values:
                want_checked = value.lower() not in ("false", "0", "off", "no", "")
                is_checked = await page.is_checked(sel)
                if is_checked != want_checked:
                    await page.click(sel)
            else:
                # Value is a label/name — find the checkbox with matching value
                match = next(
                    (v for v in group_values if v.lower() == value.lower()),
                    None,
                )
                if match and name:
                    name_escaped = self._escape_css_value(name)
                    match_escaped = self._escape_css_value(match)
                    target = f'input[name="{name_escaped}"][value="{match_escaped}"]'
                    if not await page.is_checked(target):
                        await page.click(target)
                else:
                    # Fallback: just check this one
                    if not await page.is_checked(sel):
                        await page.click(sel)
        else:
            await page.fill(sel, value)

    async def _fill_form(self, page, url: str, form_data: dict | None) -> ToolResult:
        if not form_data:
            return ToolResult(
                output="", success=False,
                error="'form_data' required (dict of selector: value).",
                error_category=ErrorCategory.VALIDATION,
            )
        if err := await self._maybe_navigate(page, url):
            return err
        filled = []
        failed = []
        for sel, value in form_data.items():
            try:
                await self._fill_field(page, sel, str(value))
                filled.append(f"  {sel} = {str(value)[:50]}")
            except Exception:
                failed.append(sel)
        if failed and not filled:
            interactive = await self._extract_interactive(page)
            return ToolResult(
                output="",
                success=False,
                error=(
                    f"Selector(s) not found: {', '.join(failed)}. "
                    f"Available elements:\n{interactive}"
                ),
                error_category=ErrorCategory.NOT_FOUND,
            )
        interactive = await self._extract_interactive(page)
        output = f"Filled {len(filled)} field(s) {self._page_summary(page)}:\n" + "\n".join(filled)
        if failed:
            output += f"\n\nFailed selectors: {', '.join(failed)}"
        output += f"\n\n--- Interactive Elements (current state) ---\n{interactive}"
        return ToolResult(output=output, success=True)

    async def _evaluate_js(self, page, url: str, script: str) -> ToolResult:
        if not script:
            return ToolResult(
                output="", success=False,
                error="'script' is required for evaluate_js.",
                error_category=ErrorCategory.VALIDATION,
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
            (r"\bdocument\s*\.\s*location\b", "document.location"),
            (r"\bwindow\s*\.\s*location\b", "window.location"),
            (r"\bdocument\s*\.\s*forms\b", "document.forms"),
            (r"\bServiceWorker\b", "ServiceWorker"),
            (r"\bindexedDB\b", "indexedDB"),
            (r"\bsendBeacon\b", "sendBeacon"),
            (r"\bfromCharCode\b", "fromCharCode"),
            (r"\bfromCodePoint\b", "fromCodePoint"),
            (r"data\s*:\s*text/html", "data:text/html"),
            (r"\bsrcdoc\s*=", "srcdoc="),
            (r"\batob\b", "atob"),
            (r"\bbtoa\b", "btoa"),
        ]
        for pattern, label in _BLOCKED_PATTERNS:
            if _re.search(pattern, script):
                return ToolResult(
                    output="", success=False,
                    error=f"Script contains blocked pattern: {label}",
                    error_category=ErrorCategory.PERMISSION,
                )

        if err := await self._maybe_navigate(page, url):
            return err

        result = await page.evaluate(script)
        output = str(result)
        if len(output) > _max_content_length():
            output = output[:_max_content_length()] + "\n[... truncated]"
        return ToolResult(output=output, success=True)

    async def _screenshot(self, page) -> ToolResult:
        """Capture current page as PNG screenshot."""
        from pathlib import Path
        from datetime import datetime, timezone

        screenshot_dir = Path(config.SCREENSHOT_DIR)
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

    async def _get_interactive_elements(self, page, url: str) -> ToolResult:
        """Return all interactive elements on the current page with selectors."""
        if err := await self._maybe_navigate(page, url):
            return err
        interactive = await self._extract_interactive(page)
        title = await page.title()
        return ToolResult(
            output=f"Page: {title} ({page.url})\n\n{interactive}",
            success=True,
        )

    async def _wait(self, page, selector: str) -> ToolResult:
        """Wait for an element to appear on the current page."""
        if not selector:
            return ToolResult(
                output="", success=False,
                error="'selector' is required for wait.",
                error_category=ErrorCategory.VALIDATION,
            )
        try:
            await page.wait_for_selector(selector, timeout=10000)
        except Exception:
            return ToolResult(
                output="", success=False,
                error=f"Timed out waiting for '{selector}' (10s).",
                retriable=True,
                error_category=ErrorCategory.TRANSIENT,
            )
        return ToolResult(
            output=f"Element '{selector}' found on {page.url}",
            success=True,
        )
