"""Focused UI screenshots — capture specific sections legibly (2026-08-20).
docker exec: python /tmp/ui_shot_section.py <tab> <heading-substring> <out-name>
Scrolls the heading into view and screenshots a bounded region below it.
"""
import asyncio
import os
import sys

FRONTEND = os.environ.get("UI_FRONTEND", "http://nova-frontend:5173")
API_KEY = os.environ.get("NOVA_API_KEY", "")
OUT = "/data/ui_shots"


async def main():
    from playwright.async_api import async_playwright
    os.makedirs(OUT, exist_ok=True)
    tab = sys.argv[1] if len(sys.argv) > 1 else "bulletin"
    heading = sys.argv[2] if len(sys.argv) > 2 else "Storylines"
    name = sys.argv[3] if len(sys.argv) > 3 else "section"
    clicks = sys.argv[4].split("|") if len(sys.argv) > 4 else []

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        ctx = await browser.new_context(viewport={"width": 1380, "height": 1600}, device_scale_factor=2)
        await ctx.add_init_script(f"try{{localStorage.setItem('nova_api_key', {API_KEY!r});}}catch(e){{}}")
        page = await ctx.new_page()
        await page.goto(FRONTEND, wait_until="domcontentloaded")
        await page.evaluate(f"localStorage.setItem('nova_api_key', {API_KEY!r})")
        await page.goto(f"{FRONTEND}/#{tab}", wait_until="networkidle", timeout=45000)
        await page.wait_for_timeout(5000)
        for c in clicks:
            c = c.strip()
            if not c:
                continue
            try:
                # exact=True so "Graph" doesn't match "Knowledge Graph".
                await page.get_by_role("button", name=c, exact=True).first.click(timeout=5000)
                await page.wait_for_timeout(3500)
            except Exception as e:
                print(f"  (click {c!r} skipped: {str(e)[:80]})")
        # Extra settle for the force-graph (physics + zoomToFit fire on engine stop).
        settle = int(os.environ.get("UI_SETTLE_MS", "1200"))
        await page.wait_for_timeout(settle)
        # Scroll the heading into view.
        try:
            el = page.get_by_text(heading, exact=False).first
            await el.scroll_into_view_if_needed(timeout=6000)
            await page.wait_for_timeout(800)
        except Exception as e:
            print(f"  (heading {heading!r} not found: {e})")
        path = os.path.join(OUT, f"{name}.png")
        await page.screenshot(path=path)  # viewport only — legible
        print(f"shot: {path}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
