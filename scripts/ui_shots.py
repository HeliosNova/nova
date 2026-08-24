"""UI screenshot harness (2026-08-20) — drives the app container's Chromium to
capture the live frontend so visual work can be verified without the Chrome
extension. Loads http://nova-frontend:5173 (Vite proxies /api to nova-app),
injects the API key into localStorage, visits each hash route, writes PNGs to
/data/ui_shots/. Run: docker exec nova-app python scripts/ui_shots.py [tabs...]
"""
import asyncio
import os
import sys

FRONTEND = os.environ.get("UI_FRONTEND", "http://nova-frontend:5173")
API_KEY = os.environ.get("NOVA_API_KEY", "")
OUT = "/data/ui_shots"
DEFAULT_TABS = ["bulletin", "learning", "knowing", "chat", "monitors"]
# Optional per-tab warm-up clicks (selector text) to expand things before shot.
VIEWPORT = {"width": 1440, "height": 2200}


async def main():
    from playwright.async_api import async_playwright
    os.makedirs(OUT, exist_ok=True)
    tabs = sys.argv[1:] or DEFAULT_TABS
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        # Seed the API key before any app code runs.
        await ctx.add_init_script(f"try{{localStorage.setItem('nova_api_key', {API_KEY!r});}}catch(e){{}}")
        page = await ctx.new_page()
        # Prime localStorage on the real origin, then reload into each route.
        await page.goto(FRONTEND, wait_until="domcontentloaded")
        await page.evaluate(f"localStorage.setItem('nova_api_key', {API_KEY!r})")
        for tab in tabs:
            url = f"{FRONTEND}/#{tab}"
            try:
                await page.goto(url, wait_until="networkidle", timeout=45000)
            except Exception:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            # Let lazy chunks + data fetches settle and force graph physics run.
            await page.wait_for_timeout(6000 if tab in ("learning", "bulletin") else 3500)
            # On learning, switch to the Graph view + KG tab if present.
            if tab == "learning":
                try:
                    await page.get_by_role("button", name="Knowledge Graph").first.click(timeout=4000)
                    await page.wait_for_timeout(1500)
                    await page.get_by_role("button", name="Graph").first.click(timeout=4000)
                    await page.wait_for_timeout(7000)  # force-graph settle
                except Exception as e:
                    print(f"  (learning graph nav skipped: {e})")
            path = os.path.join(OUT, f"{tab}.png")
            await page.screenshot(path=path, full_page=True)
            print(f"shot: {path}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
