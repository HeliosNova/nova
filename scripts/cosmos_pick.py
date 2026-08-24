"""Verify the cosmos-as-graph picking: sweep clicks over the galaxy until one
lands on a real KG node, then screenshot the entity card + 3D highlight.
Run: docker exec nova-app python /data/cosmos_pick.py
"""
import asyncio, os

FRONTEND = os.environ.get("UI_FRONTEND", "http://nova-frontend:5173")
API_KEY = os.environ.get("NOVA_API_KEY", "")
OUT = "/data/ui_shots"
VIEWPORT = {"width": 1512, "height": 945}
GL_ARGS = ["--no-sandbox", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
           "--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl"]


async def main():
    from playwright.async_api import async_playwright
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=GL_ARGS)
        ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        await ctx.add_init_script(f"try{{localStorage.setItem('nova_api_key', {API_KEY!r});}}catch(e){{}}")
        page = await ctx.new_page()
        page.on("pageerror", lambda e: print(f"  [pageerror] {e}"))
        await page.goto(FRONTEND, wait_until="domcontentloaded")
        await page.evaluate(f"localStorage.setItem('nova_api_key', {API_KEY!r})")
        await page.goto(f"{FRONTEND}/#", wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(7000)

        async def has_card():
            try:
                return await page.evaluate("() => !!document.body.innerText.match(/connections traced/)")
            except Exception:
                return False

        async def ensure_overview():
            # a stray click may have hit a waypoint and navigated; return home
            try:
                h = await page.evaluate("() => location.hash")
                if h and h not in ("", "#"):
                    await page.evaluate("() => { location.hash=''; }")
                    await page.wait_for_timeout(500)
            except Exception:
                pass

        # sweep only the dense galactic CORE (center screen) — away from the ring
        # waypoints so we pick stars, not navigate
        hit = False
        for y in range(410, 610, 16):
            for x in range(640, 880, 16):
                await page.mouse.click(x, y)
                if await has_card():
                    hit = True
                    print(f"  node picked at ({x},{y})")
                    break
                await ensure_overview()
            if hit:
                break
        print(f"  card visible: {hit}")
        await page.wait_for_timeout(800)
        await page.screenshot(path=os.path.join(OUT, "cosmos_pick.png"))
        print("shot: cosmos_pick.png")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
