"""Immersive-hero screenshot harness (2026-08-21). Captures the WebGL Bulletin
cosmos, which needs (a) real WebGL in headless Chromium (forced via ANGLE +
SwiftShader software GL) and (b) VIEWPORT captures, not full_page stitching
(a GL canvas only renders the viewport). Writes frames to /data/ui_shots/.

Run:  docker exec nova-app python /data/ui_shot_hero.py
"""
import asyncio
import os

FRONTEND = os.environ.get("UI_FRONTEND", "http://nova-frontend:5173")
API_KEY = os.environ.get("NOVA_API_KEY", "")
OUT = "/data/ui_shots"
VIEWPORT = {"width": 1512, "height": 945}  # 16:10 desktop

GL_ARGS = [
    "--no-sandbox",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-swiftshader",
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-webgl",
    "--enable-accelerated-2d-canvas",
]


async def main():
    from playwright.async_api import async_playwright
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=GL_ARGS)
        ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        await ctx.add_init_script(f"try{{localStorage.setItem('nova_api_key', {API_KEY!r});}}catch(e){{}}")
        page = await ctx.new_page()

        # surface WebGL / console errors
        page.on("console", lambda m: print(f"  [console.{m.type}] {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: print(f"  [pageerror] {e}"))

        await page.goto(FRONTEND, wait_until="domcontentloaded")
        await page.evaluate(f"localStorage.setItem('nova_api_key', {API_KEY!r})")
        await page.goto(f"{FRONTEND}/#bulletin", wait_until="domcontentloaded", timeout=45000)

        # is there a <canvas> mounted, and does it have a GL context?
        await page.wait_for_timeout(3500)
        gl = await page.evaluate(
            """() => {
                const c = document.querySelector('canvas');
                if (!c) return 'no-canvas';
                const g = c.getContext('webgl2') || c.getContext('webgl');
                return g ? ('webgl-ok ' + c.width + 'x' + c.height) : 'no-gl-context';
            }"""
        )
        print(f"  canvas/gl: {gl}")

        # settled hero (after intro + data + brief)
        await page.wait_for_timeout(4200)
        await page.screenshot(path=os.path.join(OUT, "hero.png"))
        print("shot: hero.png")

        # IMMERSIVE EVERYWHERE: a content page should show the ambient cosmos behind it.
        # NOTE: backdrop-filter over a live WebGL canvas is far too slow to composite
        # in the software renderer here (screenshot times out); disable it + CSS anims
        # for capture only. On a real GPU the frosted glass is fine.
        await page.add_style_tag(content="*{backdrop-filter:none !important;animation:none !important;transition:none !important}")
        for tab in ("learning", "settings", "documents", "actions"):
            await page.evaluate(f"window.location.hash = '{tab}'")
            await page.wait_for_timeout(4500)
            try:
                await page.screenshot(path=os.path.join(OUT, f"page_{tab}.png"), timeout=60000)
                print(f"shot: page_{tab}.png")
            except Exception as e:
                print(f"  (page_{tab} screenshot failed: {e})")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
