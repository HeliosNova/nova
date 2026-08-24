"""Cosmos-app screenshot harness (2026-08-21). The whole app is now one 3D
space; capture the overview and flying into regions. Software-WebGL forced
(ANGLE/SwiftShader); backdrop-filter disabled for region panels (too slow to
composite in software — fine on a real GPU; panels are ~95% opaque anyway).

Run: docker exec nova-app python /data/cosmos_shot.py
"""
import asyncio
import os

FRONTEND = os.environ.get("UI_FRONTEND", "http://nova-frontend:5173")
API_KEY = os.environ.get("NOVA_API_KEY", "")
OUT = "/data/ui_shots"
VIEWPORT = {"width": 1512, "height": 945}
GL_ARGS = [
    "--no-sandbox", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
    "--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl",
]

# click a button (by visible text) inside the open region panel
CLICK_BTN = """(txt) => {
  const p = document.querySelector('.animate-slide-in-right') || document;
  const b = [...p.querySelectorAll('button')].find(x => x.textContent.trim().toLowerCase() === txt.toLowerCase());
  if (b) { b.click(); return true; } return false;
}"""
CLICK_FIRST_CARD = """() => {
  const p = document.querySelector('.animate-slide-in-right'); if (!p) return false;
  const b = p.querySelector('.space-y-2 > button'); if (b) { b.click(); return true; } return false;
}"""


async def region(page, name, hash_, shots):
    await page.evaluate(f"window.location.hash = {hash_!r}")
    await page.wait_for_timeout(4200)
    await page.screenshot(path=os.path.join(OUT, f"cosmos_{name}.png"), timeout=60000)
    print(f"shot: cosmos_{name}.png")
    for label, action in shots:
        ok = await page.evaluate(action[0], action[1]) if len(action) == 2 else await page.evaluate(action[0])
        print(f"  {label}: {ok}")
        await page.wait_for_timeout(1600)
        await page.screenshot(path=os.path.join(OUT, f"cosmos_{name}_{label}.png"), timeout=60000)
        print(f"shot: cosmos_{name}_{label}.png")


async def main():
    from playwright.async_api import async_playwright
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=GL_ARGS)
        ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        await ctx.add_init_script(f"try{{localStorage.setItem('nova_api_key', {API_KEY!r});}}catch(e){{}}")
        page = await ctx.new_page()
        page.on("pageerror", lambda e: print(f"  [pageerror] {e}"))
        page.on("console", lambda m: print(f"  [console.error] {m.text}") if m.type == "error" else None)

        await page.goto(FRONTEND, wait_until="domcontentloaded")
        await page.evaluate(f"localStorage.setItem('nova_api_key', {API_KEY!r})")
        await page.goto(f"{FRONTEND}/#", wait_until="domcontentloaded", timeout=45000)

        await page.wait_for_timeout(3500)
        gl = await page.evaluate("() => { const c=document.querySelector('canvas'); if(!c) return 'no-canvas'; const g=c.getContext('webgl2')||c.getContext('webgl'); return g?('ok '+c.width+'x'+c.height):'no-gl'; }")
        print(f"  canvas/gl: {gl}")

        await page.wait_for_timeout(3000)
        await page.screenshot(path=os.path.join(OUT, "cosmos_overview.png"))
        print("shot: cosmos_overview.png")

        # backdrop-blur is too slow to composite in software GL — disable for capture
        await page.add_style_tag(content="*{backdrop-filter:none !important}")

        # KNOWLEDGE — the force graph; needs longer to load + settle physics
        await page.evaluate("window.location.hash = 'knowledge'")
        await page.wait_for_timeout(9000)
        await page.screenshot(path=os.path.join(OUT, "cosmos_knowledge.png"), timeout=60000)
        print("shot: cosmos_knowledge.png")

        # DOSSIERS — the reference organization to compare regions against
        await region(page, "dossiers", "dossiers", [
            ("reader", (CLICK_FIRST_CARD,)),
        ])

        # STORYLINES — text list (default), a reader, then the map
        await region(page, "storylines", "storylines", [
            ("reader", (CLICK_FIRST_CARD,)),
            ("back", (CLICK_BTN, "All storylines")),
            ("map", (CLICK_BTN, "Map")),
        ])
        # FORECASTS — text list (default), then the field
        await region(page, "forecasts", "forecasts", [
            ("field", (CLICK_BTN, "Field")),
        ])
        # SIGNALS — in-panel reader
        await region(page, "signals", "signals", [
            ("reader", (CLICK_FIRST_CARD,)),
        ])

        # CHAT — must NOT show the legacy 280px conversation sidebar (H1)
        await page.evaluate("window.location.hash = 'chat'")
        await page.wait_for_timeout(4000)
        await page.screenshot(path=os.path.join(OUT, "cosmos_chat.png"), timeout=60000)
        print("shot: cosmos_chat.png")

        # SYSTEMS — legacy pages must not double-header (embedded chrome)
        await page.evaluate("window.location.hash = 'systems'")
        await page.wait_for_timeout(4500)
        await page.screenshot(path=os.path.join(OUT, "cosmos_systems.png"), timeout=60000)
        print("shot: cosmos_systems.png")
        ok = await page.evaluate(CLICK_BTN, "Monitors")
        print(f"  monitors tab: {ok}")
        await page.wait_for_timeout(3500)
        await page.screenshot(path=os.path.join(OUT, "cosmos_systems_monitors.png"), timeout=60000)
        print("shot: cosmos_systems_monitors.png")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
