"""Capture the KG graph with an entity selected → the dossier/info panel."""
import asyncio, os, sys

FRONTEND = os.environ.get("UI_FRONTEND", "http://nova-frontend:5173")
API_KEY = os.environ.get("NOVA_API_KEY", "")
OUT = "/data/ui_shots"


async def main():
    from playwright.async_api import async_playwright
    os.makedirs(OUT, exist_ok=True)
    entity = sys.argv[1] if len(sys.argv) > 1 else "Microsoft"
    name = sys.argv[2] if len(sys.argv) > 2 else "kg_dossier"
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        ctx = await browser.new_context(viewport={"width": 1380, "height": 1600}, device_scale_factor=2)
        await ctx.add_init_script(f"try{{localStorage.setItem('nova_api_key', {API_KEY!r});}}catch(e){{}}")
        page = await ctx.new_page()
        await page.goto(FRONTEND, wait_until="domcontentloaded")
        await page.evaluate(f"localStorage.setItem('nova_api_key', {API_KEY!r})")
        await page.goto(f"{FRONTEND}/#learning", wait_until="networkidle", timeout=45000)
        await page.wait_for_timeout(4000)
        await page.get_by_role("button", name="Knowledge Graph", exact=True).first.click(timeout=6000)
        await page.wait_for_timeout(2000)
        await page.get_by_role("button", name="Graph", exact=True).first.click(timeout=6000)
        await page.wait_for_timeout(3000)
        # Fill the entity search + Explore → selects entity, opens info panel.
        box = page.get_by_placeholder("Entity name", exact=False).first
        await box.fill(entity, timeout=6000)
        await page.get_by_role("button", name="Explore", exact=True).first.click(timeout=6000)
        await page.wait_for_timeout(9000)  # graph reload + dossier fetch + settle
        await page.screenshot(path=os.path.join(OUT, f"{name}.png"))
        print(f"shot: {OUT}/{name}.png")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
