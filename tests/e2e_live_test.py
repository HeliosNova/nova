"""Live E2E tests - exercises Nova's tools against real services.

No Ollama required. Uses a temp SQLite database.
Tests: BrowserTool, ScreenshotTool, SkillStore, LearningEngine.
"""

import asyncio
import io
import os
import sys
import tempfile
import traceback

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Override DB_PATH and TRAINING_DATA_PATH before any imports touch config
_tmpdir = tempfile.mkdtemp(prefix="nova_e2e_")
os.environ["DB_PATH"] = os.path.join(_tmpdir, "test.db")
os.environ["TRAINING_DATA_PATH"] = os.path.join(_tmpdir, "training.jsonl")

from app.database import SafeDB
from app.tools.base import ToolResult

# -- Globals --
PASS = 0
FAIL = 0
ERRORS: list[str] = []


def report(name: str, result: ToolResult | None = None, ok: bool | None = None,
           detail: str = ""):
    """Print pass/fail for a single test."""
    global PASS, FAIL
    if ok is None and result is not None:
        ok = result.success
    symbol = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
        ERRORS.append(name)
    extra = ""
    if result and result.output:
        preview = result.output[:120].replace("\n", " | ")
        extra = f"  -> {preview}"
    elif detail:
        extra = f"  -> {detail}"
    elif result and result.error:
        extra = f"  !! {result.error}"
    print(f"  [{symbol}] {name}{extra}")


# ═══════════════════════════════════════════════════════════════════
# 1. BROWSER TOOL
# ═══════════════════════════════════════════════════════════════════

async def test_browser():
    print("\n====== 1. BrowserTool ======")
    from app.tools.browser import BrowserTool
    bt = BrowserTool()

    # 1a. navigate
    r = await bt.execute(action="navigate", url="https://example.com")
    report("navigate example.com", r)

    # 1b. get_text (whole page)
    r = await bt.execute(action="get_text", url="https://example.com")
    report("get_text (whole page)", r)
    if r.success:
        ok = "Example Domain" in r.output
        report("  content contains 'Example Domain'", ok=ok,
               detail=f"found={'Example Domain' in r.output}")

    # 1c. get_text with selector
    r = await bt.execute(action="get_text", url="https://example.com", selector="h1")
    report("get_text selector='h1'", r)
    if r.success:
        report("  h1 == 'Example Domain'", ok=r.output.strip() == "Example Domain",
               detail=repr(r.output.strip()))

    # 1d. get_links
    r = await bt.execute(action="get_links", url="https://example.com")
    report("get_links", r)
    if r.success:
        report("  found links", ok="iana.org" in r.output.lower(),
               detail=r.output[:100])

    # 1e. evaluate_js
    r = await bt.execute(
        action="evaluate_js",
        url="https://example.com",
        script="() => document.title",
    )
    report("evaluate_js (document.title)", r)
    if r.success:
        report("  title == 'Example Domain'", ok=r.output.strip() == "Example Domain",
               detail=repr(r.output.strip()))

    # 1f. evaluate_js — blocked pattern guard
    r = await bt.execute(
        action="evaluate_js",
        url="https://example.com",
        script="() => fetch('https://evil.com')",
    )
    report("evaluate_js blocked pattern", ok=not r.success,
           detail=r.error or "should have been blocked")

    # 1g. edge: missing action
    r = await bt.execute()
    report("no action -> error", ok=not r.success, detail=r.error or "")


# ═══════════════════════════════════════════════════════════════════
# 2. SCREENSHOT TOOL
# ═══════════════════════════════════════════════════════════════════

async def test_screenshot():
    print("\n====== 2. ScreenshotTool ======")
    import app.tools.screenshot as ss_mod
    from pathlib import Path

    # Redirect screenshot dir to temp
    ss_mod._SCREENSHOT_DIR = Path(_tmpdir) / "screenshots"

    from app.tools.screenshot import ScreenshotTool
    st = ScreenshotTool()

    # 2a. basic screenshot
    r = await st.execute(url="https://example.com")
    report("screenshot example.com", r)
    if r.success:
        # Verify file was created
        lines = r.output.splitlines()
        report("  output has metadata lines", ok=len(lines) >= 3, detail=str(len(lines)))

    # 2b. full_page screenshot
    r = await st.execute(url="https://example.com", full_page=True)
    report("screenshot full_page=True", r)

    # 2c. selector screenshot
    r = await st.execute(url="https://example.com", selector="h1")
    report("screenshot selector='h1'", r)

    # 2d. missing url
    r = await st.execute()
    report("no url -> error", ok=not r.success, detail=r.error or "")

    # 2e. bad selector
    r = await st.execute(url="https://example.com", selector="#nonexistent999")
    report("bad selector -> error", ok=not r.success, detail=r.error or "")

    # Check actual PNG files exist
    png_files = list((Path(_tmpdir) / "screenshots").glob("*.png"))
    report("PNG files on disk", ok=len(png_files) >= 2,
           detail=f"{len(png_files)} PNG files")


# ═══════════════════════════════════════════════════════════════════
# 3. SKILLS SYSTEM
# ═══════════════════════════════════════════════════════════════════

def test_skills():
    print("\n====== 3. SkillStore ======")
    from app.core.skills import SkillStore

    db = SafeDB(os.environ["DB_PATH"])
    db.init_schema()
    store = SkillStore(db=db)

    # 3a. create a skill
    sid = store.create_skill(
        name="weather_lookup",
        trigger_pattern=r"(?i)what(?:'s| is) the weather in (\w+)",
        steps=[{"tool": "web_search", "args_template": {"query": "weather in {capture_1}"}}],
        answer_template="The weather in {capture_1} is: {result}",
    )
    report("create_skill", ok=sid is not None and sid > 0, detail=f"id={sid}")

    # 3b. list all skills
    all_skills = store.get_all_skills()
    report("get_all_skills", ok=len(all_skills) >= 1, detail=f"count={len(all_skills)}")

    # 3c. get active skills
    active = store.get_active_skills()
    report("get_active_skills", ok=len(active) >= 1, detail=f"count={len(active)}")

    # 3d. match a query
    matched = store.get_matching_skill("What's the weather in London?")
    report("get_matching_skill (positive)", ok=matched is not None,
           detail=f"matched={matched.name if matched else None}")

    # 3e. non-matching query
    no_match = store.get_matching_skill("Tell me a joke")
    report("get_matching_skill (negative)", ok=no_match is None,
           detail=f"matched={no_match}")

    # 3f. record_use (success)
    store.record_use(sid, success=True)
    skill_after = store.get_skill(sid)
    report("record_use(success=True)", ok=skill_after.times_used == 1,
           detail=f"times_used={skill_after.times_used}, rate={skill_after.success_rate:.2f}")

    # 3g. record_use (failure) — shouldn't auto-disable yet (only 2 uses)
    store.record_use(sid, success=False)
    skill_after2 = store.get_skill(sid)
    report("record_use(success=False)", ok=skill_after2.times_used == 2 and skill_after2.enabled,
           detail=f"times_used={skill_after2.times_used}, rate={skill_after2.success_rate:.2f}, enabled={skill_after2.enabled}")

    # 3h. toggle disable
    toggled = store.toggle_skill(sid, enabled=False)
    disabled = store.get_skill(sid)
    report("toggle_skill(enabled=False)", ok=toggled and not disabled.enabled,
           detail=f"enabled={disabled.enabled}")

    # 3i. toggle re-enable (resets stats)
    store.toggle_skill(sid, enabled=True)
    reenabled = store.get_skill(sid)
    report("toggle_skill(enabled=True) resets stats",
           ok=reenabled.enabled and reenabled.times_used == 0 and reenabled.success_rate == 1.0,
           detail=f"enabled={reenabled.enabled}, uses={reenabled.times_used}, rate={reenabled.success_rate}")

    # 3j. deduplication — same trigger pattern should boost, not create new
    sid2 = store.create_skill(
        name="weather_lookup_v2",
        trigger_pattern=r"(?i)what(?:'s| is) the weather in (\w+)",
        steps=[{"tool": "web_search", "args_template": {"query": "weather {capture_1}"}}],
    )
    report("dedup — same trigger returns existing id", ok=sid2 == sid,
           detail=f"returned={sid2}, original={sid}")

    # 3k. broadness guard — overly broad pattern rejected
    broad_id = store.create_skill(
        name="too_broad",
        trigger_pattern=r".*",  # matches everything
        steps=[{"tool": "web_search", "args_template": {"query": "{query}"}}],
    )
    report("broadness guard rejects '.*'", ok=broad_id is None,
           detail=f"returned={broad_id}")

    # 3l. auto-disable after 3+ fails
    sid3 = store.create_skill(
        name="fragile_skill",
        trigger_pattern=r"(?i)convert (\d+) miles to km",
        steps=[{"tool": "calculator", "args_template": {"expr": "{capture_1} * 1.609"}}],
    )
    for _ in range(4):
        store.record_use(sid3, success=False)
    fragile = store.get_skill(sid3)
    report("auto-disable after 3+ failures",
           ok=not fragile.enabled,
           detail=f"enabled={fragile.enabled}, rate={fragile.success_rate:.2f}, uses={fragile.times_used}")

    # 3m. delete
    deleted = store.delete_skill(sid)
    gone = store.get_skill(sid)
    report("delete_skill", ok=deleted and gone is None, detail=f"deleted={deleted}")

    # cleanup sid3
    store.delete_skill(sid3)


# ═══════════════════════════════════════════════════════════════════
# 4. LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════

def test_learning():
    print("\n====== 4. LearningEngine ======")
    from app.core.learning import LearningEngine, Correction, is_likely_correction

    db = SafeDB(os.environ["DB_PATH"])
    db.init_schema()
    engine = LearningEngine(db=db)

    # 4a. regex-stage correction detection
    report("is_likely_correction('Actually, it is X')",
           ok=is_likely_correction("Actually, the capital is Canberra"))

    report("is_likely_correction('That's wrong')",
           ok=is_likely_correction("That's wrong, it should be Python 3"))

    report("is_likely_correction('Remember that I prefer Vue')",
           ok=is_likely_correction("Remember that I prefer Vue over React"))

    report("is_likely_correction('Thanks!') -> False",
           ok=not is_likely_correction("Thanks, that was helpful!"))

    report("is_likely_correction('Hello') -> False",
           ok=not is_likely_correction("Hello, how are you?"))

    # 4b. save_lesson from a manually constructed Correction
    c1 = Correction(
        user_message="Actually, Python was created by Guido van Rossum",
        previous_answer="Python was created by James Gosling",
        topic="Python creator",
        correct_answer="Python was created by Guido van Rossum",
        wrong_answer="Python was created by James Gosling",
        original_query="Who created Python?",
        lesson_text="Python was created by Guido van Rossum, not James Gosling",
    )
    lid1 = engine.save_lesson(c1)
    report("save_lesson", ok=lid1 > 0, detail=f"lesson_id={lid1}")

    # 4c. save another lesson
    c2 = Correction(
        user_message="The capital of Australia is Canberra, not Sydney",
        previous_answer="The capital of Australia is Sydney",
        topic="Capital of Australia",
        correct_answer="The capital of Australia is Canberra",
        wrong_answer="Sydney",
        original_query="What is the capital of Australia?",
        lesson_text="The capital of Australia is Canberra, not Sydney",
    )
    lid2 = engine.save_lesson(c2)
    report("save_lesson #2", ok=lid2 > 0 and lid2 != lid1, detail=f"lesson_id={lid2}")

    # 4d. deduplication — same topic+correct_answer should boost, not create new
    lid_dup = engine.save_lesson(c1)
    report("lesson dedup — same content returns same id", ok=lid_dup == lid1,
           detail=f"returned={lid_dup}, original={lid1}")

    # 4e. quality gate — rejects low-quality
    c_bad = Correction(
        user_message="hmm",
        previous_answer="something",
        topic="test",
        correct_answer="i don't know",
        wrong_answer="",
        original_query="",
    )
    lid_bad = engine.save_lesson(c_bad)
    report("quality gate rejects 'i don't know'", ok=lid_bad == -1,
           detail=f"returned={lid_bad}")

    # 4f. get_relevant_lessons
    lessons = engine.get_relevant_lessons("Who created Python programming language?")
    report("get_relevant_lessons (Python)", ok=len(lessons) >= 1,
           detail=f"count={len(lessons)}, first_topic={lessons[0].topic if lessons else 'none'}")

    # 4g. get_relevant_lessons — no match
    lessons_none = engine.get_relevant_lessons("How do I bake a cake?")
    report("get_relevant_lessons (no match)", ok=len(lessons_none) == 0,
           detail=f"count={len(lessons_none)}")

    # 4h. mark_lesson_helpful
    engine.mark_lesson_helpful(lid1)
    all_lessons = engine.get_all_lessons()
    lesson1 = next((l for l in all_lessons if l.id == lid1), None)
    report("mark_lesson_helpful", ok=lesson1 and lesson1.times_helpful >= 1,
           detail=f"times_helpful={lesson1.times_helpful if lesson1 else '?'}")

    # 4i. get_all_lessons
    all_l = engine.get_all_lessons()
    report("get_all_lessons", ok=len(all_l) >= 2, detail=f"count={len(all_l)}")

    # 4j. get_metrics
    metrics = engine.get_metrics()
    report("get_metrics", ok=metrics["total_lessons"] >= 2,
           detail=f"lessons={metrics['total_lessons']}, skills={metrics['total_skills']}")

    # 4k. save_training_pair
    engine.save_training_pair(
        query="Who created Python?",
        bad_answer="James Gosling created Python",
        good_answer="Guido van Rossum created Python",
    )
    m2 = engine.get_metrics()
    report("save_training_pair", ok=m2["training_examples"] >= 1,
           detail=f"training_examples={m2['training_examples']}")

    # 4l. decay_stale_lessons (on fresh lessons — should decay 0)
    decayed = engine.decay_stale_lessons(days=30)
    report("decay_stale_lessons (fresh -> 0 decayed)", ok=decayed == 0,
           detail=f"decayed={decayed}")

    # 4m. delete_lesson
    deleted = engine.delete_lesson(lid2)
    gone = engine.get_all_lessons()
    ids = [l.id for l in gone]
    report("delete_lesson", ok=deleted and lid2 not in ids,
           detail=f"deleted={deleted}, remaining_ids={ids}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

async def main():
    print(f"Nova Live E2E Tests")
    print(f"Temp dir: {_tmpdir}")

    # Browser tests (async)
    try:
        await test_browser()
    except Exception as e:
        print(f"  [FAIL] BrowserTool crashed: {e}")
        traceback.print_exc()

    # Screenshot tests (async)
    try:
        await test_screenshot()
    except Exception as e:
        print(f"  [FAIL] ScreenshotTool crashed: {e}")
        traceback.print_exc()

    # Skills tests (sync)
    try:
        test_skills()
    except Exception as e:
        print(f"  [FAIL] SkillStore crashed: {e}")
        traceback.print_exc()

    # Learning tests (sync)
    try:
        test_learning()
    except Exception as e:
        print(f"  [FAIL] LearningEngine crashed: {e}")
        traceback.print_exc()

    # Summary
    total = PASS + FAIL
    print(f"\n{'=' * 50}")
    print(f"Results: {PASS}/{total} passed, {FAIL} failed")
    if ERRORS:
        print(f"Failed: {', '.join(ERRORS)}")
    print(f"{'=' * 50}")

    return FAIL == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
