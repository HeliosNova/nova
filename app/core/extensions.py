"""Module-level hot-reload — Nova extends itself without rebuilding the image.

`auto_tools.py` already covers the common case (Nova writes Python *functions*
that become callable tools). This module covers the next tier: Nova writing
whole *modules* that can register monitors, parsers, helpers, or more elaborate
multi-function tools.

How it works:
  - `/data/extensions/` is a writable directory inside the container
  - Every `.py` file in there is imported on startup
  - If the module exposes `register(services)` it's called with the live
    Services bundle, so it can register tools, monitors, etc.
  - `POST /api/system/extensions/reload` re-scans and reloads — no docker
    rebuild required.

Safety: modules are loaded as `app.extensions.<filename>`. Standard Python
import semantics apply — there is no extra sandbox, in line with the owner
directive that Nova has no policy gates.

If a module fails to import, the error is logged but the rest of the
extensions still load — a single broken extension doesn't break the system.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


EXTENSIONS_DIR = Path("/data/extensions")
EXTENSIONS_PACKAGE = "app.extensions"


_loaded_modules: dict[str, object] = {}
_load_errors: dict[str, str] = {}


def _ensure_dir() -> None:
    try:
        EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("[Extensions] could not create %s: %s", EXTENSIONS_DIR, e)


def _ensure_package_loaded() -> None:
    """Make `app.extensions` resolvable so modules can import siblings."""
    if EXTENSIONS_PACKAGE in sys.modules:
        return
    try:
        # Synthetic package — no __init__.py file needed
        from types import ModuleType
        pkg = ModuleType(EXTENSIONS_PACKAGE)
        pkg.__path__ = [str(EXTENSIONS_DIR)]
        sys.modules[EXTENSIONS_PACKAGE] = pkg
    except Exception as e:
        logger.warning("[Extensions] failed to create synthetic package: %s", e)


def _import_one(path: Path, services) -> tuple[str, bool, str]:
    """Import a single .py file. Returns (name, ok, message)."""
    name = path.stem
    if name.startswith("_"):
        return name, False, "skipped (leading underscore)"
    full_name = f"{EXTENSIONS_PACKAGE}.{name}"
    try:
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            return name, False, "spec creation failed"
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        # Remove from sys.modules so a fixed re-load works cleanly
        sys.modules.pop(full_name, None)
        return name, False, f"import error: {e}"

    # Optional registration hook
    register_fn = getattr(module, "register", None)
    if callable(register_fn):
        try:
            result = register_fn(services)
            # Allow async registration
            import asyncio
            if asyncio.iscoroutine(result):
                # Best effort: schedule but don't block. Most register fns
                # should be sync.
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(result)
                except Exception:
                    pass
        except Exception as e:
            return name, False, f"register() raised: {e}"

    _loaded_modules[name] = module
    return name, True, "ok"


def load_all(services) -> dict:
    """Scan /data/extensions/, import every .py file, return load report."""
    _ensure_dir()
    _ensure_package_loaded()

    report: dict = {"loaded": [], "failed": [], "skipped": []}
    if not EXTENSIONS_DIR.exists():
        return report

    for path in sorted(EXTENSIONS_DIR.glob("*.py")):
        name, ok, msg = _import_one(path, services)
        if ok:
            report["loaded"].append(name)
            _load_errors.pop(name, None)
        elif msg.startswith("skipped"):
            report["skipped"].append({"name": name, "reason": msg})
        else:
            report["failed"].append({"name": name, "error": msg})
            _load_errors[name] = msg

    if report["loaded"]:
        logger.info("[Extensions] loaded %d module(s): %s",
                    len(report["loaded"]), ", ".join(report["loaded"]))
    if report["failed"]:
        for f in report["failed"]:
            logger.warning("[Extensions] %s failed: %s", f["name"], f["error"])
    return report


def reload_all(services) -> dict:
    """Reload every extension. Wipes cached modules first.

    importlib.reload only works on modules already imported. We bypass it and
    just re-run the spec loading, which gives us a clean slate every time.
    """
    # Drop cached modules under the extensions package
    to_drop = [k for k in sys.modules if k.startswith(f"{EXTENSIONS_PACKAGE}.")]
    for k in to_drop:
        sys.modules.pop(k, None)
    _loaded_modules.clear()
    return load_all(services)


def status() -> dict:
    """Return what's currently loaded and any persisted errors."""
    return {
        "directory": str(EXTENSIONS_DIR),
        "loaded": sorted(_loaded_modules.keys()),
        "errors": dict(_load_errors),
    }


def write_extension(filename: str, source: str) -> tuple[bool, str]:
    """Write a Python source file into /data/extensions/.

    Used by Nova's auto-extension synthesis to drop new module files. The
    filename must end in .py and contain only safe characters; the body is
    written verbatim.
    """
    if not filename.endswith(".py"):
        return False, "filename must end with .py"
    if "/" in filename or "\\" in filename or filename.startswith("."):
        return False, "filename must not contain path separators or start with '.'"
    if not source or len(source) > 50_000:
        return False, "source empty or too large (>50KB)"
    _ensure_dir()
    path = EXTENSIONS_DIR / filename
    try:
        path.write_text(source, encoding="utf-8")
    except Exception as e:
        return False, f"write failed: {e}"
    return True, str(path)
