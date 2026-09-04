r"""The checks that actually find things here, made standing.

On 2026-09-04 twelve real defects surfaced in one day. Ten were found by ad-hoc
scans and one-line SQL queries taking seconds each; three by the owner reading
output; ONE by the 3,596-test suite. The suite proves the code does what was
written. It cannot tell you that a writer stopped writing, that a monitor has
returned the identical string 101 times, or that a refactor left six names
behind.

So the scans stop being ad-hoc. Three of them are about the SOURCE and belong in
the test suite, where they run before a deploy rather than after
(`tests/test_source_sanity.py`); the runtime ones live with the liveness
registry. This module is the shared implementation and a standalone entry point:

    python scripts/sanity_scan.py [app_dir]

Exit code is the number of findings, so it drops straight into a pre-deploy gate.

What each check has already caught, so nobody deletes one as speculative:

  undefined names    Six module-level names the heartbeat mixin split left
                     behind (2026-09-04) - among them the delivery-journal
                     recovery that exists so a finished briefing survives a
                     restart. It was failing into an `except` and going quiet.
                     The suite caught one of the six.
  duplicate defs     strip_markup and its four regexes defined twice in
                     text_utils, the second silently winning (2026-09-04). Same
                     shape as the duplicate _CHECK_DISPATCH key that killed
                     scheduled Dream Consolidation for weeks.
  control bytes      A patch script wrote literal \x08 where a regex meant \b,
                     so the pattern silently matched nothing (2026-09-04).
"""
from __future__ import annotations

import ast
import builtins
import io
import pathlib
import re
import sys

_BUILTINS = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__",
    "__builtins__", "__debug__", "self", "cls",
}
_ALLOWED_CONTROL = {9, 10, 13}          # tab, newline, carriage return


def _read(path: pathlib.Path) -> str:
    return io.open(path, encoding="utf-8", errors="replace").read()


# ---------------------------------------------------------------- undefined
def _module_level_names(tree: ast.Module) -> set[str]:
    """Everything a function in this module can close over at module scope."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global):
            names.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            names.update(node.names)
    return names


def check_undefined_names(root: pathlib.Path) -> list[str]:
    """Names a module reads that nothing in that module could ever provide.

    Deliberately blunt: every binding ANYWHERE in the module counts as
    available. That cannot see a name used before assignment or shadowed in an
    inner scope, and it does not try to - a checker that reports maybes gets
    ignored, and the defect this exists for is a name that is simply not there.
    """
    out: list[str] = []
    for path in sorted(root.rglob("*.py")):
        src = _read(path)
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            out.append(f"{path}:{e.lineno}: SYNTAX ERROR: {e.msg}")
            continue
        available = _module_level_names(tree) | _BUILTINS
        seen: set[tuple[str, int]] = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            if node.id in available or (node.id, node.lineno) in seen:
                continue
            seen.add((node.id, node.lineno))
            out.append(f"{path}:{node.lineno}: undefined name '{node.id}'")
    return out


# ---------------------------------------------------------------- duplicates
def check_duplicate_definitions(root: pathlib.Path) -> list[str]:
    """A second top-level definition silently replaces the first."""
    out: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(_read(path))
        except SyntaxError:
            continue                      # reported by the undefined-name pass
        first: dict[str, int] = {}
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in names:
                if name in first:
                    out.append(f"{path}:{node.lineno}: '{name}' redefined "
                               f"(first at line {first[name]})")
                first[name] = node.lineno
    return out


# ------------------------------------------------------------ control bytes
def check_control_bytes(root: pathlib.Path) -> list[str]:
    """A literal control byte where an escape sequence was meant."""
    out: list[str] = []
    for path in sorted(root.rglob("*.py")):
        raw = io.open(path, "rb").read()
        bad = sorted({b for b in raw if b < 32 and b not in _ALLOWED_CONTROL})
        if bad:
            out.append(f"{path}: control byte(s) "
                       + ", ".join(hex(b) for b in bad))
    return out


# ------------------------------------------------------- silent write paths
_WRITE_CALL = re.compile(
    r"\b(execute|executemany|commit|INSERT|UPDATE|DELETE|\.post|\.put|\.send|"
    r"send_alert|write|dump|add_fact|add_message|store|save)\b", re.I)


def silent_write_sites(root: pathlib.Path) -> list[tuple[str, str]]:
    """(file, enclosing function) for every `except ...: pass` whose try body
    writes, sends or fetches.

    An exception swallowed around a WRITE is a lie about success: the caller
    carries on and reports the work as done. That is how the enrichment gate
    logged "0 entail-dropped" after timing out, how 8 of 15 SEC filings shipped
    unannotated, and how fact banking could have reported "0 stored" with no
    reason at all. 27 such sites existed on 2026-09-04; 12 were given a log line
    and the rest are accepted in tests/test_silent_writes.py with a reason each.
    """
    out: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        src = _read(path)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        owner: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    owner.setdefault(ln, node.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for h in node.handlers:
                if len(h.body) != 1 or not isinstance(h.body[0], ast.Pass):
                    continue
                body = "\n".join(ast.get_source_segment(src, st) or "" for st in node.body)
                if _WRITE_CALL.search(body):
                    out.append((path.as_posix(), owner.get(h.lineno, "<module>")))
    return out


CHECKS = (
    ("undefined names", check_undefined_names),
    ("duplicate definitions", check_duplicate_definitions),
    ("control bytes", check_control_bytes),
)


def scan(root: pathlib.Path) -> dict[str, list[str]]:
    return {name: fn(root) for name, fn in CHECKS}


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1] if len(argv) > 1 else "app")
    if not root.exists():
        print(f"no such directory: {root}")
        return 1
    findings = scan(root)
    total = sum(len(v) for v in findings.values())
    for name, items in findings.items():
        print(f"{name:<24}{len(items):>4}")
        for line in items[:20]:
            print(f"   {line}")
        if len(items) > 20:
            print(f"   ... and {len(items) - 20} more")
    print(f"\n{total} finding(s) in {root}")
    return total


if __name__ == "__main__":
    sys.exit(main(sys.argv))
