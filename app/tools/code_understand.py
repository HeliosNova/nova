"""code_understand — architectural code analysis tool.

Returns a structured map of a project or file: file tree, classes,
functions, imports. Lets Nova reason about codebases at the architectural
level instead of just chunked retrieval.

For Python files: uses stdlib `ast` for parsing.
For other languages: returns file tree + sizes only.
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path

from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)


# Hard limits — keep output bounded
_MAX_FILES = 200
_MAX_OUTPUT_CHARS = 12000
_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", "env", "node_modules",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".idea", ".vscode", "site-packages",
})


def _walk_python_files(root: Path, max_files: int = _MAX_FILES) -> list[Path]:
    """Yield .py files under root, skipping virtualenvs / caches / vendored deps."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip noisy directories
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                files.append(Path(dirpath) / f)
                if len(files) >= max_files:
                    return files
    return files


def _summarize_python_file(path: Path) -> dict:
    """Parse a single .py file, return structured summary.

    Returns: {
      'classes': [{'name': str, 'lineno': int, 'methods': [str]}],
      'functions': [{'name': str, 'lineno': int, 'args': [str]}],
      'imports': [str],
      'docstring': str | None,
      'lines': int,
      'error': str | None,
    }
    """
    out = {
        "classes": [], "functions": [], "imports": [],
        "docstring": None, "lines": 0, "error": None,
    }
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        out["lines"] = len(src.splitlines())
        tree = ast.parse(src)
    except Exception as e:
        out["error"] = str(e)[:120]
        return out

    out["docstring"] = (ast.get_docstring(tree) or "")[:200] or None

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                m.name for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            out["classes"].append({
                "name": node.name,
                "lineno": node.lineno,
                "methods": methods[:25],
                "method_count": len(methods),
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            out["functions"].append({
                "name": node.name,
                "lineno": node.lineno,
                "args": args[:8],
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            })
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = ", ".join(a.name for a in node.names)
            out["imports"].append(f"from {mod} import {names}")
        elif isinstance(node, ast.Import):
            out["imports"].append("import " + ", ".join(a.name for a in node.names))

    out["imports"] = out["imports"][:20]
    return out


def _format_summary(root: Path, files: list[Path], summaries: dict[str, dict],
                     mode: str = "tree") -> str:
    """Render the structured summary to text, respecting char budget."""
    lines: list[str] = [f"# Code structure: {root}\n"]
    lines.append(f"Files scanned: {len(files)} (Python only)\n")

    if mode == "tree":
        # File tree with class/function counts
        lines.append("## File tree (with class/function counts)\n")
        for f in sorted(files):
            rel = f.relative_to(root) if root in f.parents or f == root else f
            s = summaries.get(str(f), {})
            cls_n = len(s.get("classes", []))
            fn_n = len(s.get("functions", []))
            ln = s.get("lines", 0)
            err = s.get("error")
            tag = f"[ERR: {err}]" if err else f"({cls_n} classes, {fn_n} fns, {ln} lines)"
            lines.append(f"  {rel}  {tag}")
            if "\n".join(lines).__len__() > _MAX_OUTPUT_CHARS - 500:
                lines.append(f"  ... (truncated at {len(lines)} files)")
                break
    elif mode == "detailed":
        # Per-file class + function index
        for f in sorted(files):
            rel = f.relative_to(root) if root in f.parents or f == root else f
            s = summaries.get(str(f), {})
            if s.get("error"):
                lines.append(f"\n### {rel}\n  PARSE ERROR: {s['error']}")
                continue
            doc = s.get("docstring")
            lines.append(f"\n### {rel}  ({s.get('lines', 0)} lines)")
            if doc:
                lines.append(f"  \"{doc[:160]}\"")
            for c in s.get("classes", [])[:20]:
                lines.append(f"  class {c['name']}  (line {c['lineno']}, {c['method_count']} methods)")
                for m in c["methods"][:8]:
                    lines.append(f"    .{m}()")
            for fn in s.get("functions", [])[:30]:
                async_marker = "async " if fn.get("is_async") else ""
                args = ", ".join(fn.get("args", [])[:5])
                lines.append(f"  {async_marker}def {fn['name']}({args})  (line {fn['lineno']})")
            if "\n".join(lines).__len__() > _MAX_OUTPUT_CHARS - 500:
                lines.append("\n  ... (truncated for budget)")
                break

    text = "\n".join(lines)
    if len(text) > _MAX_OUTPUT_CHARS:
        text = text[:_MAX_OUTPUT_CHARS] + "\n[... truncated]"
    return text


class CodeUnderstandTool(BaseTool):
    name = "code_understand"
    description = (
        "Get architectural overview of a Python codebase or single file. "
        "Returns file tree, classes, functions, imports, docstrings — without "
        "reading every line. Use this BEFORE diving into specific files when "
        "you need to understand a project's structure or find where something "
        "lives. Two modes: 'tree' (file tree with counts), 'detailed' (per-file "
        "class/function index)."
    )
    parameters = "path: str, mode: str = 'tree'"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to a Python file or directory to analyze.",
            },
            "mode": {
                "type": "string",
                "enum": ["tree", "detailed"],
                "description": "'tree' = file tree with counts. 'detailed' = full class/function index per file.",
            },
        },
        "required": ["path"],
    }

    async def execute(self, *, path: str = "", mode: str = "tree", **kwargs) -> ToolResult:
        if not path:
            return ToolResult(
                output="", success=False,
                error="path is required",
                error_category=ErrorCategory.VALIDATION,
            )
        p = Path(path)
        if not p.exists():
            return ToolResult(
                output="", success=False,
                error=f"path not found: {path}",
                error_category=ErrorCategory.NOT_FOUND,
            )

        if mode not in ("tree", "detailed"):
            mode = "tree"

        # Collect files to summarize
        if p.is_file():
            if p.suffix != ".py":
                return ToolResult(
                    output=f"Path is not a Python file: {p.name}",
                    success=True,
                )
            files = [p]
            root = p.parent
        else:
            root = p
            files = _walk_python_files(p, max_files=_MAX_FILES)
            if not files:
                return ToolResult(
                    output=f"No Python files found under {path}",
                    success=True,
                )

        # Parse each file
        summaries: dict[str, dict] = {}
        for f in files:
            try:
                summaries[str(f)] = _summarize_python_file(f)
            except Exception as e:
                summaries[str(f)] = {"error": str(e)[:80]}

        text = _format_summary(root, files, summaries, mode=mode)
        return ToolResult(output=text, success=True)
