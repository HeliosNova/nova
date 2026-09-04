"""The scans that find what the suite cannot, run by the suite.

On 2026-09-04 twelve real defects surfaced in a day. Ten came from ad-hoc scans
taking seconds, three from the owner reading output, and ONE from the 3,596
tests. That ratio is why this file exists: these three checks cost about a
second and each has already caught something that shipped.

They belong in the suite rather than in a script nobody runs, because the point
is to fail BEFORE a deploy. Each is validated red-on-the-real-defect, not just
green on a clean tree — a check that has never been seen to fire is a check
nobody should trust.
"""
from __future__ import annotations

import pathlib

import pytest

from scripts.sanity_scan import (
    check_control_bytes,
    check_duplicate_definitions,
    check_undefined_names,
)

APP = pathlib.Path(__file__).resolve().parent.parent / "app"


def test_no_module_reads_a_name_that_does_not_exist():
    """The heartbeat mixin split left SIX of these behind (2026-09-04),
    including the delivery-journal recovery that exists so a finished briefing
    survives a restart. It raised NameError into an `except` and went quiet.
    This suite caught one of the six; this check catches all six in a second."""
    found = check_undefined_names(APP)
    assert not found, "undefined names:\n  " + "\n  ".join(found)


def test_nothing_is_defined_twice_at_module_level():
    """The second definition silently wins. strip_markup and its four regexes
    were both defined twice in text_utils (2026-09-04) — harmless only because
    the copies were identical. The duplicate _CHECK_DISPATCH key that killed
    scheduled Dream Consolidation for weeks was the same shape."""
    found = check_duplicate_definitions(APP)
    assert not found, "duplicate definitions:\n  " + "\n  ".join(found)


def test_no_source_file_carries_a_stray_control_byte():
    r"""A patch script wrote a literal \x08 where a regex meant \b, so the
    pattern matched nothing and did so silently (2026-09-04)."""
    found = check_control_bytes(APP)
    assert not found, "control bytes:\n  " + "\n  ".join(found)


# ---------------------------------------------------------------------------
# The checks themselves, proven to fire. A detector nobody has watched fail is
# a detector that quietly stops working.
# ---------------------------------------------------------------------------

def _write(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_the_undefined_check_catches_a_name_that_moved_away(tmp_path):
    """Exactly the 2026-09-04 defect: the constant moved to another module and
    the method that reads it did not."""
    root = _write(tmp_path, "m.py",
                  "def f():\n    return _MOVED_CONSTANT + 1\n")
    found = check_undefined_names(root)
    assert any("_MOVED_CONSTANT" in f for f in found), found


def test_the_undefined_check_accepts_ordinary_code(tmp_path):
    """It must not cry wolf: imports, module constants, parameters, locals,
    comprehension targets, except-as and globals are all legitimate."""
    root = _write(tmp_path, "ok.py", (
        "import os\n"
        "from typing import Any\n"
        "CONST = 1\n"
        "def f(a, *args, **kw):\n"
        "    local = a + CONST\n"
        "    xs = [y for y in range(3) if y]\n"
        "    with open(os.devnull) as fh:\n"
        "        fh.read()\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError as e:\n"
        "        print(e)\n"
        "    def inner(z: Any):\n"
        "        return z + local\n"
        "    return inner(sum(xs))\n"
    ))
    assert check_undefined_names(root) == []


def test_the_duplicate_check_catches_a_second_definition(tmp_path):
    root = _write(tmp_path, "dup.py",
                  "def g():\n    return 1\n\n\ndef g():\n    return 2\n")
    found = check_duplicate_definitions(root)
    assert any("'g' redefined" in f for f in found), found


def test_the_duplicate_check_allows_a_conditional_reassignment(tmp_path):
    """Only TOP-LEVEL redefinition is silent-overwrite; a reassignment inside a
    branch is ordinary code."""
    root = _write(tmp_path, "cond.py",
                  "import sys\n"
                  "if sys.platform == 'win32':\n"
                  "    SEP = '\\\\'\n"
                  "else:\n"
                  "    SEP = '/'\n")
    assert check_duplicate_definitions(root) == []


def test_the_control_byte_check_catches_a_mangled_escape(tmp_path):
    p = tmp_path / "bad.py"
    p.write_bytes(b'PATTERN = "\x08word"\n')
    found = check_control_bytes(tmp_path)
    assert any("bad.py" in f for f in found), found


@pytest.mark.parametrize("body", [b"a = 1\n", b"a = 1\r\n", b"a\t= 1\n"])
def test_the_control_byte_check_allows_tabs_and_newlines(tmp_path, body):
    (tmp_path / "fine.py").write_bytes(body)
    assert check_control_bytes(tmp_path) == []
