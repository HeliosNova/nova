"""A GitHub advisory row was an id, a package and two links (2026-09-04).

Every one of the 15 advisories in that day's digest carried a `description` of
"**CVE:** This vulnerability corresponds to [CVE-...](nvd.nist.gov/...)" and
nothing else, so the rendered body told a reader nothing the title had not
already said - and said it in links. The advisory's own `summary` IS the real
one-line description; it was being used as the TITLE, where the renderer
truncates it with an ellipsis.

So a stub description now yields to the summary. The bar is absolute, not
relative: a real description that happens to be SHORTER than the summary is
still the better body, and an earlier version of this rule that compared the two
lengths would have thrown those away.
"""
from __future__ import annotations

from app.monitors.rss_feeds import _ADVISORY_MIN_PROSE, _prose_len

POINTER = ("**CVE:** This vulnerability corresponds to "
           "[CVE-2026-72800](https://nvd.nist.gov/vuln/detail/CVE-2026-72800).")
REAL = ("An attacker can bypass the access control check in the kernel and read "
        "arbitrary notebooks belonging to other users.")
SUMMARY = ("SiYuan: Missing publish-access filter on getAttributeViewKeysByID "
           "discloses database column schema, plus two unscoped block-ID "
           "enumeration endpoints")


def _body_for(description: str, summary: str) -> str:
    """The rule as _fetch_gh_advisories applies it."""
    body = (description or "").strip()
    if _prose_len(body) < _ADVISORY_MIN_PROSE:
        body = summary
    return body or summary


def test_a_cve_cross_reference_is_not_a_description():
    assert _prose_len(POINTER) < _ADVISORY_MIN_PROSE
    assert _body_for(POINTER, SUMMARY) == SUMMARY


def test_a_real_description_survives_even_when_shorter_than_the_summary():
    """The whole point of an absolute bar."""
    assert len(REAL) < len(SUMMARY)
    assert _body_for(REAL, SUMMARY) == REAL


def test_an_empty_or_id_only_description_yields_to_the_summary():
    assert _body_for("", SUMMARY) == SUMMARY
    assert _body_for("GHSA-5fhr-f75j-8wr9", SUMMARY) == SUMMARY


def test_prose_length_ignores_links_and_identifiers():
    """Identifiers and URLs are not information a reader gains."""
    assert _prose_len("https://example.com/a/b/c") == 0
    assert _prose_len("CVE-2026-72800") == 0
    assert _prose_len("[label](https://example.com)") == 0
    assert _prose_len("**bold**") == len("bold")


def test_plain_prose_counts_at_face_value():
    text = "The kernel discloses protected content to anonymous readers."
    assert _prose_len(text) == len(text)
