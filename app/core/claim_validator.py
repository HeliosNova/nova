"""Post-synthesis claim validator — strip unsupported entity+fact claims.

Nova's weights carry residue from prior training on contaminated documents
(Helios Protocol blockchain specs, "Dr. Sarah Chen", etc.). Runtime regex
sanitization can't unlearn these; DPO over ~700 pairs tied v10 three times.

This validator runs AFTER `_sanitize_answer` and BEFORE token emission. It
scans the final answer for a narrow set of high-risk entity-attribution
shapes and drops the containing sentence unless its key tokens appear in
the evidence set (retrieval + KG + user facts + learned lessons + tool outputs).

Scope — only the three shapes that appeared in live probes:

1. Person-with-title-and-role: "Dr. Elena Vasquez, the founder of Helios
   Protocol". Both the person name tokens AND the org tokens must appear
   in evidence; otherwise the sentence is dropped.

2. Attribution by title: "…designed by Dr. Sarah Chen". The name tokens
   must appear in evidence.

3. Named-entity numeric specs: "Helios Protocol achieves 100,000 TPS".
   Both the org and the number must appear in evidence.

Self-architecture claims are NOT regex-policed here — that problem is
solved upstream by making `IDENTITY_AND_REASONING` interpolate
`config.LLM_MODEL` at build time so Nova's context matches reality.

Conservative v1 — prefers false negatives over false positives. The
query is intentionally excluded from evidence to prevent presupposition
attacks from self-validating.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


_PERSON_TITLE_ORG = re.compile(
    r"\b(?i:Dr|Prof|Professor|Mr|Ms|Mrs)\.?\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
    r"[^.!?\n]{0,80}?"
    r"\b(?i:creator|founder|co-?founder|inventor|CEO|CTO|CFO|architect|designer|"
    r"author|developer|director|president|chair|head)\s+of\s+"
    r"(?:the\s+)?"
    r"([A-Z][\w][\w\s\.\-]{1,60}?)(?=[.!?,\n]|$)"
)

_PERSON_BY_ATTRIB = re.compile(
    r"\b(?i:designed|created|built|invented|developed|founded|"
    r"authored|written|made|architected)\s+by\s+"
    r"(?i:Dr|Prof|Professor|Mr|Ms|Mrs)\.?\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
)

# Bare titled-person mention — catches table cells like
#   | Designer | Dr. Sarah Chen (2024) |
# where the sentence-level patterns don't match.
_PERSON_BARE = re.compile(
    r"\b(?i:Dr|Prof|Professor)\.?\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
)

_SPEC_CLAIM = re.compile(
    r"\b([A-Z][\w]*(?:\s+[A-Z][\w]*){0,3}\s+"
    r"(?:Protocol|Platform|Chain|Network|System|Framework|Blockchain))\b"
    r"[^.!?\n]{0,60}?"
    r"\b(?i:achieves?|delivers?|supports?|handles?|reaches?|processes?|"
    r"provides?|offers?|has|features?|uses?)\s+"
    r"(?i:(?:up to\s+|approximately\s+|around\s+)?)"
    r"(\d[\d,\.]*)\s*"
    r"(?i:TPS|transactions(?:\s*per\s*second)?|ms|milliseconds|"
    r"GB|MB|tokens|requests|nodes|validators)\b"
)

# Broadened coverage (added 2026-05-13). Catches name attributions without
# the explicit Dr./Prof./Mr./Ms./Mrs. title prefix that _PERSON_TITLE_ORG
# and _PERSON_BY_ATTRIB require. The same evidence-grounding check applies:
# if both the proper-noun name AND the org tokens appear in retrieval/KG/
# tool evidence, the sentence is kept. FP risk surveyed at ~0.1% over 761
# real assistant messages (a single "William Shakespeare wrote Romeo and
# Juliet" hit that would be rescued by evidence-grounding when retrieval
# contains either token). Requires 2-4 capitalized words for the name to
# avoid matching single-pronoun starts like "She is the founder of...".
_PROPER_NOUN_ROLE_OF = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
    r"(?:is|was)\s+(?:the\s+|a\s+)?"
    r"(?i:creator|founder|co-?founder|inventor|CEO|CTO|CFO|architect|designer|"
    r"author|developer|director|president|chair|head|chief|lead)\s+of\s+"
    r"(?:the\s+)?"
    r"([A-Z][\w][\w\s\.\-]{1,60}?)(?=[.!?,\n]|$)"
)

_PROPER_NOUN_VERB_ATTRIB = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+"
    r"(?i:designed|created|built|invented|developed|founded|"
    r"authored|wrote|architected|engineered)\s+"
    r"(?:the\s+)?"
    r"([A-Z][\w][\w\s\.\-]{1,60}?)(?=[.!?,\n]|$)"
)



def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def _evidence_contains(evidence_norm: str, token: str) -> bool:
    token_norm = _normalize(token)
    if not token_norm:
        return True
    return token_norm in evidence_norm


def _drop_sentence(text: str, start: int, end: int) -> tuple[str, str]:
    left_candidates: list[int] = []
    for sep in (". ", "! ", "? "):
        i = text.rfind(sep, 0, start)
        if i >= 0:
            left_candidates.append(i + len(sep))
    i = text.rfind("\n", 0, start)
    if i >= 0:
        left_candidates.append(i + 1)
    left = max(left_candidates) if left_candidates else 0

    right_candidates: list[int] = []
    for sep in (". ", "! ", "? "):
        i = text.find(sep, end)
        if i >= 0:
            right_candidates.append(i + len(sep))
    i = text.find("\n", end)
    if i >= 0:
        right_candidates.append(i + 1)
    right = min(right_candidates) if right_candidates else len(text)

    removed = text[left:right]
    return text[:left] + text[right:], removed


def build_evidence(
    retrieved_context: str = "",
    kg_facts_text: str = "",
    user_facts_text: str = "",
    tool_results: list[dict] | None = None,
    query: str = "",
    lessons_text: str = "",
) -> str:
    """Assemble grounding evidence for claim validation.

    Evidence = retrieved context + KG facts + user facts + learned lessons +
    tool outputs. Lessons are first-class grounding: the memory loop stores a
    correction as a lesson, so an answer derived from a lesson IS supported.
    Without this, the validator would strip the very answers the memory loop
    just taught (e.g. "Dr. X is based in <city>"), silently defeating the loop.

    The query is intentionally NOT included: presupposition attacks
    ("Who is Dr. X, the creator of Y?") would otherwise self-validate
    because the fabricated entity appears in the question text.
    """
    del query
    parts: list[str] = []
    if retrieved_context:
        parts.append(retrieved_context)
    if kg_facts_text:
        parts.append(kg_facts_text)
    if user_facts_text:
        parts.append(user_facts_text)
    if lessons_text:
        parts.append(lessons_text)
    if tool_results:
        for tr in tool_results:
            out = tr.get("output") if isinstance(tr, dict) else None
            if isinstance(out, str) and out:
                parts.append(out)
            elif out is not None:
                try:
                    parts.append(str(out))
                except Exception:
                    pass
    return "\n\n".join(p for p in parts if p)


def count_claim_candidates(answer: str) -> int:
    """Count how many high-risk claim patterns appear in `answer`.

    Used by RLVR signal gating: `claim_grounded` should only fire when the
    validator actually had work to do. Otherwise every non-claim response
    records value=1.0 and the signal becomes degenerate (zero variance).
    Returns the total number of pattern matches across all four shapes,
    overlap-counted — exact-overlap deduplication isn't necessary; we
    just need ">0 means there's signal to record."
    """
    if not answer:
        return 0
    n = 0
    for pat in (
        _PERSON_TITLE_ORG, _PERSON_BY_ATTRIB, _PERSON_BARE, _SPEC_CLAIM,
        _PROPER_NOUN_ROLE_OF, _PROPER_NOUN_VERB_ATTRIB,
    ):
        n += sum(1 for _ in pat.finditer(answer))
    return n


def validate_claims(
    answer: str,
    evidence: str,
    current_model_tag: str | None = None,
) -> tuple[str, list[str]]:
    """Strip sentences whose entity+fact claims aren't backed by evidence.

    Returns (cleaned_answer, list_of_stripped_reasons).
    """
    if not answer or not answer.strip():
        return answer, []

    evidence_norm = _normalize(evidence)
    to_drop: list[tuple[int, int, str]] = []

    for m in _PERSON_TITLE_ORG.finditer(answer):
        name = m.group(1)
        org = m.group(2).strip().rstrip(".")
        name_parts = [p for p in name.split() if len(p) > 1]
        org_parts = [p for p in org.split() if len(p) > 2 and p.lower() not in {"the", "and", "of"}]
        name_supported = name_parts and all(_evidence_contains(evidence_norm, p) for p in name_parts)
        org_supported = org_parts and all(_evidence_contains(evidence_norm, p) for p in org_parts)
        if not (name_supported and org_supported):
            to_drop.append((m.start(), m.end(), f"person-title-org: {name!r} of {org!r}"))

    for m in _PERSON_BY_ATTRIB.finditer(answer):
        name = m.group(1)
        name_parts = [p for p in name.split() if len(p) > 1]
        name_supported = name_parts and all(_evidence_contains(evidence_norm, p) for p in name_parts)
        if not name_supported:
            to_drop.append((m.start(), m.end(), f"attribution: by {name!r}"))

    bare_spans_seen: set[tuple[int, int]] = {(s, e) for s, e, _ in to_drop}
    for m in _PERSON_BARE.finditer(answer):
        if any(m.start() >= s and m.end() <= e for s, e, _ in to_drop):
            continue
        name = m.group(1)
        name_parts = [p for p in name.split() if len(p) > 1]
        name_supported = name_parts and all(_evidence_contains(evidence_norm, p) for p in name_parts)
        if not name_supported:
            to_drop.append((m.start(), m.end(), f"bare-titled-person: {name!r}"))

    for m in _SPEC_CLAIM.finditer(answer):
        org = m.group(1).strip()
        number = m.group(2).strip()
        org_parts = [p for p in org.split() if len(p) > 2 and p.lower() not in {"the", "and", "of"}]
        org_supported = org_parts and all(_evidence_contains(evidence_norm, p) for p in org_parts)
        number_supported = _evidence_contains(evidence_norm, number) or _evidence_contains(
            evidence_norm, number.replace(",", "")
        )
        if not (org_supported and number_supported):
            to_drop.append((m.start(), m.end(), f"spec-claim: {org!r} = {number!r}"))

    # _PROPER_NOUN_ROLE_OF and _PROPER_NOUN_VERB_ATTRIB share the same
    # grounding rule as _PERSON_TITLE_ORG: both name AND org tokens must
    # appear in evidence. Skip if a wider pattern already flagged this span.
    _STOP_ORG_WORDS = {"the", "and", "of", "a", "an"}
    for pat, label in (
        (_PROPER_NOUN_ROLE_OF, "proper-noun-role-of"),
        (_PROPER_NOUN_VERB_ATTRIB, "proper-noun-verb-attrib"),
    ):
        for m in pat.finditer(answer):
            if any(m.start() >= s and m.end() <= e for s, e, _ in to_drop):
                continue
            name = m.group(1)
            org = m.group(2).strip().rstrip(".")
            name_parts = [p for p in name.split() if len(p) > 1]
            org_parts = [p for p in org.split() if len(p) > 2 and p.lower() not in _STOP_ORG_WORDS]
            name_supported = name_parts and all(_evidence_contains(evidence_norm, p) for p in name_parts)
            org_supported = org_parts and all(_evidence_contains(evidence_norm, p) for p in org_parts)
            if not (name_supported and org_supported):
                to_drop.append((m.start(), m.end(), f"{label}: {name!r} → {org!r}"))

    if not to_drop:
        return answer, []

    to_drop.sort(key=lambda x: x[0], reverse=True)
    seen_spans: set[tuple[int, int]] = set()
    out = answer
    reasons: list[str] = []

    for start, end, reason in to_drop:
        if start >= len(out):
            continue
        out, removed = _drop_sentence(out, start, end)
        reasons.append(f"{reason} → removed: {removed.strip()[:140]!r}")

    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")

    return out.strip(), reasons
