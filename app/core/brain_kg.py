"""KG triple extraction helper — pulled out of brain.py for size hygiene.

Re-exported by brain.py so existing `from app.core.brain import _extract_kg_triples`
imports (tests, heartbeat_loop) keep working.
"""

from __future__ import annotations

import json
import logging
import re

from app.core import llm

logger = logging.getLogger(__name__)


# Confidence defaults by source provenance. Domain studies get a slight boost
# because the monitor prompts constrain the LLM to factual synthesis.
_SOURCE_CONFIDENCE: dict[str, float] = {
    "Domain Study: Science": 0.75,
    "Domain Study: Technology": 0.75,
    "Domain Study: Finance": 0.70,
    "Domain Study: Current Events": 0.65,
    "World Awareness": 0.60,
    "Curiosity Research": 0.60,
}
_DEFAULT_SOURCE_CONFIDENCE = 0.65


async def _extract_kg_triples(kg, query: str, answer: str, source_name: str = "") -> None:
    """Extract (subject, predicate, object) triples from a Q&A pair.

    Runs as a background task — failures are logged, never raised.
    Includes quality gate (heuristic pre-filter) and contradiction detection.
    """
    from app.core.kg import CANONICAL_PREDICATES, is_garbage_triple
    from app.core.prompt_optimizer import get_active_module

    predicates_str = ", ".join(sorted(CANONICAL_PREDICATES))
    _kg_default = (
        "Extract factual (subject, predicate, object) triples from this Q&A.\n"
        "Use ONLY these predicates: {predicates}\n"
        "Return a JSON array. Max 5 triples. Only verifiable facts, not opinions.\n"
        "Rate each triple's confidence: 0.3 (uncertain/speculative) to 0.95 (well-established fact).\n\n"
        "DIRECTION RULES — these predicates are NOT symmetric. The subject and object roles are fixed:\n"
        "  capital_of:  subject = CITY,    object = COUNTRY     (e.g., \"Tokyo capital_of Japan\", NEVER \"Japan capital_of Tokyo\")\n"
        "  works_at:    subject = PERSON,  object = ORG         (e.g., \"Tim Cook works_at Apple\", NEVER \"Apple works_at Tim Cook\")\n"
        "  leads:       subject = PERSON,  object = ORG         (e.g., \"Jensen Huang leads NVIDIA\", NEVER \"NVIDIA leads Jensen Huang\")\n"
        "  created_by:  subject = THING,   object = PERSON/ORG  (e.g., \"Python created_by Guido van Rossum\", NEVER \"Guido van Rossum created_by Python\")\n"
        "  invented_by: subject = THING,   object = PERSON      (e.g., \"light bulb invented_by Edison\")\n"
        "  founded_by:  subject = ORG,     object = PERSON      (e.g., \"Apple founded_by Steve Jobs\")\n"
        "  located_in:  subject = ENTITY,  object = PLACE       (e.g., \"TSMC located_in Taiwan\")\n"
        "  born_in:     subject = PERSON,  object = PLACE       (e.g., \"Einstein born_in Germany\")\n"
        "  member_of:   subject = THING,   object = LARGER_GROUP (NEVER tautological like \"SEC member_of U.S. Securities and Exchange Commission\")\n"
        "Before emitting a triple, check that the roles match the rule. If not, swap them.\n\n"
        "REJECT these triples:\n"
        "  - Tautologies where subject == object semantically (\"SEC member_of Securities and Exchange Commission\")\n"
        "  - Meta-statements about the source itself (\"Reuters is_a financial news source\" — fine; \"website is_a authoritative source\" — too vague)\n"
        "  - Underscored variable names like \"defi_tvl\", \"nvidia_gtc_2026\" — extract the real entity name instead\n"
        "  - Question-label entities (\"Domain Study: X\", \"X Intelligence\", \"monitor system\")\n"
        "If the Answer says nothing substantive, return [].\n\n"
        'Example: [{{"subject": "python", "predicate": "created_by", "object": "guido van rossum", "confidence": 0.9}}]\n\n'
        "Q: {query}\nA: {answer}"
    )
    kg_template = get_active_module("kg_extraction_prompt") or _kg_default
    # Strip monitor-name prefixes so they don't leak into entity extraction.
    clean_query = re.sub(r"^Domain Study:\s*", "", query, flags=re.IGNORECASE).strip()
    prompt = kg_template.format(
        predicates=predicates_str,
        query=clean_query or query,
        answer=answer[:1000],
    )

    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            json_prefix="[{",
        )
        if raw is None or not raw:
            return

        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict) and "triples" in data:
            data = data["triples"]
        if not isinstance(data, list):
            return

        added = 0
        for triple in data[:5]:
            if not isinstance(triple, dict):
                continue
            s = str(triple.get("subject", "")).strip()
            p = str(triple.get("predicate", "")).strip()
            o = str(triple.get("object", "")).strip()
            if not s or not p or not o or len(s) > 100 or len(o) > 100:
                continue

            if is_garbage_triple(s, p, o):
                logger.debug("KG quality gate rejected: %s %s %s", s, p, o)
                continue

            raw_conf = triple.get("confidence")
            if isinstance(raw_conf, (int, float)) and raw_conf > 0.0:
                conf = max(0.3, min(0.95, float(raw_conf)))
            else:
                conf = _SOURCE_CONFIDENCE.get(source_name, _DEFAULT_SOURCE_CONFIDENCE)

            try:
                safe = await kg.check_and_resolve_contradictions(s, p, o, conf)
                if not safe:
                    continue
            except Exception as e:
                logger.warning("KG contradiction check failed (allowing fact): %s", e)

            if await kg.add_fact(s, p, o, confidence=conf, source="extracted", provenance=source_name):
                added += 1

        if added:
            logger.info("KG: extracted %d triple(s) from Q&A (source=%r)", added, source_name or "chat")
        else:
            # No triples landed — either LLM returned [] or every triple was
            # filtered. Still useful to log at INFO so the operator can spot
            # patterns of consistently-empty extractions per source.
            logger.info("KG: 0 triple(s) extracted (source=%r)", source_name or "chat")
    except Exception as e:
        # Bumped from DEBUG to WARNING 2026-05-13 — extraction failures were
        # invisible in production logs (which run at INFO), making this loop
        # look healthy when it had been silently throwing for weeks.
        logger.warning("KG extraction failed (source=%r): %s", source_name or "chat", e)
