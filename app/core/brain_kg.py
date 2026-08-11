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


async def _extract_kg_triples(kg, query: str, answer: str, source_name: str = "",
                              *, max_answer_chars: int = 1000, max_triples: int = 5,
                              model: str | None = None,
                              trust: float | None = None) -> None:
    """Extract (subject, predicate, object) triples from a Q&A pair.

    Runs as a background task — failures are logged, never raised.
    Includes quality gate (heuristic pre-filter) and contradiction detection.

    `max_answer_chars` / `max_triples` default to the lean chat budget. The monitor
    digest path passes far larger values: a multi-paragraph domain study is the main
    KG-growth pipe, and capping it to 5 triples from the first 1000 chars threw away
    the back half of every briefing (the richer the synthesis, the more was lost).

    `model` routes extraction to a bigger model (the monitor path passes the synthesis
    model). A controlled 4-arm A/B (2026-06-29, 45 real digests) found the EXTRACTION
    step — not the cap — was the binding constraint: the 27B yields ~4.6× more grounded
    facts/digest (1.4→6.5) at 100% entity-grounding vs the 9B. The same A/B REJECTED a
    "capture actions/events" prompt relaxation — it coerced the 27B into misframed
    event-relations (direction reversals, spurious competes_with); the CURRENT prompt
    on the 27B both extracts MORE and keeps relations clean, so the prompt is unchanged.
    """
    from app.core.kg import CANONICAL_PREDICATES, is_garbage_triple
    from app.core.prompt_optimizer import get_active_module

    predicates_str = ", ".join(sorted(CANONICAL_PREDICATES))
    _kg_default = (
        "Extract factual (subject, predicate, object) triples from this Q&A.\n"
        "Use ONLY these predicates: {predicates}\n"
        "Return a JSON array. Max {max_triples} triples. Only verifiable facts, not opinions.\n"
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
        "  acquired:    subject = ACQUIRER,  object = ACQUIRED    (e.g., \"Google acquired Wiz\", NEVER reversed)\n"
        "  subsidiary_of: subject = UNIT,    object = PARENT      (e.g., \"Instagram subsidiary_of Meta\")\n"
        "  invested_in: subject = INVESTOR,  object = RECIPIENT   (e.g., \"SoftBank invested_in OpenAI\")\n"
        "  sued:        subject = PLAINTIFF, object = DEFENDANT   (e.g., \"Epic sued Apple\")\n"
        "  sanctioned:  subject = SANCTIONER, object = TARGET     (e.g., \"US sanctioned Huawei\")\n"
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
        answer=answer[:max_answer_chars],
        max_triples=max_triples,
    )

    try:
        raw = await llm.invoke_nothink(
            [{"role": "user", "content": prompt}],
            json_mode=True,
            json_prefix="[{",
            model=model,
        )
        if raw is None or not raw:
            return

        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict) and "triples" in data:
            data = data["triples"]
        if not isinstance(data, list):
            return

        added = 0
        for triple in data[:max_triples]:
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

            if await kg.add_fact(s, p, o, confidence=conf, source="extracted", provenance=source_name,
                                 trust=trust):
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
