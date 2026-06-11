"""KG consistency check — batched nightly cross-check of recent answers vs KG.

For each high-quality assistant response in the last N hours:
  1. Extract structured factual claims via LLM (entity/predicate/value).
  2. Look up the entity in the KG.
  3. If a KG fact's predicate matches but its object differs from the
     claimed value, surface the contradiction.

Per-response inline checks would add 200-500ms latency to every chat call;
running as a nightly batch is the right cost/benefit trade. Output is
logged + sent as alert via configured channels so the user sees:
  "I told you X but my KG says Y — which is right?"

This is the closure for the "ground answers to KG" loop. Without it, KG
facts can drift from what Nova actually says, and contradictions never
surface for resolution.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import config

logger = logging.getLogger(__name__)


_EXTRACT_CLAIMS_PROMPT = """Extract every factual claim from this answer as structured triples.

ANSWER:
{answer}

Output JSON: {{"claims": [{{"subject": "...", "predicate": "...", "object": "..."}}, ...]}}

Rules:
- subject = the entity the claim is about (proper noun, normalized lowercase)
- predicate = the relationship (snake_case: "depth_meters", "founded_in", "capital_of", "born_year", "located_in")
- object = the value (number+unit, date, name, place — short)
- Skip opinions, hedges, instructions, meta-commentary
- Skip claims with no concrete subject (e.g. "this is a great approach")
- Cap at 8 claims per answer
- If no factual claims, return {{"claims": []}}

Examples:
- "Mount Everest is 8848 meters tall" -> {{"subject":"mount everest","predicate":"height_meters","object":"8848"}}
- "Albert Einstein was born in 1879 in Ulm" -> 2 claims
- "I think you should try X" -> 0 claims (opinion)

JSON:"""


_JUDGE_CONTRADICTION_PROMPT = """Compare these two values for the same fact. Are they contradicting?

SUBJECT: {subject}
PREDICATE: {predicate}
ANSWERED: {answered}
KG STORED: {kg_stored}

Reply with JSON: {{"contradicts": true|false, "reason": "<one short sentence>"}}

Rules:
- "8848" vs "8848.86" → not contradicting (rounding)
- "Tokyo" vs "tokyo, japan" → not contradicting (granularity)
- "1879" vs "1955" → contradicting (different years)
- "100km" vs "62 miles" → not contradicting (unit conversion ~equivalent)
- Empty/null KG value → not contradicting (no comparison data)

JSON:"""


async def check_recent_answers(db, recent_hours: int = 24, max_answers: int = 30) -> dict:
    """Run consistency check across recent assistant responses.

    Returns:
      {
        'checked': int,
        'claims_extracted': int,
        'contradictions': list[dict],
        'errors': int,
      }
    """
    from app.core import llm
    from app.core.kg import KnowledgeGraph

    result = {
        "checked": 0,
        "claims_extracted": 0,
        "contradictions": [],
        "errors": 0,
    }

    try:
        rows = db.fetchall(
            "SELECT id, content, created_at FROM messages "
            "WHERE role = 'assistant' "
            "AND created_at > datetime('now', ?) "
            "AND LENGTH(content) > 200 "
            "ORDER BY created_at DESC LIMIT ?",
            (f"-{int(recent_hours)} hours", int(max_answers)),
        )
    except Exception as e:
        logger.warning("[KG-Consistency] failed to fetch messages: %s", e)
        return result

    if not rows:
        return result

    kg = KnowledgeGraph(db)

    for row in rows:
        msg_id = row["id"]
        content = (row["content"] or "")[:3000]
        if not content.strip():
            continue
        result["checked"] += 1

        # 1. Extract factual claims via LLM
        try:
            resp = await llm.invoke_nothink(
                [{"role": "user", "content": _EXTRACT_CLAIMS_PROMPT.format(answer=content)}],
                max_tokens=500, json_mode=True, temperature=0.0, model=config.FAST_MODEL,
            )
            obj = llm.extract_json_object(resp) or {}
            claims = obj.get("claims", []) if isinstance(obj, dict) else []
            if not isinstance(claims, list):
                continue
        except Exception as e:
            logger.debug("[KG-Consistency] claim extraction failed for msg %s: %s", msg_id, e)
            result["errors"] += 1
            continue

        result["claims_extracted"] += len(claims)

        # 2. Cross-check each claim against KG
        for claim in claims[:8]:
            if not isinstance(claim, dict):
                continue
            subject = str(claim.get("subject", "")).strip()
            predicate = str(claim.get("predicate", "")).strip()
            answered = str(claim.get("object", "")).strip()
            if not subject or not predicate or not answered:
                continue

            try:
                kg_facts = kg.query(subject, hops=0)
            except Exception as e:
                logger.debug("[KG-Consistency] KG query failed for %r: %s", subject, e)
                result["errors"] += 1
                continue

            # Find facts where predicate roughly matches
            matching = [
                f for f in kg_facts
                if (f.get("predicate") or "").lower().replace("_", " ").startswith(
                    predicate.lower().replace("_", " ")[:15]
                )
                or predicate.lower().replace("_", " ").startswith(
                    (f.get("predicate") or "").lower().replace("_", " ")[:15]
                )
            ]
            if not matching:
                continue

            # 3. LLM judge: do values contradict?
            for kg_fact in matching[:2]:
                kg_object = str(kg_fact.get("object", "")).strip()
                if not kg_object:
                    continue
                # Cheap pre-filter: if strings overlap substantially, skip judge
                if (
                    answered.lower() in kg_object.lower()
                    or kg_object.lower() in answered.lower()
                ):
                    continue
                try:
                    judge_resp = await llm.invoke_nothink(
                        [{"role": "user", "content": _JUDGE_CONTRADICTION_PROMPT.format(
                            subject=subject[:80],
                            predicate=predicate[:80],
                            answered=answered[:200],
                            kg_stored=kg_object[:200],
                        )}],
                        max_tokens=120, json_mode=True, temperature=0.0,
                        model=config.FAST_MODEL,
                    )
                    j = llm.extract_json_object(judge_resp) or {}
                    contradicts = bool(j.get("contradicts"))
                    reason = str(j.get("reason", ""))[:200]
                except Exception as e:
                    logger.debug("[KG-Consistency] judge failed: %s", e)
                    result["errors"] += 1
                    continue

                if contradicts:
                    result["contradictions"].append({
                        "msg_id": msg_id,
                        "subject": subject,
                        "predicate": predicate,
                        "answered": answered,
                        "kg_stored": kg_object,
                        "kg_fact_id": kg_fact.get("id"),
                        "reason": reason,
                    })
                    logger.warning(
                        "[KG-Consistency] contradiction: %s.%s answered=%r kg=%r — %s",
                        subject[:40], predicate[:30], answered[:60], kg_object[:60], reason,
                    )

    logger.info(
        "[KG-Consistency] checked=%d claims=%d contradictions=%d errors=%d",
        result["checked"], result["claims_extracted"],
        len(result["contradictions"]), result["errors"],
    )
    return result


async def run_kg_consistency_check(svc: Any | None = None) -> str:
    """Monitor-friendly wrapper. Returns a summary string for alerting."""
    from app.database import get_db
    from app.core.brain import get_services

    if svc is None:
        try:
            svc = get_services()
        except Exception:
            svc = None

    db = get_db()
    res = await check_recent_answers(db, recent_hours=24, max_answers=30)

    if not res["contradictions"]:
        return (
            f"KG consistency: checked {res['checked']} answers, "
            f"extracted {res['claims_extracted']} claims, no contradictions found."
        )

    lines = [
        f"KG consistency check found {len(res['contradictions'])} contradictions "
        f"(checked {res['checked']} answers, {res['claims_extracted']} claims):"
    ]
    for c in res["contradictions"][:8]:
        lines.append(
            f"  - {c['subject']}.{c['predicate']}: "
            f"I said {c['answered']!r} but KG has {c['kg_stored']!r} — {c['reason']}"
        )
    if len(res["contradictions"]) > 8:
        lines.append(f"  ... and {len(res['contradictions']) - 8} more")
    return "\n".join(lines)
