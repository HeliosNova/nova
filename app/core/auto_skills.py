"""Auto skill creation — two acquisition paths.

1. Chat-path extraction (maybe_extract_skill): when a response uses 1+ tools
   successfully, a background task asks the LLM to extract a reusable skill
   (trigger pattern + steps). Reuses ALL existing skill guards (broadness,
   regex validation, capture groups). Honest status: this path is starved —
   it requires tool-using live chat, which a monitor-driven personal AI barely
   generates (0 organic skills in Nova's lifetime, audit 2026-08-23).

2. Scheduled induction (induce_procedure_skills): mines the PROVEN substrate —
   success reflexions + repeatedly-helpful lessons — clusters by shared topic-
   keyword pairs, requires >=2 corroborating items, and LLM-distills each
   cluster into an NL PROCEDURE skill (migration 30: semantic-match-only,
   rendered into the prompt as guidance). Runs from daily maintenance,
   decoupled from chat — the same decoupling that makes auto_tools the
   working self-improvement path.

Threshold notes (chat path):
- Single-tool interactions: extracted only when the answer looks successful
  (no failure markers). This prevents caching "I couldn't find anything" patterns.
- Multi-tool interactions (2+): extracted unconditionally (same as before).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.config import config
from app.core import llm
from app.core.skills import SkillStore, _get_tool_names, _is_too_broad, _has_capture_group_mismatch

logger = logging.getLogger(__name__)

# Phrases that indicate a failed/uncertain response — don't cache these patterns.
_FAILURE_MARKERS = (
    "couldn't find",
    "could not find",
    "failed to",
    "no results",
    "i don't know",
    "i'm not sure",
    "unable to",
    "not found",
    "error occurred",
    "an error",
    "i apologize",
    "unfortunately",
    "no information",
)


async def maybe_extract_skill(
    query: str,
    tool_results: list[dict],
    final_answer: str,
    skills: SkillStore,
    quality_score: float | None = None,
) -> None:
    """Background task: attempt to extract a reusable skill from a tool interaction.

    Runs when:
    - ENABLE_AUTO_SKILL_CREATION is true
    - 1+ tool results in the interaction
    - Single-tool interactions pass a quality gate (no failure markers, OR quality >= 0.7)
    - Multi-tool interactions extracted unconditionally
    - Not a delegate-based interaction

    quality_score: reflexion quality from the response (0.0–1.0). When >= 0.7 for
    single-tool interactions, the failure-marker check is bypassed — the reflexion
    system already confirmed the response was good.

    Failures are logged, never raised.
    """
    if not config.ENABLE_AUTO_SKILL_CREATION:
        return

    if not tool_results:
        return

    # Skip if any tool was delegate (sub-agent interactions are too complex)
    if any(tr.get("tool") == "delegate" for tr in tool_results):
        return

    # Quality gate for single-tool interactions: only extract from successful responses.
    # Multi-tool interactions are assumed successful enough to attempt extraction.
    # Exception: if the reflexion quality score is >= 0.7, the response was confirmed
    # good — bypass the failure marker check.
    if len(tool_results) == 1:
        high_quality = quality_score is not None and quality_score >= 0.7
        if not high_quality:
            answer_lower = final_answer.lower()
            if any(marker in answer_lower for marker in _FAILURE_MARKERS):
                logger.debug(
                    "Auto-skill: single-tool response contains failure marker, skipping"
                )
                return

    tool_summary = json.dumps([
        {
            "tool": tr["tool"],
            "args": tr["args"],
            "output": (tr.get("output", "") or "")[:200],
        }
        for tr in tool_results
    ], indent=2)

    try:
        result = await asyncio.wait_for(
            llm.invoke_nothink(
                [
                    {
                        "role": "system",
                        "content": (
                            "You extract reusable skills from tool interactions.\n"
                            "A skill is a trigger pattern (regex) and a sequence of tool calls "
                            "that should be repeated for similar future queries.\n\n"
                            "Given a query, the tool calls used, and the final answer, decide if "
                            "this is a reusable pattern worth caching.\n\n"
                            "Respond with JSON:\n"
                            '{"name": "short_name", "trigger_pattern": "regex_for_similar_queries", '
                            '"steps": [{"tool": "tool_name", "args_template": {"key": "{query}"}, '
                            '"output_key": "result"}], '
                            '"answer_template": "Template using {result}"}\n\n'
                            "IMPORTANT:\n"
                            "- trigger_pattern must be a valid regex (not too broad)\n"
                            f"- steps must use actual tools: {', '.join(_get_tool_names())}\n"
                            "- PLACEHOLDER CONTRACT (violations get the skill rejected): inside "
                            "args_template values you may ONLY use {query} (the user's raw input), "
                            "a (?P<name>...) named group defined in YOUR trigger_pattern, or an "
                            "earlier step's output_key. Do NOT invent placeholders like {topic} "
                            "or {search_query} — use {query} instead.\n"
                            "- Only extract if this is genuinely reusable for future similar queries\n\n"
                            'If NOT reusable, respond: {"skip": true}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Query: {query}\n\n"
                            f"Tool calls:\n{tool_summary}\n\n"
                            f"Answer: {final_answer[:500]}"
                        ),
                    },
                ],
                json_mode=True,
                json_prefix="{",
                max_tokens=500,
                temperature=0.2,
            ),
            timeout=config.INTERNAL_LLM_TIMEOUT,
        )

        # Funnel logs at INFO (2026-08-19): the whole decision path was DEBUG,
        # so 0-organic-skills-ever was indistinguishable from "never fires".
        # Live probe showed the extractor working and the LLM producing skills
        # that guards then silently discarded.
        obj = llm.extract_json_object(result)
        if not obj or obj.get("skip") or not obj.get("name"):
            logger.info("Auto-skill: LLM declined to extract (query=%r)", query[:60])
            return

        def _procedure_fallback(reason: str) -> None:
            """Strict-skill extraction failed a guard — memorialize the
            interaction as a PROCEDURE skill instead of discarding it
            (migration 30). Only multi-tool interactions qualify: a one-tool
            pattern the strict path rejected isn't worth a procedure."""
            if len(tool_results) < 2:
                return
            steps_lines = []
            for i, tr in enumerate(tool_results[:5], 1):
                args = tr.get("args") or {}
                akeys = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in list(args.items())[:3])
                steps_lines.append(f"{i}. {tr.get('tool')}({akeys})")
            proc = (
                f"For queries like: {query[:200]}\n"
                "Effective tool sequence:\n"
                + "\n".join(steps_lines)
                + "\nThen synthesize the tool results into a direct answer with "
                "specific figures and named entities."
            )
            pname = (str(obj.get("name") or "procedure"))[:80]
            pid = skills.create_procedure_skill(pname, query[:200], proc, source="auto")
            if pid:
                logger.info("Auto-skill: procedure fallback created #%d '%s' (strict path failed: %s)",
                            pid, pname, reason)

        pattern = obj.get("trigger_pattern", "")
        if not pattern:
            return

        try:
            re.compile(pattern)
        except re.error:
            logger.info("Auto-skill: invalid regex %r — rejected", pattern[:80])
            _procedure_fallback("invalid regex")
            return

        if _is_too_broad(pattern):
            logger.info("Auto-skill: pattern too broad %r — rejected", pattern[:80])
            _procedure_fallback("pattern too broad")
            return

        steps = obj.get("steps", [])
        valid_tool_names = _get_tool_names()
        for step in steps:
            if not isinstance(step, dict):
                logger.info("Auto-skill: step is not a dict — rejected")
                return
            if step.get("tool") not in valid_tool_names:
                logger.info("Auto-skill: unknown tool %r — rejected", step.get("tool"))
                return
            if "args_template" not in step:
                logger.info("Auto-skill: step missing args_template — rejected")
                return
            output_key = step.get("output_key", "")
            if output_key and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", output_key):
                logger.info("Auto-skill: invalid output_key %r — rejected", output_key)
                return

        answer_template = obj.get("answer_template")

        if _has_capture_group_mismatch(pattern, steps, answer_template):
            # Repair pass before discarding: the commonest LLM slip is an
            # invented alias for the user's input ({topic}, {search_query},
            # {question}, {q}, {subject}, {input}) in args_template. Those have
            # exactly one sane binding — {query}. Substitute and re-check;
            # anything still mismatched after repair is genuinely broken.
            _ALIAS_RE = re.compile(r"\{(topic|search_query|question|q|subject|input|text|term)\}")
            repaired = False
            try:
                compiled = re.compile(pattern)
                known = {"query"} | set(compiled.groupindex.keys()) | {
                    s.get("output_key", "") for s in steps if s.get("output_key")}
                for step in steps:
                    at = step.get("args_template")
                    if not isinstance(at, dict):
                        continue
                    for k, v in list(at.items()):
                        if isinstance(v, str):
                            nv = _ALIAS_RE.sub(
                                lambda m: "{query}" if m.group(1) not in known else m.group(0), v)
                            if nv != v:
                                at[k] = nv
                                repaired = True
            except re.error:
                pass
            if not (repaired and not _has_capture_group_mismatch(pattern, steps, answer_template)):
                logger.info("Auto-skill: capture group mismatch — rejected (pattern=%r)", pattern[:80])
                _procedure_fallback("capture group mismatch")
                return
            logger.info("Auto-skill: repaired invented placeholder alias → {query}")

        skill_id = skills.create_skill(
            name=obj["name"],
            trigger_pattern=pattern,
            steps=steps,
            answer_template=answer_template,
            source="auto",
        )

        if skill_id:
            logger.info(
                "Auto-skill created: '%s' (id=%d, trigger=%s, tools=%d)",
                obj["name"], skill_id, pattern, len(tool_results),
            )
        else:
            logger.debug("Auto-skill rejected by guards: '%s'", obj["name"])

    except Exception as e:
        logger.debug("Auto-skill extraction failed: %s", e)


# ===========================================================================
# Scheduled induction from proven memory (audit 2026-08-23)
# ===========================================================================
#
# Root cause of 0-organic-skills-ever: maybe_extract_skill is gated on
# tool-using LIVE CHAT (brain.py post-response, len(tool_results)>=1), a signal
# a monitor-driven personal AI barely generates — the extractor was starved,
# not broken. auto_tools became the one working self-improvement path precisely
# because it is decoupled from chat (scheduled, mines durable state). This pass
# applies the same decoupling to skills: mine the PROVEN substrate — success
# reflexions and repeatedly-helpful lessons — cluster by topic, require >=2
# corroborating items (trajectory-pool consolidation: the ExpeL/Trace2Skill/
# MIND-Skill line), and distill each cluster into an NL PROCEDURE skill
# (migration-30 representation: no regex, matched semantically, rendered into
# the prompt as guidance — brain.py:684). Validation + the 0.94 semantic dup
# bar live in create_procedure_skill.

_INDUCTION_SYSTEM_PROMPT = (
    "You distill an agent's proven experience into ONE reusable procedure.\n"
    "Today's date is {today}.\n\n"
    "You are given several pieces of evidence (successful task reflections "
    "and repeatedly-helpful lessons) that share a topic. If they support a "
    "genuinely reusable way of handling that topic, respond with JSON:\n"
    '{{"name": "short_snake_case_name", '
    '"description": "one sentence: when this procedure applies", '
    '"procedure_text": "When to use: ...\\nSteps:\\n1. ...\\n2. ...\\n'
    'Pitfalls: ..."}}\n\n'
    "RULES:\n"
    "- Ground every step in the evidence. Do NOT invent tools, figures or "
    "steps the evidence does not support.\n"
    "- procedure_text must be 60-2000 characters, imperative voice, "
    "specific enough that following it changes behavior.\n"
    "- The procedure must generalize beyond the exact evidence queries.\n"
    'If the evidence is too thin or too incoherent: {{"skip": true}}'
)


_STEM_SUFFIXES = ("ings", "ing", "tion", "ions", "ers", "er", "ed", "es", "s")


def _stem(word: str) -> str:
    """Light suffix-stripping stem, applied to fixpoint so inflectional
    variants converge symmetrically ("answering"→"answer"→"answ" and
    "answer"→"answ" — a single pass left them unequal)."""
    while True:
        for suf in _STEM_SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 4:
                word = word[: -len(suf)]
                break
        else:
            return word


def _cluster_covered(key, existing_names: str) -> bool:
    """Is every cluster keyword already represented in an enabled skill name?

    Raw-substring comparison minted semantic twins (2026-08-24:
    'factual_question_answering' created seconds after the inductor logged
    the {'questions','factual'} cluster as covered by
    'answer_factual_questions' — "answering" is not a substring of
    "answer"). Compare stems both ways so inflectional variants count.
    """
    name_stems = {_stem(w) for w in re.findall(r"[a-z0-9]+", existing_names.lower())}
    return all(
        kw in existing_names or _stem(kw) in name_stems for kw in key
    )


async def induce_procedure_skills(db, skills: SkillStore, *, max_new: int = 2) -> int:
    """Scheduled pass: distill procedure skills from proven memory.

    Called from daily maintenance (next to distill_principles). Returns the
    number of skills created. Failures are logged, never raised.
    """
    if not config.ENABLE_AUTO_SKILL_CREATION:
        return 0

    from collections import defaultdict
    from datetime import datetime, timezone

    from app.core.principles import _topic_keywords

    # --- Gather the proven substrate (each source isolated — a missing table
    # must not kill the pass; same hardening as goal_deriver) ---
    items: list[dict] = []  # {key_text, evidence, weight}
    try:
        for r in db.fetchall(
            "SELECT task_summary, reflection, quality_score FROM reflexions "
            "WHERE quality_score >= 0.8 AND (is_eval IS NULL OR is_eval = 0) "
            "ORDER BY quality_score DESC LIMIT 60"
        ):
            items.append({
                "key_text": r["task_summary"] or "",
                "evidence": f"[reflection q={r['quality_score']:.2f}] "
                            f"{(r['task_summary'] or '')[:120]} — "
                            f"{(r['reflection'] or '')[:220]}",
                "weight": float(r["quality_score"] or 0.8),
            })
    except Exception as e:
        logger.info("Skill-induction: reflexions source unavailable: %s", e)
    try:
        for r in db.fetchall(
            "SELECT topic, lesson_text, times_helpful FROM lessons "
            "WHERE times_helpful >= 3 ORDER BY times_helpful DESC LIMIT 60"
        ):
            items.append({
                "key_text": r["topic"] or "",
                "evidence": f"[lesson helpful×{r['times_helpful']}] "
                            f"{(r['topic'] or '')[:120]} — "
                            f"{(r['lesson_text'] or '')[:220]}",
                "weight": 0.8 + min(0.2, (r["times_helpful"] or 0) / 50.0),
            })
    except Exception as e:
        logger.info("Skill-induction: lessons source unavailable: %s", e)

    # --- Cluster on ALL keyword pairs, not principles.py's top-2-alphabetical
    # key: real task summaries carry stray third words ("... task", "... best
    # practices") that push the shared nouns out of an alphabetical top-2 and
    # silently under-cluster. Every 2-subset of an item's keywords is a bucket;
    # two items corroborate if ANY pair is shared. `consumed` ensures one piece
    # of evidence backs at most one induced skill per pass (the same items
    # appear in several pair-buckets). ---
    clusters: dict[frozenset, list[int]] = defaultdict(list)
    for idx, it in enumerate(items):
        kws = sorted(_topic_keywords(it["key_text"]))
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                clusters[frozenset((kws[i], kws[j]))].append(idx)
    corroborated = [
        (key, idxs) for key, idxs in clusters.items() if len(set(idxs)) >= 2
    ]
    if not corroborated:
        logger.info(
            "Skill-induction: no corroborated clusters (%d items, %d pair-buckets)",
            len(items), len(clusters),
        )
        return 0
    corroborated.sort(
        key=lambda kv: (len(set(kv[1])), sum(items[i]["weight"] for i in set(kv[1]))),
        reverse=True,
    )
    consumed: set[int] = set()

    # Cheap pre-dedup: skip clusters whose (stemmed) keywords already appear
    # in an enabled skill's name (the real gate is the 0.94 semantic dup bar
    # inside create_procedure_skill — this just avoids burning LLM calls).
    existing_names = " ".join(
        (r["name"] or "").lower()
        for r in db.fetchall("SELECT name FROM skills WHERE enabled = 1")
    )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    created = 0
    for key, idxs in corroborated:
        if created >= max_new:
            break
        members = [items[i] for i in dict.fromkeys(idxs) if i not in consumed]
        if len(members) < 2:
            continue  # remaining evidence already backed an induced skill
        if existing_names and _cluster_covered(key, existing_names):
            logger.info("Skill-induction: cluster %s already covered by an existing skill", set(key))
            continue
        evidence = "\n".join(m["evidence"] for m in members[:8])
        try:
            result = await asyncio.wait_for(
                llm.invoke_nothink(
                    [
                        {"role": "system",
                         "content": _INDUCTION_SYSTEM_PROMPT.format(today=today)},
                        {"role": "user",
                         "content": f"Topic keywords: {', '.join(sorted(key))}\n\n"
                                    f"Evidence ({len(members)} items):\n{evidence}"},
                    ],
                    json_mode=True,
                    json_prefix="{",
                    max_tokens=700,
                    temperature=0.2,
                    num_ctx=8192,
                ),
                timeout=config.INTERNAL_LLM_TIMEOUT,
            )
            obj = llm.extract_json_object(result)
            if not obj or obj.get("skip") or not obj.get("name"):
                logger.info("Skill-induction: LLM declined for cluster %s", set(key))
                continue
            sid = skills.create_procedure_skill(
                name=str(obj.get("name") or "")[:80],
                description=str(obj.get("description") or "")[:200],
                procedure_text=str(obj.get("procedure_text") or ""),
                source="induced",
                initial_success_rate=0.6,
            )
            if sid:
                created += 1
                consumed.update(idxs)
                logger.info(
                    "Skill-induction: created procedure skill #%d '%s' from %d "
                    "corroborating items (cluster %s)",
                    sid, obj.get("name"), len(members), set(key),
                )
            else:
                logger.info(
                    "Skill-induction: create_procedure_skill rejected '%s' "
                    "(dup/length gate)", obj.get("name"),
                )
        except Exception as e:
            logger.info("Skill-induction: cluster %s failed: %s", set(key), e)
    return created
