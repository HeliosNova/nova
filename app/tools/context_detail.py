"""Context Detail Tool — on-demand fetching of full context details.

The system prompt shows compact summaries with IDs (e.g. [L42], [K7]).
This tool lets the LLM fetch full details for any summary item.
Read-only, safe to cache.
"""

from __future__ import annotations

import asyncio
import logging

from app.tools.base import BaseTool, ToolResult, ErrorCategory

logger = logging.getLogger(__name__)


class ContextDetailTool(BaseTool):
    name = "context_detail"
    description = (
        "Fetch full details for a context summary item. Use when you need more "
        "information than shown in the system prompt summaries. Pass the category "
        "and numeric ID from the summary prefix (e.g. [L42] → category='lesson', item_id=42)."
    )
    parameters = "category: str (lesson|kg_fact|user_fact|document|reflexion), item_id: int"
    input_schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["lesson", "kg_fact", "user_fact", "document", "reflexion"],
                "description": "Type of context item to fetch details for.",
            },
            "item_id": {
                "type": "integer",
                "description": "Numeric ID from the summary prefix (e.g. 42 from [L42]).",
            },
        },
        "required": ["category", "item_id"],
    }

    async def execute(self, *, category: str = "", item_id: int = 0, **kwargs) -> ToolResult:
        if not category:
            return ToolResult(
                output="", success=False,
                error="category is required",
                error_category=ErrorCategory.VALIDATION,
            )
        if not item_id:
            return ToolResult(
                output="", success=False,
                error="item_id is required",
                error_category=ErrorCategory.VALIDATION,
            )

        from app.core.brain import get_services
        svc = get_services()

        try:
            if category == "lesson":
                return await self._get_lesson(svc, item_id)
            elif category == "kg_fact":
                return await self._get_kg_fact(svc, item_id)
            elif category == "user_fact":
                return await self._get_user_fact(svc, item_id)
            elif category == "document":
                return await self._get_document(svc, item_id)
            elif category == "reflexion":
                return await self._get_reflexion(svc, item_id)
            else:
                return ToolResult(
                    output="", success=False,
                    error=f"Unknown category: {category}",
                    error_category=ErrorCategory.VALIDATION,
                )
        except Exception as e:
            logger.warning("context_detail(%s, %d) failed: %s", category, item_id, e)
            return ToolResult(
                output="", success=False,
                error=f"Failed to fetch details: {e}",
                error_category=ErrorCategory.INTERNAL,
            )

    async def _get_lesson(self, svc, item_id: int) -> ToolResult:
        if not svc.learning:
            return ToolResult(output="No learning engine available.", success=False, error="Not available", error_category=ErrorCategory.NOT_FOUND)

        row = await asyncio.to_thread(
            svc.learning._db.fetchone,
            "SELECT id, topic, wrong_answer, correct_answer, lesson_text, context, "
            "confidence, times_retrieved, times_helpful, created_at "
            "FROM lessons WHERE id = ?", (item_id,)
        )
        if not row:
            return ToolResult(output=f"Lesson #{item_id} not found.", success=False, error="Not found", error_category=ErrorCategory.NOT_FOUND)

        confidence = row["confidence"] or 0.8
        label = "HIGH" if confidence >= 0.8 else ("MED" if confidence >= 0.5 else "LOW")

        lines = [
            f"## Lesson #{row['id']} [{label}] — {row['topic']}",
            "",
            f"**Correct answer:** {row['correct_answer']}",
        ]
        if row["wrong_answer"]:
            lines.append(f"**Wrong answer:** {row['wrong_answer']}")
        if row["lesson_text"]:
            lines.append(f"**Lesson:** {row['lesson_text']}")
        if row["context"]:
            lines.append(f"**Original context:** {row['context']}")
        lines.append(f"**Confidence:** {confidence:.2f} | Retrieved {row['times_retrieved']}x | Helpful {row['times_helpful']}x")
        lines.append(f"**Created:** {row['created_at']}")

        return ToolResult(output="\n".join(lines), success=True)

    async def _get_kg_fact(self, svc, item_id: int) -> ToolResult:
        if not svc.kg:
            return ToolResult(output="No knowledge graph available.", success=False, error="Not available", error_category=ErrorCategory.NOT_FOUND)

        row = await asyncio.to_thread(
            svc.kg._db.fetchone,
            "SELECT id, subject, predicate, object, confidence, source, "
            "created_at, valid_from, valid_to, provenance, superseded_by "
            "FROM kg_facts WHERE id = ?", (item_id,)
        )
        if not row:
            return ToolResult(output=f"KG fact #{item_id} not found.", success=False, error="Not found", error_category=ErrorCategory.NOT_FOUND)

        pred = row["predicate"].replace("_", " ")
        lines = [
            f"## KG Fact #{row['id']}",
            f"**{row['subject']}** —{pred}→ **{row['object']}**",
            "",
            f"**Confidence:** {row['confidence']:.2f}",
            f"**Source:** {row['source']}",
        ]
        if row["provenance"]:
            lines.append(f"**Provenance:** {row['provenance']}")
        if row["valid_from"]:
            lines.append(f"**Valid from:** {row['valid_from']}")
        if row["valid_to"]:
            lines.append(f"**Valid to:** {row['valid_to']} (superseded)")
        if row["superseded_by"]:
            lines.append(f"**Superseded by:** fact #{row['superseded_by']}")

        # Related facts (1-hop)
        related = await asyncio.to_thread(
            svc.kg._db.fetchall,
            "SELECT id, subject, predicate, object FROM kg_facts "
            "WHERE (LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?)) "
            "AND id != ? AND valid_to IS NULL LIMIT 5",
            (row["subject"], row["subject"], item_id),
        )
        if related:
            lines.append("")
            lines.append("**Related facts:**")
            for r in related:
                p = r["predicate"].replace("_", " ")
                lines.append(f"  - [K{r['id']}] {r['subject']} —{p}→ {r['object']}")

        return ToolResult(output="\n".join(lines), success=True)

    async def _get_user_fact(self, svc, item_id: int) -> ToolResult:
        if not svc.user_facts:
            return ToolResult(output="No user facts available.", success=False, error="Not available", error_category=ErrorCategory.NOT_FOUND)

        row = await asyncio.to_thread(
            svc.user_facts._db.fetchone,
            "SELECT * FROM user_facts WHERE id = ?", (item_id,)
        )
        if not row:
            return ToolResult(output=f"User fact #{item_id} not found.", success=False, error="Not found", error_category=ErrorCategory.NOT_FOUND)

        lines = [
            f"## User Fact #{row['id']}",
            f"**Key:** {row['key']}",
            f"**Value:** {row['value']}",
        ]
        if row.get("category"):
            lines.append(f"**Category:** {row['category']}")
        if row.get("source"):
            lines.append(f"**Source:** {row['source']}")
        if row.get("confidence"):
            lines.append(f"**Confidence:** {row['confidence']:.2f}")
        if row.get("created_at"):
            lines.append(f"**Created:** {row['created_at']}")

        return ToolResult(output="\n".join(lines), success=True)

    async def _get_document(self, svc, item_id: int) -> ToolResult:
        if not svc.retriever:
            return ToolResult(output="No retriever available.", success=False, error="Not available", error_category=ErrorCategory.NOT_FOUND)

        # item_id is the chunk row id from documents table
        row = await asyncio.to_thread(
            svc.retriever._db.fetchone,
            "SELECT id, title, source, content, created_at FROM documents WHERE id = ?",
            (item_id,),
        )
        if not row:
            return ToolResult(output=f"Document chunk #{item_id} not found.", success=False, error="Not found", error_category=ErrorCategory.NOT_FOUND)

        content = row["content"] or ""
        if len(content) > 2000:
            content = content[:2000] + "\n[...truncated]"

        lines = [
            f"## Document Chunk #{row['id']}",
            f"**Title:** {row['title'] or 'Untitled'}",
            f"**Source:** {row['source'] or 'Unknown'}",
            f"**Created:** {row['created_at']}",
            "",
            content,
        ]

        return ToolResult(output="\n".join(lines), success=True)

    async def _get_reflexion(self, svc, item_id: int) -> ToolResult:
        if not svc.reflexions:
            return ToolResult(output="No reflexion store available.", success=False, error="Not available", error_category=ErrorCategory.NOT_FOUND)

        row = await asyncio.to_thread(
            svc.reflexions._db.fetchone,
            "SELECT id, task_summary, outcome, reflection, quality_score, "
            "tools_used, revision_count, created_at "
            "FROM reflexions WHERE id = ?", (item_id,)
        )
        if not row:
            return ToolResult(output=f"Reflexion #{item_id} not found.", success=False, error="Not found", error_category=ErrorCategory.NOT_FOUND)

        label = "FAILURE" if row["outcome"] == "failure" else "SUCCESS"
        lines = [
            f"## Reflexion #{row['id']} [{label}]",
            f"**Task:** {row['task_summary']}",
            f"**Outcome:** {row['outcome']}",
            f"**Quality:** {row['quality_score']:.2f}",
            "",
            f"**Reflection:** {row['reflection']}",
        ]
        if row["tools_used"]:
            lines.append(f"**Tools used:** {row['tools_used']}")
        lines.append(f"**Tool rounds:** {row['revision_count']}")
        lines.append(f"**Created:** {row['created_at']}")

        return ToolResult(output="\n".join(lines), success=True)
