"""Learning engine — correction detection, lesson storage, training data.

The differentiator: Nova learns from every correction and gets permanently smarter.

Flow: correction detected → extract lesson → save to SQLite → save training pair to JSONL
→ on future queries, relevant lessons injected into prompt → better answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import config
from app.core import llm
from app.database import get_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Correction:
    """A detected correction from the user."""
    user_message: str
    previous_answer: str
    topic: str = ""
    correct_answer: str = ""
    wrong_answer: str = ""
    original_query: str = ""    # The question that led to the wrong answer
    lesson_text: str = ""       # One-sentence lesson summary


@dataclass
class Lesson:
    """A learned lesson from a correction."""
    id: int
    topic: str
    wrong_answer: str | None
    correct_answer: str
    lesson_text: str | None
    context: str | None
    confidence: float
    times_retrieved: int
    times_helpful: int
    created_at: str | None


# ---------------------------------------------------------------------------
# Correction detection (2-stage: regex + LLM)
# ---------------------------------------------------------------------------

_CORRECTION_PATTERNS = [
    # Direct corrections
    re.compile(r"(?i)\b(?:actually|no,?\s|that'?s?\s+(?:wrong|incorrect|not\s+right))"),
    re.compile(r"(?i)\b(?:you'?re\s+wrong|that\s+is\s+(?:wrong|incorrect|false))"),
    re.compile(r"(?i)\b(?:the\s+(?:correct|right|actual)\s+(?:answer|information)\s+is)"),
    re.compile(r"(?i)\b(?:it'?s?\s+actually|it\s+should\s+be)"),
    re.compile(r"(?i)\b(?:correction:|wrong!|incorrect!|not\s+quite)"),
    # Flexible "that ... is wrong/incomplete" (catches "that data is wrong", "that info is incomplete")
    re.compile(r"(?i)\bthat\b.{1,30}\b(?:wrong|incorrect|incomplete|inaccurate|misleading)\b"),
    # Preference / procedural corrections
    re.compile(r"(?i)\bremember\s+that\b"),
    re.compile(r"(?i)\binstead\s+of\b"),
    re.compile(r"(?i)\b(?:you\s+should|always)\s+(?:use|check|search|try)\b"),
    # "you should also/always search/check" — allow adverbs between should and verb
    re.compile(r"(?i)\byou\s+should\s+(?:also|always|really)\s+(?:use|check|search|try)\b"),
    re.compile(r"(?i)\b(?:next\s+time|from\s+now\s+on|in\s+the\s+future)\b"),
    re.compile(r"(?i)\b(?:don'?t|do\s+not)\s+(?:use|do|say|recommend)\b"),
    re.compile(r"(?i)\brather\b.*\bnot\b"),
    # Data quality corrections (require context to avoid false positives like "I'm missing my dog")
    re.compile(r"(?i)\b(?:data|info(?:rmation)?|answer|result|response|it)\b.{0,20}\b(?:missed|missing|incomplete|outdated)\b"),
    re.compile(r"(?i)\b(?:you|that)\b.{0,30}\b(?:missed|missing|incomplete|outdated)\b"),
]


def is_likely_correction(text: str) -> bool:
    """Fast regex check — is this message likely a correction?"""
    return any(p.search(text) for p in _CORRECTION_PATTERNS)


# The extraction prompt — detailed with few-shot examples so Qwen doesn't leave fields empty
_EXTRACTION_PROMPT = """You extract structured corrections from conversations. The user is correcting something the assistant said wrong.

You MUST respond with a JSON object containing ALL these fields (never leave them empty):

{
  "is_correction": true,
  "topic": "brief topic (2-5 words)",
  "wrong_answer": "what the assistant said that was wrong (quote or summarize the specific wrong claim)",
  "correct_answer": "what the user says is actually correct (the specific right information)",
  "lesson_text": "one-sentence lesson: 'X, not Y' or 'The correct answer is X'"
}

If this is NOT a correction, respond: {"is_correction": false}

## Examples

Assistant said: "Python was created by James Gosling"
User says: "Actually, Python was created by Guido van Rossum"
→ {"is_correction": true, "topic": "Python creator", "wrong_answer": "Python was created by James Gosling", "correct_answer": "Python was created by Guido van Rossum", "lesson_text": "Python was created by Guido van Rossum, not James Gosling"}

Assistant said: "The capital of Australia is Sydney"
User says: "That's wrong, the capital is Canberra"
→ {"is_correction": true, "topic": "Capital of Australia", "wrong_answer": "The capital of Australia is Sydney", "correct_answer": "The capital of Australia is Canberra", "lesson_text": "The capital of Australia is Canberra, not Sydney"}

Assistant said: "The sky on Mars appears pinkish-red to yellow-brown"
User says: "Actually, the correct answer is that the sky on Mars is butterscotch colored during the day, not just pink or red"
→ {"is_correction": true, "topic": "Mars daytime sky color", "wrong_answer": "The sky on Mars appears pinkish-red to yellow-brown", "correct_answer": "The sky on Mars is butterscotch colored during the day", "lesson_text": "The Mars daytime sky is butterscotch colored, not just pink or red"}

Assistant said: "I recommend using React for that project"
User says: "Remember that I prefer Vue over React"
→ {"is_correction": true, "topic": "Framework preference", "wrong_answer": "Recommended React", "correct_answer": "User prefers Vue over React", "lesson_text": "Owner prefers Vue over React"}"""


# ---------------------------------------------------------------------------
# Learning Engine
# ---------------------------------------------------------------------------

class LearningEngine:
    """Handles correction detection, lesson storage, and training data collection."""

    def __init__(self, db=None):
        self._db = db or get_db()
        self._lessons_collection = None
        self._training_lock = asyncio.Lock()

    def _get_lessons_collection(self):
        """Lazy-init ChromaDB collection for semantic lesson search."""
        if self._lessons_collection is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
                self._lessons_collection = client.get_or_create_collection(
                    name="lessons",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning("Failed to init lessons ChromaDB collection: %s", e)
                return None
        return self._lessons_collection

    def reindex_lessons(self) -> int:
        """One-time backfill of existing lessons into ChromaDB. Returns count indexed."""
        collection = self._get_lessons_collection()
        if collection is None:
            return 0
        # Skip if collection already has data
        if collection.count() > 0:
            logger.info("Lessons collection already has %d entries, skipping reindex", collection.count())
            return 0

        all_lessons = self._db.fetchall("SELECT * FROM lessons")
        if not all_lessons:
            return 0

        ids = []
        documents = []
        metadatas = []
        for row in all_lessons:
            topic = row["topic"] or ""
            correct_answer = row["correct_answer"] or ""
            lesson_text = _row_get(row, "lesson_text")
            searchable = f"{topic} {correct_answer} {lesson_text}".strip()
            if not searchable:
                continue
            ids.append(str(row["id"]))
            documents.append(searchable)
            metadatas.append({"topic": topic, "confidence": row["confidence"]})

        if ids:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            logger.info("Reindexed %d lessons into ChromaDB", len(ids))
        return len(ids)

    async def detect_correction(
        self,
        user_message: str,
        previous_answer: str,
        original_query: str = "",
    ) -> Correction | None:
        """Detect if user_message is correcting previous_answer.

        Stage 1: regex check (fast)
        Stage 2: LLM confirmation + extraction (only if regex matches)
        """
        if not is_likely_correction(user_message):
            return None

        # Stage 2: Use LLM to confirm and extract the correction
        try:
            prompt_content = (
                f"Assistant said: \"{previous_answer[:500]}\"\n\n"
                f"User says: \"{user_message}\""
            )
            logger.info(
                "Correction LLM input: assistant=%d chars, user='%s'",
                min(len(previous_answer), 500),
                user_message[:120],
            )

            result = await asyncio.wait_for(
                llm.invoke_nothink(
                    [
                        {"role": "system", "content": _EXTRACTION_PROMPT},
                        {"role": "user", "content": prompt_content},
                    ],
                    json_mode=True,
                    json_prefix="{",
                ),
                timeout=15.0,
            )

            logger.info("Correction LLM raw output: %s", result[:500])

            obj = llm.extract_json_object(result)
            logger.info("Correction LLM parsed: %s", obj)

            if not obj or not obj.get("is_correction"):
                return None

            correct_answer = (obj.get("correct_answer") or "").strip()
            wrong_answer = (obj.get("wrong_answer") or "").strip()
            topic = (obj.get("topic") or "").strip()
            lesson_text = (obj.get("lesson_text") or "").strip()

            # Validation — if correct_answer is still empty, extract from user message
            if not correct_answer:
                correct_answer = _extract_answer_from_message(user_message)

            # If we still have nothing useful, build from the user message
            if not lesson_text:
                if correct_answer and wrong_answer:
                    lesson_text = f"{correct_answer}, not {wrong_answer}"
                elif correct_answer:
                    lesson_text = correct_answer

            # Final validation — we need at minimum a topic and either
            # correct_answer or lesson_text
            if not topic and not correct_answer and not lesson_text:
                logger.warning("Correction extraction produced empty fields, using fallback")
                return _fallback_correction(user_message, previous_answer, original_query)

            return Correction(
                user_message=user_message,
                previous_answer=previous_answer,
                topic=topic or "general",
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                original_query=original_query,
                lesson_text=lesson_text,
            )

        except asyncio.TimeoutError:
            logger.warning("Correction LLM timed out after 15s — skipping")
            return None
        except Exception as e:
            logger.warning("LLM correction detection failed: %s — skipping", e)
            return None

    def save_lesson(self, correction: Correction) -> int:
        """Save a lesson from a correction. Returns lesson ID, or -1 if rejected."""
        # Quality gate — reject garbage corrections
        if not _is_quality_content(correction.correct_answer):
            logger.warning("Lesson rejected: low-quality correct_answer '%s'", (correction.correct_answer or "")[:50])
            return -1

        # Build lesson_text if not already set
        lesson_text = correction.lesson_text
        if not lesson_text:
            if correction.correct_answer and correction.wrong_answer:
                lesson_text = f"{correction.correct_answer}, not {correction.wrong_answer}"
            elif correction.correct_answer:
                lesson_text = correction.correct_answer

        topic = correction.topic or "general"
        correct_answer = correction.correct_answer or ""

        # Deduplication — fuzzy match on topic + correct_answer
        existing = self._find_similar_lesson(topic, correct_answer)
        if existing:
            new_conf = min(1.0, existing["confidence"] + 0.1)
            self._db.execute(
                "UPDATE lessons SET confidence = ? WHERE id = ?",
                (new_conf, existing["id"]),
            )
            logger.info("Lesson dedup: boosted #%d confidence to %.2f", existing["id"], new_conf)
            return existing["id"]

        cursor = self._db.execute(
            """INSERT INTO lessons
               (topic, wrong_answer, correct_answer, lesson_text, context, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                topic,
                correction.wrong_answer or "",
                correct_answer,
                lesson_text or "",
                correction.original_query or correction.user_message,
                0.8,
            ),
        )
        lesson_id = cursor.lastrowid
        logger.info(
            "Saved lesson #%d: topic='%s', lesson='%s'",
            lesson_id, correction.topic, (lesson_text or "")[:80]
        )

        # Add to ChromaDB for semantic search
        try:
            collection = self._get_lessons_collection()
            if collection is not None:
                searchable = f"{topic} {correct_answer} {lesson_text or ''}".strip()
                if searchable:
                    collection.add(
                        ids=[str(lesson_id)],
                        documents=[searchable],
                        metadatas=[{"topic": topic, "confidence": 0.8}],
                    )
        except Exception as e:
            logger.warning("Failed to add lesson #%d to ChromaDB: %s", lesson_id, e)

        # Bound the lessons table
        self._prune_lessons()

        return lesson_id

    def get_relevant_lessons(self, query: str, limit: int | None = None) -> list[Lesson]:
        """Get lessons relevant to a query using hybrid retrieval (vector + keyword + RRF)."""
        limit = limit or config.MAX_LESSONS_IN_PROMPT

        all_lessons = self._db.fetchall(
            "SELECT * FROM lessons ORDER BY times_helpful DESC, confidence DESC LIMIT 500"
        )

        if not all_lessons:
            return []

        # Build a lookup by ID for fast access
        lesson_by_id: dict[int, dict] = {row["id"]: row for row in all_lessons}

        # --- Vector search via ChromaDB ---
        vector_ranked: list[int] = []  # lesson IDs in ranked order
        try:
            collection = self._get_lessons_collection()
            if collection is not None and collection.count() > 0:
                results = collection.query(
                    query_texts=[query],
                    n_results=min(limit * 2, collection.count()),
                    include=["distances"],
                )
                if results and results["ids"] and results["ids"][0]:
                    distances = results.get("distances", [[]])[0]
                    for i, id_str in enumerate(results["ids"][0]):
                        # Filter high-distance results (cosine: 0=identical, 2=opposite)
                        if distances and i < len(distances) and distances[i] > 0.7:
                            continue
                        try:
                            lid = int(id_str)
                            if lid in lesson_by_id:
                                vector_ranked.append(lid)
                        except ValueError:
                            continue
        except Exception as e:
            logger.warning("Vector lesson search failed: %s", e)

        # --- Keyword search (existing overlap logic) ---
        query_words = _normalize_words(query)
        keyword_ranked: list[int] = []
        if query_words:
            scored = []
            for row in all_lessons:
                topic_words = _normalize_words(row["topic"] or "")
                context_words = _normalize_words(row["context"] or "")
                correct_words = _normalize_words(row["correct_answer"] or "")
                lesson_words = _normalize_words(_row_get(row, "lesson_text"))
                all_lesson_words = topic_words | context_words | correct_words | lesson_words
                overlap = len((query_words & all_lesson_words) - _STOP_WORDS)
                non_stop_query = query_words - _STOP_WORDS
                if overlap >= 2 or (overlap == 1 and len(non_stop_query) <= 1):
                    scored.append((overlap, row["id"]))
            scored.sort(key=lambda x: x[0], reverse=True)
            keyword_ranked = [lid for _, lid in scored]

        # --- Reciprocal Rank Fusion (k=60) ---
        rrf_scores: dict[int, float] = {}
        k = config.RRF_K
        for rank, lid in enumerate(vector_ranked):
            rrf_scores[lid] = rrf_scores.get(lid, 0) + 1.0 / (k + rank + 1)
        for rank, lid in enumerate(keyword_ranked):
            rrf_scores[lid] = rrf_scores.get(lid, 0) + 1.0 / (k + rank + 1)

        if not rrf_scores:
            return []

        # Sort by RRF score descending, then filter by minimum relevance
        _MIN_RRF_SCORE = 0.015
        sorted_ids = sorted(rrf_scores, key=lambda lid: rrf_scores[lid], reverse=True)
        sorted_ids = [lid for lid in sorted_ids if rrf_scores[lid] >= _MIN_RRF_SCORE]

        lessons = []
        retrieved_ids = []
        for lid in sorted_ids[:limit]:
            row = lesson_by_id[lid]
            lessons.append(Lesson(
                id=row["id"],
                topic=row["topic"],
                wrong_answer=row["wrong_answer"],
                correct_answer=row["correct_answer"],
                lesson_text=_row_get(row, "lesson_text"),
                context=row["context"],
                confidence=row["confidence"],
                times_retrieved=row["times_retrieved"],
                times_helpful=row["times_helpful"],
                created_at=row["created_at"],
            ))
            retrieved_ids.append(row["id"])

        # Batch update retrieval counts + last_retrieved_at (avoid N+1)
        if retrieved_ids:
            placeholders = ",".join("?" for _ in retrieved_ids)
            self._db.execute(
                f"UPDATE lessons SET times_retrieved = times_retrieved + 1, "
                f"last_retrieved_at = CURRENT_TIMESTAMP WHERE id IN ({placeholders})",
                tuple(retrieved_ids),
            )

        return lessons

    def mark_lesson_helpful(self, lesson_id: int) -> None:
        """Mark a lesson as helpful (was used and the user didn't correct again).

        Uses dampened adjustments — delta scales by 1/(1+times_helpful) so
        early uses have big impact, later uses converge.
        """
        row = self._db.fetchone(
            "SELECT times_helpful FROM lessons WHERE id = ?", (lesson_id,),
        )
        times = row["times_helpful"] if row else 0
        delta = 0.05 / (1 + times)
        self._db.execute(
            "UPDATE lessons SET times_helpful = times_helpful + 1, "
            "confidence = MIN(1.0, confidence + ?) WHERE id = ?",
            (delta, lesson_id),
        )

    def mark_lesson_unhelpful(self, lesson_id: int) -> None:
        """Reduce confidence when a lesson was used but the answer was poor.

        Uses dampened adjustments — delta scales by 1/(1+times_helpful).
        """
        row = self._db.fetchone(
            "SELECT times_helpful FROM lessons WHERE id = ?", (lesson_id,),
        )
        times = row["times_helpful"] if row else 0
        delta = 0.05 / (1 + times)
        self._db.execute(
            "UPDATE lessons SET confidence = MAX(0.1, confidence - ?) WHERE id = ?",
            (delta, lesson_id),
        )

    async def save_training_pair(
        self,
        query: str,
        bad_answer: str,
        good_answer: str,
        channel: str = "api",
        confidence: float = 1.0,
    ) -> None:
        """Append a DPO-ready training pair to JSONL.

        Format: {"query": ..., "chosen": ..., "rejected": ..., "timestamp": ...}
        This is directly usable for DPO/ORPO fine-tuning with Unsloth.
        Uses asyncio.Lock for thread safety.

        Anti-poisoning (OWASP ASI06): only channels listed in
        TRAINING_DATA_CHANNELS may generate training data, and external
        channels require confidence >= 0.8.
        """
        # Skip if we don't have meaningful data
        if not query.strip() or not good_answer.strip():
            logger.warning("Skipping training pair: empty query or good_answer")
            return

        # Channel gate — only allowed channels produce training data
        allowed_channels = {
            c.strip().lower()
            for c in config.TRAINING_DATA_CHANNELS.split(",")
            if c.strip()
        }
        if allowed_channels and channel.lower() not in allowed_channels:
            logger.info("Skipping training pair: channel '%s' not in TRAINING_DATA_CHANNELS", channel)
            return

        # Confidence gate — external channels need high confidence
        if channel.lower() != "api" and confidence < 0.8:
            logger.info(
                "Skipping training pair from channel '%s': confidence %.2f < 0.8",
                channel, confidence,
            )
            return

        # Quality gate — don't save garbage as training data
        if not _is_quality_content(good_answer):
            logger.warning("Skipping training pair: low-quality good_answer")
            return

        path = Path(config.TRAINING_DATA_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "query": query,
            "chosen": good_answer,
            "rejected": bad_answer,
            "timestamp": datetime.now().isoformat(),
        }

        async with self._training_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Rotation: keep most recent 80% when over limit
            _rotate_training_data(path)

        logger.info("Saved training pair to %s", path)

    def get_all_lessons(self, limit: int = 100) -> list[Lesson]:
        """Get all lessons (for admin/API use)."""
        rows = self._db.fetchall(
            "SELECT * FROM lessons ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [
            Lesson(
                id=row["id"],
                topic=row["topic"],
                wrong_answer=row["wrong_answer"],
                correct_answer=row["correct_answer"],
                lesson_text=_row_get(row, "lesson_text"),
                context=row["context"],
                confidence=row["confidence"],
                times_retrieved=row["times_retrieved"],
                times_helpful=row["times_helpful"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_learning_summary(self, hours: int = 24) -> dict:
        """Summarize all learning activity in the last `hours` hours.

        Returns counts and sample items for lessons, skills, custom tools,
        reflexions, training pairs, and degraded skills.
        """
        window = f"-{hours} hours"

        # New lessons
        new_lessons = self._db.fetchall(
            "SELECT topic, lesson_text FROM lessons "
            "WHERE created_at > datetime('now', ?) ORDER BY created_at DESC",
            (window,),
        )

        # New skills
        new_skills = self._db.fetchall(
            "SELECT name, trigger_pattern FROM skills "
            "WHERE created_at > datetime('now', ?) ORDER BY created_at DESC",
            (window,),
        )

        # New custom tools
        new_tools = self._db.fetchall(
            "SELECT name, description FROM custom_tools "
            "WHERE created_at > datetime('now', ?) ORDER BY created_at DESC",
            (window,),
        )

        # New reflexions
        new_reflexions = self._db.fetchall(
            "SELECT task_summary, quality_score FROM reflexions "
            "WHERE created_at > datetime('now', ?) ORDER BY created_at DESC",
            (window,),
        )

        # Degraded skills — enabled but struggling (success_rate < 0.5, 3+ uses)
        degraded = self._db.fetchall(
            "SELECT name, success_rate, times_used FROM skills "
            "WHERE enabled = 1 AND success_rate < 0.5 AND times_used >= 3",
        )

        # Training pairs added in window (count by scanning JSONL timestamps)
        new_training = 0
        try:
            path = Path(config.TRAINING_DATA_PATH)
            if path.exists():
                from datetime import timedelta
                cutoff_iso = (datetime.now() - timedelta(hours=hours)).isoformat()
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("timestamp", "") >= cutoff_iso:
                                new_training += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
        except Exception as e:
            logger.warning("Failed to count new training pairs: %s", e)

        totals = self.get_metrics()

        return {
            "new_lessons": [
                {"topic": r["topic"], "lesson": _row_get(r, "lesson_text")}
                for r in new_lessons
            ],
            "new_skills": [
                {"name": r["name"], "trigger": r["trigger_pattern"]}
                for r in new_skills
            ],
            "new_tools": [
                {"name": r["name"], "description": r["description"]}
                for r in new_tools
            ],
            "new_reflexions": [
                {"task": r["task_summary"], "score": r["quality_score"]}
                for r in new_reflexions
            ],
            "degraded_skills": [
                {"name": r["name"], "success_rate": r["success_rate"], "uses": r["times_used"]}
                for r in degraded
            ],
            "new_training_pairs": new_training,
            "totals": totals,
        }

    def get_metrics(self) -> dict:
        """Get learning metrics."""
        lesson_count = self._db.fetchone("SELECT COUNT(*) as c FROM lessons")
        skill_count = self._db.fetchone("SELECT COUNT(*) as c FROM skills")
        last_correction = self._db.fetchone(
            "SELECT created_at FROM lessons ORDER BY created_at DESC LIMIT 1"
        )

        training_count = 0
        path = Path(config.TRAINING_DATA_PATH)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                training_count = sum(1 for _ in f)

        return {
            "total_lessons": lesson_count["c"] if lesson_count else 0,
            "total_skills": skill_count["c"] if skill_count else 0,
            "total_corrections": lesson_count["c"] if lesson_count else 0,
            "training_examples": training_count,
            "last_correction_date": (
                last_correction["created_at"] if last_correction else None
            ),
        }

    def delete_lesson(self, lesson_id: int) -> bool:
        """Delete a lesson."""
        cursor = self._db.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))
        if cursor.rowcount > 0:
            try:
                collection = self._get_lessons_collection()
                if collection is not None:
                    collection.delete(ids=[str(lesson_id)])
            except Exception as e:
                logger.warning("Failed to delete lesson #%d from ChromaDB: %s", lesson_id, e)
            return True
        return False

    def _find_similar_lesson(self, topic: str, correct_answer: str) -> dict | None:
        """Find an existing lesson similar enough to be a duplicate.

        Fast path: exact match on (topic, correct_answer).
        Slow path: normalized word overlap (Jaccard >= 0.7).
        """
        # Fast path — exact match
        exact = self._db.fetchone(
            "SELECT id, confidence FROM lessons WHERE topic = ? AND correct_answer = ?",
            (topic, correct_answer),
        )
        if exact:
            return exact

        # Slow path — fuzzy word overlap
        new_words = _normalize_words(topic + " " + correct_answer) - _STOP_WORDS
        if len(new_words) < 2:
            return None

        candidates = self._db.fetchall(
            "SELECT id, confidence, topic, correct_answer FROM lessons"
        )
        for row in candidates:
            existing_words = (
                _normalize_words((row["topic"] or "") + " " + (row["correct_answer"] or ""))
                - _STOP_WORDS
            )
            if not existing_words:
                continue
            overlap = len(new_words & existing_words)
            union = len(new_words | existing_words)
            if union > 0 and overlap / union >= 0.85:
                return row

        return None

    def _prune_lessons(self) -> None:
        """If lessons exceed MAX_LESSONS, delete oldest low-confidence ones.

        Also removes pruned lessons from ChromaDB to prevent phantom vector results.
        """
        count_row = self._db.fetchone("SELECT COUNT(*) as c FROM lessons")
        count = count_row["c"] if count_row else 0
        if count <= config.MAX_LESSONS:
            return

        excess = count - config.MAX_LESSONS
        # Collect IDs before deleting so we can sync ChromaDB
        to_delete = self._db.fetchall(
            """SELECT id FROM lessons
               ORDER BY confidence ASC, times_helpful ASC, created_at ASC
               LIMIT ?""",
            (excess,),
        )
        if not to_delete:
            return

        delete_ids = [row["id"] for row in to_delete]
        placeholders = ",".join("?" for _ in delete_ids)
        with self._db.transaction() as tx:
            tx.execute(
                f"DELETE FROM lessons WHERE id IN ({placeholders})",
                tuple(delete_ids),
            )

        # Sync: remove from ChromaDB too
        try:
            collection = self._get_lessons_collection()
            if collection is not None:
                collection.delete(ids=[str(lid) for lid in delete_ids])
        except Exception as e:
            logger.warning("Failed to sync lesson prune to ChromaDB: %s", e)

        logger.info("Pruned %d lessons (over %d limit)", excess, config.MAX_LESSONS)

    def decay_stale_lessons(self, days: int = 30, factor: float = 0.95) -> int:
        """Reduce confidence for lessons not retrieved in `days` days.

        Decays lessons that either:
        - Were never retrieved and are older than `days`, OR
        - Were last retrieved more than 60 days ago.

        Returns the number of lessons decayed.
        """
        rows = self._db.fetchall(
            """SELECT id, confidence FROM lessons
               WHERE (times_retrieved = 0 AND created_at < datetime('now', ?))
                  OR (last_retrieved_at IS NOT NULL AND last_retrieved_at < datetime('now', '-60 days'))""",
            (f"-{days} days",),
        )
        decayed = 0
        for row in rows:
            new_conf = max(0.1, row["confidence"] * factor)
            if new_conf < row["confidence"]:
                self._db.execute(
                    "UPDATE lessons SET confidence = ? WHERE id = ?",
                    (round(new_conf, 4), row["id"]),
                )
                decayed += 1
        if decayed:
            logger.info("Decayed confidence on %d stale lessons", decayed)
        return decayed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_get(row, key: str, default: str = "") -> str:
    """Safely get a column from a sqlite3.Row (which has no .get() method)."""
    try:
        val = row[key]
        return val if val else default
    except (IndexError, KeyError):
        return default


from app.core.text_utils import STOP_WORDS as _STOP_WORDS, normalize_words as _normalize_words  # noqa: E402


def _extract_answer_from_message(message: str) -> str:
    """Try to extract the 'correct' information from a correction message.

    Handles patterns like:
    - "Actually, X" → X
    - "The correct answer is X" → X
    - "It should be X" → X
    """
    patterns = [
        r"(?i)actually,?\s+(.+?)(?:\.|$)",
        r"(?i)the\s+(?:correct|right|actual)\s+(?:answer|info(?:rmation)?)\s+is\s+(.+?)(?:\.|$)",
        r"(?i)it\s+should\s+be\s+(.+?)(?:\.|$)",
        r"(?i)(?:no|nope),?\s+(?:it'?s?|the answer is)\s+(.+?)(?:\.|$)",
        r"(?i)remember\s+that\s+(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1).strip()
    # Last resort: return the whole message (it IS the correction)
    return message.strip()


_ERROR_PHRASES = [
    "i don't know",
    "i'm not sure",
    "i cannot",
    "i wasn't able",
    "could you rephrase",
    "error occurred",
    "failed to",
]


def _is_quality_content(text: str) -> bool:
    """Check if text is high enough quality to save as a lesson or training pair."""
    if not text or len(text.strip()) < 10:
        return False
    lower = text.lower()
    return not any(phrase in lower for phrase in _ERROR_PHRASES)


def _rotate_training_data(path: Path) -> None:
    """If training data exceeds MAX_TRAINING_PAIRS, keep most recent entries up to the limit."""
    try:
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= config.MAX_TRAINING_PAIRS:
            return
        keep = config.MAX_TRAINING_PAIRS
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[-keep:])
        logger.info("Rotated training data: %d → %d lines", len(lines), keep)
    except Exception as e:
        logger.warning("Training data rotation failed: %s", e)


def _fallback_correction(
    user_message: str,
    previous_answer: str,
    original_query: str,
) -> Correction:
    """Build a correction when LLM extraction fails.

    Extracts what it can from the user message via regex.
    """
    correct_answer = _extract_answer_from_message(user_message)

    # Build a reasonable lesson_text
    lesson_text = correct_answer
    if len(lesson_text) > 200:
        lesson_text = lesson_text[:200]

    return Correction(
        user_message=user_message,
        previous_answer=previous_answer,
        topic="general",
        correct_answer=correct_answer,
        wrong_answer=previous_answer[:200] if previous_answer else "",
        original_query=original_query,
        lesson_text=lesson_text,
    )
