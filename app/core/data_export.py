"""Unified data export/import with HMAC-SHA256 signing.

Extends the skill-only signing to lessons and KG facts.
Provides bundle export (all data types in one signed envelope).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class SignatureError(Exception):
    """Raised when signature verification fails."""


# ---------------------------------------------------------------------------
# Signing primitives (shared across all data types)
# ---------------------------------------------------------------------------

def load_key(path: str | Path) -> bytes:
    """Load a hex-encoded 32-byte signing key from a file."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    return bytes.fromhex(raw)


def canonical_json(data: dict) -> bytes:
    """Canonical JSON: sorted keys, no whitespace, UTF-8."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_data(data: dict, key: bytes, *, exclude_fields: tuple[str, ...] = ("signature",)) -> str:
    """Create HMAC-SHA256 signature of canonical JSON, excluding specified fields."""
    payload = {k: v for k, v in data.items() if k not in exclude_fields}
    return hmac.new(key, canonical_json(payload), hashlib.sha256).hexdigest()


def verify_data(data: dict, signature: str, key: bytes, *, exclude_fields: tuple[str, ...] = ("signature",)) -> bool:
    """Verify HMAC-SHA256 signature. Constant-time comparison."""
    expected = sign_data(data, key, exclude_fields=exclude_fields)
    return hmac.compare_digest(expected, signature)


def generate_key() -> str:
    """Generate a random 32-byte hex-encoded key for HMAC signing."""
    import os
    return os.urandom(32).hex()


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _export_meta() -> dict:
    """Common metadata fields for all exports."""
    try:
        from app import __version__
        version = __version__
    except Exception:
        version = "unknown"
    return {
        "version": "1.0",
        "author": "nova",
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "nova_version": version,
    }


def _check_signature(data: dict, data_type: str, verify_key_path: str | None, require_signed: bool) -> None:
    """Common signature verification logic. Raises SignatureError on failure."""
    signature = data.get("signature")

    if signature and verify_key_path:
        key = load_key(verify_key_path)
        if not verify_data(data, signature, key):
            raise SignatureError(f"Invalid signature for {data_type}")
        logger.info("Signature verified for %s", data_type)
    elif signature and not verify_key_path and require_signed:
        raise SignatureError(f"{data_type} has signature but no verification key provided")
    elif not signature and require_signed:
        raise SignatureError(f"{data_type} is unsigned but signed import required")


# ---------------------------------------------------------------------------
# Lesson export/import
# ---------------------------------------------------------------------------

def export_lesson(lesson, key_path: str | None = None) -> dict:
    """Export a Lesson to a signed dict.

    Args:
        lesson: Lesson dataclass or dict with topic/correct_answer/etc.
        key_path: Optional path to hex-encoded HMAC key file.
    """
    if hasattr(lesson, "topic"):
        data = {
            "topic": lesson.topic,
            "wrong_answer": lesson.wrong_answer or "",
            "correct_answer": lesson.correct_answer or "",
            "lesson_text": lesson.lesson_text or "",
            "confidence": lesson.confidence if hasattr(lesson, "confidence") else 0.8,
            "context": lesson.context or "" if hasattr(lesson, "context") else "",
        }
    else:
        data = {
            "topic": lesson.get("topic", ""),
            "wrong_answer": lesson.get("wrong_answer", ""),
            "correct_answer": lesson.get("correct_answer", ""),
            "lesson_text": lesson.get("lesson_text", ""),
            "confidence": lesson.get("confidence", 0.8),
            "context": lesson.get("context", ""),
        }
    data.update(_export_meta())

    if key_path:
        key = load_key(key_path)
        data["signature"] = sign_data(data, key)

    return data


def export_all_lessons(db, key_path: str | None = None) -> list[dict]:
    """Export all lessons from the database."""
    rows = db.fetchall("SELECT * FROM lessons ORDER BY id")
    return [export_lesson(dict(row), key_path) for row in (rows or [])]


def import_lesson(data: dict, db, verify_key_path: str | None = None) -> int:
    """Import a lesson dict. Returns lesson ID or -1 if skipped (duplicate).

    Raises SignatureError / ValueError on failure.
    """
    from app.config import config

    # Validate required fields
    required = ("topic", "correct_answer")
    missing = [f for f in required if not data.get(f)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    require_signed = getattr(config, "REQUIRE_SIGNED_LESSONS", False)
    _check_signature(data, f"lesson '{data['topic']}'", verify_key_path, require_signed)

    # Dedup
    existing = db.fetchone(
        "SELECT id FROM lessons WHERE LOWER(topic) = LOWER(?) AND LOWER(correct_answer) = LOWER(?)",
        (data["topic"], data["correct_answer"]),
    )
    if existing:
        logger.info("Skipping duplicate lesson '%s' (id=%d)", data["topic"], existing["id"])
        return -1

    cursor = db.execute(
        """INSERT INTO lessons (topic, wrong_answer, correct_answer, lesson_text, context, confidence)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            data["topic"],
            data.get("wrong_answer", ""),
            data["correct_answer"],
            data.get("lesson_text", ""),
            data.get("context", ""),
            data.get("confidence", 0.8),
        ),
    )
    lesson_id = cursor.lastrowid
    logger.info("Imported lesson '%s' as #%d", data["topic"], lesson_id)
    return lesson_id


# ---------------------------------------------------------------------------
# KG fact export/import
# ---------------------------------------------------------------------------

def export_kg_fact(fact, key_path: str | None = None) -> dict:
    """Export a KG Fact to a signed dict."""
    if hasattr(fact, "subject"):
        data = {
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "confidence": fact.confidence if hasattr(fact, "confidence") else 0.8,
            "source": fact.source if hasattr(fact, "source") else "extracted",
            "valid_from": fact.valid_from or "" if hasattr(fact, "valid_from") else "",
            "valid_to": fact.valid_to or "" if hasattr(fact, "valid_to") else "",
            "provenance": fact.provenance or "" if hasattr(fact, "provenance") else "",
        }
    else:
        data = {
            "subject": fact.get("subject", ""),
            "predicate": fact.get("predicate", ""),
            "object": fact.get("object", ""),
            "confidence": fact.get("confidence", 0.8),
            "source": fact.get("source", "extracted"),
            "valid_from": fact.get("valid_from", ""),
            "valid_to": fact.get("valid_to", ""),
            "provenance": fact.get("provenance", ""),
        }
    data.update(_export_meta())

    if key_path:
        key = load_key(key_path)
        data["signature"] = sign_data(data, key)

    return data


def export_all_kg_facts(db, key_path: str | None = None) -> list[dict]:
    """Export all current KG facts from the database."""
    rows = db.fetchall(
        "SELECT * FROM kg_facts WHERE valid_to IS NULL AND superseded_by IS NULL ORDER BY id"
    )
    return [export_kg_fact(dict(row), key_path) for row in (rows or [])]


def import_kg_fact(data: dict, db, verify_key_path: str | None = None) -> int:
    """Import a KG fact dict. Returns fact ID or -1 if skipped.

    Raises SignatureError / ValueError on failure.
    """
    from app.config import config

    required = ("subject", "predicate", "object")
    missing = [f for f in required if not data.get(f)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    require_signed = getattr(config, "REQUIRE_SIGNED_KG_FACTS", False)
    _check_signature(
        data,
        f"KG fact '{data['subject']} {data['predicate']} {data['object']}'",
        verify_key_path,
        require_signed,
    )

    # Dedup
    existing = db.fetchone(
        "SELECT id FROM kg_facts WHERE LOWER(subject) = LOWER(?) AND predicate = ? "
        "AND LOWER(object) = LOWER(?) AND valid_to IS NULL",
        (data["subject"], data["predicate"], data["object"]),
    )
    if existing:
        logger.info("Skipping duplicate KG fact (id=%d)", existing["id"])
        return -1

    cursor = db.execute(
        """INSERT INTO kg_facts (subject, predicate, object, confidence, source, valid_from, provenance)
           VALUES (?, ?, ?, ?, ?, datetime('now'), ?)""",
        (
            data["subject"],
            data["predicate"],
            data["object"],
            data.get("confidence", 0.8),
            data.get("source", "imported"),
            data.get("provenance", "import"),
        ),
    )
    fact_id = cursor.lastrowid
    logger.info("Imported KG fact #%d: %s %s %s", fact_id, data["subject"], data["predicate"], data["object"])
    return fact_id


# ---------------------------------------------------------------------------
# Bundle export/import
# ---------------------------------------------------------------------------

def export_bundle(db, key_path: str | None = None, *, include_skills: bool = True) -> dict:
    """Export all data types in a single signed envelope."""
    from app.core.skill_export import export_all_skills

    bundle = {
        "format": "nova_export_v1",
        **_export_meta(),
        "lessons": export_all_lessons(db),
        "kg_facts": export_all_kg_facts(db),
    }
    if include_skills:
        bundle["skills"] = export_all_skills(db)

    if key_path:
        key = load_key(key_path)
        bundle["signature"] = sign_data(bundle, key)

    return bundle


def import_bundle(data: dict, db, verify_key_path: str | None = None) -> dict:
    """Import a bundle. Returns counts dict.

    Raises SignatureError if bundle signature fails verification.
    """
    if data.get("format") != "nova_export_v1":
        raise ValueError(f"Unknown bundle format: {data.get('format')}")

    # Verify bundle-level signature first
    signature = data.get("signature")
    if signature and verify_key_path:
        key = load_key(verify_key_path)
        if not verify_data(data, signature, key):
            raise SignatureError("Invalid bundle signature")
        logger.info("Bundle signature verified")
    elif signature and not verify_key_path:
        logger.warning("Bundle has signature but no verification key — skipping verification")
    elif not signature:
        logger.info("Bundle is unsigned")

    counts = {"lessons": 0, "kg_facts": 0, "skills": 0, "errors": 0}

    # Import lessons
    for item in data.get("lessons", []):
        try:
            result = import_lesson(item, db)
            if result > 0:
                counts["lessons"] += 1
        except (ValueError, SignatureError) as e:
            logger.warning("Skipping lesson: %s", e)
            counts["errors"] += 1

    # Import KG facts
    for item in data.get("kg_facts", []):
        try:
            result = import_kg_fact(item, db)
            if result > 0:
                counts["kg_facts"] += 1
        except (ValueError, SignatureError) as e:
            logger.warning("Skipping KG fact: %s", e)
            counts["errors"] += 1

    # Import skills
    if "skills" in data:
        from app.core.skill_export import import_skill
        for item in data["skills"]:
            try:
                result = import_skill(item, db, verify_key_path)
                if result > 0:
                    counts["skills"] += 1
            except Exception as e:
                logger.warning("Skipping skill: %s", e)
                counts["errors"] += 1

    logger.info("Bundle import complete: %s", counts)
    return counts
