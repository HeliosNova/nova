"""Skill import/export with HMAC-SHA256 signature verification.

Exports skills to portable JSON dicts, optionally signed.
Imports skills with signature verification and deduplication.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import config
from app.core.skills import SkillStore

logger = logging.getLogger(__name__)


class SkillSignatureError(Exception):
    """Raised when skill signature verification fails."""


# ---------------------------------------------------------------------------
# Signing utilities
# ---------------------------------------------------------------------------

def _load_key(path: str | Path) -> bytes:
    """Load a hex-encoded signing key from a file."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    return bytes.fromhex(raw)


def _canonical_json(data: dict) -> bytes:
    """Canonical JSON: sorted keys, no whitespace, UTF-8."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_skill(skill_data: dict, key: bytes) -> str:
    """Create HMAC-SHA256 signature of canonical skill JSON.

    The 'signature' field is excluded from the signed payload.
    """
    payload = {k: v for k, v in skill_data.items() if k != "signature"}
    return hmac.new(key, _canonical_json(payload), hashlib.sha256).hexdigest()


def verify_skill(skill_data: dict, signature: str, key: bytes) -> bool:
    """Verify HMAC-SHA256 signature of a skill dict."""
    expected = sign_skill(skill_data, key)
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_skill(skill, private_key_path: str | None = None) -> dict:
    """Export a single Skill to a JSON-serializable dict.

    Args:
        skill: A Skill dataclass instance (from SkillStore).
        private_key_path: Optional path to hex-encoded HMAC key file.

    Returns:
        Dict with name, description (trigger), trigger, steps, template,
        version, author, created_at. Optionally includes signature.
    """
    # Derive description from trigger + template rather than repeating the name
    desc_parts = []
    if skill.trigger_pattern:
        desc_parts.append(f"Triggered by: {skill.trigger_pattern}")
    if skill.answer_template:
        desc_parts.append(f"Template: {skill.answer_template[:100]}")
    description = "; ".join(desc_parts) if desc_parts else skill.name

    data = {
        "name": skill.name,
        "description": description,
        "trigger": skill.trigger_pattern,
        "steps": skill.steps,
        "template": skill.answer_template,
        "version": "1.0",
        "author": "nova",
        "created_at": skill.created_at or datetime.now(timezone.utc).isoformat(),
    }

    if private_key_path:
        key = _load_key(private_key_path)
        data["signature"] = sign_skill(data, key)

    return data


def export_all_skills(db, private_key_path: str | None = None) -> list[dict]:
    """Export all skills from the database.

    Args:
        db: SafeDB instance.
        private_key_path: Optional path to hex-encoded HMAC key file.

    Returns:
        List of exported skill dicts.
    """
    store = SkillStore(db)
    skills = store.get_all_skills(limit=9999)
    return [export_skill(s, private_key_path) for s in skills]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("name", "trigger", "steps")


def import_skill(data: dict, db, verify_key_path: str | None = None) -> int:
    """Import a single skill dict into the database.

    Args:
        data: Skill dict with at least name, trigger, steps.
        db: SafeDB instance.
        verify_key_path: Optional path to hex-encoded HMAC key for verification.

    Returns:
        New skill ID on success, -1 if skipped (duplicate).

    Raises:
        ValueError: Missing required fields.
        SkillSignatureError: Signature verification failed or unsigned
            when REQUIRE_SIGNED_SKILLS is True.
    """
    # Validate required fields
    missing = [f for f in _REQUIRED_FIELDS if f not in data or not data[f]]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    # Signature verification
    signature = data.get("signature")
    require_signed = getattr(config, "REQUIRE_SIGNED_SKILLS", False)

    if signature and verify_key_path:
        key = _load_key(verify_key_path)
        if not verify_skill(data, signature, key):
            raise SkillSignatureError(
                f"Invalid signature for skill '{data['name']}'"
            )
        logger.info("Signature verified for skill '%s'", data["name"])
    elif signature and not verify_key_path and require_signed:
        raise SkillSignatureError(
            f"Skill '{data['name']}' has signature but no verification key path provided"
        )
    elif not signature and require_signed:
        raise SkillSignatureError(
            f"Skill '{data['name']}' is unsigned but REQUIRE_SIGNED_SKILLS is enabled"
        )

    # Dedup: skip if skill with same name already exists
    existing = db.fetchone(
        "SELECT id FROM skills WHERE LOWER(name) = LOWER(?)",
        (data["name"],),
    )
    if existing:
        logger.info("Skipping duplicate skill '%s' (id=%d)", data["name"], existing["id"])
        return -1

    # Parse steps if they're a JSON string
    steps = data["steps"]
    if isinstance(steps, str):
        steps = json.loads(steps)

    # Insert via SkillStore
    store = SkillStore(db)
    skill_id = store.create_skill(
        name=data["name"],
        trigger_pattern=data["trigger"],
        steps=steps,
        answer_template=data.get("template"),
    )

    if skill_id is None:
        logger.warning("Skill '%s' rejected by SkillStore guards", data["name"])
        return -1

    logger.info("Imported skill '%s' as #%d", data["name"], skill_id)
    return skill_id


def import_skills_from_file(
    path: str, db, verify_key_path: str | None = None
) -> int:
    """Import skills from a JSON file.

    The file can contain a single skill dict or a list of skill dicts.

    Args:
        path: Path to the JSON file.
        db: SafeDB instance.
        verify_key_path: Optional path to hex-encoded HMAC key.

    Returns:
        Count of successfully imported skills.
    """
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)

    if isinstance(data, dict):
        data = [data]

    imported = 0
    for item in data:
        try:
            result = import_skill(item, db, verify_key_path)
            if result > 0:
                imported += 1
        except (ValueError, SkillSignatureError) as e:
            logger.warning("Skipping skill during import: %s", e)
        except Exception as e:
            logger.warning("Unexpected error importing skill: %s", e)

    logger.info("Imported %d/%d skill(s) from %s", imported, len(data), path)
    return imported


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def generate_key() -> str:
    """Generate a random 32-byte hex-encoded key for HMAC signing."""
    import os
    return os.urandom(32).hex()
