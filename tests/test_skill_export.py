"""Tests for skill import/export with signature verification, and external skill loader."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from unittest.mock import patch

from app.core.skill_export import (
    SkillSignatureError,
    export_all_skills,
    export_skill,
    generate_key,
    import_skill,
    import_skills_from_file,
    sign_skill,
    verify_skill,
)
from app.core.skills import SkillStore


@pytest.fixture
def allow_unsigned():
    """Temporarily set REQUIRE_SIGNED_SKILLS=False for unsigned import tests."""
    with patch("app.core.skill_export.config") as mock_config:
        mock_config.REQUIRE_SIGNED_SKILLS = False
        yield mock_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_key_file(tmp_path: Path) -> tuple[Path, bytes]:
    """Create a temp key file and return (path, raw_key_bytes)."""
    key_hex = generate_key()
    key_path = tmp_path / "test.key"
    key_path.write_text(key_hex + "\n", encoding="utf-8")
    return key_path, bytes.fromhex(key_hex)


def _insert_skill(db, name="unit_convert", trigger=r"\bconvert\b.*\bto\b", steps=None):
    """Insert a skill directly and return its ID."""
    store = SkillStore(db)
    return store.create_skill(
        name=name,
        trigger_pattern=trigger,
        steps=steps or [{"tool": "calculator", "args_template": {"expr": "{query}"}}],
        answer_template="The result is {result}",
    )


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

class TestExportSkill:
    def test_export_single(self, db):
        skill_id = _insert_skill(db)
        store = SkillStore(db)
        skill = store.get_skill(skill_id)

        data = export_skill(skill)
        assert data["name"] == "unit_convert"
        assert data["trigger"] == skill.trigger_pattern
        assert data["steps"] == skill.steps
        assert data["template"] == skill.answer_template
        assert data["version"] == "1.0"
        assert data["author"] == "nova"
        assert "created_at" in data
        assert "signature" not in data

    def test_export_with_signature(self, db, tmp_path):
        key_path, _ = _make_key_file(tmp_path)
        skill_id = _insert_skill(db)
        store = SkillStore(db)
        skill = store.get_skill(skill_id)

        data = export_skill(skill, private_key_path=str(key_path))
        assert "signature" in data
        assert len(data["signature"]) == 64  # SHA-256 hex digest

    def test_export_all(self, db):
        _insert_skill(db, name="skill_a", trigger=r"\bconvert\b.*\bto\b")
        _insert_skill(db, name="skill_b", trigger=r"\bcalculate\b.*\bsum\b")
        result = export_all_skills(db)
        assert len(result) == 2
        names = {d["name"] for d in result}
        assert names == {"skill_a", "skill_b"}


# ---------------------------------------------------------------------------
# Signing tests
# ---------------------------------------------------------------------------

class TestSigning:
    def test_sign_and_verify(self):
        key = os.urandom(32)
        data = {"name": "test", "trigger": ".*", "steps": []}
        sig = sign_skill(data, key)
        assert verify_skill(data, sig, key)

    def test_verify_wrong_key(self):
        key1 = os.urandom(32)
        key2 = os.urandom(32)
        data = {"name": "test", "trigger": ".*", "steps": []}
        sig = sign_skill(data, key1)
        assert not verify_skill(data, sig, key2)

    def test_verify_tampered_data(self):
        key = os.urandom(32)
        data = {"name": "test", "trigger": ".*", "steps": []}
        sig = sign_skill(data, key)
        data["name"] = "tampered"
        assert not verify_skill(data, sig, key)

    def test_signature_field_excluded(self):
        """The 'signature' field itself should not be part of the signed payload."""
        key = os.urandom(32)
        data = {"name": "test", "trigger": ".*", "steps": []}
        sig = sign_skill(data, key)
        data["signature"] = sig
        # sign_skill should exclude "signature" and produce the same result
        assert sign_skill(data, key) == sig


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImportSkill:
    def test_import_unsigned(self, db, allow_unsigned):
        data = {
            "name": "web_lookup",
            "trigger": r"\blookup\b",
            "steps": [{"tool": "web_search", "args_template": {"q": "{query}"}}],
            "template": "Found: {result}",
        }
        skill_id = import_skill(data, db)
        assert skill_id > 0
        store = SkillStore(db)
        skill = store.get_skill(skill_id)
        assert skill.name == "web_lookup"

    def test_import_with_valid_signature(self, db, tmp_path):
        key_path, key = _make_key_file(tmp_path)
        data = {
            "name": "signed_skill",
            "trigger": r"\bsigned\b",
            "steps": [{"tool": "calculator", "args_template": {"expr": "1+1"}}],
        }
        data["signature"] = sign_skill(data, key)
        skill_id = import_skill(data, db, verify_key_path=str(key_path))
        assert skill_id > 0

    def test_import_with_invalid_signature(self, db, tmp_path):
        key_path, _ = _make_key_file(tmp_path)
        data = {
            "name": "bad_sig",
            "trigger": r"\bbad\b",
            "steps": [{"tool": "calculator", "args_template": {}}],
            "signature": "0" * 64,
        }
        with pytest.raises(SkillSignatureError, match="Invalid signature"):
            import_skill(data, db, verify_key_path=str(key_path))

    def test_import_unsigned_when_required(self, db, monkeypatch):
        # Config is frozen, so set env var and reload
        monkeypatch.setenv("REQUIRE_SIGNED_SKILLS", "true")
        import importlib
        import app.config
        importlib.reload(app.config)
        import app.core.skill_export as skill_export_mod
        monkeypatch.setattr(skill_export_mod, "config", app.config.config)

        data = {
            "name": "unsigned",
            "trigger": r"\bunsigned\b",
            "steps": [{"tool": "web_search", "args_template": {}}],
        }
        with pytest.raises(SkillSignatureError, match="unsigned"):
            import_skill(data, db)

    def test_import_dedup(self, db, allow_unsigned):
        data = {
            "name": "dup_skill",
            "trigger": r"\bduplicate\b",
            "steps": [{"tool": "calculator", "args_template": {}}],
        }
        first = import_skill(data, db)
        assert first > 0
        second = import_skill(data, db)
        assert second == -1

    def test_import_missing_fields(self, db):
        with pytest.raises(ValueError, match="Missing required fields"):
            import_skill({"name": "incomplete"}, db)

    def test_import_from_file_single(self, db, tmp_path, allow_unsigned):
        data = {
            "name": "from_file",
            "trigger": r"\bfile\b",
            "steps": [{"tool": "web_search", "args_template": {"q": "{query}"}}],
        }
        path = tmp_path / "skills.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        count = import_skills_from_file(str(path), db)
        assert count == 1

    def test_import_from_file_list(self, db, tmp_path, allow_unsigned):
        data = [
            {
                "name": "list_a",
                "trigger": r"\blist_a\b",
                "steps": [{"tool": "web_search", "args_template": {}}],
            },
            {
                "name": "list_b",
                "trigger": r"\blist_b\b",
                "steps": [{"tool": "calculator", "args_template": {}}],
            },
        ]
        path = tmp_path / "skills.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        count = import_skills_from_file(str(path), db)
        assert count == 2


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_export_import_roundtrip(self, db, tmp_path):
        """Export skills, import into a fresh DB, verify data matches."""
        _insert_skill(db, name="roundtrip_skill", trigger=r"\broundtrip\b")
        key_path, _ = _make_key_file(tmp_path)

        exported = export_all_skills(db, private_key_path=str(key_path))
        assert len(exported) == 1

        # Write to file
        path = tmp_path / "export.json"
        path.write_text(json.dumps(exported), encoding="utf-8")

        # Import into fresh DB
        from app.database import SafeDB
        db2 = SafeDB(str(tmp_path / "fresh.db"))
        db2.init_schema()

        count = import_skills_from_file(str(path), db2, verify_key_path=str(key_path))
        assert count == 1

        store2 = SkillStore(db2)
        skills = store2.get_all_skills()
        assert len(skills) == 1
        assert skills[0].name == "roundtrip_skill"
        assert skills[0].trigger_pattern == r"\broundtrip\b"
        db2.close()

    def test_roundtrip_unsigned(self, db, tmp_path, allow_unsigned):
        """Round-trip without signing."""
        _insert_skill(db, name="plain_skill", trigger=r"\bplain\b")
        exported = export_all_skills(db)

        path = tmp_path / "plain.json"
        path.write_text(json.dumps(exported), encoding="utf-8")

        from app.database import SafeDB
        db2 = SafeDB(str(tmp_path / "fresh2.db"))
        db2.init_schema()

        count = import_skills_from_file(str(path), db2)
        assert count == 1
        db2.close()


# ===========================================================================
# External Skill Loader (from test_skill_loader)
# ===========================================================================

import tempfile

from app.core.skill_loader import (
    _parse_frontmatter,
    _check_requirements,
    _map_tools,
    load_skills,
    match_skill,
    format_skill_summaries,
    format_skill_body,
    ExternalSkill,
)


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        content = """---
name: my-skill
description: A test skill
---

# Instructions

Do something useful.
"""
        meta, body = _parse_frontmatter(content)
        assert meta["name"] == "my-skill"
        assert meta["description"] == "A test skill"
        assert "Instructions" in body

    def test_no_frontmatter(self):
        content = "# Just markdown\n\nNo frontmatter here."
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_list_values(self):
        content = """---
name: test
allowed-tools: [Bash, Read, Write]
---

Body.
"""
        meta, body = _parse_frontmatter(content)
        assert isinstance(meta["allowed-tools"], list)
        assert "Bash" in meta["allowed-tools"]


class TestCheckRequirements:
    def test_no_requirements(self):
        met, missing = _check_requirements({})
        assert met is True

    def test_env_var_present(self):
        os.environ["TEST_SKILL_VAR"] = "yes"
        try:
            met, missing = _check_requirements({
                "metadata": {"openclaw": {"requires": {"env": ["TEST_SKILL_VAR"]}}}
            })
            assert met is True
        finally:
            del os.environ["TEST_SKILL_VAR"]

    def test_env_var_missing(self):
        met, missing = _check_requirements({
            "metadata": {"openclaw": {"requires": {"env": ["NONEXISTENT_VAR_12345"]}}}
        })
        assert met is False
        assert "env:NONEXISTENT_VAR_12345" in missing


class TestMapTools:
    def test_maps_known_tools(self):
        mapping = _map_tools({"allowed-tools": ["Bash(curl:*)", "Read", "WebSearch"]})
        assert mapping["Bash(curl:*)"] == "shell_exec"
        assert mapping["Read"] == "file_ops"
        assert mapping["WebSearch"] == "web_search"

    def test_empty(self):
        assert _map_tools({}) == {}


class TestLoadSkills:
    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("""---
name: todoist
description: Manage your Todoist tasks
---

# Todoist Skill

Use the Todoist API to manage tasks.
""")
            skills = load_skills(tmpdir)
            assert len(skills) == 1
            assert skills[0].name == "todoist"

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_skills(tmpdir) == []

    def test_nonexistent_directory(self):
        assert load_skills("/nonexistent/path/12345") == []


class TestMatchSkill:
    def _make_skill(self, name, description):
        from app.core.text_utils import normalize_words
        skill = ExternalSkill(
            name=name,
            description=description,
            directory="/tmp",
            body="# Instructions\n\nDo something.",
        )
        skill._description_words = normalize_words(f"{name} {description}")
        return skill

    def test_matches_relevant_query(self):
        skills = [
            self._make_skill("todoist", "Manage Todoist tasks and to-do lists"),
            self._make_skill("github", "Create and manage GitHub issues and PRs"),
        ]
        result = match_skill("create a new task in todoist", skills)
        assert result is not None
        assert result.name == "todoist"

    def test_no_match_below_threshold(self):
        skills = [self._make_skill("todoist", "Manage Todoist tasks")]
        result = match_skill("what is quantum computing", skills)
        assert result is None

    def test_empty_skills(self):
        assert match_skill("hello", []) is None


class TestFormatFunctions:
    def test_format_summaries(self):
        from app.core.text_utils import normalize_words
        skill = ExternalSkill(
            name="test-skill",
            description="A test skill for testing",
            directory="/tmp",
            body="body",
            requirements_met=True,
        )
        skill._description_words = normalize_words("test-skill A test skill for testing")
        text = format_skill_summaries([skill])
        assert "test-skill" in text
        assert "External Skills" in text

    def test_format_body(self):
        skill = ExternalSkill(
            name="my-skill",
            description="desc",
            directory="/tmp",
            body="# Do this\n\nFollow these steps.",
        )
        text = format_skill_body(skill)
        assert "Active Skill: my-skill" in text
        assert "Follow these steps" in text
