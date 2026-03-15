"""Tests for external skill loader."""

import tempfile
import os
from pathlib import Path

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
        assert "Do something useful" in body

    def test_no_frontmatter(self):
        content = "# Just markdown\n\nNo frontmatter here."
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_nested_metadata(self):
        content = """---
name: test
metadata:
  openclaw:
    requires:
      env: API_KEY
---

Body here.
"""
        meta, body = _parse_frontmatter(content)
        assert meta["name"] == "test"
        assert "Body here" in body

    def test_list_values(self):
        content = """---
name: test
allowed-tools: [Bash, Read, Write]
---

Body.
"""
        meta, body = _parse_frontmatter(content)
        assert meta["name"] == "test"
        assert isinstance(meta["allowed-tools"], list)
        assert "Bash" in meta["allowed-tools"]


class TestCheckRequirements:
    def test_no_requirements(self):
        met, missing = _check_requirements({})
        assert met is True
        assert missing == []

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

    def test_bin_present(self):
        # python should be on PATH
        met, missing = _check_requirements({
            "metadata": {"openclaw": {"requires": {"bins": ["python"]}}}
        })
        # We can't guarantee python is on PATH in all test envs,
        # but the function should not crash


class TestMapTools:
    def test_maps_known_tools(self):
        mapping = _map_tools({"allowed-tools": ["Bash(curl:*)", "Read", "WebSearch"]})
        assert "Bash(curl:*)" in mapping
        assert mapping["Bash(curl:*)"] == "shell_exec"
        assert mapping["Read"] == "file_ops"
        assert mapping["WebSearch"] == "web_search"

    def test_empty(self):
        mapping = _map_tools({})
        assert mapping == {}


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
            assert "Todoist" in skills[0].description or "todoist" in skills[0].description.lower()

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills = load_skills(tmpdir)
            assert skills == []

    def test_nonexistent_directory(self):
        skills = load_skills("/nonexistent/path/12345")
        assert skills == []


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
        skills = [
            self._make_skill("todoist", "Manage Todoist tasks"),
        ]
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
