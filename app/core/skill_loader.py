"""External skill loader — loads AgentSkills / OpenClaw SKILL.md files.

Scans a skills/ directory for */SKILL.md files, parses YAML frontmatter,
checks requirements, and makes skills available for prompt injection.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from app.config import config
from app.core.text_utils import normalize_words

logger = logging.getLogger(__name__)


@dataclass
class ExternalSkill:
    """A loaded external skill from a SKILL.md file."""
    name: str
    description: str
    directory: str
    body: str                          # Full markdown body (instructions)
    metadata: dict = field(default_factory=dict)
    requirements_met: bool = True
    tool_mapping: dict = field(default_factory=dict)  # OpenClaw tool -> Nova tool
    _description_words: set = field(default_factory=set, repr=False)


# Maps OpenClaw/AgentSkills tool names to Nova tool names
_TOOL_MAP = {
    "bash": "shell_exec",
    "read": "file_ops",
    "write": "file_ops",
    "edit": "file_ops",
    "glob": "file_ops",
    "grep": "file_ops",
    "webfetch": "browser",
    "websearch": "web_search",
    "agent": "delegate",
}

# Regex for YAML frontmatter
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Try to use PyYAML if available (handles nested structures correctly)
try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _parse_frontmatter_fallback(yaml_text: str) -> dict:
    """Simple YAML parser — handles flat and 1-level nested key: value.

    Fallback when PyYAML is not installed.
    """
    metadata: dict = {}
    current_key = None

    for line in yaml_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())

        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            if indent == 0:
                current_key = key
                if val:
                    if val.startswith("[") and val.endswith("]"):
                        metadata[key] = [v.strip().strip("'\"") for v in val[1:-1].split(",")]
                    else:
                        metadata[key] = val.strip("'\"")
                else:
                    metadata[key] = {}
            elif current_key:
                if not isinstance(metadata.get(current_key), dict):
                    metadata[current_key] = {}
                if val:
                    metadata[current_key][key] = val.strip("'\"")
                else:
                    metadata[current_key][key] = {}
        elif stripped.startswith("- ") and current_key:
            if not isinstance(metadata.get(current_key), list):
                metadata[current_key] = []
            metadata[current_key].append(stripped[2:].strip().strip("'\""))

    return metadata


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a SKILL.md file.

    Uses PyYAML if available (handles deep nesting), falls back to simple parser.
    Returns (metadata_dict, body_markdown).
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    yaml_text = match.group(1)
    body = content[match.end():]

    if _HAS_YAML:
        try:
            metadata = _yaml.safe_load(yaml_text)
            if not isinstance(metadata, dict):
                metadata = {}
            return metadata, body
        except Exception:
            pass  # Fall through to fallback parser

    return _parse_frontmatter_fallback(yaml_text), body


def _check_requirements(metadata: dict) -> tuple[bool, list[str]]:
    """Check if skill requirements are met.

    Returns (all_met, list_of_missing).
    """
    missing = []
    requires = metadata.get("metadata", {})
    if isinstance(requires, dict):
        requires = requires.get("openclaw", requires)
        if isinstance(requires, dict):
            requires = requires.get("requires", {})

    if not isinstance(requires, dict):
        return True, []

    # Check env vars
    env_vars = requires.get("env", [])
    if isinstance(env_vars, str):
        env_vars = [env_vars]
    if isinstance(env_vars, list):
        for var in env_vars:
            if not os.environ.get(str(var)):
                missing.append(f"env:{var}")

    # Check bins on PATH
    bins = requires.get("bins", [])
    if isinstance(bins, str):
        bins = [bins]
    if isinstance(bins, list):
        for binary in bins:
            if not shutil.which(str(binary)):
                missing.append(f"bin:{binary}")

    return len(missing) == 0, missing


def _map_tools(metadata: dict) -> dict[str, str]:
    """Map OpenClaw allowed-tools to Nova tool names."""
    allowed = metadata.get("allowed-tools", [])
    if isinstance(allowed, str):
        allowed = [allowed]
    if not isinstance(allowed, list):
        return {}

    mapping = {}
    for tool_spec in allowed:
        # Parse "Bash(curl:*)" -> "bash"
        tool_name = str(tool_spec).split("(")[0].strip().lower()
        if tool_name in _TOOL_MAP:
            mapping[tool_spec] = _TOOL_MAP[tool_name]

    return mapping


def load_skills(skills_dir: str | None = None) -> list[ExternalSkill]:
    """Scan skills directory for SKILL.md files, parse and validate them.

    Returns list of loaded ExternalSkill objects.
    """
    skills_dir = skills_dir or config.SKILLS_DIR
    skills_path = Path(skills_dir)

    if not skills_path.exists():
        logger.debug("Skills directory does not exist: %s", skills_dir)
        return []

    skills = []
    for skill_md in skills_path.glob("*/SKILL.md"):
        try:
            content = skill_md.read_text(encoding="utf-8", errors="replace")
            metadata, body = _parse_frontmatter(content)

            name = metadata.get("name", skill_md.parent.name)
            description = metadata.get("description", "")

            if not description:
                # Use first non-empty line of body as description
                for line in body.split("\n"):
                    line = line.strip().lstrip("#").strip()
                    if line and not line.startswith("---"):
                        description = line[:200]
                        break

            requirements_met, missing = _check_requirements(metadata)
            if missing:
                logger.info("Skill '%s' missing requirements: %s", name, missing)

            tool_mapping = _map_tools(metadata)

            # Sanitize skill body for prompt injection
            if config.ENABLE_INJECTION_DETECTION:
                from app.core.injection import sanitize_content
                body = sanitize_content(body, context="skill")

            skill = ExternalSkill(
                name=name,
                description=description,
                directory=str(skill_md.parent),
                body=body,
                metadata=metadata,
                requirements_met=requirements_met,
                tool_mapping=tool_mapping,
            )
            skill._description_words = normalize_words(f"{name} {description}")
            skills.append(skill)
            logger.info("Loaded external skill: '%s' (reqs_met=%s)", name, requirements_met)

        except Exception as e:
            logger.warning("Failed to load skill from %s: %s", skill_md, e)

    logger.info("Loaded %d external skill(s) from %s", len(skills), skills_dir)
    return skills


def match_skill(query: str, skills: list[ExternalSkill], threshold: float = 0.3) -> ExternalSkill | None:
    """Find the best matching external skill for a query using Jaccard overlap.

    Returns the top match if similarity >= threshold, else None.
    """
    if not skills:
        return None

    query_words = normalize_words(query)
    if not query_words:
        return None

    best_skill = None
    best_score = 0.0

    for skill in skills:
        if not skill.requirements_met:
            continue
        if not skill._description_words:
            continue

        intersection = len(query_words & skill._description_words)
        union = len(query_words | skill._description_words)
        if union == 0:
            continue

        score = intersection / union
        if score > best_score:
            best_score = score
            best_skill = skill

    if best_score >= threshold and best_skill:
        return best_skill
    return None


def format_skill_summaries(skills: list[ExternalSkill]) -> str:
    """Format skill summaries for inclusion in the system prompt (~100 tokens each)."""
    if not skills:
        return ""

    lines = []
    for skill in skills:
        if not skill.requirements_met:
            continue
        lines.append(f"- **{skill.name}**: {skill.description[:150]}")

    if not lines:
        return ""
    return "## Available External Skills\n\n" + "\n".join(lines)


def format_skill_body(skill: ExternalSkill) -> str:
    """Format a matched skill's full body for context injection."""
    return (
        f"## Active Skill: {skill.name}\n\n"
        f"{skill.body[:3000]}"
    )
