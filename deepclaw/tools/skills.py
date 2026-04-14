"""Local skill management for DeepClaw.

Provides:
  - skills_list: browse installed local skills
  - skill_view: read one installed skill
  - skill_create: create a new local skill
  - skill_update: update an existing local skill
  - skill_install: import a skill from a local file or directory
  - skill_delete: remove an installed local skill
"""

import re
import shutil
from pathlib import Path
from typing import Any

from deepclaw.config import CONFIG_DIR

SKILLS_DIR = CONFIG_DIR / "skills"

_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SKILL_FILE_NAME = "SKILL.md"
_FRONTMATTER_DESCRIPTION_RE = re.compile(r"^description:\s*(.+)$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def available() -> bool:
    """This plugin uses only the Python standard library, so it is always available."""
    return True


def _ensure_skills_dir() -> Path:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    return SKILLS_DIR


def _validate_skill_name(name: str) -> str:
    normalized = (name or "").strip()
    if not _SKILL_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "Invalid skill name. Use lowercase letters, numbers, hyphens, or underscores "
            "(max 64 chars, must start with a letter or number)."
        )
    return normalized


def _skill_dir(name: str) -> Path:
    return _ensure_skills_dir() / _validate_skill_name(name)


def _skill_file(name: str) -> Path:
    return _skill_dir(name) / _SKILL_FILE_NAME


def _extract_description(content: str) -> str:
    if content.startswith("---\n"):
        end = content.find("\n---\n", 4)
        if end != -1:
            frontmatter = content[4:end]
            match = _FRONTMATTER_DESCRIPTION_RE.search(frontmatter)
            if match:
                return match.group(1).strip().strip('"').strip("'")
    match = _HEADING_RE.search(content)
    if match:
        return match.group(1).strip()
    first_nonempty = next((line.strip() for line in content.splitlines() if line.strip()), "")
    return first_nonempty[:120]


def _default_skill_content(name: str, description: str) -> str:
    title = name.replace("-", " ").replace("_", " ").title()
    return (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"## When to Use\n"
        f"- Describe when this skill should be used\n\n"
        f"## Workflow\n"
        f"1. Describe the first step\n"
        f"2. Describe the second step\n"
    )


def skills_list() -> dict[str, Any]:
    """List installed local skills under ~/.deepclaw/skills.

    Returns:
        Dictionary with installed skills, including name, description, and path.
    """
    skills_dir = _ensure_skills_dir()
    skills = []
    for path in sorted(skills_dir.iterdir()):
        if not path.is_dir():
            continue
        skill_file = path / _SKILL_FILE_NAME
        if not skill_file.is_file():
            continue
        content = skill_file.read_text(encoding="utf-8")
        skills.append(
            {
                "name": path.name,
                "description": _extract_description(content),
                "path": str(skill_file),
            }
        )
    return {"skills": skills, "count": len(skills), "skills_dir": str(skills_dir)}


def skill_view(name: str) -> dict[str, Any]:
    """Read an installed skill's SKILL.md file.

    Args:
        name: Skill name under ~/.deepclaw/skills.

    Returns:
        Dictionary with the skill content and metadata.
    """
    try:
        skill_file = _skill_file(name)
    except ValueError as e:
        return {"error": str(e), "name": name}
    if not skill_file.is_file():
        return {"error": f"Skill not found: {name}", "name": name}
    content = skill_file.read_text(encoding="utf-8")
    return {
        "name": skill_file.parent.name,
        "path": str(skill_file),
        "description": _extract_description(content),
        "content": content,
    }


def skill_create(
    name: str,
    description: str,
    content: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a new local skill under ~/.deepclaw/skills.

    Args:
        name: Skill directory name.
        description: Short skill description.
        content: Full SKILL.md content. If omitted, a template is created.
        overwrite: Replace an existing skill if True (default: False).

    Returns:
        Dictionary describing the created skill.
    """
    try:
        skill_dir = _skill_dir(name)
    except ValueError as e:
        return {"error": str(e), "name": name}

    skill_file = skill_dir / _SKILL_FILE_NAME
    if skill_file.exists() and not overwrite:
        return {
            "error": f"Skill already exists: {name}. Use overwrite=True or skill_update().",
            "name": name,
            "path": str(skill_file),
        }

    skill_dir.mkdir(parents=True, exist_ok=True)
    final_content = content if content is not None else _default_skill_content(name, description)
    skill_file.write_text(final_content, encoding="utf-8")
    return {
        "success": True,
        "action": "created",
        "name": name,
        "path": str(skill_file),
        "description": _extract_description(final_content),
    }


def skill_update(name: str, content: str) -> dict[str, Any]:
    """Update an existing local skill's SKILL.md content.

    Args:
        name: Existing skill name.
        content: Complete replacement content for SKILL.md.

    Returns:
        Dictionary describing the updated skill.
    """
    try:
        skill_file = _skill_file(name)
    except ValueError as e:
        return {"error": str(e), "name": name}
    if not skill_file.is_file():
        return {"error": f"Skill not found: {name}", "name": name}
    skill_file.write_text(content, encoding="utf-8")
    return {
        "success": True,
        "action": "updated",
        "name": name,
        "path": str(skill_file),
        "description": _extract_description(content),
    }


def skill_install(
    source_path: str, name: str | None = None, overwrite: bool = False
) -> dict[str, Any]:
    """Install/import a skill from a local file or directory.

    Supported sources:
    - a directory containing SKILL.md
    - a standalone SKILL.md file

    Args:
        source_path: Local file or directory path to import from.
        name: Optional destination skill name. Defaults to the source directory/file stem.
        overwrite: Replace an existing installed skill if True.

    Returns:
        Dictionary describing the installed skill.
    """
    src = Path(source_path).expanduser().resolve()
    if not src.exists():
        return {"error": f"Source path not found: {src}", "source_path": source_path}

    if src.is_dir():
        src_skill_file = src / _SKILL_FILE_NAME
        default_name = src.name
    else:
        src_skill_file = src
        default_name = src.parent.name if src.parent.name != src.anchor else src.stem.lower()

    if src_skill_file.name != _SKILL_FILE_NAME or not src_skill_file.is_file():
        return {
            "error": "Source must be a SKILL.md file or a directory containing SKILL.md.",
            "source_path": source_path,
        }

    dest_name = name or default_name
    try:
        dest_dir = _skill_dir(dest_name)
    except ValueError as e:
        return {"error": str(e), "name": dest_name, "source_path": source_path}

    if dest_dir.exists():
        if not overwrite:
            return {
                "error": f"Skill already exists: {dest_name}. Use overwrite=True to replace it.",
                "name": dest_name,
                "path": str(dest_dir / _SKILL_FILE_NAME),
            }
        shutil.rmtree(dest_dir)

    if src.is_dir():
        shutil.copytree(src, dest_dir)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_skill_file, dest_dir / _SKILL_FILE_NAME)

    installed_skill_file = dest_dir / _SKILL_FILE_NAME
    content = installed_skill_file.read_text(encoding="utf-8")
    return {
        "success": True,
        "action": "installed",
        "name": dest_dir.name,
        "path": str(installed_skill_file),
        "description": _extract_description(content),
        "source_path": str(src),
    }


def skill_delete(name: str) -> dict[str, Any]:
    """Delete an installed local skill directory under ~/.deepclaw/skills.

    Args:
        name: Existing skill name.

    Returns:
        Dictionary describing the deleted skill.
    """
    try:
        dest_dir = _skill_dir(name)
    except ValueError as e:
        return {"error": str(e), "name": name}

    skill_file = dest_dir / _SKILL_FILE_NAME
    if not skill_file.is_file():
        return {"error": f"Skill not found: {name}", "name": name}

    shutil.rmtree(dest_dir)
    return {
        "success": True,
        "action": "deleted",
        "name": name,
        "path": str(skill_file),
    }


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [skills_list, skill_view, skill_create, skill_update, skill_install, skill_delete]
