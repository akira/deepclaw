"""Local skill management for DeepClaw.

Provides:
  - skills_list: browse installed local skills
  - skills_search_remote: search or browse skills from skills.sh
  - skill_view: read one installed skill
  - skill_create: create a new local skill
  - skill_update: update an existing local skill
  - skill_install: import a skill from a local file, directory, or skills.sh page
  - skill_delete: remove an installed local skill
"""

import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from deepclaw.config import CONFIG_DIR

SKILLS_DIR = CONFIG_DIR / "skills"

_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SKILL_FILE_NAME = "SKILL.md"
_FRONTMATTER_DESCRIPTION_RE = re.compile(r"^description:\s*(.+)$", re.MULTILINE)
_FRONTMATTER_FIELD_RE = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.+?)\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_SKILLS_SH_PREFIXES = ("https://skills.sh/", "http://skills.sh/", "skills.sh/")
_SKILLS_SH_INSTALL_CMD_RE = re.compile(
    r"npx\s+skills\s+add\s+(?P<repo>https?://github\.com/[^\s<]+|[^\s<]+)"
    r"(?:\s+--skill\s+(?P<skill>[^\s<]+))?",
    re.IGNORECASE,
)
_GITHUB_API_BASE = "https://api.github.com"
_SKILLS_SH_API_SEARCH_URL = "https://skills.sh/api/search"
_HTTP_USER_AGENT = "DeepClaw/0.1 (+https://github.com/akira/deepclaw)"
_REQUIRED_SKILL_SECTIONS = ("When to Use", "Deterministic First", "Verification")


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
    fields = _frontmatter_fields(content)
    description = fields.get("description", "").strip()
    if description:
        return description
    match = _HEADING_RE.search(content)
    if match:
        return match.group(1).strip()
    first_nonempty = next((line.strip() for line in content.splitlines() if line.strip()), "")
    return first_nonempty[:120]


def _split_frontmatter(content: str) -> tuple[str, str]:
    if not content.startswith("---\n"):
        return "", content
    end = content.find("\n---\n", 4)
    if end == -1:
        return "", content
    return content[4:end], content[end + len("\n---\n") :]


def _frontmatter_fields(content: str) -> dict[str, str]:
    frontmatter, _ = _split_frontmatter(content)
    if not frontmatter:
        return {}
    try:
        import yaml

        data = yaml.safe_load(frontmatter)
        if isinstance(data, dict):
            return {
                str(key).strip(): str(value).strip()
                for key, value in data.items()
                if value is not None
            }
    except Exception:
        pass
    fields: dict[str, str] = {}
    for match in _FRONTMATTER_FIELD_RE.finditer(frontmatter):
        fields[match.group("key").strip()] = match.group("value").strip().strip('"').strip("'")
    return fields


def _has_heading(body: str, heading: str) -> bool:
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    return bool(pattern.search(body))


def _iter_skill_dirs() -> list[Path]:
    skills_dir = _ensure_skills_dir()
    return sorted(
        path for path in skills_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
    )


def _skill_inventory_entry(skill_dir: Path) -> dict[str, Any]:
    skill_file = skill_dir / _SKILL_FILE_NAME
    content = skill_file.read_text(encoding="utf-8") if skill_file.is_file() else ""
    fields = _frontmatter_fields(content) if content else {}
    _, body = _split_frontmatter(content)
    missing_sections = [
        section for section in _REQUIRED_SKILL_SECTIONS if not _has_heading(body, section)
    ]
    frontmatter_name = fields.get("name", "")
    frontmatter_description = fields.get("description", "")
    loadable = bool(frontmatter_name and frontmatter_description)
    issues: list[str] = []
    if not skill_file.is_file():
        issues.append("missing SKILL.md")
    elif not loadable:
        issues.append("missing required 'name' or 'description' in YAML frontmatter")
    if frontmatter_name and frontmatter_name != skill_dir.name:
        issues.append(
            f"frontmatter name '{frontmatter_name}' does not match directory '{skill_dir.name}'"
        )
    return {
        "name": skill_dir.name,
        "path": str(skill_file),
        "content": content,
        "description": frontmatter_description or _extract_description(content),
        "frontmatter_name": frontmatter_name,
        "frontmatter_description": frontmatter_description,
        "loadable": loadable,
        "missing_sections": missing_sections,
        "issues": issues,
    }


def _skill_inventory() -> list[dict[str, Any]]:
    return [_skill_inventory_entry(skill_dir) for skill_dir in _iter_skill_dirs()]


def _default_install_name(src: Path) -> str:
    """Return the default installed skill name for a source path."""
    if src.is_dir():
        return src.name
    return src.parent.name if src.parent.name else src.stem.lower()


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
        f"## Deterministic First\n"
        f"- If code, a script, or a direct tool call can answer this reliably, use that before reasoning in-model\n"
        f"- Push repeatable precision work out of latent reasoning and into deterministic execution\n\n"
        f"## Workflow\n"
        f"1. Describe the first step\n"
        f"2. Describe the second step\n\n"
        f"## Verification\n"
        f"- Describe how to verify the skill worked correctly\n"
    )


def _http_get(url: str, *, accept: str | None = None) -> bytes:
    headers = {"User-Agent": _HTTP_USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


def _http_get_json(url: str, *, accept: str = "application/vnd.github+json") -> Any:
    return json.loads(_http_get(url, accept=accept))


def _github_api_url(repo: str, path: str) -> str:
    quoted_path = urllib.parse.quote(path.strip("/"), safe="/")
    return f"{_GITHUB_API_BASE}/repos/{repo}/contents/{quoted_path}"


def _extract_repo_slug(repo_value: str) -> str | None:
    normalized = repo_value.strip()
    if normalized.startswith("https://github.com/"):
        normalized = normalized[len("https://github.com/") :]
    normalized = normalized.strip("/")
    parts = normalized.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return None


def _skills_sh_identifier(source: str) -> str | None:
    normalized = (source or "").strip()
    for prefix in _SKILLS_SH_PREFIXES:
        if normalized.startswith(prefix):
            identifier = normalized[len(prefix) :].strip("/")
            return identifier or None
    return None


def _resolve_skills_sh_source(source: str) -> dict[str, str]:
    identifier = _skills_sh_identifier(source)
    if not identifier or identifier.count("/") < 2:
        raise ValueError(
            "skills.sh sources must look like https://skills.sh/<owner>/<repo>/<skill>."
        )

    html = _http_get(f"https://skills.sh/{identifier}").decode("utf-8", errors="replace")
    parts = identifier.split("/", 2)
    repo = f"{parts[0]}/{parts[1]}"
    skill_path = parts[2]

    match = _SKILLS_SH_INSTALL_CMD_RE.search(html)
    if match:
        repo = _extract_repo_slug(match.group("repo") or "") or repo
        skill_path = (match.group("skill") or skill_path).strip()

    return {
        "identifier": identifier,
        "repo": repo,
        "skill_path": skill_path.strip("/"),
        "detail_url": f"https://skills.sh/{identifier}",
    }


def _download_github_directory(repo: str, path: str) -> dict[str, bytes]:
    try:
        payload = _http_get_json(_github_api_url(repo, path))
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, list):
        return {}

    files: dict[str, bytes] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("type")
        entry_name = entry.get("name")
        entry_path = entry.get("path")
        if not isinstance(entry_name, str) or not isinstance(entry_path, str):
            continue
        if entry_type == "file":
            download_url = entry.get("download_url")
            if not isinstance(download_url, str) or not download_url:
                continue
            files[entry_name] = _http_get(download_url)
        elif entry_type == "dir":
            nested = _download_github_directory(repo, entry_path)
            for nested_name, nested_content in nested.items():
                files[f"{entry_name}/{nested_name}"] = nested_content
    return files


def _find_github_skill_path(repo: str, skill_name: str) -> str | None:
    repo_meta = _http_get_json(f"{_GITHUB_API_BASE}/repos/{repo}")
    default_branch = repo_meta.get("default_branch") if isinstance(repo_meta, dict) else None
    if not isinstance(default_branch, str) or not default_branch:
        default_branch = "main"

    tree_url = (
        f"{_GITHUB_API_BASE}/repos/{repo}/git/trees/"
        f"{urllib.parse.quote(default_branch, safe='')}?recursive=1"
    )
    tree_payload = _http_get_json(tree_url)
    entries = tree_payload.get("tree") if isinstance(tree_payload, dict) else None
    if not isinstance(entries, list):
        return None

    suffix = f"/{skill_name}/SKILL.md"
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        if path.endswith(suffix) or path == f"{skill_name}/SKILL.md":
            return path[: -len("/SKILL.md")]
    return None


def _install_remote_bundle(dest_dir: Path, files: dict[str, bytes]) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        target = dest_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return dest_dir / _SKILL_FILE_NAME


def skills_search_remote(query: str = "", limit: int = 10) -> dict[str, Any]:
    """Search or browse remotely hosted skills from skills.sh.

    Args:
        query: Search string. Leave blank to browse popular skills.
        limit: Max results to return (1-20, default 10).

    Returns:
        Dictionary with matching remote skills and install URLs.
    """
    try:
        safe_limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        safe_limit = 10

    params = {"limit": safe_limit}
    normalized_query = (query or "").strip()
    if normalized_query:
        params["q"] = normalized_query
    url = f"{_SKILLS_SH_API_SEARCH_URL}?{urllib.parse.urlencode(params)}"

    try:
        payload = _http_get_json(url, accept="application/json")
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        return {
            "error": f"Could not query skills.sh: {e}",
            "query": normalized_query,
            "limit": safe_limit,
        }

    items = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        items = []

    results = []
    for item in items[:safe_limit]:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        name = item.get("name")
        source = item.get("source")
        skill_id = item.get("skillId")
        installs = item.get("installs")
        if not all(isinstance(v, str) and v for v in (identifier, name, source, skill_id)):
            continue
        results.append(
            {
                "name": name,
                "identifier": identifier,
                "repo": source,
                "skill_path": skill_id,
                "installs": installs if isinstance(installs, int) else None,
                "url": f"https://skills.sh/{identifier}",
            }
        )

    return {
        "query": normalized_query,
        "count": len(results),
        "limit": safe_limit,
        "source": "skills.sh",
        "skills": results,
    }


def skills_list() -> dict[str, Any]:
    """List installed local skills under ~/.deepclaw/skills.

    Returns:
        Dictionary with skill metadata.
    """
    skills = [
        {
            "name": entry["name"],
            "path": entry["path"],
            "description": entry["description"],
        }
        for entry in _skill_inventory()
        if Path(entry["path"]).is_file()
    ]

    return {
        "count": len(skills),
        "skills": skills,
        "root": str(_ensure_skills_dir()),
        "skills_dir": str(_ensure_skills_dir()),
    }


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
    """Install/import a skill from a local path or a skills.sh page.

    Supported sources:
    - a directory containing SKILL.md
    - a standalone SKILL.md file
    - a skills.sh URL like https://skills.sh/<owner>/<repo>/<skill>

    Args:
        source_path: Local file/directory path or skills.sh URL to import from.
        name: Optional destination skill name. Defaults to the source directory/file stem.
        overwrite: Replace an existing installed skill if True.

    Returns:
        Dictionary describing the installed skill.
    """
    is_skills_sh = _skills_sh_identifier(source_path) is not None
    src: Path | None = None
    remote_source: dict[str, str] | None = None

    if is_skills_sh:
        try:
            remote_source = _resolve_skills_sh_source(source_path)
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as e:
            return {"error": f"Could not fetch skills.sh source: {e}", "source_path": source_path}
        default_name = remote_source["skill_path"].split("/")[-1]
    else:
        src = Path(source_path).expanduser().resolve()
        if not src.exists():
            return {"error": f"Source path not found: {src}", "source_path": source_path}

        src_skill_file = src / _SKILL_FILE_NAME if src.is_dir() else src
        default_name = _default_install_name(src)

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

    try:
        if remote_source is not None:
            resolved_skill_path = remote_source["skill_path"]
            files = _download_github_directory(remote_source["repo"], resolved_skill_path)
            if _SKILL_FILE_NAME not in files:
                fallback_path = _find_github_skill_path(
                    remote_source["repo"], resolved_skill_path.split("/")[-1]
                )
                if fallback_path and fallback_path != resolved_skill_path:
                    resolved_skill_path = fallback_path
                    files = _download_github_directory(remote_source["repo"], resolved_skill_path)
            if _SKILL_FILE_NAME not in files:
                return {
                    "error": (
                        "Downloaded skill bundle does not contain SKILL.md: "
                        f"{remote_source['repo']} / {remote_source['skill_path']}"
                    ),
                    "source_path": source_path,
                }
            installed_skill_file = _install_remote_bundle(dest_dir, files)
            resolved_source_path = remote_source["detail_url"]
            remote_source = dict(remote_source)
            remote_source["resolved_skill_path"] = resolved_skill_path
        else:
            assert src is not None
            if src.is_dir():
                shutil.copytree(src, dest_dir)
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_skill_file, dest_dir / _SKILL_FILE_NAME)
            installed_skill_file = dest_dir / _SKILL_FILE_NAME
            resolved_source_path = str(src)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as e:
        shutil.rmtree(dest_dir, ignore_errors=True)
        return {"error": f"Failed to install skill: {e}", "source_path": source_path}

    content = installed_skill_file.read_text(encoding="utf-8")
    result = {
        "success": True,
        "action": "installed",
        "name": dest_dir.name,
        "path": str(installed_skill_file),
        "description": _extract_description(content),
        "source_path": resolved_source_path,
    }
    if remote_source is not None:
        result.update(
            {
                "source_repo": remote_source["repo"],
                "source_skill_path": remote_source["skill_path"],
                "resolved_skill_path": remote_source.get(
                    "resolved_skill_path", remote_source["skill_path"]
                ),
                "source_identifier": remote_source["identifier"],
            }
        )
    return result


def skill_delete(name: str) -> dict[str, Any]:
    """Delete an installed local skill by name.

    Args:
        name: Skill directory name.

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


def skills_audit() -> dict[str, Any]:
    """Audit local skills for duplicate descriptions and missing required sections."""
    inventory = _skill_inventory()
    duplicates: dict[str, list[str]] = {}
    by_description: dict[str, list[str]] = {}
    for entry in inventory:
        description = (
            entry.get("frontmatter_description") or entry.get("description") or ""
        ).strip()
        if not description:
            continue
        by_description.setdefault(description.casefold(), []).append(entry["name"])
    duplicate_descriptions = []
    for normalized, names in sorted(by_description.items()):
        if len(names) < 2:
            continue
        canonical = next(
            (
                (entry.get("frontmatter_description") or entry.get("description") or "").strip()
                for entry in inventory
                if entry["name"] == names[0]
            ),
            normalized,
        )
        duplicate_descriptions.append({"description": canonical, "skills": sorted(names)})
        duplicates[normalized] = names

    skills = [
        {
            "name": entry["name"],
            "path": entry["path"],
            "missing_sections": entry["missing_sections"],
            "issues": entry["issues"],
        }
        for entry in inventory
    ]
    missing_required_sections_count = sum(1 for entry in inventory if entry["missing_sections"])

    return {
        "count": len(inventory),
        "skills": skills,
        "duplicate_descriptions": duplicate_descriptions,
        "duplicate_descriptions_count": len(duplicate_descriptions),
        "skills_missing_required_sections_count": missing_required_sections_count,
        "required_sections": list(_REQUIRED_SKILL_SECTIONS),
        "root": str(_ensure_skills_dir()),
    }


def skills_check_resolvable() -> dict[str, Any]:
    """Check whether installed skills are loadable by SkillsMiddleware-style rules."""
    inventory = _skill_inventory()
    unresolvable = []
    for entry in inventory:
        if entry["loadable"]:
            continue
        reason = next(
            (
                issue
                for issue in entry["issues"]
                if "missing required 'name' or 'description'" in issue
            ),
            "; ".join(entry["issues"]) or "skill is not loadable",
        )
        unresolvable.append(
            {
                "name": entry["name"],
                "path": entry["path"],
                "reason": reason,
            }
        )

    loadable_count = sum(1 for entry in inventory if entry["loadable"])
    return {
        "count": len(inventory),
        "loadable_count": loadable_count,
        "unresolvable_count": len(unresolvable),
        "unresolvable": unresolvable,
        "root": str(_ensure_skills_dir()),
    }


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [
        skills_list,
        skills_search_remote,
        skills_audit,
        skills_check_resolvable,
        skill_view,
        skill_create,
        skill_update,
        skill_install,
        skill_delete,
    ]
