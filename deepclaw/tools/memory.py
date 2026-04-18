"""Explicit memory management tools backed by AGENTS.md.

Always available. Provides:
  - memory_add: append a durable fact to a named section
  - memory_replace: replace existing text in memory
  - memory_remove: remove existing text from memory
  - memory_search: search memory for matching lines
"""

from pathlib import Path
from typing import Any

from deepclaw import agent as agent_module


def available() -> bool:
    """Always available — memory tools only use the local AGENTS.md file."""
    return True


def _memory_file() -> Path:
    path = agent_module.MEMORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(agent_module.DEFAULT_MEMORY_SEED, encoding="utf-8")
    return path


def _read_text() -> str:
    return _memory_file().read_text(encoding="utf-8")


def _write_text(text: str) -> None:
    _memory_file().write_text(text, encoding="utf-8")


def _find_section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    target = f"## {section.strip()}"
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == target:
            start = idx
            break
    if start is None:
        return None

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return start, end


def memory_add(content: str, section: str = "Notes") -> dict[str, Any]:
    """Add a durable memory entry to AGENTS.md.

    Args:
        content: Compact fact to remember for future sessions.
        section: Section heading to append to (default: "Notes").

    Returns:
        Confirmation with the updated section and file path.
    """
    content = content.strip()
    section = section.strip() or "Notes"
    if not content:
        return {"error": "Content must not be empty."}

    path = _memory_file()
    lines = _read_text().splitlines()
    bullet = f"- {content}"
    bounds = _find_section_bounds(lines, section)

    if bounds is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([f"## {section}", bullet])
    else:
        _start, end = bounds
        insert_at = end
        while insert_at > 0 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines.insert(insert_at, bullet)

    _write_text("\n".join(lines).rstrip() + "\n")
    return {"status": "added", "section": section, "path": str(path), "content": content}


def memory_replace(old_text: str, new_text: str) -> dict[str, Any]:
    """Replace a unique snippet in AGENTS.md.

    Args:
        old_text: Exact text to replace. Must match uniquely.
        new_text: Replacement text.

    Returns:
        Confirmation of the replacement or an error if not found / ambiguous.
    """
    old_text = old_text.strip()
    if not old_text:
        return {"error": "old_text must not be empty."}

    text = _read_text()
    matches = text.count(old_text)
    if matches == 0:
        return {"error": f"Text not found: {old_text}"}
    if matches > 1:
        return {"error": f"Text is ambiguous ({matches} matches): {old_text}"}

    _write_text(text.replace(old_text, new_text, 1))
    return {"status": "replaced", "path": str(_memory_file()), "old_text": old_text}


def memory_remove(text: str) -> dict[str, Any]:
    """Remove a unique memory entry or snippet from AGENTS.md.

    Args:
        text: Exact text to remove. Must match uniquely.

    Returns:
        Confirmation of removal or an error if not found / ambiguous.
    """
    text = text.strip()
    if not text:
        return {"error": "text must not be empty."}

    original = _read_text()
    lines = original.splitlines()
    matching_lines = [idx for idx, line in enumerate(lines) if text in line]

    if len(matching_lines) == 1:
        del lines[matching_lines[0]]
        while lines and lines[-1] == "":
            lines.pop()
        _write_text("\n".join(lines) + "\n")
        return {"status": "removed", "path": str(_memory_file()), "text": text}

    matches = original.count(text)
    if matches == 0:
        return {"error": f"Text not found: {text}"}
    if matches > 1:
        return {"error": f"Text is ambiguous ({matches} matches): {text}"}

    _write_text(original.replace(text, "", 1))
    return {"status": "removed", "path": str(_memory_file()), "text": text}


def memory_search(query: str) -> dict[str, Any]:
    """Search AGENTS.md for matching lines.

    Args:
        query: Case-insensitive search string.

    Returns:
        Matching lines with line numbers for quick inspection.
    """
    query = query.strip()
    if not query:
        return {"error": "query must not be empty."}

    matches = []
    needle = query.casefold()
    for line_number, line in enumerate(_read_text().splitlines(), start=1):
        if needle in line.casefold():
            matches.append({"line_number": line_number, "line": line})

    return {"query": query, "count": len(matches), "matches": matches, "path": str(_memory_file())}


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [memory_add, memory_replace, memory_remove, memory_search]
