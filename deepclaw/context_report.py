"""Context breakdown helpers for DeepClaw status reporting."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from deepagents.middleware.memory import MEMORY_SYSTEM_PROMPT
from deepagents.middleware.skills import SKILLS_SYSTEM_PROMPT
from langchain_core.messages import messages_to_dict
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deepclaw import agent as agent_module
from deepclaw.agent import (
    DEFAULT_SOUL,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    OPENAI_MODEL_GUIDANCE_MODELS,
    TOOL_USE_ENFORCEMENT,
    _shell_backend,
)
from deepclaw.config import CHECKPOINTER_DB_PATH, DeepClawConfig
from deepclaw.local_context import DETECT_CONTEXT_SCRIPT, ExecutableBackend
from deepclaw.tools import discover_tools

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency fallback
    tiktoken = None


@dataclass(frozen=True)
class ContextSection:
    name: str
    chars: int
    tokens: int
    details: str


def _model_token_name(model: str) -> str:
    if not model:
        return ""
    return model.split(":", 1)[-1]


def estimate_tokens(text: str, model: str = "") -> int:
    """Estimate tokens for text, preferring tiktoken when available."""
    if not text:
        return 0

    if tiktoken is not None:
        token_name = _model_token_name(model)
        try:
            encoding = (
                tiktoken.encoding_for_model(token_name)
                if token_name
                else tiktoken.get_encoding("cl100k_base")
            )
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    return max(1, len(text) // 4)


def _format_size(chars: int, tokens: int) -> str:
    return f"{chars:,} chars (~{tokens:,} tok)"


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _active_soul_text() -> tuple[str, str]:
    if agent_module.SOUL_FILE.exists():
        return _safe_read_text(agent_module.SOUL_FILE).strip(), str(agent_module.SOUL_FILE)
    return DEFAULT_SOUL.strip(), f"default ({agent_module.SOUL_FILE} not created yet)"


def _load_latest_checkpoint(thread_id: str) -> dict[str, Any] | None:
    if not thread_id or not CHECKPOINTER_DB_PATH.exists():
        return None

    query = (
        "select type, checkpoint from checkpoints "
        "where thread_id = ? and checkpoint_ns = '' "
        "order by rowid desc limit 1"
    )
    with sqlite3.connect(CHECKPOINTER_DB_PATH) as conn:
        row = conn.execute(query, (thread_id,)).fetchone()

    if row is None:
        return None

    row_type, blob = row
    if blob is None:
        return None

    serde = JsonPlusSerializer()
    return serde.loads_typed((row_type or "msgpack", blob))


def _serialize_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _memory_prompt(memory_contents: dict[str, str], sources: list[str]) -> str:
    if not memory_contents:
        return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

    sections = [f"{path}\n{memory_contents[path]}" for path in sources if memory_contents.get(path)]
    agent_memory = "\n\n".join(sections) if sections else "(No memory loaded)"
    return MEMORY_SYSTEM_PROMPT.format(agent_memory=agent_memory)


def _extract_frontmatter(path: Path) -> dict[str, Any]:
    text = _safe_read_text(path)
    if not text.startswith("---\n"):
        return {}

    try:
        _, rest = text.split("---\n", 1)
        frontmatter, _body = rest.split("\n---\n", 1)
    except ValueError:
        return {}

    try:
        data = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return {}

    return data if isinstance(data, dict) else {}


def _skill_metadata_from_disk(skills_dir: Path) -> list[dict[str, Any]]:
    if not skills_dir.is_dir():
        return []

    result: list[dict[str, Any]] = []
    for skill_dir in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            continue
        frontmatter = _extract_frontmatter(skill_file)
        name = str(frontmatter.get("name") or skill_dir.name)
        description = str(frontmatter.get("description") or "")
        allowed_tools = frontmatter.get("allowed-tools")
        if isinstance(allowed_tools, str):
            allowed_tools_list = [item for item in allowed_tools.split() if item]
        elif isinstance(allowed_tools, list):
            allowed_tools_list = [str(item) for item in allowed_tools]
        else:
            allowed_tools_list = []
        result.append(
            {
                "name": name,
                "description": description,
                "path": str(skill_file),
                "allowed_tools": allowed_tools_list,
            }
        )
    return result


def _format_skills_locations(sources: list[str]) -> str:
    lines = []
    for index, source_path in enumerate(sources):
        suffix = " (higher priority)" if index == len(sources) - 1 else ""
        name = Path(source_path.rstrip("/")).name.capitalize()
        lines.append(f"**{name} Skills**: `{source_path}`{suffix}")
    return "\n".join(lines)


def _format_skills_list(skills_metadata: list[dict[str, Any]], sources: list[str]) -> str:
    if not skills_metadata:
        joined = " or ".join(sources)
        return f"(No skills available yet. You can create skills in {joined})"

    lines: list[str] = []
    for skill in skills_metadata:
        description = skill.get("description") or ""
        lines.append(f"- **{skill['name']}**: {description}".rstrip())
        allowed_tools = skill.get("allowed_tools") or []
        if allowed_tools:
            lines.append(f"  -> Allowed tools: {', '.join(str(tool) for tool in allowed_tools)}")
        lines.append(f"  -> Read `{skill['path']}` for full instructions")
    return "\n".join(lines)


def _skills_prompt(skills_metadata: list[dict[str, Any]], sources: list[str]) -> str:
    return SKILLS_SYSTEM_PROMPT.format(
        skills_locations=_format_skills_locations(sources),
        skills_list=_format_skills_list(skills_metadata, sources),
        skills_load_warnings="",
    )


def _get_local_context(config: DeepClawConfig, cached: str | None) -> tuple[str, str]:
    if cached:
        return cached, "cached"

    backend = _shell_backend(config)
    if not isinstance(backend, ExecutableBackend):
        return "", "unavailable"

    try:
        result = backend.execute(DETECT_CONTEXT_SCRIPT, timeout=30)
    except Exception:
        return "", "unavailable"

    if result.exit_code != 0 or not result.output:
        return "", "unavailable"

    return result.output.strip(), "fresh"


def _tool_schemas(model: str) -> tuple[list[str], str]:
    tool_schemas = []
    tool_names = []
    for tool in discover_tools():
        tool_names.append(getattr(tool, "__name__", tool.__class__.__name__))
        tool_schemas.append(convert_to_openai_tool(tool))
    return tool_names, _serialize_json(tool_schemas)


def _system_prompt_sections(config: DeepClawConfig) -> list[tuple[str, str, str]]:
    sections: list[tuple[str, str, str]] = []

    soul_text, soul_source = _active_soul_text()
    if soul_text:
        sections.append(("SOUL.md", soul_text, soul_source))

    sections.append(("Tool Use Enforcement", TOOL_USE_ENFORCEMENT.strip(), "built-in"))

    model_name = (config.model or "").lower()
    if any(token in model_name for token in OPENAI_MODEL_GUIDANCE_MODELS):
        sections.append(
            (
                "OpenAI/Codex execution guidance",
                OPENAI_MODEL_EXECUTION_GUIDANCE.strip(),
                "built-in",
            )
        )

    return sections


def build_context_report(config: DeepClawConfig, thread_id: str) -> str:
    """Build a human-readable context breakdown for the current thread."""
    checkpoint = _load_latest_checkpoint(thread_id)
    channel_values = checkpoint.get("channel_values", {}) if checkpoint else {}

    system_layers = _system_prompt_sections(config)
    system_sections = [
        ContextSection(
            name=name,
            chars=len(text),
            tokens=estimate_tokens(text, config.model),
            details=details,
        )
        for name, text, details in system_layers
    ]
    base_system_prompt = "\n\n".join(text for _name, text, _details in system_layers if text)

    memory_sources = [str(agent_module.MEMORY_FILE)]
    memory_contents = channel_values.get("memory_contents") or {}
    if not memory_contents and agent_module.MEMORY_FILE.exists():
        memory_contents = {str(agent_module.MEMORY_FILE): _safe_read_text(agent_module.MEMORY_FILE)}
    memory_prompt = _memory_prompt(memory_contents, memory_sources)

    skills_sources = [str(agent_module.SKILLS_DIR)]
    raw_skills_metadata = channel_values.get("skills_metadata")
    if isinstance(raw_skills_metadata, list):
        skills_metadata = [dict(item) for item in raw_skills_metadata]
        skills_state = "cached"
    else:
        skills_metadata = _skill_metadata_from_disk(agent_module.SKILLS_DIR)
        skills_state = "disk"
    skills_prompt = _skills_prompt(skills_metadata, skills_sources)

    local_context_text, local_context_state = _get_local_context(
        config,
        channel_values.get("local_context")
        if isinstance(channel_values.get("local_context"), str)
        else None,
    )

    tool_names, tool_schemas_json = _tool_schemas(config.model or "")

    messages = channel_values.get("messages") or []
    serialized_messages = _serialize_json(messages_to_dict(messages)) if messages else "[]"

    prompt_total = base_system_prompt + "\n\n" + memory_prompt + "\n\n" + skills_prompt
    if local_context_text:
        prompt_total = f"{prompt_total}\n\n{local_context_text}"

    combined_tokens = (
        estimate_tokens(prompt_total, config.model)
        + estimate_tokens(tool_schemas_json, config.model)
        + estimate_tokens(serialized_messages, config.model)
    )
    combined_chars = len(prompt_total) + len(tool_schemas_json) + len(serialized_messages)

    lines = [
        "🧠 Context breakdown",
        f"Workspace: {Path(config.workspace_root).expanduser()}",
        f"Model: {config.model or 'not set'}",
        f"Thread ID: {thread_id}",
        f"Checkpoint DB: {CHECKPOINTER_DB_PATH}",
        "",
        "System prompt layers (in order):",
    ]

    for index, section in enumerate(system_sections, start=1):
        lines.append(
            f"{index}. {section.name}: {_format_size(section.chars, section.tokens)} | {section.details}"
        )

    system_total_tokens = estimate_tokens(base_system_prompt, config.model)
    lines.append(
        f"Base system prompt total: {_format_size(len(base_system_prompt), system_total_tokens)}"
    )
    lines.append("")

    lines.extend(
        [
            "Injected context:",
            f"• MemoryMiddleware: {_format_size(len(memory_prompt), estimate_tokens(memory_prompt, config.model))} | {len(memory_contents)} source(s)",
            f"• SkillsMiddleware: {_format_size(len(skills_prompt), estimate_tokens(skills_prompt, config.model))} | {len(skills_metadata)} skill(s) from {skills_state}",
            f"• LocalContextMiddleware: {_format_size(len(local_context_text), estimate_tokens(local_context_text, config.model))} | {local_context_state}",
            f"• Tool schemas JSON: {_format_size(len(tool_schemas_json), estimate_tokens(tool_schemas_json, config.model))} | {len(tool_names)} tool(s)",
            f"• Thread messages: {_format_size(len(serialized_messages), estimate_tokens(serialized_messages, config.model))} | {len(messages)} message(s)",
            "",
        ]
    )

    cached_keys = sorted(channel_values.keys()) if channel_values else []
    lines.append(
        f"Checkpoint state keys: {', '.join(cached_keys) if cached_keys else '(none yet)'}"
    )
    if skills_metadata:
        skill_names = ", ".join(str(skill.get("name", "unknown")) for skill in skills_metadata)
        lines.append(f"Skills in context: {skill_names}")
    else:
        lines.append("Skills in context: (none)")
    lines.append(f"Tools: {', '.join(tool_names)}")
    lines.append("")
    lines.append(
        f"Estimated active context subtotal: {_format_size(combined_chars, combined_tokens)}"
    )
    lines.append(
        "(Includes prompt text, tool schemas, and serialized thread messages; excludes provider-side framing.)"
    )

    return "\n".join(lines)


__all__ = ["build_context_report", "estimate_tokens"]
