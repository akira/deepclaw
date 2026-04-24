"""Internal runtime hygiene helpers for non-prompt state and artifact offloading."""

from __future__ import annotations

import contextlib
import contextvars
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepclaw.config import CONFIG_DIR

RUNTIME_DIR = CONFIG_DIR / "runtime"
ARTIFACTS_DIR = RUNTIME_DIR / "artifacts"
USER_INPUT_MAX_CHARS = 8_000
TOOL_RESULT_MAX_CHARS = 12_000
PREVIEW_MAX_CHARS = 280

_THREAD_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_runtime_state_var: contextvars.ContextVar[RuntimeState | None] = contextvars.ContextVar(
    "deepclaw_runtime_state",
    default=None,
)


@dataclass(slots=True)
class ArtifactRecord:
    """Metadata for an offloaded runtime artifact."""

    path: Path
    kind: str
    original_chars: int
    preview_chars: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeState:
    """Per-invocation runtime state for non-prompt metadata and artifacts."""

    thread_id: str
    invocation_id: str
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@contextlib.contextmanager
def bind_runtime_state(thread_id: str):
    """Bind a fresh per-invocation runtime state to the current async context."""

    state = RuntimeState(thread_id=thread_id, invocation_id=uuid.uuid4().hex)
    token = _runtime_state_var.set(state)
    try:
        yield state
    finally:
        _runtime_state_var.reset(token)


def get_runtime_state() -> RuntimeState | None:
    """Return the current runtime state, if one is bound."""

    return _runtime_state_var.get()


def _thread_slug(thread_id: str) -> str:
    slug = _THREAD_SLUG_RE.sub("-", thread_id).strip("-._")
    return slug or "thread"


def _artifact_dir(thread_id: str, category: str) -> Path:
    path = ARTIFACTS_DIR / _thread_slug(thread_id) / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def _preview_text(text: str, limit: int = PREVIEW_MAX_CHARS) -> str:
    compact = text.strip()
    if len(compact) <= limit:
        return compact
    half = max(limit // 2, 40)
    return f"{compact[:half].rstrip()}\n...\n{compact[-half:].lstrip()}"


def _artifact_name(kind: str) -> str:
    return f"{kind}-{uuid.uuid4().hex[:12]}.txt"


def write_text_artifact(
    text: str,
    *,
    category: str,
    kind: str,
    thread_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactRecord:
    """Persist text as a runtime artifact and record it in the active state when present."""

    state = get_runtime_state()
    resolved_thread_id = thread_id or (state.thread_id if state is not None else "detached")
    artifact_path = _artifact_dir(resolved_thread_id, category) / _artifact_name(kind)
    artifact_path.write_text(text, encoding="utf-8")

    record = ArtifactRecord(
        path=artifact_path,
        kind=kind,
        original_chars=len(text),
        preview_chars=min(len(text), PREVIEW_MAX_CHARS),
        metadata=dict(metadata or {}),
    )
    if state is not None:
        state.artifacts.append(record)
        state.metadata["artifact_count"] = len(state.artifacts)
    return record


def offload_text_if_oversized(
    text: str,
    *,
    category: str,
    kind: str,
    label: str,
    max_chars: int,
    thread_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, ArtifactRecord | None]:
    """Replace oversized text with a compact artifact pointer and preview."""

    if len(text) <= max_chars:
        return text, None

    record = write_text_artifact(
        text,
        category=category,
        kind=kind,
        thread_id=thread_id,
        metadata=metadata,
    )
    preview = _preview_text(text)
    replacement = (
        f"[DeepClaw offloaded {label} ({len(text)} chars) to runtime artifact "
        f"`{record.path}`. Use `read_file` on that path to recover the full content if needed.]"
    )
    if preview:
        replacement += f"\n\nPreview:\n{preview}"
    return replacement, record


def offload_user_input(
    text: str, *, thread_id: str | None = None
) -> tuple[str, ArtifactRecord | None]:
    """Offload oversized user messages before agent invocation."""

    return offload_text_if_oversized(
        text,
        category="user-input",
        kind="user-input",
        label="user input",
        max_chars=USER_INPUT_MAX_CHARS,
        thread_id=thread_id,
        metadata={"source": "gateway"},
    )


def offload_tool_result(
    tool_name: str,
    text: str,
    *,
    thread_id: str | None = None,
) -> tuple[str, ArtifactRecord | None]:
    """Offload oversized tool results before they are reintroduced into the prompt."""

    return offload_text_if_oversized(
        text,
        category="tool-results",
        kind=tool_name or "tool-result",
        label=f"tool result from {tool_name or 'unknown tool'}",
        max_chars=TOOL_RESULT_MAX_CHARS,
        thread_id=thread_id,
        metadata={"tool_name": tool_name},
    )
