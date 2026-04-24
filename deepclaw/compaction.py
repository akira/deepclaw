"""Thread compaction helpers built around structured continuity checkpoints."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from deepclaw.state_continuity import (
    REFERENCE_ONLY_NOTE,
    build_continuity_checkpoint,
    ensure_working_state,
)

COMPACTION_ROOT = Path("~/.deepclaw/workspace/compactions").expanduser()
_REFERENCE_ONLY_HEADER = "[HISTORICAL HANDOFF SUMMARY — REFERENCE ONLY]"


@dataclass(slots=True)
class CompactionArtifacts:
    raw_history_path: str
    checkpoint_path: str
    summary_path: str


@dataclass(slots=True)
class CompactionResult:
    old_thread_id: str
    new_thread_id: str
    summary_text: str
    summary_artifact_path: str
    checkpoint_artifact_path: str
    raw_history_artifact_paths: list[str]
    reason: str


@dataclass(slots=True)
class ThreadSnapshot:
    messages: list[AnyMessage]
    working_state: dict[str, Any]
    continuity_checkpoint: dict[str, Any]


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "thread"


async def get_thread_snapshot(checkpointer, thread_id: str) -> ThreadSnapshot:
    """Return checkpointed thread state for compaction purposes."""
    if checkpointer is None:
        return ThreadSnapshot(
            messages=[], working_state=ensure_working_state({}), continuity_checkpoint={}
        )

    checkpoint_tuple = await checkpointer.aget_tuple(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    if checkpoint_tuple is None:
        return ThreadSnapshot(
            messages=[], working_state=ensure_working_state({}), continuity_checkpoint={}
        )

    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])
    raw_working_state = channel_values.get("working_state", {})
    working_state = ensure_working_state(raw_working_state)
    continuity_checkpoint = channel_values.get("continuity_checkpoint")
    if not isinstance(continuity_checkpoint, dict):
        continuity_checkpoint = build_continuity_checkpoint(working_state)

    return ThreadSnapshot(
        messages=list(messages) if isinstance(messages, list) else [],
        working_state=working_state,
        continuity_checkpoint=dict(continuity_checkpoint),
    )


async def compact_thread(
    checkpointer,
    *,
    thread_id: str,
    chat_id: str,
    reason: str,
    source: str = "telegram",
) -> CompactionResult | None:
    """Create raw-history/checkpoint artifacts and return a fresh thread rotation plan."""
    snapshot = await get_thread_snapshot(checkpointer, thread_id)
    non_system_messages = [
        message for message in snapshot.messages if not isinstance(message, SystemMessage)
    ]
    if len(non_system_messages) < 2:
        return None

    artifacts = write_compaction_artifacts(
        snapshot=snapshot,
        thread_id=thread_id,
        chat_id=chat_id,
        reason=reason,
        source=source,
    )
    summary_text = Path(artifacts.summary_path).read_text(encoding="utf-8")
    return CompactionResult(
        old_thread_id=thread_id,
        new_thread_id=str(uuid.uuid4()),
        summary_text=summary_text,
        summary_artifact_path=artifacts.summary_path,
        checkpoint_artifact_path=artifacts.checkpoint_path,
        raw_history_artifact_paths=[artifacts.raw_history_path],
        reason=reason,
    )


def write_compaction_artifacts(
    *,
    snapshot: ThreadSnapshot,
    thread_id: str,
    chat_id: str,
    reason: str,
    source: str = "telegram",
) -> CompactionArtifacts:
    """Persist raw history, continuity checkpoint, and handoff summary artifacts."""
    artifacts_dir = COMPACTION_ROOT / source / _safe_slug(chat_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    raw_history_path = artifacts_dir / f"{stamp}-{_safe_slug(thread_id)}-raw-history.md"
    checkpoint_path = artifacts_dir / f"{stamp}-{_safe_slug(thread_id)}-checkpoint.json"
    summary_path = artifacts_dir / f"{stamp}-{_safe_slug(thread_id)}-handoff.md"

    raw_history_path.write_text(_format_messages_markdown(snapshot.messages), encoding="utf-8")
    checkpoint_payload = {
        "thread_id": thread_id,
        "reason": reason,
        "captured_at": datetime.now(UTC).isoformat(),
        "continuity_checkpoint": snapshot.continuity_checkpoint,
        "working_state": snapshot.working_state,
    }
    checkpoint_path.write_text(
        json.dumps(checkpoint_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_text = build_reference_handoff_summary(
        continuity_checkpoint=snapshot.continuity_checkpoint,
        thread_id=thread_id,
        reason=reason,
        raw_history_path=str(raw_history_path),
        checkpoint_path=str(checkpoint_path),
        message_count=len(snapshot.messages),
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    return CompactionArtifacts(
        raw_history_path=str(raw_history_path),
        checkpoint_path=str(checkpoint_path),
        summary_path=str(summary_path),
    )


def build_reference_handoff_summary(
    *,
    continuity_checkpoint: dict[str, Any],
    thread_id: str,
    reason: str,
    raw_history_path: str,
    checkpoint_path: str,
    message_count: int,
) -> str:
    """Render the one-shot historical handoff reinjected into a fresh thread."""
    current_goal = continuity_checkpoint.get("current_goal") or "(not captured)"
    next_action = continuity_checkpoint.get("next_action") or "(not captured)"
    blockers = continuity_checkpoint.get("active_blockers") or []
    relevant_files = continuity_checkpoint.get("relevant_files") or []
    pending_verification = continuity_checkpoint.get("pending_verification") or []
    artifact_refs = continuity_checkpoint.get("artifact_refs") or []
    reference_items = continuity_checkpoint.get("reference_summary", {}).get("items") or []

    def _bullets(items: list[Any], *, empty: str) -> list[str]:
        if not items:
            return [f"- {empty}"]
        return [f"- {item}" for item in items]

    artifact_lines = []
    for artifact in artifact_refs[:6]:
        if not isinstance(artifact, dict):
            continue
        label = artifact.get("label") or artifact.get("kind") or "artifact"
        path = artifact.get("path") or "(unknown path)"
        artifact_lines.append(f"- {label}: `{path}`")
    if not artifact_lines:
        artifact_lines.append("- none recorded")

    lines = [
        _REFERENCE_ONLY_HEADER,
        "This summary was generated when DeepClaw compacted and rotated the prior thread.",
        REFERENCE_ONLY_NOTE,
        "Use it as background only. If a detail matters, inspect the artifacts instead of assuming.",
        "",
        "## Prior Thread",
        f"- thread_id: {thread_id}",
        f"- reason: {reason}",
        f"- message_count: {message_count}",
        f"- continuity_checkpoint_artifact: `{checkpoint_path}`",
        f"- raw_history_artifact: `{raw_history_path}`",
        "",
        "## Continuity Checkpoint",
        f"- current_goal: {current_goal}",
        f"- next_action: {next_action}",
        f"- verification_status: {continuity_checkpoint.get('verification_status', 'unknown')}",
        "",
        "## Active Blockers",
        *_bullets(blockers, empty="none"),
        "",
        "## Relevant Files",
        *_bullets(relevant_files, empty="none recorded"),
        "",
        "## Pending Verification",
        *_bullets(pending_verification, empty="none"),
        "",
        "## Referenced Artifacts",
        *artifact_lines,
        "",
        "## Historical Notes",
        *_bullets(reference_items, empty="none"),
        "",
        "## Recoverability",
        "- Use `read_file` on the checkpoint or raw-history artifacts to recover compacted-away detail.",
        "- Reconcile any historical summary against the live thread before acting on it.",
    ]
    return "\n".join(lines).strip() + "\n"


def _text_from_message(msg: AnyMessage) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_role(msg: AnyMessage) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, ToolMessage):
        return "tool"
    if isinstance(msg, SystemMessage):
        return "system"
    return getattr(msg, "type", msg.__class__.__name__.lower())


def _format_messages_markdown(messages: list[AnyMessage]) -> str:
    """Serialize checkpoint messages into a readable raw-history artifact."""
    lines = [
        "# DeepClaw Thread History Artifact",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Message count: {len(messages)}",
        "",
    ]
    for idx, msg in enumerate(messages, start=1):
        role = _message_role(msg)
        lines.append(f"## Message {idx} — {role}")
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None) or "tool"
            lines.append(f"tool_name: {tool_name}")
        lines.append("")
        lines.append(_text_from_message(msg).strip() or "(empty)")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
