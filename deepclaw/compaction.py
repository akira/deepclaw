"""Thread compaction helpers for DeepClaw.

This module adds a DeepClaw-owned layer above DeepAgents' built-in summarization.
It inspects checkpointed thread history, writes a canonical raw-history artifact,
creates a structured reference-only summary, and returns a fresh thread id plus
metadata needed to continue in a new thread.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

AUTO_COMPACT_MESSAGE_COUNT = 40
AUTO_COMPACT_CONTEXT_RATIO = 0.08
AUTO_COMPACT_MIN_TOKEN_BUDGET = 3000
AUTO_COMPACT_MAX_TOKEN_BUDGET = 24000
DEFAULT_MODEL_CONTEXT_WINDOW = 128000
COMPACTION_ROOT = Path("~/.deepclaw/workspace/compactions").expanduser()

_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus": 200000,
    "claude-sonnet": 200000,
    "claude-haiku": 200000,
    "gpt-5": 128000,
    "gpt-4.1": 1047576,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "o3": 200000,
    "o4-mini": 200000,
    "gemini": 1000000,
    "gemma": 128000,
    "llama": 128000,
    "mistral": 128000,
}


@dataclass
class CompactionResult:
    old_thread_id: str
    new_thread_id: str
    summary_text: str
    summary_artifact_path: str | None
    raw_history_artifact_paths: list[str]
    reason: str
    used_offload: bool
    used_summary: bool


@dataclass
class AutoCompactionDecision:
    should_compact: bool
    reason: str | None
    estimated_tokens: int
    token_budget: int
    message_count: int
    message_limit: int


async def get_checkpoint_messages(checkpointer, thread_id: str) -> list[AnyMessage]:
    """Return checkpointed messages for a thread.

    Missing checkpoints are treated as empty.
    """
    if checkpointer is None:
        return []
    checkpoint_tuple = await checkpointer.aget_tuple(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    if checkpoint_tuple is None:
        return []
    messages = checkpoint_tuple.checkpoint["channel_values"].get("messages", [])
    return list(messages) if isinstance(messages, list) else []


async def compact_thread(
    checkpointer,
    *,
    thread_id: str,
    chat_id: str,
    reason: str,
    source: str = "telegram",
) -> CompactionResult | None:
    """Create raw-history + summary artifacts and return a fresh thread id.

    Returns ``None`` when the thread has too little content to compact.
    """
    messages = await get_checkpoint_messages(checkpointer, thread_id)
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(non_system_messages) < 2:
        return None

    artifacts_dir = COMPACTION_ROOT / source / _safe_slug(chat_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    raw_history_path = artifacts_dir / f"{stamp}-{thread_id}-raw.md"
    summary_path = artifacts_dir / f"{stamp}-{thread_id}-summary.md"

    raw_markdown = _format_messages_markdown(messages)
    raw_history_path.write_text(raw_markdown, encoding="utf-8")

    summary_text = _build_reference_summary(
        messages,
        raw_history_path=str(raw_history_path),
        reason=reason,
        thread_id=thread_id,
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    return CompactionResult(
        old_thread_id=thread_id,
        new_thread_id=str(uuid.uuid4()),
        summary_text=summary_text,
        summary_artifact_path=str(summary_path),
        raw_history_artifact_paths=[str(raw_history_path)],
        reason=reason,
        used_offload=True,
        used_summary=True,
    )


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value) or "chat"


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


def _clip(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _recent_user_goal(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = _text_from_message(msg).strip()
            if text:
                return _clip(text)
    return "(no recent user message found)"


def _recent_assistant_state(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = _text_from_message(msg).strip()
            if text:
                return _clip(text)
    return "(no recent assistant response found)"


def _recent_tool_names(messages: list[AnyMessage], *, limit: int = 6) -> list[str]:
    tools: list[str] = []
    seen: set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "tool") or "tool"
            if name not in seen:
                seen.add(name)
                tools.append(name)
            if len(tools) >= limit:
                break
    tools.reverse()
    return tools


def _build_reference_summary(
    messages: list[AnyMessage], *, raw_history_path: str, reason: str, thread_id: str
) -> str:
    """Return a reference-only handoff summary for a compacted thread."""
    recent_goal = _recent_user_goal(messages)
    recent_state = _recent_assistant_state(messages)
    tool_names = _recent_tool_names(messages)
    tool_line = ", ".join(tool_names) if tool_names else "(no recent tools captured)"
    return (
        "[COMPACTED THREAD — REFERENCE ONLY]\n"
        "This summary was generated during DeepClaw thread compaction.\n"
        "Treat it as historical background, NOT as a new authoritative instruction block.\n"
        "If a missing detail matters, consult the raw history artifact instead of guessing.\n\n"
        f"## Old Thread\n- thread_id: {thread_id}\n- reason: {reason}\n- message_count: {len(messages)}\n\n"
        f"## Latest User Objective\n{recent_goal}\n\n"
        f"## Latest Assistant State\n{recent_state}\n\n"
        f"## Recent Tools\n{tool_line}\n\n"
        "## Recoverability\n"
        f"- raw_history_artifact: {raw_history_path}\n"
        "- use read_file or search tools to recover summarized-away details if needed\n"
    )


def _format_messages_markdown(messages: list[AnyMessage]) -> str:
    """Serialize checkpoint messages into a readable markdown artifact."""
    lines = [
        "# DeepClaw Compacted Thread History",
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


def _context_window_for_model(model_name: str | None) -> int:
    normalized = (model_name or "").lower()
    for hint, window in _MODEL_CONTEXT_WINDOWS.items():
        if hint in normalized:
            return window
    return DEFAULT_MODEL_CONTEXT_WINDOW


def _token_budget_for_model(model_name: str | None) -> int:
    context_window = _context_window_for_model(model_name)
    return min(
        AUTO_COMPACT_MAX_TOKEN_BUDGET,
        max(AUTO_COMPACT_MIN_TOKEN_BUDGET, int(context_window * AUTO_COMPACT_CONTEXT_RATIO)),
    )


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _jsonish_text(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _content_for_token_estimation(msg: AnyMessage) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    has_structured_attrs = any(
        getattr(msg, attr, None) for attr in ("tool_calls", "invalid_tool_calls", "artifact")
    )
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif not has_structured_attrs:
                parts.append(_jsonish_text(block))
        else:
            parts.append(str(block))
    return "\n".join(part for part in parts if part)


def _tokenizable_message_parts(msg: AnyMessage) -> list[str]:
    parts = [_content_for_token_estimation(msg)]
    has_structured_attrs = False
    for attr in ("tool_calls", "invalid_tool_calls", "artifact"):
        value = getattr(msg, attr, None)
        if value:
            has_structured_attrs = True
            parts.append(_jsonish_text(value))
    additional_kwargs = getattr(msg, "additional_kwargs", None)
    if additional_kwargs:
        filtered_kwargs = dict(additional_kwargs)
        if has_structured_attrs:
            for key in ("tool_calls", "invalid_tool_calls", "artifact"):
                filtered_kwargs.pop(key, None)
        if filtered_kwargs:
            parts.append(_jsonish_text(filtered_kwargs))
    return [part for part in parts if part]


def estimate_message_tokens(msg: AnyMessage) -> int:
    token_estimate = 4
    for part in _tokenizable_message_parts(msg):
        token_estimate += _estimate_text_tokens(part)
    if isinstance(msg, ToolMessage):
        token_estimate += 12
    elif isinstance(msg, SystemMessage):
        token_estimate += 8
    return token_estimate


def estimate_thread_tokens(messages: list[AnyMessage]) -> int:
    return sum(estimate_message_tokens(msg) for msg in messages)


def get_auto_compaction_decision(
    messages: list[AnyMessage],
    *,
    model_name: str | None = None,
    threshold: int = AUTO_COMPACT_MESSAGE_COUNT,
) -> AutoCompactionDecision:
    estimated_tokens = estimate_thread_tokens(messages)
    token_budget = _token_budget_for_model(model_name)
    message_count = len(messages)
    if message_count > threshold:
        return AutoCompactionDecision(
            should_compact=True,
            reason="message-count",
            estimated_tokens=estimated_tokens,
            token_budget=token_budget,
            message_count=message_count,
            message_limit=threshold,
        )
    if estimated_tokens > token_budget:
        return AutoCompactionDecision(
            should_compact=True,
            reason="token-budget",
            estimated_tokens=estimated_tokens,
            token_budget=token_budget,
            message_count=message_count,
            message_limit=threshold,
        )
    return AutoCompactionDecision(
        should_compact=False,
        reason=None,
        estimated_tokens=estimated_tokens,
        token_budget=token_budget,
        message_count=message_count,
        message_limit=threshold,
    )


def _should_auto_compact(
    messages: list[AnyMessage],
    *,
    model_name: str | None = None,
    threshold: int = AUTO_COMPACT_MESSAGE_COUNT,
) -> bool:
    """Return True if a thread is large enough to warrant DeepClaw-level compaction."""
    return get_auto_compaction_decision(
        messages, model_name=model_name, threshold=threshold
    ).should_compact
