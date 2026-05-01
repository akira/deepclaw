"""Shared message gateway that invokes the agent and streams responses back through channels.

This module is channel-agnostic — it depends only on the Channel ABC,
so adding Discord/Slack/etc. requires zero changes here.
"""

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from deepclaw.channels.base import Channel, IncomingMessage
from deepclaw.safety import redact_secrets

logger = logging.getLogger(__name__)

CURSOR_INDICATOR = "\u258c"
THINKING_MESSAGE = "Thinking..."
_QUEUE_PROGRESS_LIMIT = 8
_TOOL_ICONS = {
    "skill_view": "📚",
    "skills_list": "📚",
    "todo": "📋",
    "terminal": "💻",
    "execute": "💻",
    "search_files": "🔎",
    "read_file": "📖",
    "patch": "🔧",
    "write_file": "📝",
    "browser_navigate": "🌐",
    "browser_click": "🖱️",
    "browser_snapshot": "🌐",
    "browser_vision": "🖼️",
    "vision_analyze": "🖼️",
}

# Phrases that suggest the model is describing a future action instead of calling a tool
_NARRATION_OPENERS = (
    "i'll ",
    "i will ",
    "let me ",
    "i'm going to ",
    "i am going to ",
    "i can ",
    "i need to ",
    "i should ",
    "now i'll ",
    "next i'll ",
    "first i'll ",
    "i'll now ",
)
_ACTION_WORDS = (
    "check",
    "look",
    "search",
    "find",
    "read",
    "write",
    "create",
    "run",
    "execute",
    "analyze",
    "fetch",
    "get",
    "update",
    "install",
    "clone",
    "open",
    "scan",
    "list",
    "review",
    "examine",
    "call",
    "use",
    "save",
    "edit",
    "patch",
    "modify",
    "restart",
)
# Pre-compiled regex for whole-word action matching — avoids substring false positives
# such as "use" in "because", "get" in "forget", "list" in "listen".
_ACTION_RE = re.compile(r"\b(" + "|".join(_ACTION_WORDS) + r")\b")
_NUDGE_MESSAGE = (
    "You described an action but did not call any tools. "
    "Please call the appropriate tool now to carry out what you described."
)
_MUTATION_REQUEST_WORDS = (
    "update",
    "save",
    "edit",
    "patch",
    "modify",
    "write",
    "add",
    "create",
    "install",
    "restart",
    "remember",
)
_MUTATION_REQUEST_RE = re.compile(r"\b(" + "|".join(_MUTATION_REQUEST_WORDS) + r")\b")
_COMPLETION_CLAIM_WORDS = (
    "done",
    "updated",
    "saved",
    "edited",
    "patched",
    "modified",
    "added",
    "created",
    "installed",
    "restarted",
    "incorporated",
)
_COMPLETION_CLAIM_RE = re.compile(r"\b(" + "|".join(_COMPLETION_CLAIM_WORDS) + r")\b")
_MEMORY_REQUEST_PATTERNS = (
    re.compile(r"\bremember\b"),
    re.compile(r"\bprefer\b"),
    re.compile(r"\bpreference\b"),
    re.compile(r"\bfrom now on\b"),
    re.compile(r"\bwhen i ask you to\b"),
)
_MEMORY_ACK_PATTERNS = (
    re.compile(r"\bi(?:'ll| will) remember\b"),
    re.compile(r"\bi(?:'ll| will) prefer\b"),
    re.compile(r"\bfrom now on\b"),
    re.compile(r"\bi(?:'ll| will) treat\b"),
    re.compile(r"\btreat .* as\b"),
    re.compile(r"\bprefer that workflow\b"),
    re.compile(r"\bpreference going forward\b"),
)


def _extract_pending_interrupt(state: Any, thread_id: str) -> dict[str, Any] | None:
    """Return the first safety-review interrupt payload, if any."""
    interrupts = getattr(state, "interrupts", ()) or ()
    for interrupt in interrupts:
        value = getattr(interrupt, "value", None)
        if not isinstance(value, Mapping):
            continue
        if value.get("type") != "safety_review":
            continue
        approval_keys = value.get("approval_keys")
        return {
            "id": getattr(interrupt, "id", None),
            "thread_id": thread_id,
            "type": value.get("type"),
            "tool": value.get("tool"),
            "command": value.get("command"),
            "approval_keys": list(approval_keys) if isinstance(approval_keys, list) else [],
            "warning": value.get("warning"),
            "message": value.get("message") or "Safety review required.",
        }
    return None


def _format_pending_interrupt_message(pending: Mapping[str, Any]) -> str:
    """Format the Telegram-visible prompt for a pending safety review."""
    base = str(pending.get("message") or "Safety review required.").strip()
    suffix = "Use /approve, /approve session, or /deny <reason> to respond."
    return f"{base}\n\n{suffix}" if base else suffix


def _normalize_text(text: str) -> str:
    return text.lower().replace("’", "'").replace("‘", "'")


def _truncate_preview(text: str, limit: int = 48) -> str:
    """Return a single-line truncated preview."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _safe_preview(text: Any, limit: int = 48) -> str:
    """Redact secrets first, then produce a short single-line preview."""
    return _truncate_preview(redact_secrets(str(text)), limit=limit)


def _format_tool_summary(tool_name: str, tool_args: Any) -> str | None:
    """Return a short user-visible summary for a tool call."""
    if not isinstance(tool_args, Mapping):
        return None

    if tool_name == "skill_view":
        name = tool_args.get("name")
        return f'"{name}"' if name else None

    if tool_name == "todo":
        todos = tool_args.get("todos")
        if isinstance(todos, list):
            return f'"planning {len(todos)} task(s)"'
        return '"read current task list"'

    if tool_name in {"terminal", "execute"}:
        command = tool_args.get("command")
        return f'"{_safe_preview(command)}"' if command else None

    if tool_name == "search_files":
        pattern = tool_args.get("pattern")
        return f'"{_safe_preview(pattern)}"' if pattern else None

    if tool_name in {"read_file", "write_file"}:
        path = tool_args.get("path")
        return f'"{path}"' if path else None

    if tool_name == "patch":
        path = tool_args.get("path")
        if path:
            suffix = ""
            if tool_args.get("replace_all"):
                suffix = " (all)"
            return f'"{path}"{suffix}'
        if tool_args.get("patch"):
            return '"multi-file patch"'
        return None

    if tool_name == "browser_navigate":
        url = tool_args.get("url")
        return f'"{_safe_preview(url)}"' if url else None

    for key in ("file_path", "name", "question", "expression", "url", "text"):
        value = tool_args.get(key)
        if value:
            return f'"{_safe_preview(value)}"'
    return None


def _format_tool_progress_line(tool_name: str, tool_args: Any) -> str:
    """Format a compact status line for an in-flight tool call."""
    icon = _TOOL_ICONS.get(tool_name, "🔧")
    summary = _format_tool_summary(tool_name, tool_args)
    return f"{icon} {tool_name}: {summary}" if summary else f"{icon} {tool_name}"


def _block_progress_key(block: Mapping[str, Any], tool_name: str) -> str | None:
    """Return a stable-ish key for chunk/final tool-call deduping when available."""
    block_id = block.get("id")
    if block_id:
        return str(block_id)
    index = block.get("index")
    if index is not None:
        return f"{tool_name}:{index}"
    return None


def _emit_progress_line(
    accumulated: str,
    queue_snapshot: dict[str, Any],
    line: str,
    *,
    now: float,
) -> tuple[str, int]:
    """Append a user-visible progress line to both live text and queue snapshot."""
    line = redact_secrets(line)
    accumulated, delta = _append_progress_line(accumulated, line)
    queue_snapshot["progress_lines"] = _append_progress_history(
        queue_snapshot.get("progress_lines", []), line
    )
    queue_snapshot["updated_at"] = now
    return accumulated, delta


def _extract_message_tool_calls(message_obj: Any) -> list[Mapping[str, Any]]:
    """Return structured tool calls attached to a streamed message object, if any."""
    resolved: list[Mapping[str, Any]] = []
    tool_calls = getattr(message_obj, "tool_calls", None)
    if isinstance(tool_calls, list):
        resolved.extend(tc for tc in tool_calls if isinstance(tc, Mapping))

    tool_call_chunks = getattr(message_obj, "tool_call_chunks", None)
    if isinstance(tool_call_chunks, list):
        resolved.extend(tc for tc in tool_call_chunks if isinstance(tc, Mapping))

    return resolved


def _coerce_tool_args(candidate_args: Any) -> dict[str, Any]:
    """Best-effort normalization of tool args from mappings or JSON strings."""
    if isinstance(candidate_args, Mapping) and candidate_args:
        return dict(candidate_args)
    if isinstance(candidate_args, str):
        text = candidate_args.strip()
        if not text:
            return {}
        with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
            parsed = json.loads(text)
            if isinstance(parsed, Mapping) and parsed:
                return dict(parsed)
    return {}


def _resolve_tool_args(
    block: Mapping[str, Any],
    tool_name: str,
    tool_calls: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve tool args from a stream block, falling back to message-level tool_calls."""
    block_args = block.get("args")
    if resolved := _coerce_tool_args(block_args):
        return resolved

    block_id = block.get("id")
    block_index = block.get("index")
    for tool_call in tool_calls:
        if tool_call.get("name") != tool_name:
            continue
        if block_id and tool_call.get("id") not in {None, block_id}:
            continue
        if block_index is not None and tool_call.get("index") not in {None, block_index}:
            continue
        if resolved := _coerce_tool_args(tool_call.get("args")):
            return resolved

    for tool_call in tool_calls:
        if tool_call.get("name") != tool_name:
            continue
        if resolved := _coerce_tool_args(tool_call.get("args")):
            return resolved

    return {}


def _append_progress_line(accumulated: str, line: str) -> tuple[str, int]:
    """Append a progress line, collapsing consecutive duplicates with a count."""
    if not accumulated:
        new_text = line
        return new_text, len(new_text)

    lines = accumulated.split("\n")
    last = lines[-1]
    duplicate_match = re.fullmatch(rf"{re.escape(line)} \(×(\d+)\)", last)
    if last == line:
        lines[-1] = f"{line} (×2)"
        new_text = "\n".join(lines)
        return new_text, len(lines[-1]) - len(last)
    if duplicate_match:
        count = int(duplicate_match.group(1)) + 1
        lines[-1] = f"{line} (×{count})"
        new_text = "\n".join(lines)
        return new_text, len(lines[-1]) - len(last)

    new_text = accumulated + "\n" + line
    return new_text, len(line) + 1


def _append_progress_history(
    history: list[str], line: str, *, limit: int = _QUEUE_PROGRESS_LIMIT
) -> list[str]:
    """Append a progress line to queue history, collapsing consecutive duplicates."""
    if not history:
        return [line]

    last = history[-1]
    duplicate_match = re.fullmatch(rf"{re.escape(line)} \(×(\d+)\)", last)
    if last == line:
        history = [*history[:-1], f"{line} (×2)"]
    elif duplicate_match:
        count = int(duplicate_match.group(1)) + 1
        history = [*history[:-1], f"{line} (×{count})"]
    else:
        history = [*history, line]

    if len(history) > limit:
        history = history[-limit:]
    return history


def _redacted_pending_approval(pending: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a redacted copy of pending approval data for queue/status display."""
    if not isinstance(pending, Mapping):
        return None
    redacted: dict[str, Any] = {}
    for key, value in pending.items():
        redacted[key] = redact_secrets(value) if isinstance(value, str) else value
    return redacted


def _looks_like_narration(text: str) -> bool:
    """Return True if text describes a tool action without having called any tools."""
    lower = _normalize_text(text)
    has_opener = any(
        lower.lstrip().startswith(op)
        or f"\n{op}" in lower
        or f". {op}" in lower
        or f"! {op}" in lower
        or f"? {op}" in lower
        for op in _NARRATION_OPENERS
    )
    has_action = bool(_ACTION_RE.search(lower))
    return has_opener and has_action


def _looks_like_false_completion(user_text: str, assistant_text: str) -> bool:
    """Return True when the user asked for a mutation and the assistant claimed completion without tools."""
    user_lower = _normalize_text(user_text)
    assistant_lower = _normalize_text(assistant_text)
    requested_mutation = bool(_MUTATION_REQUEST_RE.search(user_lower))
    claimed_completion = bool(_COMPLETION_CLAIM_RE.search(assistant_lower))
    return requested_mutation and claimed_completion


def _looks_like_memory_request(user_text: str, assistant_text: str) -> bool:
    """Return True when the user asked to remember a preference and the assistant only acknowledged it."""
    user_lower = _normalize_text(user_text)
    assistant_lower = _normalize_text(assistant_text)
    requested_memory = any(pattern.search(user_lower) for pattern in _MEMORY_REQUEST_PATTERNS)
    acknowledged_preference = any(
        pattern.search(assistant_lower) for pattern in _MEMORY_ACK_PATTERNS
    )
    claimed_completion = bool(_COMPLETION_CLAIM_RE.search(assistant_lower))
    return requested_memory and acknowledged_preference and not claimed_completion


def chunk_message(text: str, limit: int = 4096) -> list[str]:
    """Split text into chunks that fit within a channel's message size limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at last newline within limit
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


class _GatewayInactivityTimeout(RuntimeError):
    """Raised when the agent stops producing stream activity for too long."""

    def __init__(self, idle_seconds: float, last_tool: str | None) -> None:
        super().__init__(f"Agent inactive for {idle_seconds:.1f}s")
        self.idle_seconds = idle_seconds
        self.last_tool = last_tool


class Gateway:
    """Shared message handler that invokes the agent and streams responses back through channels."""

    def __init__(
        self,
        agent,
        streaming_config,
        *,
        max_turns: int = 500,
        gateway_timeout: float = 1800,
        gateway_timeout_warning: float = 900,
    ):
        self.agent = agent
        self.streaming_config = streaming_config
        self.max_turns = max(0, int(max_turns))
        self.gateway_timeout = None if gateway_timeout <= 0 else float(gateway_timeout)
        self.gateway_timeout_warning = (
            None if gateway_timeout_warning <= 0 else float(gateway_timeout_warning)
        )
        self._queue_state: dict[str, dict[str, Any]] = {}

    def get_queue_snapshot(self, chat_id: str) -> dict[str, Any] | None:
        """Return a copy of the latest queue/progress snapshot for a chat."""
        snapshot = self._queue_state.get(chat_id)
        if snapshot is None:
            return None
        copied = dict(snapshot)
        copied["progress_lines"] = list(snapshot.get("progress_lines", []))
        pending = snapshot.get("pending_approval")
        copied["pending_approval"] = dict(pending) if isinstance(pending, Mapping) else pending
        return copied

    @staticmethod
    def _run_config(
        thread_id: str,
        *,
        chat_id: str,
        channel_name: str,
        recursion_limit: int | None = None,
    ) -> dict[str, Any]:
        """Build per-invocation config for checkpointing and trace metadata."""
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "active_thread_id": thread_id,
                "chat_id": chat_id,
                "channel": channel_name,
                "updated_at": datetime.now(tz=UTC).isoformat(),
            },
        }
        if recursion_limit and recursion_limit > 0:
            config["recursion_limit"] = recursion_limit
        return config

    async def _edit_redacted_message(
        self, channel: Channel, chat_id: str, message_id: str, text: str
    ) -> None:
        """Redact secrets before editing a streamed message."""
        await channel.edit_message(chat_id, message_id, redact_secrets(text))

    async def _send_redacted_message(self, channel: Channel, chat_id: str, text: str) -> str:
        """Redact secrets before sending a message."""
        return await channel.send(chat_id, redact_secrets(text))

    async def _get_pending_interrupt(
        self,
        channel: Channel,
        thread_id: str,
        *,
        chat_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Inspect the graph state for a pending safety-review interrupt."""
        get_state = getattr(self.agent, "aget_state", None)
        if get_state is None:
            return None
        config = self._run_config(
            thread_id,
            chat_id=chat_id or thread_id,
            channel_name=channel.name,
        )
        try:
            state = await get_state(config)
        except Exception:
            logger.exception("Failed to inspect graph state for thread %s", thread_id)
            return None
        return _extract_pending_interrupt(state, thread_id)

    async def _stream_graph_input(
        self,
        channel: Channel,
        chat_id: str,
        thread_id: str,
        graph_input: dict[str, Any] | Command,
        *,
        original_user_text: str | None,
    ) -> dict[str, Any] | None:
        """Stream one graph invocation and return pending safety review info, if any."""
        msg_id = await self._send_redacted_message(channel, chat_id, THINKING_MESSAGE)

        stop_typing = asyncio.Event()

        async def _typing_heartbeat() -> None:
            """Keep Telegram typing indicator alive while agent is processing."""
            while not stop_typing.is_set():
                with contextlib.suppress(Exception):
                    await channel.send_typing(chat_id)
                try:
                    await asyncio.wait_for(stop_typing.wait(), timeout=4.5)
                except TimeoutError:
                    continue

        typing_task = asyncio.create_task(_typing_heartbeat())

        accumulated = ""
        last_edit_time = time.monotonic()
        chars_since_edit = 0
        last_display = THINKING_MESSAGE
        limit = channel.message_limit
        now_ts = time.time()
        queue_snapshot: dict[str, Any] = {
            "chat_id": chat_id,
            "thread_id": thread_id,
            "status": "running",
            "started_at": now_ts,
            "updated_at": now_ts,
            "user_text_preview": _safe_preview(original_user_text or "approval resume", limit=120),
            "progress_lines": [],
            "pending_approval": None,
            "final_text_preview": None,
        }
        self._queue_state[chat_id] = queue_snapshot

        async def _stream_once(payload: dict[str, Any] | Command) -> bool:
            """Stream one graph pass. Returns True if any tool calls were seen."""
            nonlocal accumulated, last_edit_time, chars_since_edit, last_display
            tool_calls_seen = False
            chunk_progress: dict[str, str] = {}
            last_chunk_line: str | None = None
            last_render_was_tool = False
            last_tool_name: str | None = None
            run_config = self._run_config(
                thread_id,
                chat_id=chat_id,
                channel_name=channel.name,
                recursion_limit=self.max_turns,
            )
            stream = self.agent.astream(
                payload,
                config=run_config,
                stream_mode="messages",
            )
            stream_iter = stream.__aiter__()
            pending_chunk: asyncio.Task[Any] | None = None
            last_activity = time.monotonic()
            warning_sent = False
            try:
                while True:
                    if self.gateway_timeout is None:
                        wait_timeout = None
                    else:
                        idle_seconds = time.monotonic() - last_activity
                        warning_wait: float | None = None
                        if (
                            not warning_sent
                            and self.gateway_timeout_warning is not None
                            and self.gateway_timeout_warning < self.gateway_timeout
                        ):
                            if idle_seconds >= self.gateway_timeout_warning:
                                warning_sent = True
                                warning_minutes = max(1, int(self.gateway_timeout_warning // 60))
                                remaining_seconds = max(0.0, self.gateway_timeout - idle_seconds)
                                remaining_minutes = max(1, int(remaining_seconds // 60))
                                await self._send_redacted_message(
                                    channel,
                                    chat_id,
                                    (
                                        f"⚠️ No activity for {warning_minutes} min. "
                                        f"If the agent does not respond soon, it will be timed out in "
                                        f"{remaining_minutes} min. You can continue waiting, or use "
                                        "/clear."
                                    ),
                                )
                            else:
                                warning_wait = self.gateway_timeout_warning - idle_seconds
                        remaining = self.gateway_timeout - idle_seconds
                        if remaining <= 0:
                            raise _GatewayInactivityTimeout(idle_seconds, last_tool_name)
                        wait_timeout = remaining
                        if warning_wait is not None:
                            wait_timeout = min(wait_timeout, warning_wait)

                    try:
                        if pending_chunk is None:
                            pending_chunk = asyncio.create_task(anext(stream_iter))
                        if wait_timeout is None:
                            chunk = await asyncio.shield(pending_chunk)
                        else:
                            chunk = await asyncio.wait_for(
                                asyncio.shield(pending_chunk), timeout=wait_timeout
                            )
                        pending_chunk = None
                    except StopAsyncIteration:
                        pending_chunk = None
                        break
                    except TimeoutError:
                        continue

                    last_activity = time.monotonic()

                    if not isinstance(chunk, tuple) or len(chunk) != 2:
                        continue
                    message_obj, _metadata = chunk

                    if isinstance(message_obj, ToolMessage):
                        tool_name = getattr(message_obj, "name", "unknown")
                        content = message_obj.content
                        preview = redact_secrets(str(content)[:200]) if content else "(empty)"
                        logger.info("Tool result [%s]: %s", tool_name, preview)
                        continue

                    if not hasattr(message_obj, "content_blocks"):
                        continue

                    message_tool_calls = _extract_message_tool_calls(message_obj)

                    for block in message_obj.content_blocks:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")

                        if block_type in ("tool_call", "tool_call_chunk"):
                            tool_calls_seen = True
                            tool_name = block.get("name")
                            if tool_name:
                                last_tool_name = str(tool_name)
                                tool_args = _resolve_tool_args(block, tool_name, message_tool_calls)
                                args_preview = (
                                    redact_secrets(str(tool_args)[:200]) if tool_args else ""
                                )
                                logger.info("Tool call [%s]: %s", tool_name, args_preview)
                                if not tool_args:
                                    block_preview = redact_secrets(str(dict(block))[:300])
                                    tool_calls_preview = redact_secrets(
                                        str(message_tool_calls)[:500]
                                    )
                                    additional_kwargs = redact_secrets(
                                        str(getattr(message_obj, "additional_kwargs", {}))[:500]
                                    )
                                    response_metadata = redact_secrets(
                                        str(getattr(message_obj, "response_metadata", {}))[:500]
                                    )
                                    logger.warning(
                                        "Tool call [%s] missing resolved args; block=%s message_tool_calls=%s additional_kwargs=%s response_metadata=%s message_type=%s",
                                        tool_name,
                                        block_preview,
                                        tool_calls_preview,
                                        additional_kwargs,
                                        response_metadata,
                                        type(message_obj).__name__,
                                    )
                                tool_line = _format_tool_progress_line(tool_name, tool_args)
                                tool_summary = _format_tool_summary(tool_name, tool_args)
                                progress_key = _block_progress_key(block, tool_name)
                                now_wall = time.time()
                                if block_type == "tool_call_chunk":
                                    if tool_summary:
                                        if progress_key is not None:
                                            previous_line = chunk_progress.get(progress_key)
                                            if previous_line != tool_line:
                                                accumulated, delta = _emit_progress_line(
                                                    accumulated,
                                                    queue_snapshot,
                                                    tool_line,
                                                    now=now_wall,
                                                )
                                                chars_since_edit += delta
                                                chunk_progress[progress_key] = tool_line
                                                last_render_was_tool = True
                                        elif last_chunk_line != tool_line:
                                            accumulated, delta = _emit_progress_line(
                                                accumulated,
                                                queue_snapshot,
                                                tool_line,
                                                now=now_wall,
                                            )
                                            chars_since_edit += delta
                                            last_chunk_line = tool_line
                                            last_render_was_tool = True
                                    else:
                                        if progress_key is not None:
                                            previous_line = chunk_progress.get(progress_key)
                                            if previous_line != tool_line:
                                                accumulated, delta = _emit_progress_line(
                                                    accumulated,
                                                    queue_snapshot,
                                                    tool_line,
                                                    now=now_wall,
                                                )
                                                chars_since_edit += delta
                                                chunk_progress[progress_key] = tool_line
                                                last_render_was_tool = True
                                        elif last_chunk_line != tool_line:
                                            accumulated, delta = _emit_progress_line(
                                                accumulated,
                                                queue_snapshot,
                                                tool_line,
                                                now=now_wall,
                                            )
                                            chars_since_edit += delta
                                            last_chunk_line = tool_line
                                            last_render_was_tool = True
                                else:
                                    chunk_line = (
                                        chunk_progress.pop(progress_key, None)
                                        if progress_key is not None
                                        else last_chunk_line
                                    )
                                    if chunk_line != tool_line:
                                        accumulated, delta = _emit_progress_line(
                                            accumulated,
                                            queue_snapshot,
                                            tool_line,
                                            now=now_wall,
                                        )
                                        chars_since_edit += delta
                                        last_render_was_tool = True
                                    if progress_key is None:
                                        last_chunk_line = None
                        elif block_type == "text":
                            text = block.get("text", "")
                            if not text:
                                continue
                            if (
                                last_render_was_tool
                                and accumulated
                                and not accumulated.endswith("\n")
                            ):
                                accumulated += "\n"
                                chars_since_edit += 1
                            accumulated += text
                            queue_snapshot["updated_at"] = time.time()
                            chars_since_edit += len(text)
                            last_render_was_tool = False
                        else:
                            continue

                        now = time.monotonic()
                        elapsed = now - last_edit_time
                        if (
                            elapsed >= self.streaming_config.edit_interval
                            or chars_since_edit >= self.streaming_config.buffer_threshold
                        ):
                            display = accumulated + CURSOR_INDICATOR
                            if len(display) <= limit and display != last_display:
                                await self._edit_redacted_message(channel, chat_id, msg_id, display)
                                last_display = display
                            last_edit_time = time.monotonic()
                            chars_since_edit = 0
            finally:
                if pending_chunk is not None and not pending_chunk.done():
                    pending_chunk.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await pending_chunk
                aclose = getattr(stream_iter, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()

            return tool_calls_seen

        pending: dict[str, Any] | None = None
        try:
            tool_calls_seen = await _stream_once(graph_input)
            pending = await self._get_pending_interrupt(channel, thread_id, chat_id=chat_id)

            if (
                original_user_text
                and pending is None
                and not tool_calls_seen
                and (
                    _looks_like_narration(accumulated)
                    or _looks_like_false_completion(original_user_text, accumulated)
                    or _looks_like_memory_request(original_user_text, accumulated)
                )
            ):
                logger.info(
                    "Narration/false-completion/memory-ack without tool-call detected — sending nudge"
                )
                accumulated = ""
                last_edit_time = time.monotonic()
                chars_since_edit = 0
                await _stream_once({"messages": [{"role": "user", "content": _NUDGE_MESSAGE}]})
                pending = await self._get_pending_interrupt(channel, thread_id, chat_id=chat_id)
        except GraphRecursionError:
            logger.warning("Agent hit recursion limit for thread %s", thread_id)
            accumulated = (
                f"⚠️ The agent hit the maximum turn limit ({self.max_turns}) and was stopped. "
                "Try breaking the task into smaller steps, or use /clear to start fresh."
            )
            queue_snapshot["status"] = "error"
            queue_snapshot["pending_approval"] = None
            queue_snapshot["updated_at"] = time.time()
        except _GatewayInactivityTimeout as exc:
            logger.warning(
                "Agent inactive for %.1fs on thread %s (last_tool=%s)",
                exc.idle_seconds,
                thread_id,
                exc.last_tool or "none",
            )
            timeout_minutes = max(1, int(exc.idle_seconds // 60))
            if exc.last_tool:
                accumulated = (
                    f"⏱️ The agent was inactive for {timeout_minutes} min and was stopped. "
                    f"It appeared stuck while using `{exc.last_tool}`. Try again, or use /clear to start fresh."
                )
            else:
                accumulated = (
                    f"⏱️ The agent was inactive for {timeout_minutes} min and was stopped. "
                    "It may have been waiting on a model or tool response. Try again, or use /clear to start fresh."
                )
            queue_snapshot["status"] = "error"
            queue_snapshot["pending_approval"] = None
            queue_snapshot["updated_at"] = time.time()
        except Exception:
            logger.exception("Agent streaming failed")
            accumulated = accumulated or "Sorry, something went wrong processing your message."
            queue_snapshot["status"] = "error"
            queue_snapshot["pending_approval"] = None
            queue_snapshot["updated_at"] = time.time()
        finally:
            stop_typing.set()
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

        if pending is not None:
            response_text = _format_pending_interrupt_message(pending)
            queue_snapshot["status"] = "awaiting_approval"
            queue_snapshot["pending_approval"] = _redacted_pending_approval(pending)
        else:
            response_text = accumulated if accumulated else "(no response)"
            if queue_snapshot.get("status") != "error":
                queue_snapshot["status"] = "completed"
            queue_snapshot["pending_approval"] = None
        response_text = redact_secrets(response_text)
        queue_snapshot["updated_at"] = time.time()
        queue_snapshot["final_text_preview"] = _safe_preview(response_text, limit=160)

        chunks = chunk_message(response_text, limit)
        if chunks[0] != last_display:
            await self._edit_redacted_message(channel, chat_id, msg_id, chunks[0])
        for extra_chunk in chunks[1:]:
            await self._send_redacted_message(channel, chat_id, extra_chunk)

        return pending

    async def handle_message(
        self, channel: Channel, message: IncomingMessage, thread_id: str
    ) -> dict[str, Any] | None:
        """Process an inbound message: invoke agent, stream response, deliver via channel."""
        logger.info("Received message from chat %s: %s", message.chat_id, message.text[:80])

        try:
            from deepclaw.tools.cron import set_chat_context

            set_chat_context(channel.name, message.chat_id)
        except ImportError:
            pass

        return await self._stream_graph_input(
            channel,
            message.chat_id,
            thread_id,
            {"messages": [{"role": "user", "content": message.text}]},
            original_user_text=message.text,
        )

    async def resume_interrupt(
        self,
        channel: Channel,
        *,
        chat_id: str,
        thread_id: str,
        decision: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Resume a pending graph interrupt with a human decision."""
        logger.info(
            "Resuming interrupt for chat %s thread %s with %s", chat_id, thread_id, decision
        )
        return await self._stream_graph_input(
            channel,
            chat_id,
            thread_id,
            Command(resume=decision),
            original_user_text=None,
        )
