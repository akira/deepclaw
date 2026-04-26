"""Shared message gateway that invokes the agent and streams responses back through channels.

This module is channel-agnostic — it depends only on the Channel ABC,
so adding Discord/Slack/etc. requires zero changes here.
"""

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepclaw.channels.base import Channel, IncomingMessage
from deepclaw.safety import redact_secrets

logger = logging.getLogger(__name__)

CURSOR_INDICATOR = "\u258c"
THINKING_MESSAGE = "Thinking..."

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
        return {
            "id": getattr(interrupt, "id", None),
            "thread_id": thread_id,
            "type": value.get("type"),
            "tool": value.get("tool"),
            "command": value.get("command"),
            "warning": value.get("warning"),
            "message": value.get("message") or "Safety review required.",
        }
    return None


def _format_pending_interrupt_message(pending: Mapping[str, Any]) -> str:
    """Format the Telegram-visible prompt for a pending safety review."""
    base = str(pending.get("message") or "Safety review required.").strip()
    suffix = "Use /approve to continue or /deny <reason> to reject."
    return f"{base}\n\n{suffix}" if base else suffix


def _normalize_text(text: str) -> str:
    return text.lower().replace("’", "'").replace("‘", "'")


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


class Gateway:
    """Shared message handler that invokes the agent and streams responses back through channels."""

    def __init__(self, agent, streaming_config):
        self.agent = agent
        self.streaming_config = streaming_config

    @staticmethod
    def _run_config(thread_id: str, *, chat_id: str, channel_name: str) -> dict[str, Any]:
        """Build per-invocation config for checkpointing and trace metadata."""
        return {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "active_thread_id": thread_id,
                "chat_id": chat_id,
                "channel": channel_name,
            },
        }

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
        limit = channel.message_limit

        async def _stream_once(payload: dict[str, Any] | Command) -> bool:
            """Stream one graph pass. Returns True if any tool calls were seen."""
            nonlocal accumulated, last_edit_time, chars_since_edit
            tool_calls_seen = False

            run_config = self._run_config(
                thread_id,
                chat_id=chat_id,
                channel_name=channel.name,
            )
            async for chunk in self.agent.astream(
                payload,
                config=run_config,
                stream_mode="messages",
            ):
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

                for block in message_obj.content_blocks:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")

                    if block_type in ("tool_call", "tool_call_chunk"):
                        tool_calls_seen = True
                        tool_name = block.get("name")
                        if tool_name:
                            tool_args = block.get("args", {})
                            args_preview = redact_secrets(str(tool_args)[:200]) if tool_args else ""
                            logger.info("Tool call [%s]: %s", tool_name, args_preview)
                            tool_line = f"\n\U0001f527 {tool_name}\n"
                            accumulated += tool_line
                            chars_since_edit += len(tool_line)
                    elif block_type == "text":
                        text = block.get("text", "")
                        if not text:
                            continue
                        accumulated += text
                        chars_since_edit += len(text)
                    else:
                        continue

                    now = time.monotonic()
                    elapsed = now - last_edit_time
                    if (
                        elapsed >= self.streaming_config.edit_interval
                        or chars_since_edit >= self.streaming_config.buffer_threshold
                    ):
                        display = accumulated + CURSOR_INDICATOR
                        if len(display) <= limit:
                            await self._edit_redacted_message(channel, chat_id, msg_id, display)
                        last_edit_time = time.monotonic()
                        chars_since_edit = 0

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
        except Exception:
            logger.exception("Agent streaming failed")
            accumulated = accumulated or "Sorry, something went wrong processing your message."
        finally:
            stop_typing.set()
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

        if pending is not None:
            response_text = _format_pending_interrupt_message(pending)
        else:
            response_text = accumulated if accumulated else "(no response)"
        response_text = redact_secrets(response_text)

        chunks = chunk_message(response_text, limit)
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
