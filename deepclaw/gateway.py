"""Shared message gateway that invokes the agent and streams responses back through channels.

This module is channel-agnostic — it depends only on the Channel ABC,
so adding Discord/Slack/etc. requires zero changes here.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime

from langchain_core.messages import ToolMessage

from deepclaw.auth import get_thread_state
from deepclaw.channels.base import Channel, IncomingMessage
from deepclaw.compaction import (
    AutoCompactionDecision,
    CompactionResult,
    get_auto_compaction_decision,
    get_checkpoint_messages,
)
from deepclaw.compaction import compact_thread as compact_thread_impl
from deepclaw.safety import redact_secrets

logger = logging.getLogger(__name__)

CURSOR_INDICATOR = "\u258c"
THINKING_MESSAGE = "Thinking..."
_REFERENCE_SUMMARY_LABEL = "[REFERENCE-ONLY SUMMARY FROM PREVIOUS THREAD]"

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
)
_ACTION_RE = re.compile(r"\b(" + "|".join(_ACTION_WORDS) + r")\b")
_NUDGE_MESSAGE = (
    "You described an action but did not call any tools. "
    "Please call the appropriate tool now to carry out what you described."
)


def _looks_like_narration(text: str) -> bool:
    """Return True if text describes a tool action without having called any tools."""
    lower = text.lower()
    has_opener = any(
        lower.lstrip().startswith(op) or f"\n{op}" in lower for op in _NARRATION_OPENERS
    )
    has_action = bool(_ACTION_RE.search(lower))
    return has_opener and has_action


def chunk_message(text: str, limit: int = 4096) -> list[str]:
    """Split text into chunks that fit within a channel's message size limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


class Gateway:
    """Shared message handler that invokes the agent and streams responses back through channels."""

    def __init__(
        self,
        agent,
        streaming_config,
        *,
        checkpointer=None,
        thread_state_store: dict | None = None,
        persist_thread_state: Callable[[dict], None] | None = None,
        model_name: str | None = None,
    ):
        self.agent = agent
        self.streaming_config = streaming_config
        self.checkpointer = checkpointer
        self.thread_state_store = thread_state_store
        self.persist_thread_state = persist_thread_state
        self.model_name = model_name

    def _persist_thread_store(self) -> None:
        if self.persist_thread_state is None or self.thread_state_store is None:
            return
        self.persist_thread_state(self.thread_state_store)

    async def _edit_redacted_message(
        self, channel: Channel, chat_id: str, message_id: str, text: str
    ) -> None:
        """Redact secrets before editing a streamed message."""
        await channel.edit_message(chat_id, message_id, redact_secrets(text))

    async def _send_redacted_message(self, channel: Channel, chat_id: str, text: str) -> str:
        """Redact secrets before sending a message."""
        return await channel.send(chat_id, redact_secrets(text))

    def _peek_pending_summary(self, chat_id: str) -> str | None:
        if self.thread_state_store is None:
            return None
        state = get_thread_state(self.thread_state_store, chat_id)
        summary = state.get("pending_summary_text")
        return str(summary) if summary else None

    def _clear_pending_summary(self, chat_id: str) -> None:
        if self.thread_state_store is None:
            return
        state = get_thread_state(self.thread_state_store, chat_id)
        if not state.get("pending_summary_text"):
            return
        state["pending_summary_text"] = None
        self._persist_thread_store()

    async def compact_thread(
        self, chat_id: str, thread_id: str, *, reason: str
    ) -> CompactionResult | None:
        """Compact the current thread and rotate chat metadata to a fresh thread id."""
        result = await compact_thread_impl(
            self.checkpointer,
            thread_id=thread_id,
            chat_id=chat_id,
            reason=reason,
        )
        if result is None or self.thread_state_store is None:
            return result

        state = get_thread_state(self.thread_state_store, chat_id)
        state["current_thread_id"] = result.new_thread_id
        state["parent_thread_id"] = result.old_thread_id
        state["summary_artifact_path"] = result.summary_artifact_path
        state["raw_history_artifact_paths"] = deepcopy(result.raw_history_artifact_paths)
        state["pending_summary_text"] = result.summary_text
        state["last_compacted_at"] = datetime.now(UTC).isoformat()
        state["last_compaction_reason"] = result.reason
        self._persist_thread_store()
        return result

    async def _maybe_auto_compact(
        self, message: IncomingMessage, thread_id: str
    ) -> tuple[str, CompactionResult | None]:
        if self.checkpointer is None or self.thread_state_store is None:
            return thread_id, None
        messages = await get_checkpoint_messages(self.checkpointer, thread_id)
        decision: AutoCompactionDecision = get_auto_compaction_decision(
            messages, model_name=self.model_name
        )
        if not decision.should_compact:
            return thread_id, None
        result = await self.compact_thread(
            message.chat_id, thread_id, reason=f"auto-{decision.reason}"
        )
        if result is None:
            return thread_id, None
        logger.info(
            "Compacted chat %s thread %s -> %s (%s, est_tokens=%s, budget=%s, messages=%s)",
            message.chat_id,
            result.old_thread_id,
            result.new_thread_id,
            result.reason,
            decision.estimated_tokens,
            decision.token_budget,
            decision.message_count,
        )
        return result.new_thread_id, result

    def _build_effective_user_content(self, chat_id: str, text: str) -> tuple[str, bool]:
        summary = self._peek_pending_summary(chat_id)
        if not summary:
            return text, False
        return f"{_REFERENCE_SUMMARY_LABEL}\n{summary}\n\n[NEW USER MESSAGE]\n{text}", True

    async def handle_message(
        self, channel: Channel, message: IncomingMessage, thread_id: str
    ) -> None:
        """Process an inbound message: invoke agent, stream response, deliver via channel."""
        logger.info("Received message from chat %s: %s", message.chat_id, message.text[:80])

        try:
            from deepclaw.tools.cron import set_chat_context

            set_chat_context(channel.name, message.chat_id)
        except ImportError:
            pass

        thread_id, _auto_compaction = await self._maybe_auto_compact(message, thread_id)
        effective_text, used_pending_summary = self._build_effective_user_content(
            message.chat_id, message.text
        )

        await channel.send_typing(message.chat_id)
        msg_id = await self._send_redacted_message(channel, message.chat_id, THINKING_MESSAGE)

        accumulated = ""
        last_edit_time = time.monotonic()
        chars_since_edit = 0
        limit = channel.message_limit

        async def _stream_once(input_messages: list[dict]) -> bool:
            nonlocal accumulated, last_edit_time, chars_since_edit
            tool_calls_seen = False

            async for chunk in self.agent.astream(
                {"messages": input_messages},
                config={"configurable": {"thread_id": thread_id}},
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
                            await self._edit_redacted_message(
                                channel, message.chat_id, msg_id, display
                            )
                        last_edit_time = time.monotonic()
                        chars_since_edit = 0

            return tool_calls_seen

        try:
            tool_calls_seen = await _stream_once([{"role": "user", "content": effective_text}])
            if not tool_calls_seen and _looks_like_narration(accumulated):
                logger.info("Narration-without-tool-call detected — sending nudge")
                accumulated = ""
                last_edit_time = time.monotonic()
                chars_since_edit = 0
                await _stream_once([{"role": "user", "content": _NUDGE_MESSAGE}])
            if used_pending_summary:
                self._clear_pending_summary(message.chat_id)
        except Exception:
            logger.exception("Agent streaming failed")
            accumulated = accumulated or "Sorry, something went wrong processing your message."

        response_text = accumulated if accumulated else "(no response)"
        response_text = redact_secrets(response_text)
        chunks = chunk_message(response_text, limit)
        await self._edit_redacted_message(channel, message.chat_id, msg_id, chunks[0])
        for extra_chunk in chunks[1:]:
            await self._send_redacted_message(channel, message.chat_id, extra_chunk)
