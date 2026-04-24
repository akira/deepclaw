"""Shared message gateway that invokes the agent and streams responses back through channels.

This module is channel-agnostic — it depends only on the Channel ABC,
so adding Discord/Slack/etc. requires zero changes here.
"""

from __future__ import annotations

import logging
import re
import time
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import ToolMessage

from deepclaw.auth import get_thread_state
from deepclaw.channels.base import Channel, IncomingMessage
from deepclaw.compaction import CompactionResult, get_thread_snapshot
from deepclaw.compaction import compact_thread as compact_thread_impl
from deepclaw.runtime_controller import (
    HARD_RECOVERY_PROMPT_MAX_CHARS,
    RuntimeRoute,
    ThreadSnapshotView,
    build_rebuild_prompt,
    choose_runtime_route,
    estimate_prompt_budget,
    truncate_reference_summary,
)
from deepclaw.runtime_hygiene import (
    bind_runtime_state,
    offload_text_if_oversized,
    offload_user_input,
)
from deepclaw.safety import redact_secrets

logger = logging.getLogger(__name__)

CURSOR_INDICATOR = "\u258c"
THINKING_MESSAGE = "Thinking..."
_REFERENCE_SUMMARY_LABEL = "[DeepClaw historical handoff — reference only]"

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
# Pre-compiled regex for whole-word action matching — avoids substring false positives
# such as "use" in "because", "get" in "forget", "list" in "listen".
_ACTION_RE = re.compile(r"\b(" + "|".join(_ACTION_WORDS) + r")\b")
_NUDGE_MESSAGE = (
    "You described an action but did not call any tools. "
    "Please call the appropriate tool now to carry out what you described."
)


class PreparedRuntimeInput(tuple):
    __slots__ = ()

    @property
    def thread_id(self) -> str:
        return self[0]

    @property
    def content(self) -> str:
        return self[1]

    @property
    def injected_summary(self) -> bool:
        return self[2]

    @property
    def route(self) -> RuntimeRoute:
        return self[3]

    @classmethod
    def create(
        cls,
        *,
        thread_id: str,
        content: str,
        injected_summary: bool,
        route: RuntimeRoute,
    ) -> PreparedRuntimeInput:
        return cls((thread_id, content, injected_summary, route))


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
        # Try to split at last newline within limit
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


class Gateway:
    """Shared message handler that invokes the agent and streams responses back through channels."""

    def __init__(self, agent, streaming_config, *, checkpointer=None, thread_state_store=None):
        self.agent = agent
        self.streaming_config = streaming_config
        self.checkpointer = checkpointer
        self.thread_state_store = thread_state_store

    def _persist_thread_store(self) -> None:
        if self.thread_state_store is None:
            return
        from deepclaw.auth import save_thread_ids

        save_thread_ids(self.thread_state_store)

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

    def _build_effective_user_content(self, chat_id: str, text: str) -> tuple[str, bool]:
        summary = self._peek_pending_summary(chat_id)
        if not summary:
            return text, False
        return f"{_REFERENCE_SUMMARY_LABEL}\n{summary}\n\n[NEW USER MESSAGE]\n{text}", True

    async def _prepare_runtime_input(
        self,
        *,
        chat_id: str,
        thread_id: str,
        text: str,
    ) -> PreparedRuntimeInput:
        thread_state = (
            get_thread_state(self.thread_state_store, chat_id)
            if self.thread_state_store is not None
            else None
        )
        pending_summary = (
            str(thread_state.get("pending_summary_text") or "") if thread_state else ""
        )
        snapshot = None
        if self.checkpointer is not None:
            raw_snapshot = await get_thread_snapshot(self.checkpointer, thread_id)
            snapshot = ThreadSnapshotView(
                messages=raw_snapshot.messages,
                working_state=raw_snapshot.working_state,
                continuity_checkpoint=raw_snapshot.continuity_checkpoint,
            )

        estimate = estimate_prompt_budget(
            snapshot=snapshot,
            user_text=text,
            pending_summary=pending_summary,
        )
        plan = choose_runtime_route(
            estimate,
            can_compact=self.checkpointer is not None,
        )
        logger.info(
            "Runtime route %s for chat %s thread %s (estimated=%d chars): %s",
            plan.route.value,
            chat_id,
            thread_id,
            estimate.estimated_total_chars,
            plan.reason,
        )

        if plan.route is RuntimeRoute.FITS:
            effective_text, injected_summary = self._build_effective_user_content(chat_id, text)
            return PreparedRuntimeInput.create(
                thread_id=thread_id,
                content=effective_text,
                injected_summary=injected_summary,
                route=plan.route,
            )

        if plan.route is RuntimeRoute.TRUNCATE_ARTIFACTS_ONLY and pending_summary:
            truncated_summary = truncate_reference_summary(pending_summary)
            return PreparedRuntimeInput.create(
                thread_id=thread_id,
                content=(
                    f"{_REFERENCE_SUMMARY_LABEL}\n{truncated_summary}\n\n[NEW USER MESSAGE]\n{text}"
                ),
                injected_summary=True,
                route=plan.route,
            )

        compacted = None
        if plan.route in {
            RuntimeRoute.COMPACT_THEN_CONTINUE,
            RuntimeRoute.REBUILD_FROM_STATE,
            RuntimeRoute.HARD_OVERFLOW_RECOVERY,
        }:
            compacted = await self.compact_thread(chat_id, thread_id, reason="runtime_budget")
            if compacted is not None:
                thread_id = compacted.new_thread_id
                if self.thread_state_store is not None:
                    thread_state = get_thread_state(self.thread_state_store, chat_id)

        if plan.route is RuntimeRoute.COMPACT_THEN_CONTINUE and compacted is not None:
            return PreparedRuntimeInput.create(
                thread_id=thread_id,
                content=(
                    f"{_REFERENCE_SUMMARY_LABEL}\n{compacted.summary_text}\n\n"
                    f"[NEW USER MESSAGE]\n{text}"
                ),
                injected_summary=True,
                route=plan.route,
            )

        recovery_route = (
            plan.route if compacted is not None else RuntimeRoute.HARD_OVERFLOW_RECOVERY
        )
        rebuild_prompt = build_rebuild_prompt(
            user_text=text,
            snapshot=snapshot,
            thread_state=thread_state,
            handoff_summary=compacted.summary_text if compacted is not None else pending_summary,
            route=recovery_route,
        )

        if plan.route is RuntimeRoute.REBUILD_FROM_STATE and compacted is not None:
            return PreparedRuntimeInput.create(
                thread_id=thread_id,
                content=rebuild_prompt,
                injected_summary=True,
                route=plan.route,
            )

        recovery_prompt, artifact = offload_text_if_oversized(
            rebuild_prompt,
            category="context-recovery",
            kind="runtime-recovery",
            label="runtime recovery prompt",
            max_chars=HARD_RECOVERY_PROMPT_MAX_CHARS,
            thread_id=thread_id,
            metadata={"chat_id": chat_id, "route": recovery_route.value},
        )
        if artifact is not None:
            logger.info("Offloaded hard-overflow recovery prompt to %s", artifact.path)
        return PreparedRuntimeInput.create(
            thread_id=thread_id,
            content=recovery_prompt,
            injected_summary=bool(compacted is not None or pending_summary),
            route=RuntimeRoute.HARD_OVERFLOW_RECOVERY,
        )

    async def compact_thread(
        self, chat_id: str, thread_id: str, *, reason: str, source: str = "telegram"
    ) -> CompactionResult | None:
        """Compact the current thread and rotate chat metadata to a fresh thread id."""
        result = await compact_thread_impl(
            self.checkpointer,
            thread_id=thread_id,
            chat_id=chat_id,
            reason=reason,
            source=source,
        )
        if result is None or self.thread_state_store is None:
            return result

        state = get_thread_state(self.thread_state_store, chat_id)
        state["current_thread_id"] = result.new_thread_id
        state["parent_thread_id"] = result.old_thread_id
        state["pending_summary_text"] = result.summary_text
        state["summary_artifact_path"] = result.summary_artifact_path
        state["checkpoint_artifact_path"] = result.checkpoint_artifact_path
        state["raw_history_artifact_paths"] = deepcopy(result.raw_history_artifact_paths)
        state["last_compacted_at"] = datetime.now(UTC).isoformat()
        state["last_compaction_reason"] = result.reason
        self._persist_thread_store()
        return result

    async def handle_message(
        self, channel: Channel, message: IncomingMessage, thread_id: str
    ) -> None:
        """Process an inbound message: invoke agent, stream response, deliver via channel."""
        logger.info("Received message from chat %s: %s", message.chat_id, message.text[:80])

        with bind_runtime_state(thread_id):
            # Set chat context so tools (e.g., cron) know where to deliver results
            try:
                from deepclaw.tools.cron import set_chat_context

                set_chat_context(channel.name, message.chat_id)
            except ImportError:
                pass

            await channel.send_typing(message.chat_id)

            msg_id = await self._send_redacted_message(channel, message.chat_id, THINKING_MESSAGE)

            accumulated = ""
            last_edit_time = time.monotonic()
            chars_since_edit = 0
            limit = channel.message_limit

            async def _stream_once(input_messages: list[dict[str, Any]]) -> bool:
                """Stream one agent turn. Returns True if any tool calls were seen."""
                nonlocal accumulated, last_edit_time, chars_since_edit
                tool_calls_seen = False

                async for chunk in self.agent.astream(
                    {"messages": input_messages},
                    config={"configurable": {"thread_id": thread_id}},
                    stream_mode="messages",
                ):
                    # astream with stream_mode="messages" yields (message, metadata) tuples
                    if not isinstance(chunk, tuple) or len(chunk) != 2:
                        continue
                    message_obj, _metadata = chunk

                    # Log tool results from ToolMessages
                    if isinstance(message_obj, ToolMessage):
                        tool_name = getattr(message_obj, "name", "unknown")
                        content = message_obj.content
                        preview = redact_secrets(str(content)[:200]) if content else "(empty)"
                        logger.info("Tool result [%s]: %s", tool_name, preview)
                        continue

                    # Only process AI messages with content_blocks
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
                                args_preview = (
                                    redact_secrets(str(tool_args)[:200]) if tool_args else ""
                                )
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

            injected_summary = False
            try:
                prepared = await self._prepare_runtime_input(
                    chat_id=message.chat_id,
                    thread_id=thread_id,
                    text=message.text,
                )
                thread_id = prepared.thread_id
                injected_summary = prepared.injected_summary
                user_content, artifact = offload_user_input(prepared.content, thread_id=thread_id)
                if artifact is not None:
                    logger.info("Offloaded oversized user input to %s", artifact.path)
                tool_calls_seen = await _stream_once([{"role": "user", "content": user_content}])

                # If the model described an action without calling any tools, nudge it once.
                if not tool_calls_seen and _looks_like_narration(accumulated):
                    logger.info("Narration-without-tool-call detected — sending nudge")
                    accumulated = ""
                    last_edit_time = time.monotonic()
                    chars_since_edit = 0
                    await _stream_once([{"role": "user", "content": _NUDGE_MESSAGE}])
            except Exception:
                logger.exception("Agent streaming failed")
                accumulated = accumulated or "Sorry, something went wrong processing your message."

            if injected_summary and accumulated:
                self._clear_pending_summary(message.chat_id)

            response_text = accumulated if accumulated else "(no response)"
            response_text = redact_secrets(response_text)

            # Final delivery: send complete text (possibly chunked)
            chunks = chunk_message(response_text, limit)
            await self._edit_redacted_message(channel, message.chat_id, msg_id, chunks[0])
            for extra_chunk in chunks[1:]:
                await self._send_redacted_message(channel, message.chat_id, extra_chunk)
