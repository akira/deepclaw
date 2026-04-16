"""Shared message gateway that invokes the agent and streams responses back through channels.

This module is channel-agnostic — it depends only on the Channel ABC,
so adding Discord/Slack/etc. requires zero changes here.
"""

import logging
import time

from langchain_core.messages import ToolMessage

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
)
_NUDGE_MESSAGE = (
    "You described an action but did not call any tools. "
    "Please call the appropriate tool now to carry out what you described."
)


def _looks_like_narration(text: str) -> bool:
    """Return True if text describes a tool action without having called any tools."""
    lower = text.lower()
    has_opener = any(
        lower.lstrip().startswith(op) or f"\n{op}" in lower
        for op in _NARRATION_OPENERS
    )
    has_action = any(word in lower for word in _ACTION_WORDS)
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

    def __init__(self, agent, streaming_config):
        self.agent = agent
        self.streaming_config = streaming_config

    async def _edit_redacted_message(
        self, channel: Channel, chat_id: str, message_id: str, text: str
    ) -> None:
        """Redact secrets before editing a streamed message."""
        await channel.edit_message(chat_id, message_id, redact_secrets(text))

    async def _send_redacted_message(self, channel: Channel, chat_id: str, text: str) -> str:
        """Redact secrets before sending a message."""
        return await channel.send(chat_id, redact_secrets(text))

    async def handle_message(
        self, channel: Channel, message: IncomingMessage, thread_id: str
    ) -> None:
        """Process an inbound message: invoke agent, stream response, deliver via channel."""
        logger.info("Received message from chat %s: %s", message.chat_id, message.text[:80])

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

        async def _stream_once(input_messages: list[dict]) -> bool:
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
            tool_calls_seen = await _stream_once(
                [{"role": "user", "content": message.text}]
            )

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

        response_text = accumulated if accumulated else "(no response)"
        response_text = redact_secrets(response_text)

        # Final delivery: send complete text (possibly chunked)
        chunks = chunk_message(response_text, limit)
        await self._edit_redacted_message(channel, message.chat_id, msg_id, chunks[0])
        for extra_chunk in chunks[1:]:
            await self._send_redacted_message(channel, message.chat_id, extra_chunk)
