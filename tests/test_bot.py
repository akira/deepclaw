"""Tests for deepclaw bot modules (auth, channels.telegram, gateway)."""

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError
from telegram import InlineKeyboardMarkup

from deepclaw.auth import is_user_allowed
from deepclaw.channels.telegram import (
    ACTIVE_RUNS_KEY,
    ALLOWED_USERS_KEY,
    CONFIG_KEY,
    GATEWAY_KEY,
    LAST_MESSAGE_KEY,
    MODEL_OVERRIDE_KEY,
    PENDING_APPROVALS_KEY,
    TELEGRAM_MESSAGE_LIMIT,
    THREAD_IDS_KEY,
    TelegramBotChannel,
    TelegramChannel,
    _build_incoming_text,
    _format_remote_skills,
    _format_skills_list,
    _looks_like_supported_image,
    _parse_skills_command,
    _pending_approval_markup,
    _validate_model,
    authorize_chat,
    cmd_approval_callback,
    cmd_approve,
    cmd_clear,
    cmd_context,
    cmd_deny,
    cmd_memory,
    cmd_model,
    cmd_queue,
    cmd_resume,
    cmd_retry,
    cmd_sessions,
    cmd_skills,
    cmd_soul,
    cmd_status,
    cmd_stop,
    cmd_uptime,
    get_thread_id,
    handle_message,
)
from deepclaw.config import DeepClawConfig
from deepclaw.gateway import (
    CURSOR_INDICATOR,
    Gateway,
    _append_progress_history,
    _append_progress_line,
    _extract_message_tool_calls,
    _format_tool_progress_line,
    _looks_like_false_completion,
    _looks_like_memory_request,
    _looks_like_narration,
    _resolve_tool_args,
    chunk_message,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_update(user_id: int = 1, username: str | None = None):
    """Build a minimal mock Update with effective_user set."""
    user = SimpleNamespace(id=user_id, username=username)
    update = MagicMock()
    update.effective_user = user
    return update


def _make_context(thread_ids: dict | None = None):
    """Build a minimal mock context with bot_data."""
    ctx = MagicMock()
    bot_data: dict = {}
    if thread_ids is not None:
        bot_data[THREAD_IDS_KEY] = dict(thread_ids)
    ctx.bot_data = bot_data
    return ctx


# ---------------------------------------------------------------------------
# chunk_message
# ---------------------------------------------------------------------------


class TestChunkMessage:
    def test_empty_string(self):
        assert chunk_message("") == [""]

    def test_short_string(self):
        assert chunk_message("hello") == ["hello"]

    def test_exact_limit(self):
        text = "a" * TELEGRAM_MESSAGE_LIMIT
        assert chunk_message(text) == [text]

    def test_over_limit_splits(self):
        text = "a" * (TELEGRAM_MESSAGE_LIMIT + 10)
        chunks = chunk_message(text)
        assert len(chunks) == 2
        assert "".join(chunks) == text

    def test_split_at_newline(self):
        line1 = "a" * (TELEGRAM_MESSAGE_LIMIT - 10)
        line2 = "b" * 20
        text = f"{line1}\n{line2}"
        chunks = chunk_message(text)
        assert len(chunks) == 2
        assert chunks[0] == line1
        assert chunks[1] == line2

    def test_no_newline_splits_at_limit(self):
        text = "x" * (TELEGRAM_MESSAGE_LIMIT * 2 + 5)
        chunks = chunk_message(text)
        assert len(chunks) == 3
        assert chunks[0] == "x" * TELEGRAM_MESSAGE_LIMIT
        assert chunks[1] == "x" * TELEGRAM_MESSAGE_LIMIT
        assert chunks[2] == "x" * 5

    @pytest.mark.parametrize("limit", [10, 50, 100])
    def test_custom_limit(self, limit):
        text = "a" * (limit + 5)
        chunks = chunk_message(text, limit=limit)
        assert len(chunks) == 2
        assert chunks[0] == "a" * limit
        assert chunks[1] == "a" * 5


# ---------------------------------------------------------------------------
# get_thread_id
# ---------------------------------------------------------------------------


class TestGetThreadId:
    def test_missing_mapping_generates_and_persists_uuid(self, monkeypatch):
        ctx = _make_context()
        saved: dict[str, str] = {}

        def _save(thread_ids):
            saved.update(thread_ids)

        monkeypatch.setattr(
            "deepclaw.channels.telegram.save_thread_ids",
            _save,
        )
        monkeypatch.setattr("deepclaw.channels.telegram.uuid.uuid4", lambda: "generated-uuid")

        thread_id = get_thread_id(ctx, "12345")

        assert thread_id == "generated-uuid"
        assert ctx.bot_data[THREAD_IDS_KEY]["12345"] == "generated-uuid"
        assert saved == {"12345": "generated-uuid"}

    def test_missing_mapping_logs_warning(self, monkeypatch):
        ctx = _make_context()
        monkeypatch.setattr("deepclaw.channels.telegram.save_thread_ids", lambda _thread_ids: None)
        monkeypatch.setattr("deepclaw.channels.telegram.uuid.uuid4", lambda: "generated-uuid")
        warnings: list[str] = []

        def _capture(message, *args, **_kwargs):
            warnings.append(message % args if args else message)

        monkeypatch.setattr("deepclaw.channels.telegram.logger.warning", _capture)

        thread_id = get_thread_id(ctx, "12345")

        assert thread_id == "generated-uuid"
        assert warnings == [
            "Missing thread ID mapping for chat 12345; generated replacement thread generated-uuid"
        ]

    def test_returns_custom_thread_id(self):
        custom_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        ctx = _make_context(thread_ids={"12345": custom_id})
        assert get_thread_id(ctx, "12345") == custom_id

    def test_other_chat_unaffected(self):
        ctx = _make_context(thread_ids={"12345": "custom-id"})
        assert get_thread_id(ctx, "12345") == "custom-id"


# ---------------------------------------------------------------------------
# is_user_allowed
# ---------------------------------------------------------------------------


class TestIsUserAllowed:
    def test_locked_mode_empty_set(self):
        """Empty allowlist means no users are paired — deny all."""
        update = _make_update(user_id=1)
        assert is_user_allowed(update, set()) is False

    def test_allowed_by_id(self):
        update = _make_update(user_id=42)
        assert is_user_allowed(update, {"42"}) is True

    def test_allowed_by_username(self):
        update = _make_update(user_id=1, username="alice")
        assert is_user_allowed(update, {"alice"}) is True

    def test_rejected_user(self):
        update = _make_update(user_id=999, username="eve")
        assert is_user_allowed(update, {"42", "alice"}) is False

    def test_no_effective_user(self):
        update = MagicMock()
        update.effective_user = None
        assert is_user_allowed(update, {"42"}) is False

    def test_user_without_username_rejected(self):
        update = _make_update(user_id=1, username=None)
        assert is_user_allowed(update, {"alice"}) is False


# ---------------------------------------------------------------------------
# authorize_chat — private chat enforcement
# ---------------------------------------------------------------------------


def _make_update_with_chat(chat_type: str, user_id: int = 1):
    """Build a mock Update with effective_chat.type set."""
    update = _make_update(user_id=user_id)
    update.effective_chat = SimpleNamespace(id=123, type=chat_type)
    return update


class TestAuthorizeChat:
    def test_private_chat_allowed(self):
        update = _make_update_with_chat("private")
        assert authorize_chat(update) is True

    def test_group_chat_rejected(self):
        update = _make_update_with_chat("group")
        assert authorize_chat(update) is False

    def test_supergroup_chat_rejected(self):
        update = _make_update_with_chat("supergroup")
        assert authorize_chat(update) is False

    def test_channel_rejected(self):
        update = _make_update_with_chat("channel")
        assert authorize_chat(update) is False

    def test_no_effective_chat(self):
        update = MagicMock()
        update.effective_chat = None
        assert authorize_chat(update) is False


class _FakeStreamingChannel:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []
        self.edits: list[tuple[str, str, str]] = []

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def message_limit(self) -> int:
        return TELEGRAM_MESSAGE_LIMIT

    async def send_typing(self, chat_id: str) -> None:
        return None

    async def send(self, chat_id: str, text: str) -> str:
        self.sent.append((chat_id, text))
        return "msg-1"

    async def edit_message(self, chat_id: str, message_id: str, text: str) -> None:
        self.edits.append((chat_id, message_id, text))


class TestTelegramBotChannel:
    @pytest.mark.asyncio
    async def test_send_short_message(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=100))

        channel = TelegramBotChannel(bot)

        message_id = await channel.send("123", "hello")

        bot.send_message.assert_awaited_once_with(chat_id=123, text="hello")
        assert message_id == "100"

    @pytest.mark.asyncio
    async def test_send_chunks_overlong_messages(self):
        text = "a" * (TELEGRAM_MESSAGE_LIMIT + 5)
        bot = MagicMock()
        bot.send_message = AsyncMock(
            side_effect=[
                SimpleNamespace(message_id=101),
                SimpleNamespace(message_id=102),
            ]
        )

        channel = TelegramBotChannel(bot)

        message_id = await channel.send("123", text)

        expected_chunks = chunk_message(text, TELEGRAM_MESSAGE_LIMIT)
        assert bot.send_message.await_count == 2
        assert [call.kwargs for call in bot.send_message.await_args_list] == [
            {"chat_id": 123, "text": chunk} for chunk in expected_chunks
        ]
        assert message_id == "101"


class TestTelegramChannel:
    @pytest.mark.asyncio
    async def test_send_uses_reply_text_for_message_updates(self):
        update = _make_slash_update(text="hello")
        ctx = _make_slash_context()
        channel = TelegramChannel(update, ctx)

        message_id = await channel.send("1", "Thinking...")

        update.message.reply_text.assert_awaited_once_with("Thinking...")
        assert message_id == "200"

    @pytest.mark.asyncio
    async def test_send_falls_back_to_callback_message_reply_text(self):
        update = _make_callback_update(data="safety:approve")
        ctx = _make_slash_context()
        channel = TelegramChannel(update, ctx)

        message_id = await channel.send("1", "Thinking...")

        update.callback_query.message.reply_text.assert_awaited_once_with("Thinking...")
        assert message_id == "201"


class _FakeStreamingAgent:
    def __init__(self, chunks, interrupts=()):
        self._chunks = chunks
        self._interrupts = interrupts
        self.astream_calls = []
        self.aget_state_calls = []

    async def astream(self, *args, **kwargs):
        self.astream_calls.append((args, kwargs))
        for chunk in self._chunks:
            yield chunk

    async def aget_state(self, config):
        self.aget_state_calls.append(config)
        return SimpleNamespace(interrupts=self._interrupts)


class TestGatewayProgressFormatting:
    def test_format_tool_progress_line_uses_icons_and_summaries(self):
        assert _format_tool_progress_line("skill_view", {"name": "systematic-debugging"}) == (
            '📚 skill_view: "systematic-debugging"'
        )
        assert _format_tool_progress_line("todo", {"todos": [{}, {}, {}]}) == (
            '📋 todo: "planning 3 task(s)"'
        )
        assert _format_tool_progress_line("patch", {"path": "/tmp/file.py"}) == (
            '🔧 patch: "/tmp/file.py"'
        )

    def test_append_progress_line_collapses_duplicates(self):
        accumulated, delta1 = _append_progress_line("", '🔧 patch: "/tmp/file.py"')
        accumulated, delta2 = _append_progress_line(accumulated, '🔧 patch: "/tmp/file.py"')
        accumulated, delta3 = _append_progress_line(accumulated, '🔧 patch: "/tmp/file.py"')

        assert accumulated == '🔧 patch: "/tmp/file.py" (×3)'
        assert delta1 > 0
        assert delta2 > 0
        assert delta3 >= 0

    def test_append_progress_history_collapses_duplicates(self):
        history = _append_progress_history([], '🔧 patch: "/tmp/file.py"')
        history = _append_progress_history(history, '🔧 patch: "/tmp/file.py"')
        history = _append_progress_history(history, '🔧 patch: "/tmp/file.py"')

        assert history == ['🔧 patch: "/tmp/file.py" (×3)']

    def test_resolve_tool_args_falls_back_to_message_tool_calls(self):
        block = {"type": "tool_call", "id": "call-1", "name": "execute", "args": {}}
        tool_calls = [
            {
                "id": "call-1",
                "name": "execute",
                "args": {"command": "sudo apt-get install ripgrep"},
            }
        ]

        assert _resolve_tool_args(block, "execute", tool_calls) == {
            "command": "sudo apt-get install ripgrep"
        }

    def test_resolve_tool_args_parses_json_from_tool_call_chunks(self):
        block = {"type": "tool_call_chunk", "id": "call-1", "name": "execute", "args": ""}
        tool_calls = [
            {
                "id": "call-1",
                "name": "execute",
                "args": '{"command": "sudo apt-get install ripgrep"}',
                "index": 0,
            }
        ]

        assert _resolve_tool_args(block, "execute", tool_calls) == {
            "command": "sudo apt-get install ripgrep"
        }

    def test_extract_message_tool_calls_includes_tool_call_chunks(self):
        message = SimpleNamespace(
            tool_calls=[{"id": "call-1", "name": "read_file", "args": {}}],
            tool_call_chunks=[
                SimpleNamespace(
                    id="call-1",
                    name="read_file",
                    args='{"path": "/home/ubuntu/deepclaw/README.md"}',
                    index=0,
                )
            ],
        )

        assert _extract_message_tool_calls(message) == [
            {"id": "call-1", "name": "read_file", "args": {}},
            {
                "id": "call-1",
                "name": "read_file",
                "args": '{"path": "/home/ubuntu/deepclaw/README.md"}',
                "index": 0,
            },
        ]

    def test_resolve_tool_args_parses_json_string_from_tool_call_chunks(self):
        block = {"type": "tool_call_chunk", "id": "call-1", "name": "read_file", "args": ""}
        tool_calls = [
            {
                "id": "call-1",
                "name": "read_file",
                "args": '{"path": "/home/ubuntu/deepclaw/README.md"}',
            }
        ]

        assert _resolve_tool_args(block, "read_file", tool_calls) == {
            "path": "/home/ubuntu/deepclaw/README.md"
        }


class TestGatewayRedaction:
    @pytest.mark.asyncio
    async def test_streaming_redacts_before_edit_and_final_send(self):
        secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
        agent = _FakeStreamingAgent(
            [
                (SimpleNamespace(content_blocks=[{"type": "text", "text": secret}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="show secret")

        await gateway.handle_message(channel, incoming, "thread-1")

        assert channel.sent == [("123", "Thinking...")]
        assert channel.edits
        assert all(secret not in text for _, _, text in channel.edits)
        assert any("[REDACTED]" in text for _, _, text in channel.edits)

    @pytest.mark.asyncio
    async def test_gateway_passes_recursion_limit_from_max_turns(self):
        agent = _FakeStreamingAgent(
            [(SimpleNamespace(content_blocks=[{"type": "text", "text": "Done."}]), {})]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming, max_turns=7)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="do thing")

        await gateway.handle_message(channel, incoming, "thread-1")

        config = agent.astream_calls[0][1]["config"]
        assert config["recursion_limit"] == 7

    @pytest.mark.asyncio
    async def test_gateway_reports_turn_limit_when_graph_hits_recursion_limit(self):
        class _RecursionAgent:
            async def astream(self, *_args, **_kwargs):
                raise GraphRecursionError("recursion boom")
                yield  # pragma: no cover

            async def aget_state(self, _config):
                return SimpleNamespace(interrupts=())

        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=_RecursionAgent(), streaming_config=streaming, max_turns=5)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="loop forever")

        await gateway.handle_message(channel, incoming, "thread-1")

        final_text = channel.edits[-1][2]
        assert "maximum turn limit (5)" in final_text
        assert "/clear to start fresh" in final_text

    @pytest.mark.asyncio
    async def test_gateway_reports_inactivity_timeout(self):
        class _SlowAgent:
            async def astream(self, *_args, **_kwargs):
                await asyncio.sleep(0.05)
                yield (SimpleNamespace(content_blocks=[{"type": "text", "text": "late"}]), {})

            async def aget_state(self, _config):
                return SimpleNamespace(interrupts=())

        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(
            agent=_SlowAgent(),
            streaming_config=streaming,
            gateway_timeout=0.01,
            gateway_timeout_warning=0,
        )
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="wait")

        await gateway.handle_message(channel, incoming, "thread-1")

        final_text = channel.edits[-1][2]
        assert "inactive for" in final_text
        assert "Try again, or use /clear to start fresh" in final_text

    @pytest.mark.asyncio
    async def test_gateway_sends_inactivity_warning_before_timeout(self):
        class _SlowAgent:
            async def astream(self, *_args, **_kwargs):
                await asyncio.sleep(0.03)
                yield (SimpleNamespace(content_blocks=[{"type": "text", "text": "late"}]), {})

            async def aget_state(self, _config):
                return SimpleNamespace(interrupts=())

        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(
            agent=_SlowAgent(),
            streaming_config=streaming,
            gateway_timeout=0.02,
            gateway_timeout_warning=0.01,
        )
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="wait")

        await gateway.handle_message(channel, incoming, "thread-1")

        assert any("No activity for" in text for _, text in channel.sent)
        assert "inactive for" in channel.edits[-1][2]

    @pytest.mark.asyncio
    async def test_tool_logging_redacts_sensitive_args_and_results(self, monkeypatch):
        secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"

        agent = _FakeStreamingAgent(
            [
                (
                    SimpleNamespace(
                        content_blocks=[
                            {"type": "tool_call", "name": "read_file", "args": {"token": secret}},
                            {"type": "text", "text": "done"},
                        ]
                    ),
                    {},
                ),
                (
                    ToolMessage(content=f"token={secret}", name="read_file", tool_call_id="call-1"),
                    {},
                ),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="show secret")
        logged_messages: list[str] = []

        def _capture_log(msg, *args, **_kwargs):
            logged_messages.append(msg % args if args else msg)

        monkeypatch.setattr("deepclaw.gateway.logger.info", _capture_log)

        await gateway.handle_message(channel, incoming, "thread-1")

        assert logged_messages
        assert all(secret not in entry for entry in logged_messages)
        assert any("[REDACTED]" in entry for entry in logged_messages if "Tool" in entry)
        assert all(
            CURSOR_INDICATOR not in text or "[REDACTED]" in text for _, _, text in channel.edits
        )

    @pytest.mark.asyncio
    async def test_streaming_status_updates_show_formatted_tool_progress(self):
        agent = _FakeStreamingAgent(
            [
                (
                    SimpleNamespace(
                        content_blocks=[
                            {
                                "type": "tool_call",
                                "name": "skill_view",
                                "args": {"name": "systematic-debugging"},
                            },
                            {
                                "type": "tool_call",
                                "name": "skill_view",
                                "args": {"name": "deepclaw-development"},
                            },
                            {
                                "type": "tool_call",
                                "name": "terminal",
                                "args": {
                                    "command": "TRACE_ID='20260426T195032456878Z019dcb57-f508-7650-8e34-5ad6d9612d75' journalctl --user -u deepclaw.service --no-pager"
                                },
                            },
                            {
                                "type": "tool_call",
                                "name": "patch",
                                "args": {"path": "/tmp/file.py"},
                            },
                            {
                                "type": "tool_call",
                                "name": "patch",
                                "args": {"path": "/tmp/file.py"},
                            },
                        ]
                    ),
                    {},
                ),
                (SimpleNamespace(content_blocks=[{"type": "text", "text": "Done."}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="show progress")

        await gateway.handle_message(channel, incoming, "thread-1")

        edits_text = "\n".join(text for _, _, text in channel.edits)
        assert '📚 skill_view: "systematic-debugging"' in edits_text
        assert '📚 skill_view: "deepclaw-development"' in edits_text
        assert '💻 terminal: "TRACE_ID=' in edits_text
        assert '🔧 patch: "/tmp/file.py" (×2)' in edits_text

        snapshot = gateway.get_queue_snapshot("123")
        assert snapshot is not None
        assert snapshot["status"] == "completed"
        assert snapshot["user_text_preview"] == "show progress"
        assert '🔧 patch: "/tmp/file.py" (×2)' in snapshot["progress_lines"]

    @pytest.mark.asyncio
    async def test_streaming_queue_snapshot_redacts_sensitive_previews(self):
        secret = "sk-ant-secretvalue1234567890abcd"
        agent = _FakeStreamingAgent(
            [
                (
                    SimpleNamespace(
                        content_blocks=[
                            {
                                "type": "tool_call",
                                "name": "execute",
                                "args": {"command": f"export OPENAI_API_KEY={secret}"},
                            },
                        ]
                    ),
                    {},
                ),
                (SimpleNamespace(content_blocks=[{"type": "text", "text": f"done {secret}"}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text=f"use key {secret}")

        await gateway.handle_message(channel, incoming, "thread-1")

        snapshot = gateway.get_queue_snapshot("123")
        assert snapshot is not None
        assert secret not in snapshot["user_text_preview"]
        assert all(secret not in line for line in snapshot["progress_lines"])
        assert secret not in (snapshot["final_text_preview"] or "")

    @pytest.mark.asyncio
    async def test_streaming_preserves_error_queue_status(self, monkeypatch):
        class _RaisingAgent:
            async def astream(self, *_args, **_kwargs):
                raise RuntimeError("boom")
                yield  # pragma: no cover

        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=_RaisingAgent(), streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="trigger failure")
        monkeypatch.setattr(gateway, "_get_pending_interrupt", AsyncMock(return_value=None))

        await gateway.handle_message(channel, incoming, "thread-1")

        snapshot = gateway.get_queue_snapshot("123")
        assert snapshot is not None
        assert snapshot["status"] == "error"

    @pytest.mark.asyncio
    async def test_streaming_shows_resolved_chunk_progress_without_duplicate_final_call(self):
        agent = _FakeStreamingAgent(
            [
                (
                    SimpleNamespace(
                        tool_calls=[
                            {
                                "id": "call-1",
                                "name": "execute",
                                "args": {"command": "sudo apt-get install ripgrep"},
                            }
                        ],
                        content_blocks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-1",
                                "name": "execute",
                                "args": {},
                            },
                            {
                                "type": "tool_call",
                                "id": "call-1",
                                "name": "execute",
                                "args": {},
                            },
                        ],
                    ),
                    {},
                ),
                (SimpleNamespace(content_blocks=[{"type": "text", "text": "Done."}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="install ripgrep")

        await gateway.handle_message(channel, incoming, "thread-1")

        edits_text = "\n".join(text for _, _, text in channel.edits)
        assert '💻 execute: "sudo apt-get install ripgrep"' in edits_text
        assert "💻 execute (×2)" not in edits_text
        assert "\n💻 execute\n" not in edits_text

        snapshot = gateway.get_queue_snapshot("123")
        assert snapshot is not None
        assert snapshot["progress_lines"] == ['💻 execute: "sudo apt-get install ripgrep"']

    @pytest.mark.asyncio
    async def test_streaming_shows_chunk_progress_even_without_final_tool_call(self):
        agent = _FakeStreamingAgent(
            [
                (
                    SimpleNamespace(
                        tool_calls=[
                            {
                                "id": "call-1",
                                "name": "execute",
                                "args": {"command": "sudo apt-get install ripgrep"},
                            }
                        ],
                        content_blocks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-1",
                                "name": "execute",
                                "args": {},
                            },
                        ],
                    ),
                    {},
                ),
                (SimpleNamespace(content_blocks=[{"type": "text", "text": "Done."}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="install ripgrep")

        await gateway.handle_message(channel, incoming, "thread-1")

        edits_text = "\n".join(text for _, _, text in channel.edits)
        assert '💻 execute: "sudo apt-get install ripgrep"' in edits_text

        snapshot = gateway.get_queue_snapshot("123")
        assert snapshot is not None
        assert snapshot["progress_lines"] == ['💻 execute: "sudo apt-get install ripgrep"']

    @pytest.mark.asyncio
    async def test_streaming_shows_generic_tool_name_when_stream_has_no_args(self):
        agent = _FakeStreamingAgent(
            [
                (
                    SimpleNamespace(
                        tool_calls=[{"id": "call-1", "name": "execute", "args": {}}],
                        content_blocks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-1",
                                "name": "execute",
                                "args": "",
                            },
                            {
                                "type": "tool_call",
                                "id": "call-1",
                                "name": "execute",
                                "args": "",
                            },
                        ],
                    ),
                    {},
                ),
                (SimpleNamespace(content_blocks=[{"type": "text", "text": "Done."}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="run command")

        await gateway.handle_message(channel, incoming, "thread-1")

        edits_text = "\n".join(text for _, _, text in channel.edits)
        assert "💻 execute" in edits_text

        snapshot = gateway.get_queue_snapshot("123")
        assert snapshot is not None
        assert snapshot["progress_lines"] == ["💻 execute"]

    @pytest.mark.asyncio
    async def test_streaming_surfaces_child_subgraph_tool_progress(self):
        child_message = SimpleNamespace(
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "web_search",
                    "args": {"query": "site:sf.eater.com best restaurants"},
                }
            ],
            content_blocks=[
                {
                    "type": "tool_call_chunk",
                    "id": "call-1",
                    "name": "web_search",
                    "args": {},
                },
                {
                    "type": "text",
                    "text": "Let me search the web for the list.",
                },
            ],
        )
        agent = _FakeStreamingAgent(
            [
                (("task:child-1",), (child_message, {})),
                (SimpleNamespace(content_blocks=[{"type": "text", "text": "Done."}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="find restaurants")

        await gateway.handle_message(channel, incoming, "thread-1")

        edits_text = "\n".join(text for _, _, text in channel.edits)
        assert "🔧 task" not in edits_text
        assert '🔎 web_search: "site:sf.eater.com best restaurants"' in edits_text
        assert "Let me search the web for the list." not in edits_text
        assert agent.astream_calls[0][1]["subgraphs"] is True

        snapshot = gateway.get_queue_snapshot("123")
        assert snapshot is not None
        assert snapshot["progress_lines"] == ['🔎 web_search: "site:sf.eater.com best restaurants"']

    @pytest.mark.asyncio
    async def test_streaming_does_not_warn_for_empty_tool_call_chunks(self, monkeypatch):
        agent = _FakeStreamingAgent(
            [
                (
                    SimpleNamespace(
                        tool_calls=[{"id": "call-1", "name": "read_file", "args": {}}],
                        tool_call_chunks=[
                            SimpleNamespace(id="call-1", name="read_file", args="", index=0)
                        ],
                        content_blocks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-1",
                                "name": "read_file",
                                "args": "",
                                "index": 0,
                            }
                        ],
                    ),
                    {},
                ),
                (SimpleNamespace(content_blocks=[{"type": "text", "text": "Done."}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="read file")
        warnings: list[str] = []

        def _capture_warning(msg, *args, **_kwargs):
            warnings.append(msg % args if args else msg)

        monkeypatch.setattr("deepclaw.gateway.logger.warning", _capture_warning)

        await gateway.handle_message(channel, incoming, "thread-1")

        assert warnings == []

    @pytest.mark.asyncio
    async def test_streaming_inserts_newline_between_tool_progress_and_text(self):
        agent = _FakeStreamingAgent(
            [
                (
                    SimpleNamespace(
                        tool_calls=[{"id": "call-1", "name": "execute", "args": {}}],
                        content_blocks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-1",
                                "name": "execute",
                                "args": "",
                            }
                        ],
                    ),
                    {},
                ),
                (SimpleNamespace(content_blocks=[{"type": "text", "text": "Done."}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="run command")

        await gateway.handle_message(channel, incoming, "thread-1")

        final_text = channel.edits[-1][2]
        assert "💻 execute\nDone." in final_text

    @pytest.mark.asyncio
    async def test_streaming_skips_duplicate_edits_for_unchanged_progress_text(self):
        agent = _FakeStreamingAgent(
            [
                (
                    SimpleNamespace(
                        tool_calls=[{"id": "call-1", "name": "execute", "args": {}}],
                        content_blocks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-1",
                                "name": "execute",
                                "args": "",
                            }
                        ],
                    ),
                    {},
                ),
                (
                    SimpleNamespace(
                        tool_calls=[{"id": "call-1", "name": "execute", "args": {}}],
                        content_blocks=[
                            {
                                "type": "tool_call_chunk",
                                "id": "call-1",
                                "name": "execute",
                                "args": "",
                            }
                        ],
                    ),
                    {},
                ),
                (SimpleNamespace(content_blocks=[{"type": "text", "text": "Done."}]), {}),
            ]
        )
        streaming = SimpleNamespace(edit_interval=0.0, buffer_threshold=1)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="run command")

        await gateway.handle_message(channel, incoming, "thread-1")

        edit_texts = [text for _, _, text in channel.edits]
        assert edit_texts.count("💻 execute▌") == 1

    @pytest.mark.asyncio
    async def test_returns_pending_safety_review_when_graph_interrupts(self):
        interrupt = SimpleNamespace(
            id="interrupt-1",
            value={
                "type": "safety_review",
                "tool": "execute",
                "command": "python -c 'print(1)'",
                "warning": "Inline Python execution",
                "message": "⚠️ Potentially dangerous command\n\nApprove or deny?",
            },
        )
        agent = _FakeStreamingAgent([], interrupts=(interrupt,))
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="123", text="run eval")

        pending = await gateway.handle_message(channel, incoming, "thread-1")

        assert pending == {
            "id": "interrupt-1",
            "thread_id": "thread-1",
            "type": "safety_review",
            "tool": "execute",
            "command": "python -c 'print(1)'",
            "approval_keys": [],
            "warning": "Inline Python execution",
            "message": "⚠️ Potentially dangerous command\n\nApprove or deny?",
        }
        config = agent.astream_calls[0][1]["config"]
        assert config["configurable"] == {"thread_id": "thread-1"}
        assert config["metadata"]["active_thread_id"] == "thread-1"
        assert config["metadata"]["chat_id"] == "123"
        assert config["metadata"]["channel"] == "telegram"
        assert "updated_at" in config["metadata"]
        state_config = agent.aget_state_calls[0]
        assert state_config["configurable"] == {"thread_id": "thread-1"}
        assert state_config["metadata"]["active_thread_id"] == "thread-1"
        assert state_config["metadata"]["chat_id"] == "123"
        assert state_config["metadata"]["channel"] == "telegram"
        assert "updated_at" in state_config["metadata"]
        assert channel.edits[-1][2].endswith(
            "Use /approve, /approve session, or /deny <reason> to respond."
        )

    @pytest.mark.asyncio
    async def test_resume_interrupt_uses_command_resume_payload(self):
        agent = _FakeStreamingAgent(
            [(SimpleNamespace(content_blocks=[{"type": "text", "text": "Approved and ran."}]), {})]
        )
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()

        pending = await gateway.resume_interrupt(
            channel,
            chat_id="123",
            thread_id="thread-1",
            decision={"type": "approve"},
        )

        assert pending is None
        args, kwargs = agent.astream_calls[0]
        assert args[0].resume == {"type": "approve"}
        config = kwargs["config"]
        assert config["configurable"] == {"thread_id": "thread-1"}
        assert config["metadata"]["active_thread_id"] == "thread-1"
        assert config["metadata"]["chat_id"] == "123"
        assert config["metadata"]["channel"] == "telegram"
        assert "updated_at" in config["metadata"]
        assert channel.edits[-1][2] == "Approved and ran."


# ---------------------------------------------------------------------------
# Narration detection
# ---------------------------------------------------------------------------


class TestLooksLikeNarration:
    @pytest.mark.parametrize(
        "text",
        [
            "I'll check the file for you.",
            "Let me search for that.",
            "I will run the tests now.",
            "I'm going to read the directory.",
            "I need to find the relevant code.",
            "I should look at the logs.",
            "Now I'll execute the command.",
            "First I'll scan the codebase.",
            "Absolutely. I’ll update the skill with this learning.",
            "I’ll save this to the skill now.",
            "Let me edit the prompt and restart the service.",
        ],
    )
    def test_detects_narration(self, text):
        assert _looks_like_narration(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "The answer is 42.",
            "Here is the summary of the results.",
            "Done! The file has been created.",
            "The tests all passed.",
            "",
            "Sure, happy to help.",
            # Devin review: substring false positives — word-boundary matching required
            "I'll explain this because it's important.",  # "use" in "because"
            "I should clarify because there's a misunderstanding.",
            "Let me think about this because it's tricky.",
            "I'll forget about that.",  # "get" in "forget"
            "I'll listen to your explanation.",  # "list" in "listen"
            "Let me pause for a moment.",  # "use" in "pause"
        ],
    )
    def test_ignores_final_answers(self, text):
        assert _looks_like_narration(text) is False

    def test_multiline_narration_in_body(self):
        text = "Here is what I found.\n\nLet me now check the logs for errors."
        assert _looks_like_narration(text) is True

    def test_tool_call_present_does_not_affect_function(self):
        # _looks_like_narration is a pure text check; tool-call tracking is in Gateway
        text = "I'll use the search tool to find it."
        assert _looks_like_narration(text) is True


class TestLooksLikeFalseCompletion:
    @pytest.mark.parametrize(
        ("user_text", "assistant_text"),
        [
            (
                "Can you update your skills with learning on how to deal with redacted content?",
                "Done — I've incorporated that behavior into my working rules.",
            ),
            (
                "Please save this to your skill",
                "I updated the skill with this workflow.",
            ),
            (
                "Edit your prompt and restart",
                "Updated the prompt and restarted the service.",
            ),
        ],
    )
    def test_detects_false_completion_on_mutation_request(self, user_text, assistant_text):
        assert _looks_like_false_completion(user_text, assistant_text) is True

    @pytest.mark.parametrize(
        ("user_text", "assistant_text"),
        [
            ("What is the answer?", "Done — the answer is 42."),
            ("Explain how to handle redacted output", "Done — here is the explanation."),
            ("Can you update the skill?", "I can update the skill if you want."),
        ],
    )
    def test_ignores_non_mutations_or_non_completion_claims(self, user_text, assistant_text):
        assert _looks_like_false_completion(user_text, assistant_text) is False


class TestLooksLikeMemoryRequest:
    @pytest.mark.parametrize(
        ("user_text", "assistant_text"),
        [
            (
                "Can you remember that when I ask you to push the pr use deepclaw dev skill and also open a pr as well",
                'Got it — I’ll treat "push the PR" as using the deepclaw-development skill and opening the PR automatically.',
            ),
            (
                "From now on, prefer the deepclaw-development skill for PR pushes",
                "Absolutely — from now on I’ll prefer that workflow.",
            ),
            (
                "Please remember I prefer Telegram for reports",
                "Understood — I’ll remember that preference going forward.",
            ),
        ],
    )
    def test_detects_memory_request_ack_without_tools(self, user_text, assistant_text):
        assert _looks_like_memory_request(user_text, assistant_text) is True

    @pytest.mark.parametrize(
        ("user_text", "assistant_text"),
        [
            (
                "Can you remember that when I ask you to push the pr use deepclaw dev skill and also open a pr as well",
                "I saved that preference to memory.",
            ),
            ("What do you remember about me?", "You prefer Telegram for reports."),
            ("Prefer concise replies", "Okay."),
        ],
    )
    def test_ignores_non_memory_request_or_non_acknowledgement(self, user_text, assistant_text):
        assert _looks_like_memory_request(user_text, assistant_text) is False


# ---------------------------------------------------------------------------
# Narration nudge in Gateway
# ---------------------------------------------------------------------------


class _MultiCallStreamingAgent:
    """Agent that returns different chunk sequences on successive astream() calls."""

    def __init__(self, call_responses: list[list]):
        self._responses = iter(call_responses)
        self.call_count = 0

    async def astream(self, *_args, **_kwargs):
        self.call_count += 1
        chunks = next(self._responses)
        for chunk in chunks:
            yield chunk


class TestGatewayNarrationNudge:
    @pytest.mark.asyncio
    async def test_nudges_when_narration_and_no_tools(self):
        """When the model narrates without calling tools, the gateway retries with a nudge."""
        narration_chunk = (
            SimpleNamespace(
                content_blocks=[{"type": "text", "text": "I'll check the file for you."}]
            ),
            {},
        )
        recovery_chunk = (
            SimpleNamespace(
                content_blocks=[
                    {"type": "tool_call", "name": "read_file", "args": {"path": "/tmp/x"}},
                    {"type": "text", "text": "Done."},
                ]
            ),
            {},
        )
        agent = _MultiCallStreamingAgent([[narration_chunk], [recovery_chunk]])
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="42", text="check the file")

        await gateway.handle_message(channel, incoming, "thread-1")

        assert agent.call_count == 2

    @pytest.mark.asyncio
    async def test_no_nudge_when_tool_was_called(self):
        """When at least one tool fires, no nudge is sent even if the text has narration phrases."""
        chunk = (
            SimpleNamespace(
                content_blocks=[
                    {"type": "tool_call", "name": "read_file", "args": {}},
                    {"type": "text", "text": "I'll now check the output."},
                ]
            ),
            {},
        )
        agent = _MultiCallStreamingAgent([[chunk]])
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="42", text="read the file")

        await gateway.handle_message(channel, incoming, "thread-1")

        assert agent.call_count == 1

    @pytest.mark.asyncio
    async def test_no_nudge_for_plain_answer(self):
        """A plain text answer without narration patterns does not trigger a retry."""
        chunk = (
            SimpleNamespace(content_blocks=[{"type": "text", "text": "The answer is 42."}]),
            {},
        )
        agent = _MultiCallStreamingAgent([[chunk]])
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="42", text="what is the answer?")

        await gateway.handle_message(channel, incoming, "thread-1")

        assert agent.call_count == 1

    @pytest.mark.asyncio
    async def test_nudges_when_false_completion_and_no_tools(self):
        """When the model claims completion for a mutation request without tools, the gateway retries with a nudge."""
        false_completion_chunk = (
            SimpleNamespace(
                content_blocks=[
                    {
                        "type": "text",
                        "text": "Done — I've incorporated that behavior into my working rules.",
                    }
                ]
            ),
            {},
        )
        recovery_chunk = (
            SimpleNamespace(
                content_blocks=[
                    {"type": "tool_call", "name": "edit_file", "args": {"file_path": "/tmp/x"}},
                    {"type": "text", "text": "Done."},
                ]
            ),
            {},
        )
        agent = _MultiCallStreamingAgent([[false_completion_chunk], [recovery_chunk]])
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(
            chat_id="42",
            text="Can you update your skills with learning on how to deal with redacted content?",
        )

        await gateway.handle_message(channel, incoming, "thread-1")

        assert agent.call_count == 2

    @pytest.mark.asyncio
    async def test_nudge_response_is_delivered(self):
        """After a nudge, the recovery response text is what gets sent to the channel."""
        narration_chunk = (
            SimpleNamespace(
                content_blocks=[{"type": "text", "text": "I will search for the answer."}]
            ),
            {},
        )
        recovery_chunk = (
            SimpleNamespace(content_blocks=[{"type": "text", "text": "Found it: 42."}]),
            {},
        )
        agent = _MultiCallStreamingAgent([[narration_chunk], [recovery_chunk]])
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(chat_id="42", text="find the answer")

        await gateway.handle_message(channel, incoming, "thread-1")

        final_edit = channel.edits[-1][2]
        assert "Found it: 42." in final_edit

    @pytest.mark.asyncio
    async def test_nudges_when_memory_request_ack_and_no_tools(self):
        """Preference/memory acknowledgements without tools should trigger one retry."""
        memory_ack_chunk = (
            SimpleNamespace(
                content_blocks=[
                    {
                        "type": "text",
                        "text": 'Got it — I’ll treat "push the PR" as using the deepclaw-development skill and opening the PR automatically.',
                    }
                ]
            ),
            {},
        )
        recovery_chunk = (
            SimpleNamespace(
                content_blocks=[
                    {"type": "tool_call", "name": "memory_add", "args": {"content": "..."}},
                    {"type": "text", "text": "Saved."},
                ]
            ),
            {},
        )
        agent = _MultiCallStreamingAgent([[memory_ack_chunk], [recovery_chunk]])
        streaming = SimpleNamespace(edit_interval=999.0, buffer_threshold=999)
        gateway = Gateway(agent=agent, streaming_config=streaming)
        channel = _FakeStreamingChannel()
        incoming = SimpleNamespace(
            chat_id="42",
            text="Can you remember that when I ask you to push the pr use deepclaw dev skill and also open a pr as well",
        )

        await gateway.handle_message(channel, incoming, "thread-1")

        assert agent.call_count == 2


# ---------------------------------------------------------------------------
# Telegram media handling
# ---------------------------------------------------------------------------


class TestTelegramMediaHelpers:
    def test_looks_like_supported_image_accepts_image_mime(self):
        document = SimpleNamespace(mime_type="image/png", file_name="upload.bin")
        assert _looks_like_supported_image(document) is True

    def test_looks_like_supported_image_accepts_known_suffix(self):
        document = SimpleNamespace(mime_type="application/octet-stream", file_name="photo.webp")
        assert _looks_like_supported_image(document) is True

    def test_looks_like_supported_image_rejects_other_files(self):
        document = SimpleNamespace(mime_type="application/pdf", file_name="report.pdf")
        assert _looks_like_supported_image(document) is False

    @pytest.mark.asyncio
    async def test_build_incoming_text_for_plain_text(self):
        update = _make_slash_update(text="hello")
        text, error = await _build_incoming_text(update)
        assert error is None
        assert text == "hello"

    @pytest.mark.asyncio
    async def test_build_incoming_text_for_photo(self, tmp_path):
        update = _make_slash_update(text="")
        update.message.caption = "What does this say?"
        file_obj = MagicMock()

        async def _fake_download_to_drive(*, custom_path):
            Path(custom_path).write_bytes(b"img")
            return custom_path

        file_obj.download_to_drive = AsyncMock(side_effect=_fake_download_to_drive)
        photo = MagicMock()
        photo.get_file = AsyncMock(return_value=file_obj)
        update.message.photo = [photo]

        with patch("deepclaw.channels.telegram._UPLOADS_DIR", tmp_path):
            text, error = await _build_incoming_text(update)

        assert error is None
        assert "What does this say?" in text
        assert "Attached image saved at local path:" in text
        assert "vision_analyze" in text
        file_obj.download_to_drive.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_build_incoming_text_rejects_unsupported_document(self):
        update = _make_slash_update(text="")
        update.message.document = SimpleNamespace(
            mime_type="application/pdf", file_name="report.pdf"
        )
        text, error = await _build_incoming_text(update)
        assert text is None
        assert "not supported yet" in error


class TestTelegramMediaHandleMessage:
    @pytest.mark.asyncio
    async def test_handle_message_blocks_while_task_running(self):
        update = _make_slash_update(text="hello")
        task = asyncio.create_task(asyncio.sleep(60))
        gateway = MagicMock()
        gateway.handle_message = AsyncMock()
        ctx = _make_slash_context(extra={GATEWAY_KEY: gateway, ACTIVE_RUNS_KEY: {"1": task}})

        try:
            await handle_message(update, ctx)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        update.message.reply_text.assert_awaited_once_with(
            "A task is already running. Use /stop to interrupt it before sending something new."
        )
        gateway.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_message_routes_photo_to_gateway(self, tmp_path):
        update = _make_slash_update(text="")
        update.message.caption = "Please inspect this"
        file_obj = MagicMock()

        async def _fake_download_to_drive(*, custom_path):
            Path(custom_path).write_bytes(b"img")
            return custom_path

        file_obj.download_to_drive = AsyncMock(side_effect=_fake_download_to_drive)
        photo = MagicMock()
        photo.get_file = AsyncMock(return_value=file_obj)
        update.message.photo = [photo]

        gateway = MagicMock()
        gateway.handle_message = AsyncMock()
        ctx = _make_slash_context(extra={GATEWAY_KEY: gateway})

        with patch("deepclaw.channels.telegram._UPLOADS_DIR", tmp_path):
            await handle_message(update, ctx)

        gateway.handle_message.assert_awaited_once()
        _channel, incoming, thread_id = gateway.handle_message.await_args.args
        assert thread_id == ctx.bot_data[THREAD_IDS_KEY]["1"]
        assert thread_id != "1"
        assert incoming.chat_id == "1"
        assert incoming.source == "telegram"
        assert "Please inspect this" in incoming.text
        assert "vision_analyze" in incoming.text
        assert ctx.bot_data[LAST_MESSAGE_KEY]["1"] == incoming.text

    @pytest.mark.asyncio
    async def test_handle_message_replies_for_unsupported_document(self):
        update = _make_slash_update(text="")
        update.message.document = SimpleNamespace(
            mime_type="application/pdf", file_name="report.pdf"
        )
        gateway = MagicMock()
        gateway.handle_message = AsyncMock()
        ctx = _make_slash_context(extra={GATEWAY_KEY: gateway})

        await handle_message(update, ctx)

        update.message.reply_text.assert_awaited_once()
        gateway.handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# New slash command helpers
# ---------------------------------------------------------------------------


def _make_slash_update(user_id: int = 1, text: str = "/cmd", username: str = "alice"):
    """Build a mock Update suitable for slash command tests."""
    user = SimpleNamespace(id=user_id, username=username)
    chat = SimpleNamespace(id=user_id, type="private")
    message = MagicMock()
    message.reply_text = AsyncMock(return_value=SimpleNamespace(message_id=200))
    message.text = text
    message.caption = None
    message.photo = []
    message.document = None
    update = MagicMock()
    update.effective_user = user
    update.effective_chat = chat
    update.message = message
    update.callback_query = None
    return update


def _make_callback_update(user_id: int = 1, data: str = "approve", username: str = "alice"):
    """Build a mock Update for inline keyboard callback tests."""
    user = SimpleNamespace(id=user_id, username=username)
    chat = SimpleNamespace(id=user_id, type="private")
    message = MagicMock()
    message.reply_text = AsyncMock(return_value=SimpleNamespace(message_id=201))
    message.edit_reply_markup = AsyncMock()
    callback_query = MagicMock()
    callback_query.data = data
    callback_query.answer = AsyncMock()
    callback_query.message = message
    update = MagicMock()
    update.effective_user = user
    update.effective_chat = chat
    update.message = None
    update.callback_query = callback_query
    return update


def _make_slash_context(
    user_id: int = 1,
    model: str = "anthropic:claude-test",
    extra: dict | None = None,
):
    """Build a mock context for slash command tests."""
    config = DeepClawConfig(model=model)
    bot_data: dict = {
        ALLOWED_USERS_KEY: {"1"},
        CONFIG_KEY: config,
    }
    if extra:
        bot_data.update(extra)
    ctx = MagicMock()
    ctx.bot_data = bot_data
    return ctx


# ---------------------------------------------------------------------------
# /clear
# ---------------------------------------------------------------------------


class TestCmdClear:
    @pytest.mark.asyncio
    async def test_clear_creates_new_thread(self):
        update = _make_slash_update()
        ctx = _make_slash_context()
        await cmd_clear(update, ctx)
        update.message.reply_text.assert_called_once_with("Conversation cleared.")
        assert "1" in ctx.bot_data[THREAD_IDS_KEY]

    @pytest.mark.asyncio
    async def test_clear_removes_last_message(self):
        update = _make_slash_update()
        ctx = _make_slash_context(extra={LAST_MESSAGE_KEY: {"1": "hello"}})
        await cmd_clear(update, ctx)
        assert ctx.bot_data.get(LAST_MESSAGE_KEY, {}).get("1") is None


# ---------------------------------------------------------------------------
# /model
# ---------------------------------------------------------------------------


class TestValidateModel:
    def test_valid_anthropic(self):
        assert _validate_model("anthropic:claude-sonnet-4-6") is None

    def test_valid_openai(self):
        assert _validate_model("openai:gpt-5.3-codex") is None

    def test_valid_deepinfra(self):
        assert _validate_model("deepinfra:deepseek-ai/DeepSeek-V3") is None

    def test_valid_baseten(self):
        assert _validate_model("baseten:moonshotai/Kimi-K2-Instruct-0905") is None

    def test_missing_colon(self):
        assert _validate_model("claude-sonnet") is not None

    def test_empty_model_name(self):
        assert _validate_model("anthropic:") is not None

    def test_unknown_provider(self):
        assert _validate_model("fakeprovider:some-model") is not None


class TestCmdModel:
    @pytest.mark.asyncio
    async def test_model_no_arg_shows_config_model(self):
        update = _make_slash_update(text="/model")
        ctx = _make_slash_context(model="anthropic:claude-sonnet")
        await cmd_model(update, ctx)
        call_args = update.message.reply_text.call_args[0][0]
        assert "anthropic:claude-sonnet" in call_args
        assert "no override" in call_args

    @pytest.mark.asyncio
    async def test_model_with_arg_sets_override(self):
        update = _make_slash_update(text="/model anthropic:claude-opus")
        ctx = _make_slash_context()
        ctx.bot_data["checkpointer_resolved"] = MagicMock()
        with patch("deepclaw.channels.telegram.create_agent", return_value=MagicMock()):
            await cmd_model(update, ctx)
        assert ctx.bot_data[MODEL_OVERRIDE_KEY] == "anthropic:claude-opus"
        call_args = update.message.reply_text.call_args[0][0]
        assert "anthropic:claude-opus" in call_args

    @pytest.mark.asyncio
    async def test_model_shows_override_when_set(self):
        update = _make_slash_update(text="/model")
        ctx = _make_slash_context(extra={MODEL_OVERRIDE_KEY: "openai:gpt-5"})
        await cmd_model(update, ctx)
        call_args = update.message.reply_text.call_args[0][0]
        assert "openai:gpt-5" in call_args


# ---------------------------------------------------------------------------
# /memory
# ---------------------------------------------------------------------------


class TestCmdMemory:
    @pytest.mark.asyncio
    async def test_memory_file_not_found(self, tmp_path):
        update = _make_slash_update()
        ctx = _make_slash_context()
        missing = tmp_path / "AGENTS.md"
        with patch("deepclaw.agent.MEMORY_FILE", missing):
            await cmd_memory(update, ctx)
        update.message.reply_text.assert_called_once_with("No memory file found.")

    @pytest.mark.asyncio
    async def test_memory_file_exists(self, tmp_path):
        memory_file = tmp_path / "AGENTS.md"
        memory_file.write_text("# Lessons\n- Always test.")
        update = _make_slash_update()
        ctx = _make_slash_context()
        with patch("deepclaw.agent.MEMORY_FILE", memory_file):
            await cmd_memory(update, ctx)
        update.message.reply_text.assert_called_once()
        assert "Always test" in update.message.reply_text.call_args[0][0]


# ---------------------------------------------------------------------------
# /soul
# ---------------------------------------------------------------------------


class TestCmdSoul:
    @pytest.mark.asyncio
    async def test_soul_file_not_found(self, tmp_path):
        update = _make_slash_update()
        ctx = _make_slash_context()
        missing = tmp_path / "SOUL.md"
        with patch("deepclaw.agent.SOUL_FILE", missing):
            await cmd_soul(update, ctx)
        update.message.reply_text.assert_called_once_with("No SOUL.md found.")

    @pytest.mark.asyncio
    async def test_soul_file_exists(self, tmp_path):
        soul_file = tmp_path / "SOUL.md"
        soul_file.write_text("Be genuinely helpful.")
        update = _make_slash_update()
        ctx = _make_slash_context()
        with patch("deepclaw.agent.SOUL_FILE", soul_file):
            await cmd_soul(update, ctx)
        update.message.reply_text.assert_called_once()
        assert "genuinely helpful" in update.message.reply_text.call_args[0][0]


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


class TestCmdStatus:
    @pytest.mark.asyncio
    async def test_status_shows_thread_and_model(self):
        update = _make_slash_update(text="/status")
        ctx = _make_slash_context(model="openai:gpt-5")

        await cmd_status(update, ctx)

        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "Chat ID: 1" in text
        assert "Thread ID:" in text
        assert "Model: openai:gpt-5" in text
        assert "Allowlist:" in text
        assert "🧠 Context breakdown" not in text


class TestCmdSessions:
    @pytest.mark.asyncio
    async def test_sessions_shows_current_thread_and_prompt_preview(self):
        update = _make_slash_update(text="/sessions")
        ctx = _make_slash_context(extra={THREAD_IDS_KEY: {"1": "thread-current"}})
        sessions = [
            {
                "thread_id": "thread-current",
                "updated_at": "2026-04-27T05:00:00+00:00",
                "created_at": "2026-04-27T04:00:00+00:00",
                "message_count": 4,
                "initial_prompt": "Build session resume support for Telegram",
                "checkpoint_count": 2,
            },
            {
                "thread_id": "thread-older",
                "updated_at": "2026-04-27T03:00:00+00:00",
                "created_at": "2026-04-27T02:00:00+00:00",
                "message_count": 2,
                "initial_prompt": "Older thread prompt",
                "checkpoint_count": 1,
            },
        ]

        with patch(
            "deepclaw.channels.telegram.list_sessions_for_chat",
            return_value=sessions,
        ):
            await cmd_sessions(update, ctx)

        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "Recent sessions for this chat" in text
        assert "thread-current"[:8] in text
        assert "(current)" in text
        assert "Build session resume support for Telegram" in text
        assert "Msgs: 4" in text
        assert "Older thread prompt" in text

    @pytest.mark.asyncio
    async def test_sessions_accepts_explicit_limit_argument(self):
        update = _make_slash_update(text="/sessions 25")
        ctx = _make_slash_context(extra={THREAD_IDS_KEY: {"1": "thread-current"}})

        with patch(
            "deepclaw.channels.telegram.list_sessions_for_chat",
            return_value=[],
        ) as list_sessions:
            await cmd_sessions(update, ctx)

        list_sessions.assert_called_once_with("1", limit=25)
        update.message.reply_text.assert_awaited_once_with(
            "No saved sessions found for this chat yet."
        )

    @pytest.mark.asyncio
    async def test_sessions_rejects_invalid_limit_argument(self):
        update = _make_slash_update(text="/sessions lots")
        ctx = _make_slash_context()

        await cmd_sessions(update, ctx)

        update.message.reply_text.assert_awaited_once_with(
            "Usage: /sessions [limit]\nExample: /sessions 25"
        )

    @pytest.mark.asyncio
    async def test_sessions_reports_when_none_found(self):
        update = _make_slash_update(text="/sessions")
        ctx = _make_slash_context()

        with patch("deepclaw.channels.telegram.list_sessions_for_chat", return_value=[]):
            await cmd_sessions(update, ctx)

        update.message.reply_text.assert_awaited_once_with(
            "No saved sessions found for this chat yet."
        )

    @pytest.mark.asyncio
    async def test_sessions_does_not_create_thread_mapping_when_missing(self):
        update = _make_slash_update(text="/sessions")
        ctx = _make_slash_context(extra={THREAD_IDS_KEY: {}})

        with (
            patch("deepclaw.channels.telegram.list_sessions_for_chat", return_value=[]),
            patch("deepclaw.channels.telegram.save_thread_ids") as save_thread_ids,
        ):
            await cmd_sessions(update, ctx)

        assert ctx.bot_data[THREAD_IDS_KEY] == {}
        save_thread_ids.assert_not_called()


class TestCmdResume:
    @pytest.mark.asyncio
    async def test_resume_specific_thread_updates_mapping_and_clears_chat_state(self):
        update = _make_slash_update(text="/resume thread-older")
        ctx = _make_slash_context(
            extra={
                THREAD_IDS_KEY: {"1": "thread-current"},
                LAST_MESSAGE_KEY: {"1": "hello"},
                PENDING_APPROVALS_KEY: {"1": {"id": "pending-1"}},
            }
        )

        with (
            patch("deepclaw.channels.telegram.session_belongs_to_chat", return_value=True),
            patch("deepclaw.channels.telegram.save_thread_ids") as save_thread_ids,
        ):
            await cmd_resume(update, ctx)

        assert ctx.bot_data[THREAD_IDS_KEY]["1"] == "thread-older"
        assert ctx.bot_data[LAST_MESSAGE_KEY] == {}
        assert ctx.bot_data[PENDING_APPROVALS_KEY] == {}
        save_thread_ids.assert_called_once_with({"1": "thread-older"})
        update.message.reply_text.assert_called_once()
        assert "Resumed session thread-older" in update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_resume_without_arg_uses_most_recent_session(self):
        update = _make_slash_update(text="/resume")
        ctx = _make_slash_context(extra={THREAD_IDS_KEY: {"1": "thread-current"}})

        with (
            patch(
                "deepclaw.channels.telegram.get_most_recent_session_for_chat",
                return_value={"thread_id": "thread-recent"},
            ),
            patch("deepclaw.channels.telegram.save_thread_ids"),
        ):
            await cmd_resume(update, ctx)

        assert ctx.bot_data[THREAD_IDS_KEY]["1"] == "thread-recent"
        update.message.reply_text.assert_called_once()
        assert "thread-recent" in update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_resume_unknown_thread_suggests_similar_matches(self):
        update = _make_slash_update(text="/resume thread-miss")
        ctx = _make_slash_context()

        with (
            patch("deepclaw.channels.telegram.session_belongs_to_chat", return_value=False),
            patch(
                "deepclaw.channels.telegram.find_similar_sessions_for_chat",
                return_value=["thread-missing-1", "thread-missing-2"],
            ),
        ):
            await cmd_resume(update, ctx)

        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "thread-miss" in text
        assert "thread-missing-1" in text
        assert "thread-missing-2" in text


class TestCmdContext:
    @pytest.mark.asyncio
    async def test_context_includes_context_breakdown(self):
        update = _make_slash_update(text="/context")
        ctx = _make_slash_context(model="openai:gpt-5")

        with patch(
            "deepclaw.channels.telegram.build_context_report",
            return_value="🧠 Context breakdown\nEstimated active context subtotal: 123 chars (~45 tok)",
        ):
            await cmd_context(update, ctx)

        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "Chat ID: 1" in text
        assert "Thread ID:" in text
        assert "Model: openai:gpt-5" in text
        assert "🧠 Context breakdown" in text
        assert "Estimated active context subtotal" in text


class TestCmdQueue:
    @pytest.mark.asyncio
    async def test_queue_pending_approval_without_snapshot_sets_state(self):
        update = _make_slash_update(text="/queue")
        gateway = MagicMock()
        gateway.get_queue_snapshot.return_value = None
        ctx = _make_slash_context(
            extra={
                GATEWAY_KEY: gateway,
                PENDING_APPROVALS_KEY: {
                    "1": {
                        "tool": "execute",
                        "warning": "sudo requires approval",
                        "command": "sudo apt-get install ripgrep",
                    }
                },
            }
        )

        await cmd_queue(update, ctx)

        text = update.message.reply_text.call_args[0][0]
        assert "State: awaiting_approval" in text
        assert "Pending approval:" in text

    @pytest.mark.asyncio
    async def test_queue_redacts_sensitive_pending_approval_fields(self):
        secret = "sk-ant-secretvalue1234567890abcd"
        update = _make_slash_update(text="/queue")
        gateway = MagicMock()
        gateway.get_queue_snapshot.return_value = {
            "status": "awaiting_approval",
            "started_at": 900.0,
            "updated_at": 995.0,
            "user_text_preview": f"install {secret}",
            "progress_lines": [f'💻 terminal: "export OPENAI_API_KEY={secret}"'],
            "pending_approval": {
                "tool": "execute",
                "warning": f"contains {secret}",
                "command": f"export OPENAI_API_KEY={secret}",
            },
            "final_text_preview": f"Waiting for {secret}",
        }
        ctx = _make_slash_context(extra={GATEWAY_KEY: gateway})

        with patch("deepclaw.channels.telegram.time.time", return_value=1000.0):
            await cmd_queue(update, ctx)

        text = update.message.reply_text.call_args[0][0]
        assert secret not in text
        assert "[REDACTED]" in text

    @pytest.mark.asyncio
    async def test_queue_shows_recent_tool_activity_and_pending_approval(self):
        update = _make_slash_update(text="/queue")
        gateway = MagicMock()
        gateway.get_queue_snapshot.return_value = {
            "status": "awaiting_approval",
            "started_at": 900.0,
            "updated_at": 995.0,
            "user_text_preview": "install ripgrep",
            "progress_lines": [
                '📚 skill_view: "deepclaw-development"',
                '💻 terminal: "sudo apt-get install ripgrep"',
            ],
            "pending_approval": {
                "tool": "execute",
                "warning": "sudo requires approval",
                "command": "sudo apt-get install ripgrep",
            },
            "final_text_preview": "Waiting for approval.",
        }
        ctx = _make_slash_context(
            model="openai:gpt-5",
            extra={
                GATEWAY_KEY: gateway,
                PENDING_APPROVALS_KEY: {
                    "1": {
                        "tool": "execute",
                        "warning": "sudo requires approval",
                        "command": "sudo apt-get install ripgrep",
                    }
                },
            },
        )

        with patch("deepclaw.channels.telegram.time.time", return_value=1000.0):
            await cmd_queue(update, ctx)

        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "📋 Queue" in text
        assert "State: awaiting_approval" in text
        assert "Request: install ripgrep" in text
        assert '📚 skill_view: "deepclaw-development"' in text
        assert '💻 terminal: "sudo apt-get install ripgrep"' in text
        assert "Pending approval:" in text
        assert "sudo requires approval" in text
        assert "/approve to continue" in text

    @pytest.mark.asyncio
    async def test_queue_reports_idle_when_no_activity(self):
        update = _make_slash_update(text="/queue")
        gateway = MagicMock()
        gateway.get_queue_snapshot.return_value = None
        ctx = _make_slash_context(extra={GATEWAY_KEY: gateway})

        await cmd_queue(update, ctx)

        text = update.message.reply_text.call_args[0][0]
        assert "📋 Queue" in text
        assert "State: idle" in text
        assert "No active task or recent tool activity" in text


# ---------------------------------------------------------------------------
# /uptime
# ---------------------------------------------------------------------------


class TestCmdUptime:
    @pytest.mark.asyncio
    async def test_uptime_format(self):
        update = _make_slash_update()
        ctx = _make_slash_context()
        with (
            patch("deepclaw.channels.telegram._BOT_START_TIME", 0),
            patch("deepclaw.channels.telegram.time") as mock_time,
        ):
            mock_time.time.return_value = 3661  # 1h 1m 1s
            await cmd_uptime(update, ctx)
        call_args = update.message.reply_text.call_args[0][0]
        assert "Uptime:" in call_args
        assert "1h" in call_args
        assert "1m" in call_args


# ---------------------------------------------------------------------------
# /retry
# ---------------------------------------------------------------------------


class TestCmdRetry:
    @pytest.mark.asyncio
    async def test_retry_no_previous_message(self):
        update = _make_slash_update()
        ctx = _make_slash_context()
        await cmd_retry(update, ctx)
        update.message.reply_text.assert_called_once_with("No previous message to retry.")

    @pytest.mark.asyncio
    async def test_retry_blocked_while_approval_pending(self):
        update = _make_slash_update(text="/retry")
        ctx = _make_slash_context(
            extra={
                LAST_MESSAGE_KEY: {"1": "run something"},
                PENDING_APPROVALS_KEY: {"1": {"id": "interrupt-1", "thread_id": "thread-1"}},
            }
        )

        await cmd_retry(update, ctx)

        update.message.reply_text.assert_called_once_with(
            "A safety approval is pending. Use /approve, /approve session, or /deny <reason> to respond."
        )

    @pytest.mark.asyncio
    async def test_retry_blocked_while_task_running(self):
        update = _make_slash_update(text="/retry")
        task = asyncio.create_task(asyncio.sleep(60))
        ctx = _make_slash_context(
            extra={
                LAST_MESSAGE_KEY: {"1": "run something"},
                ACTIVE_RUNS_KEY: {"1": task},
            }
        )

        try:
            await cmd_retry(update, ctx)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        update.message.reply_text.assert_called_once_with(
            "A task is already running. Use /stop to interrupt it before sending something new."
        )


class TestCmdStop:
    @pytest.mark.asyncio
    async def test_stop_without_active_task(self):
        update = _make_slash_update(text="/stop")
        ctx = _make_slash_context()

        await cmd_stop(update, ctx)

        update.message.reply_text.assert_called_once_with("No active task to stop.")

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self):
        update = _make_slash_update(text="/stop")
        blocker = asyncio.Event()

        async def _run_forever():
            await blocker.wait()

        task = asyncio.create_task(_run_forever())
        ctx = _make_slash_context(extra={ACTIVE_RUNS_KEY: {"1": task}})

        await cmd_stop(update, ctx)

        assert task.cancelled()
        assert ctx.bot_data[ACTIVE_RUNS_KEY] == {}
        update.message.reply_text.assert_called_once_with("Stopped the current task.")


# ---------------------------------------------------------------------------
# /approve and /deny
# ---------------------------------------------------------------------------


class TestSafetyApprovalCommands:
    def test_pending_approval_markup_has_once_session_and_deny_buttons(self):
        markup = _pending_approval_markup("interrupt-1")

        assert isinstance(markup, InlineKeyboardMarkup)
        rows = markup.inline_keyboard
        assert len(rows) == 1
        assert rows[0][0].text == "Approve once"
        assert rows[0][0].callback_data == "safety:approve_once:interrupt-1"
        assert rows[0][1].text == "Approve session"
        assert rows[0][1].callback_data == "safety:approve_session:interrupt-1"
        assert rows[0][2].text == "Deny"
        assert rows[0][2].callback_data == "safety:deny:interrupt-1"

    @pytest.mark.asyncio
    async def test_approve_without_pending_request(self):
        update = _make_slash_update(text="/approve")
        ctx = _make_slash_context()

        await cmd_approve(update, ctx)

        update.message.reply_text.assert_called_once_with(
            "No pending safety approval for this chat."
        )

    @pytest.mark.asyncio
    async def test_deny_without_pending_request(self):
        update = _make_slash_update(text="/deny")
        ctx = _make_slash_context()

        await cmd_deny(update, ctx)

        update.message.reply_text.assert_called_once_with(
            "No pending safety approval for this chat."
        )

    @pytest.mark.asyncio
    async def test_approve_resumes_pending_interrupt_and_clears_it(self):
        update = _make_slash_update(text="/approve")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
                "approval_keys": ["dangerous:code_injection"],
            }
        }
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=None)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        with patch(
            "deepclaw.channels.telegram.aadd_thread_approved_keys", new=AsyncMock()
        ) as mock_store:
            await cmd_approve(update, ctx)

        gateway.resume_interrupt.assert_awaited_once()
        _, kwargs = gateway.resume_interrupt.await_args
        assert kwargs["thread_id"] == "thread-1"
        assert kwargs["decision"] == {"type": "approve", "scope": "once"}
        mock_store.assert_not_awaited()
        assert ctx.bot_data[PENDING_APPROVALS_KEY] == {}
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_approve_session_resumes_pending_interrupt_with_session_scope(self):
        update = _make_slash_update(text="/approve session")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
                "approval_keys": ["dangerous:code_injection"],
            }
        }
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=None)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        with patch(
            "deepclaw.channels.telegram.aadd_thread_approved_keys",
            new=AsyncMock(return_value=True),
        ) as mock_store:
            await cmd_approve(update, ctx)

        _, kwargs = gateway.resume_interrupt.await_args
        assert kwargs["decision"] == {"type": "approve", "scope": "session"}
        mock_store.assert_awaited_once_with("thread-1", ["dangerous:code_injection"])

    @pytest.mark.asyncio
    async def test_approve_surfaces_follow_up_pending_review_with_buttons(self):
        update = _make_slash_update(text="/approve")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        }
        next_pending = {
            "id": "interrupt-2",
            "thread_id": "thread-1",
            "type": "safety_review",
            "message": "Need another approval",
        }
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=next_pending)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        await cmd_approve(update, ctx)

        assert ctx.bot_data[PENDING_APPROVALS_KEY]["1"]["id"] == "interrupt-2"
        update.message.reply_text.assert_awaited_once()
        _, kwargs = update.message.reply_text.await_args
        assert (
            kwargs["reply_markup"].inline_keyboard[0][0].callback_data
            == "safety:approve_once:interrupt-2"
        )
        assert (
            kwargs["reply_markup"].inline_keyboard[0][1].callback_data
            == "safety:approve_session:interrupt-2"
        )
        assert (
            kwargs["reply_markup"].inline_keyboard[0][2].callback_data == "safety:deny:interrupt-2"
        )

    @pytest.mark.asyncio
    async def test_approve_run_can_be_stopped_while_resume_interrupt_is_in_flight(self):
        approve_update = _make_slash_update(text="/approve")
        stop_update = _make_slash_update(text="/stop")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        }
        blocker = asyncio.Event()

        async def _resume_interrupt(*_args, **_kwargs):
            await blocker.wait()

        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(side_effect=_resume_interrupt)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        approve_task = asyncio.create_task(cmd_approve(approve_update, ctx))
        await asyncio.sleep(0)

        assert ctx.bot_data[ACTIVE_RUNS_KEY]["1"] is approve_task

        await cmd_stop(stop_update, ctx)
        await approve_task

        assert approve_task.done()
        assert not approve_task.cancelled()
        assert ctx.bot_data[ACTIVE_RUNS_KEY] == {}
        stop_update.message.reply_text.assert_called_once_with("Stopped the current task.")

    @pytest.mark.asyncio
    async def test_deny_resumes_pending_interrupt_with_reason(self):
        update = _make_slash_update(text="/deny too risky")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        }
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=None)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        await cmd_deny(update, ctx)

        gateway.resume_interrupt.assert_awaited_once()
        _, kwargs = gateway.resume_interrupt.await_args
        assert kwargs["decision"] == {"type": "reject", "message": "too risky"}
        assert ctx.bot_data[PENDING_APPROVALS_KEY] == {}

    @pytest.mark.asyncio
    async def test_callback_approve_resumes_pending_interrupt_and_clears_buttons(self):
        update = _make_callback_update(data="safety:approve_once:interrupt-1")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        }
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=None)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        await cmd_approval_callback(update, ctx)

        update.callback_query.answer.assert_awaited_once()
        gateway.resume_interrupt.assert_awaited_once()
        _, kwargs = gateway.resume_interrupt.await_args
        assert kwargs["decision"] == {"type": "approve", "scope": "once"}
        update.callback_query.message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
        assert ctx.bot_data[PENDING_APPROVALS_KEY] == {}

    @pytest.mark.asyncio
    async def test_callback_approve_session_resumes_pending_interrupt_with_session_scope(self):
        update = _make_callback_update(data="safety:approve_session:interrupt-1")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        }
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=None)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        await cmd_approval_callback(update, ctx)

        _, kwargs = gateway.resume_interrupt.await_args
        assert kwargs["decision"] == {"type": "approve", "scope": "session"}

    @pytest.mark.asyncio
    async def test_callback_approve_surfaces_follow_up_pending_review_with_buttons(self):
        update = _make_callback_update(data="safety:approve_once:interrupt-1")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        }
        next_pending = {
            "id": "interrupt-2",
            "thread_id": "thread-1",
            "type": "safety_review",
            "message": "Need another approval",
        }
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=next_pending)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        await cmd_approval_callback(update, ctx)

        assert ctx.bot_data[PENDING_APPROVALS_KEY]["1"]["id"] == "interrupt-2"
        update.callback_query.message.reply_text.assert_awaited_once()
        _, kwargs = update.callback_query.message.reply_text.await_args
        assert (
            kwargs["reply_markup"].inline_keyboard[0][0].callback_data
            == "safety:approve_once:interrupt-2"
        )
        assert (
            kwargs["reply_markup"].inline_keyboard[0][1].callback_data
            == "safety:approve_session:interrupt-2"
        )
        assert (
            kwargs["reply_markup"].inline_keyboard[0][2].callback_data == "safety:deny:interrupt-2"
        )

    @pytest.mark.asyncio
    async def test_callback_deny_rejects_pending_interrupt(self):
        update = _make_callback_update(data="safety:deny:interrupt-1")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        }
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=None)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        await cmd_approval_callback(update, ctx)

        _, kwargs = gateway.resume_interrupt.await_args
        assert kwargs["decision"] == {
            "type": "reject",
            "message": "Command rejected via inline deny button",
        }

    @pytest.mark.asyncio
    async def test_callback_stale_button_does_not_resume_newer_pending_interrupt(self):
        update = _make_callback_update(data="safety:approve_once:old-interrupt")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        }
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=None)
        ctx = _make_slash_context(extra={PENDING_APPROVALS_KEY: pending, GATEWAY_KEY: gateway})

        await cmd_approval_callback(update, ctx)

        gateway.resume_interrupt.assert_not_awaited()
        update.callback_query.message.reply_text.assert_awaited_once_with(
            "That approval button is stale. Please use the latest safety review prompt."
        )

    @pytest.mark.asyncio
    async def test_callback_keeps_buttons_when_another_active_run_blocks_resume(self):
        update = _make_callback_update(data="safety:approve_once:interrupt-1")
        pending = {
            "1": {
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        }
        blocker = asyncio.create_task(asyncio.sleep(60))
        gateway = MagicMock()
        gateway.resume_interrupt = AsyncMock(return_value=None)
        ctx = _make_slash_context(
            extra={
                PENDING_APPROVALS_KEY: pending,
                GATEWAY_KEY: gateway,
                ACTIVE_RUNS_KEY: {"1": blocker},
            }
        )

        try:
            await cmd_approval_callback(update, ctx)
        finally:
            blocker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await blocker

        gateway.resume_interrupt.assert_not_awaited()
        update.callback_query.message.reply_text.assert_awaited_once_with(
            "A task is already running. Use /stop to interrupt it before sending something new."
        )
        update.callback_query.message.edit_reply_markup.assert_not_awaited()
        assert ctx.bot_data[PENDING_APPROVALS_KEY] == pending

    @pytest.mark.asyncio
    async def test_handle_message_stores_pending_safety_review(self):
        update = _make_slash_update(text="run evals")
        gateway = MagicMock()
        gateway.handle_message = AsyncMock(
            return_value={
                "id": "interrupt-1",
                "thread_id": "thread-1",
                "type": "safety_review",
                "message": "Approve or deny?",
            }
        )
        ctx = _make_slash_context(extra={GATEWAY_KEY: gateway})

        await handle_message(update, ctx)

        assert ctx.bot_data[PENDING_APPROVALS_KEY]["1"]["id"] == "interrupt-1"
        assert update.message.reply_text.await_count == 1
        _, kwargs = update.message.reply_text.await_args
        assert (
            kwargs["reply_markup"].inline_keyboard[0][0].callback_data
            == "safety:approve_once:interrupt-1"
        )

    @pytest.mark.asyncio
    async def test_handle_message_blocks_new_requests_while_approval_pending(self):
        update = _make_slash_update(text="do something else")
        gateway = MagicMock()
        gateway.handle_message = AsyncMock()
        ctx = _make_slash_context(
            extra={
                GATEWAY_KEY: gateway,
                PENDING_APPROVALS_KEY: {
                    "1": {
                        "id": "interrupt-1",
                        "thread_id": "thread-1",
                        "type": "safety_review",
                        "message": "Approve or deny?",
                    }
                },
            }
        )

        await handle_message(update, ctx)

        gateway.handle_message.assert_not_awaited()
        update.message.reply_text.assert_called_once_with(
            "A safety approval is pending. Use /approve, /approve session, or /deny <reason> to respond."
        )


# ---------------------------------------------------------------------------
# /skills
# ---------------------------------------------------------------------------


class TestSkillsCommandHelpers:
    def test_parse_skills_defaults_to_browse(self):
        assert _parse_skills_command("/skills") == ("browse", "")

    def test_parse_skills_with_subcommand_and_args(self):
        assert _parse_skills_command("/skills view deepclaw-development") == (
            "view",
            "deepclaw-development",
        )

    def test_parse_skills_delete_alias(self):
        assert _parse_skills_command("/skills rm demo-skill") == ("rm", "demo-skill")

    def test_format_skills_list(self, monkeypatch):
        monkeypatch.setattr(
            "deepclaw.channels.telegram.skills_list",
            lambda: {
                "count": 2,
                "skills": [
                    {"name": "a", "description": "first"},
                    {"name": "b", "description": "second"},
                ],
            },
        )
        text = _format_skills_list()
        assert "Installed skills (2):" in text
        assert "- a: first" in text
        assert "- b: second" in text

    def test_format_remote_skills(self):
        text = _format_remote_skills(
            {
                "query": "browser",
                "skills": [
                    {
                        "name": "agent-browser",
                        "repo": "vercel-labs/agent-browser",
                        "installs": 195769,
                        "url": "https://skills.sh/vercel-labs/agent-browser/agent-browser",
                    }
                ],
            }
        )
        assert "skills.sh results for: browser" in text
        assert "agent-browser" in text
        assert "195,769 installs" in text


class TestCmdSkills:
    @pytest.mark.asyncio
    async def test_skills_default_lists_installed_skills(self, monkeypatch):
        update = _make_slash_update(text="/skills")
        ctx = _make_slash_context()
        monkeypatch.setattr(
            "deepclaw.channels.telegram.skills_list",
            lambda: {
                "count": 1,
                "skills": [{"name": "deepclaw-development", "description": "DeepClaw dev"}],
            },
        )

        await cmd_skills(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "deepclaw-development" in reply

    @pytest.mark.asyncio
    async def test_skills_view_returns_content(self, monkeypatch):
        update = _make_slash_update(text="/skills view demo-skill")
        ctx = _make_slash_context()
        monkeypatch.setattr(
            "deepclaw.channels.telegram.skill_view",
            lambda name: {
                "name": name,
                "path": "/tmp/demo-skill/SKILL.md",
                "content": "# Demo Skill\n\nHello",
            },
        )

        await cmd_skills(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "Skill: demo-skill" in reply
        assert "Hello" in reply

    @pytest.mark.asyncio
    async def test_skills_create_creates_skill(self, monkeypatch):
        update = _make_slash_update(text="/skills create demo-skill | Demo description")
        ctx = _make_slash_context()
        monkeypatch.setattr(
            "deepclaw.channels.telegram.skill_create",
            lambda name, description: {
                "name": name,
                "path": f"/tmp/{name}/SKILL.md",
                "description": description,
            },
        )

        await cmd_skills(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "Created skill: demo-skill" in reply

    @pytest.mark.asyncio
    async def test_skills_search_returns_remote_results(self, monkeypatch):
        update = _make_slash_update(text="/skills search browser")
        ctx = _make_slash_context()
        monkeypatch.setattr(
            "deepclaw.channels.telegram.skills_search_remote",
            lambda query: {
                "query": query,
                "skills": [
                    {
                        "name": "agent-browser",
                        "repo": "vercel-labs/agent-browser",
                        "installs": 195769,
                        "url": "https://skills.sh/vercel-labs/agent-browser/agent-browser",
                    }
                ],
            },
        )

        await cmd_skills(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "skills.sh results for: browser" in reply
        assert "agent-browser" in reply

    @pytest.mark.asyncio
    async def test_skills_audit_returns_summary(self, monkeypatch):
        update = _make_slash_update(text="/skills audit")
        ctx = _make_slash_context()
        monkeypatch.setattr(
            "deepclaw.channels.telegram.skills_audit",
            lambda: {
                "count": 2,
                "duplicate_descriptions_count": 1,
                "skills_missing_required_sections_count": 1,
                "skills_missing_required_env_declarations_count": 1,
                "duplicate_descriptions": [
                    {"description": "Shared description", "skills": ["a", "b"]}
                ],
                "skills": [
                    {
                        "name": "a",
                        "missing_sections": ["Verification"],
                        "undeclared_required_env_vars": ["LANGSMITH_API_KEY"],
                    },
                    {"name": "b", "missing_sections": [], "undeclared_required_env_vars": []},
                ],
            },
        )

        await cmd_skills(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "Skill audit" in reply
        assert "duplicate descriptions: 1" in reply
        assert "missing required sections: 1" in reply
        assert "missing env declarations: 1" in reply

    @pytest.mark.asyncio
    async def test_skills_resolvable_returns_summary(self, monkeypatch):
        update = _make_slash_update(text="/skills resolvable")
        ctx = _make_slash_context()
        monkeypatch.setattr(
            "deepclaw.channels.telegram.skills_check_resolvable",
            lambda: {
                "count": 2,
                "loadable_count": 1,
                "unresolvable_count": 1,
                "unresolvable": [
                    {"name": "bad-skill", "reason": "missing required 'name' or 'description'"}
                ],
            },
        )

        await cmd_skills(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "Resolvable skills" in reply
        assert "loadable: 1 / 2" in reply
        assert "bad-skill" in reply

    @pytest.mark.asyncio
    async def test_skills_install_uses_optional_name(self, monkeypatch):
        update = _make_slash_update(text="/skills install /tmp/source-skill/SKILL.md | imported")
        ctx = _make_slash_context()

        calls = []

        def _fake_install(source_path, name=None):
            calls.append((source_path, name))
            return {
                "name": name or "source-skill",
                "path": "/tmp/imported/SKILL.md",
                "source_path": source_path,
            }

        monkeypatch.setattr("deepclaw.channels.telegram.skill_install", _fake_install)

        await cmd_skills(update, ctx)

        assert calls == [("/tmp/source-skill/SKILL.md", "imported")]
        reply = update.message.reply_text.call_args[0][0]
        assert "Installed skill: imported" in reply

    @pytest.mark.asyncio
    async def test_skills_delete_removes_skill(self, monkeypatch):
        update = _make_slash_update(text="/skills delete demo-skill")
        ctx = _make_slash_context()
        monkeypatch.setattr(
            "deepclaw.channels.telegram.skill_delete",
            lambda name: {"name": name, "path": f"/tmp/{name}/SKILL.md"},
        )

        await cmd_skills(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "Deleted skill: demo-skill" in reply

    @pytest.mark.asyncio
    async def test_skills_unknown_subcommand_shows_usage(self):
        update = _make_slash_update(text="/skills nonsense")
        ctx = _make_slash_context()

        await cmd_skills(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "Usage: /skills" in reply
        assert "audit" in reply
        assert "resolvable" in reply
