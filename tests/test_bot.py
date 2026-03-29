"""Tests for deepclaw bot modules (auth, channels.telegram, gateway)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage

from deepclaw.auth import is_user_allowed
from deepclaw.channels.telegram import (
    TELEGRAM_MESSAGE_LIMIT,
    THREAD_IDS_KEY,
    authorize_chat,
    get_thread_id,
)
from deepclaw.gateway import CURSOR_INDICATOR, Gateway, chunk_message

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
    def test_default_returns_chat_id(self):
        ctx = _make_context()
        assert get_thread_id(ctx, "12345") == "12345"

    def test_returns_custom_thread_id(self):
        custom_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        ctx = _make_context(thread_ids={"12345": custom_id})
        assert get_thread_id(ctx, "12345") == custom_id

    def test_other_chat_unaffected(self):
        ctx = _make_context(thread_ids={"12345": "custom-id"})
        assert get_thread_id(ctx, "99999") == "99999"


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


class _FakeStreamingAgent:
    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, *_args, **_kwargs):
        for chunk in self._chunks:
            yield chunk


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
