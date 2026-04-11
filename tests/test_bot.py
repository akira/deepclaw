"""Tests for deepclaw bot modules (auth, channels.telegram, gateway)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from deepclaw.auth import is_user_allowed
from deepclaw.channels.telegram import (
    ALLOWED_USERS_KEY,
    CONFIG_KEY,
    GATEWAY_KEY,
    LAST_MESSAGE_KEY,
    MODEL_OVERRIDE_KEY,
    TELEGRAM_MESSAGE_LIMIT,
    THREAD_IDS_KEY,
    _build_incoming_text,
    _looks_like_supported_image,
    _validate_model,
    authorize_chat,
    cmd_clear,
    cmd_memory,
    cmd_model,
    cmd_retry,
    cmd_soul,
    cmd_uptime,
    get_thread_id,
    handle_message,
)
from deepclaw.config import DeepClawConfig
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
        assert thread_id == "1"
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
    message.reply_text = AsyncMock()
    message.text = text
    message.caption = None
    message.photo = []
    message.document = None
    update = MagicMock()
    update.effective_user = user
    update.effective_chat = chat
    update.message = message
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
