"""Tests for deepclaw.bot module."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from deepclaw.bot import (
    TELEGRAM_MESSAGE_LIMIT,
    THREAD_IDS_KEY,
    chunk_message,
    get_thread_id,
    is_user_allowed,
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
