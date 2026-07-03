"""Live end-to-end smoke tests against real Telegram.

The bot runs in-process with production handlers and a ScriptedAgent brain
(no LLM cost), polling real Telegram with a dedicated test bot token. A
Telethon userbot plays the human: it pairs, chats, awaits streamed replies,
and clicks inline approval buttons. This verifies what the offline harness
cannot: Telegram's server accepts our MarkdownV2, buttons are clickable,
and streamed edits arrive as real message updates.

Opt-in only — skipped unless DEEPCLAW_E2E_TELEGRAM=1. See tests/e2e/README.md
for one-time credential setup. Never point DEEPCLAW_E2E_BOT_TOKEN at a
production bot: the tests reset pairing state for every run.
"""

import asyncio
import os
import time
import uuid

import pytest
import pytest_asyncio
from telegram import Update
from telegram.ext import Application

from deepclaw.channels.telegram import _STREAM_MESSAGES, register_handlers
from tests.telegram_harness import (
    ScriptedAgent,
    Turn,
    safety_interrupt,
    seed_application,
    text_chunk,
)

LIVE_ENABLED = os.getenv("DEEPCLAW_E2E_TELEGRAM") == "1"
REQUIRED_ENV_VARS = (
    "DEEPCLAW_E2E_BOT_TOKEN",
    "TELEGRAM_API_ID",
    "TELEGRAM_API_HASH",
    "TELETHON_SESSION",
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not LIVE_ENABLED,
        reason="live Telegram e2e disabled; set DEEPCLAW_E2E_TELEGRAM=1 to enable",
    ),
    pytest.mark.asyncio,
]


def _require_env() -> dict[str, str]:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        pytest.fail(
            "DEEPCLAW_E2E_TELEGRAM=1 but required env vars are missing: "
            + ", ".join(missing)
            + " (see tests/e2e/README.md)"
        )
    return {name: os.environ[name] for name in REQUIRED_ENV_VARS}


class LiveBot:
    """An in-process DeepClaw bot polling real Telegram with a scripted brain."""

    def __init__(self, app: Application, username: str, pairing_code: str) -> None:
        self.app = app
        self.username = username
        self.pairing_code = pairing_code


@pytest_asyncio.fixture
async def live_bot(monkeypatch):
    """Factory fixture: starts the bot polling real Telegram, stops it on teardown."""
    env = _require_env()
    monkeypatch.setattr("deepclaw.channels.telegram.save_thread_ids", lambda *_: None)
    monkeypatch.setattr("deepclaw.channels.telegram.save_allowed_users", lambda *_: None)
    _STREAM_MESSAGES.clear()
    bots: list[LiveBot] = []

    async def _make(agent: ScriptedAgent, *, allowed_users: set[str] | None = None) -> LiveBot:
        app = Application.builder().token(env["DEEPCLAW_E2E_BOT_TOKEN"]).build()
        register_handlers(app)
        pairing_code = uuid.uuid4().hex[:8]
        seed_application(
            app,
            agent,
            allowed_users=allowed_users or set(),
            pairing_code=pairing_code,
            # Gentler than the offline harness: at most ~1 edit/second so the
            # streaming path is exercised without tripping flood control.
            edit_interval=1.0,
            buffer_threshold=1,
        )
        await app.initialize()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        await app.start()
        bot = LiveBot(app, app.bot.username, pairing_code)
        bots.append(bot)
        return bot

    yield _make

    for bot in bots:
        await bot.app.updater.stop()
        await bot.app.stop()
        await bot.app.shutdown()
    _STREAM_MESSAGES.clear()


@pytest_asyncio.fixture
async def userbot():
    """Telethon client authorized via a pre-generated StringSession."""
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    env = _require_env()
    client = TelegramClient(
        StringSession(env["TELETHON_SESSION"]),
        int(env["TELEGRAM_API_ID"]),
        env["TELEGRAM_API_HASH"],
    )
    await client.connect()
    if not await client.is_user_authorized():
        pytest.fail("TELETHON_SESSION is not authorized; regenerate it (see tests/e2e/README.md)")
    yield client
    await client.disconnect()


async def _wait_for_bot_message(client, entity, contains: str, *, min_id: int, timeout: float = 90):
    """Poll the chat until a bot message newer than min_id contains the text.

    Re-fetches message content each poll, so it observes both fresh sends and
    in-place streaming edits.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        messages = await client.get_messages(entity, limit=10, min_id=min_id)
        for message in messages:
            if not message.out and contains in (message.raw_text or ""):
                return message
        await asyncio.sleep(1.5)
    pytest.fail(f"No bot message containing {contains!r} arrived within {timeout}s")


async def _wait_for_bot_buttons(client, entity, *, min_id: int, timeout: float = 90):
    """Poll the chat until a bot message newer than min_id carries inline buttons."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        messages = await client.get_messages(entity, limit=10, min_id=min_id)
        for message in messages:
            if not message.out and message.buttons:
                return message
        await asyncio.sleep(1.5)
    pytest.fail(f"No bot message with inline buttons arrived within {timeout}s")


async def test_live_pair_and_streamed_reply(live_bot, userbot):
    agent = ScriptedAgent(
        [Turn(chunks=[text_chunk("Hello from the "), text_chunk("live e2e test!")])]
    )
    bot = await live_bot(agent)
    entity = await userbot.get_entity(bot.username)

    sent = await userbot.send_message(entity, f"/pair {bot.pairing_code}")
    await _wait_for_bot_message(userbot, entity, "Paired successfully", min_id=sent.id)

    sent = await userbot.send_message(entity, "hi there")
    reply = await _wait_for_bot_message(
        userbot, entity, "Hello from the live e2e test!", min_id=sent.id
    )
    assert not reply.out
    assert agent.astream_payloads[0]["messages"][0]["content"] == "hi there"


async def test_live_markdown_reply_is_accepted_and_rendered(live_bot, userbot):
    markdown_reply = (
        "## Formatting check\n\n"
        "This has **bold text**, `inline code`, and a [link](https://example.com).\n\n"
        "```python\nprint('hello')\n```\n\n"
        "- item one\n"
        "- item two with special chars: 2 * 3 = 6!\n"
    )
    agent = ScriptedAgent([Turn(chunks=[text_chunk(markdown_reply)])])
    me = await userbot.get_me()
    bot = await live_bot(agent, allowed_users={str(me.id)})
    entity = await userbot.get_entity(bot.username)

    sent = await userbot.send_message(entity, "show me formatting")
    reply = await _wait_for_bot_message(userbot, entity, "Formatting check", min_id=sent.id)

    # Telegram accepted the MarkdownV2 payload (a parse failure would have
    # produced the plaintext fallback with no entities) and parsed it into
    # formatting entities that every client renders.
    assert "bold text" in reply.raw_text
    assert "print('hello')" in reply.raw_text
    assert reply.entities, "expected Telegram to parse formatting entities"
    entity_types = {type(item).__name__ for item in reply.entities}
    assert "MessageEntityBold" in entity_types
    assert {"MessageEntityCode", "MessageEntityPre"} & entity_types


async def test_live_approval_buttons_click_and_resume(live_bot, userbot):
    agent = ScriptedAgent(
        [
            Turn(
                chunks=[text_chunk("This needs approval.")],
                interrupts=(safety_interrupt(interrupt_id="live-int-1"),),
            ),
            Turn(chunks=[text_chunk("Approved and completed.")]),
        ]
    )
    me = await userbot.get_me()
    bot = await live_bot(agent, allowed_users={str(me.id)})
    entity = await userbot.get_entity(bot.username)

    sent = await userbot.send_message(entity, "run the risky thing")
    prompt = await _wait_for_bot_buttons(userbot, entity, min_id=sent.id)
    labels = [button.text for row in prompt.buttons for button in row]
    assert any("Approve" in label for label in labels)

    await prompt.click(text=next(label for label in labels if "once" in label.lower()))
    reply = await _wait_for_bot_message(userbot, entity, "Approved and completed", min_id=prompt.id)
    assert not reply.out
    assert len(agent.astream_payloads) == 2
