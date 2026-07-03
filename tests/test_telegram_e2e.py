"""Offline end-to-end tests for the Telegram handler chain.

Each test feeds real telegram.Update objects through Application.process_update
with production handlers registered, a real Gateway, a scripted agent, and a
fake Bot API transport. See tests/telegram_harness.py.
"""

import json

import pytest
import pytest_asyncio
from langgraph.types import Command

from deepclaw.auth import REJECTION_MESSAGE
from deepclaw.channels.telegram import (
    _STREAM_MESSAGES,
    ACTIVE_RUNS_KEY,
    PENDING_APPROVALS_KEY,
    QUEUED_RUNS_KEY,
    THREAD_IDS_KEY,
)
from tests.telegram_harness import (
    DEFAULT_CHAT_ID,
    Pause,
    ScriptedAgent,
    Turn,
    build_harness,
    safety_interrupt,
    text_chunk,
)

CHAT = str(DEFAULT_CHAT_ID)


@pytest_asyncio.fixture
async def tg(monkeypatch):
    """Factory fixture: builds harnesses, isolates persistence, cleans up."""
    monkeypatch.setattr("deepclaw.channels.telegram.save_thread_ids", lambda *_: None)
    monkeypatch.setattr("deepclaw.channels.telegram.save_allowed_users", lambda *_: None)
    _STREAM_MESSAGES.clear()
    harnesses = []

    async def _make(**kwargs):
        harness = await build_harness(**kwargs)
        harnesses.append(harness)
        return harness

    yield _make

    for harness in harnesses:
        await harness.shutdown()
    _STREAM_MESSAGES.clear()


@pytest.mark.asyncio
async def test_pair_then_message_streams_reply(tg):
    agent = ScriptedAgent([Turn(chunks=[text_chunk("Hello "), text_chunk("world!")])])
    harness = await tg(agent=agent, allowed_users=set(), pairing_code="codecode")

    await harness.process(harness.command_update("/pair codecode"))
    assert any("Paired successfully" in text for text in harness.api.sent_texts())
    assert "1" in harness.app.bot_data["allowed_users"]
    assert harness.app.bot_data["pairing_code"] != "codecode"

    await harness.process(harness.text_update("hi there"))
    await harness.drain()

    sent = harness.api.sent_texts()
    assert any(text == "Thinking..." for text in sent)
    edits = harness.api.edited_texts()
    assert edits, "expected streaming edits"
    assert "Hello world" in edits[-1]
    assert harness.app.bot_data[THREAD_IDS_KEY].get(CHAT)
    assert agent.astream_payloads[0]["messages"][0]["content"] == "hi there"


@pytest.mark.asyncio
async def test_unpaired_user_rejected(tg):
    agent = ScriptedAgent([])
    harness = await tg(agent=agent, allowed_users=set())

    await harness.process(harness.text_update("hello?"))
    await harness.drain()

    assert any(REJECTION_MESSAGE in text for text in harness.api.sent_texts())
    assert agent.astream_payloads == []


@pytest.mark.asyncio
async def test_approval_interrupt_then_inline_approve_resumes(tg):
    agent = ScriptedAgent(
        [
            Turn(
                chunks=[text_chunk("About to run a command.")],
                interrupts=(safety_interrupt(interrupt_id="int-1"),),
            ),
            Turn(chunks=[text_chunk("Command approved and done.")]),
        ]
    )
    harness = await tg(agent=agent)

    await harness.process(harness.text_update("clean up my scratch dir"))
    await harness.drain()

    pending = harness.app.bot_data[PENDING_APPROVALS_KEY].get(CHAT)
    assert pending is not None
    assert pending["id"] == "int-1"

    markup_sends = [
        params
        for params in harness.api.calls_for("sendMessage")
        if "safety:approve_once:int-1" in json.dumps(params.get("reply_markup", {}))
    ]
    assert markup_sends, "expected approval prompt with inline buttons"

    await harness.process(harness.callback_update("safety:approve_once:int-1"))
    await harness.drain()

    assert harness.api.calls_for("answerCallbackQuery")
    assert len(agent.astream_payloads) == 2
    resume = agent.astream_payloads[1]
    assert isinstance(resume, Command)
    assert resume.resume == {"type": "approve", "scope": "once"}
    assert CHAT not in harness.app.bot_data[PENDING_APPROVALS_KEY]
    assert harness.api.calls_for("editMessageReplyMarkup"), "expected buttons cleared"
    delivered = harness.api.edited_texts() + harness.api.sent_texts()
    assert any("Command approved and done" in text for text in delivered)


@pytest.mark.asyncio
async def test_stop_mid_run_cancels_and_notifies(tg):
    pause = Pause()
    agent = ScriptedAgent([Turn(chunks=[text_chunk("Working on it"), pause])])
    harness = await tg(agent=agent)

    await harness.process(harness.text_update("do something slow"))
    await agent.started.wait()

    await harness.process(harness.command_update("/stop"))
    await harness.drain()

    assert any("Stopped the current task." in text for text in harness.api.sent_texts())
    assert not harness.app.bot_data.get(ACTIVE_RUNS_KEY, {})
    assert CHAT not in _STREAM_MESSAGES


@pytest.mark.asyncio
async def test_message_queued_during_active_run_then_drained(tg):
    pause = Pause()
    agent = ScriptedAgent(
        [
            Turn(chunks=[text_chunk("First response"), pause]),
            Turn(chunks=[text_chunk("Second response")]),
        ]
    )
    harness = await tg(agent=agent)

    await harness.process(harness.text_update("first task"))
    await agent.started.wait()

    await harness.process(harness.text_update("second task"))
    await harness.wait_until(
        lambda: any("Queued request #1" in text for text in harness.api.sent_texts())
    )
    assert len(agent.astream_payloads) == 1

    pause.release()
    await harness.drain()

    assert len(agent.astream_payloads) == 2
    assert agent.astream_payloads[1]["messages"][0]["content"] == "second task"
    assert CHAT not in harness.app.bot_data.get(QUEUED_RUNS_KEY, {})
    delivered = harness.api.edited_texts() + harness.api.sent_texts()
    assert any("Second response" in text for text in delivered)


@pytest.mark.asyncio
async def test_queue_full_rejects_sixth_request(tg):
    pause = Pause()
    agent = ScriptedAgent(
        [Turn(chunks=[text_chunk("First response"), pause])]
        + [Turn(chunks=[text_chunk(f"Queued response {i}")]) for i in range(1, 6)]
    )
    harness = await tg(agent=agent)

    await harness.process(harness.text_update("first task"))
    await agent.started.wait()

    for i in range(1, 6):
        await harness.process(harness.text_update(f"queued task {i}"))
    await harness.wait_until(
        lambda: any("Queued request #5" in text for text in harness.api.sent_texts())
    )

    await harness.process(harness.text_update("one too many"))
    await harness.wait_until(
        lambda: any("Queue is full" in text for text in harness.api.sent_texts())
    )

    pause.release()
    await harness.drain()

    assert len(agent.astream_payloads) == 6
    queued_texts = [payload["messages"][0]["content"] for payload in agent.astream_payloads[1:]]
    assert queued_texts == [f"queued task {i}" for i in range(1, 6)]


@pytest.mark.asyncio
async def test_retry_after_on_edit_falls_back_to_fresh_send(tg):
    agent = ScriptedAgent([Turn(chunks=[text_chunk("Hello"), text_chunk(" world")])])
    harness = await tg(agent=agent)
    harness.api.enqueue_error(
        "editMessageText", status=429, description="Flood control exceeded", retry_after=1
    )

    await harness.process(harness.text_update("hi"))
    await harness.drain()

    assert len(harness.api.calls_for("editMessageText")) == 1
    assert any("Hello world" in text for text in harness.api.sent_texts())
    snapshot = harness.gateway.get_queue_snapshot(CHAT)
    assert snapshot is not None
    assert snapshot["final_delivery_mode"] == "fresh_send"
    assert snapshot["stream_degraded_reason"] == "edit_rate_limited"


@pytest.mark.asyncio
async def test_fatal_agent_error_surfaces_to_user(tg):
    agent = ScriptedAgent([Turn(raises=RuntimeError("boom"))])
    harness = await tg(agent=agent)

    await harness.process(harness.text_update("hi"))
    await harness.drain()

    delivered = harness.api.edited_texts() + harness.api.sent_texts()
    assert any("Run failed: RuntimeError: boom" in text for text in delivered)
    assert not harness.app.bot_data.get(ACTIVE_RUNS_KEY, {})


@pytest.mark.asyncio
async def test_fatal_send_error_notifies_user(tg):
    agent = ScriptedAgent([Turn(chunks=[text_chunk("never delivered")])])
    harness = await tg(agent=agent)
    harness.api.enqueue_error("sendMessage", status=400, description="Bad Request: chat not found")

    await harness.process(harness.text_update("hi"))
    await harness.drain()

    assert any(text.startswith("Run failed:") for text in harness.api.sent_texts())
    assert not harness.app.bot_data.get(ACTIVE_RUNS_KEY, {})


@pytest.mark.asyncio
async def test_command_routing_smoke(tg):
    agent = ScriptedAgent([])
    harness = await tg(agent=agent)

    await harness.process(harness.command_update("/help"))
    await harness.process(harness.command_update("/uptime"))
    await harness.process(harness.command_update("/status"))

    sent = harness.api.sent_texts()
    assert any(text.startswith("Available commands:") for text in sent)
    assert any(text.startswith("Uptime:") for text in sent)
    assert any(f"Chat ID: {CHAT}" in text for text in sent)
