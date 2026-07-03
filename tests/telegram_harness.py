"""Offline end-to-end harness for the Telegram integration.

Builds a real python-telegram-bot Application wired with the production
handlers (deepclaw.channels.telegram.register_handlers), backed by a
FakeTelegramRequest that answers Bot API calls in-process and a ScriptedAgent
behind the real Gateway. Updates flow through Application.process_update, so
handler routing, streaming edits, approval interrupts, queueing, and
rate-limit fallbacks all execute exactly as in production — without network,
tokens, or an LLM.
"""

import asyncio
import json
import time
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from telegram import Update
from telegram.ext import Application
from telegram.request import BaseRequest, RequestData
from telegram.warnings import PTBUserWarning

from deepclaw.channels.telegram import (
    ALLOWED_USERS_KEY,
    CONFIG_KEY,
    GATEWAY_KEY,
    PAIRING_CODE_KEY,
    THREAD_IDS_KEY,
    register_handlers,
)
from deepclaw.config import DeepClawConfig, TelegramConfig, TelegramStreamingConfig
from deepclaw.gateway import Gateway

BOT_ID = 424242
BOT_USERNAME = "deepclaw_test_bot"
DEFAULT_CHAT_ID = 1001
DEFAULT_USER_ID = 1


class FakeTelegramRequest(BaseRequest):
    """In-process Bot API stub: records every call and returns canned payloads.

    Returning a non-2xx body makes python-telegram-bot raise the genuine error
    (e.g. RetryAfter for a 429 with a retry_after parameter), so production
    error paths execute unmocked.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._errors: dict[str, list[tuple[int, bytes]]] = {}
        self._next_message_id = 1000

    @property
    def read_timeout(self) -> float | None:
        return None

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def enqueue_error(
        self,
        api_method: str,
        *,
        status: int = 429,
        description: str = "Too Many Requests",
        retry_after: int | None = None,
    ) -> None:
        """Queue an error response for the next call to the given API method."""
        body: dict[str, Any] = {"ok": False, "error_code": status, "description": description}
        if retry_after is not None:
            body["parameters"] = {"retry_after": retry_after}
        self._errors.setdefault(api_method, []).append((status, json.dumps(body).encode()))

    def calls_for(self, api_method: str) -> list[dict[str, Any]]:
        return [params for method, params in self.calls if method == api_method]

    def sent_texts(self) -> list[str]:
        return [str(params.get("text", "")) for params in self.calls_for("sendMessage")]

    def edited_texts(self) -> list[str]:
        return [str(params.get("text", "")) for params in self.calls_for("editMessageText")]

    def _message_result(self, params: dict[str, Any]) -> dict[str, Any]:
        self._next_message_id += 1
        message_id = params.get("message_id", self._next_message_id)
        chat_id = int(params.get("chat_id", DEFAULT_CHAT_ID))
        result: dict[str, Any] = {
            "message_id": int(message_id),
            "date": int(time.time()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {
                "id": BOT_ID,
                "is_bot": True,
                "first_name": "DeepClaw",
                "username": BOT_USERNAME,
            },
        }
        if "text" in params:
            result["text"] = str(params["text"])
        return result

    async def do_request(
        self,
        url: str,
        method: str,
        request_data: RequestData | None = None,
        read_timeout=None,
        write_timeout=None,
        connect_timeout=None,
        pool_timeout=None,
    ) -> tuple[int, bytes]:
        api_method = url.rsplit("/", 1)[-1]
        params = dict(request_data.parameters) if request_data is not None else {}
        self.calls.append((api_method, params))

        queued = self._errors.get(api_method)
        if queued:
            return queued.pop(0)

        result: Any
        if api_method == "getMe":
            result = {
                "id": BOT_ID,
                "is_bot": True,
                "first_name": "DeepClaw",
                "username": BOT_USERNAME,
                "can_join_groups": True,
                "can_read_all_group_messages": False,
                "supports_inline_queries": False,
            }
        elif api_method in {
            "sendMessage",
            "editMessageText",
            "sendPhoto",
            "sendDocument",
            "sendVoice",
            "sendAudio",
            "sendVideo",
        }:
            result = self._message_result(params)
        elif api_method in {
            "sendChatAction",
            "answerCallbackQuery",
            "editMessageReplyMarkup",
            "deleteMessage",
            "setMyCommands",
        }:
            result = True
        elif api_method == "getMyCommands":
            result = []
        else:
            body = {
                "ok": False,
                "error_code": 404,
                "description": f"Not Found: method {api_method}",
            }
            return 404, json.dumps(body).encode()

        return 200, json.dumps({"ok": True, "result": result}).encode()


class Pause:
    """Sentinel chunk: the scripted agent blocks here until release() is called."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def release(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


@dataclass
class Turn:
    """One scripted agent invocation: chunks to stream, then optional outcomes."""

    chunks: list[Any] = field(default_factory=list)
    interrupts: tuple[Any, ...] = ()
    raises: Exception | None = None


def text_chunk(text: str) -> tuple[Any, dict]:
    """Build a streamed text chunk in the shape the gateway consumes."""
    return (SimpleNamespace(content_blocks=[{"type": "text", "text": text}]), {})


def tool_chunk(name: str, args: dict | None = None) -> tuple[Any, dict]:
    """Build a streamed tool-call chunk in the shape the gateway consumes."""
    block = {"type": "tool_call", "id": f"call-{name}", "name": name, "args": args or {}}
    return (SimpleNamespace(content_blocks=[block]), {})


def safety_interrupt(
    interrupt_id: str = "int-1",
    command: str = "rm -rf /tmp/scratch",
    approval_keys: tuple[str, ...] = ("execute:rm",),
) -> SimpleNamespace:
    """Build a safety-review interrupt matching gateway._extract_pending_interrupt."""
    return SimpleNamespace(
        id=interrupt_id,
        value={
            "type": "safety_review",
            "tool": "execute",
            "command": command,
            "approval_keys": list(approval_keys),
            "warning": "Dangerous command pattern",
            "message": f"Safety review required for:\n{command}\nApprove?",
        },
    )


class ScriptedAgent:
    """Multi-turn fake agent driven through the real Gateway.

    Each astream() call consumes the next Turn; aget_state() reports the
    current turn's interrupts so approval flows work across resume calls.
    """

    def __init__(self, turns: list[Turn]) -> None:
        self._turns = list(turns)
        self._current: Turn | None = None
        self.astream_payloads: list[Any] = []
        self.started = asyncio.Event()

    async def astream(self, payload, config=None, stream_mode=None, subgraphs=False):
        self.astream_payloads.append(payload)
        if not self._turns:
            raise AssertionError("ScriptedAgent ran out of scripted turns")
        turn = self._turns.pop(0)
        self._current = turn
        self.started.set()
        for item in turn.chunks:
            if isinstance(item, Pause):
                await item.wait()
                continue
            yield item
        if turn.raises is not None:
            raise turn.raises

    async def aget_state(self, config):
        turn = self._current
        return SimpleNamespace(interrupts=tuple(turn.interrupts) if turn else ())


class TrackedApplication(Application):
    """Application that records tasks spawned for block=False handlers.

    Application defines __slots__, so create_task cannot be monkeypatched on an
    instance; subclassing via ApplicationBuilder.application_class() is the
    supported extension point.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tracked_handler_tasks: list[asyncio.Task] = []

    def create_task(self, coroutine, update=None, *, name=None):
        # PTB warns that tasks created while the app is not running are not
        # auto-awaited; the harness awaits them explicitly via drain().
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PTBUserWarning)
            task = super().create_task(coroutine, update=update, name=name)
        self.tracked_handler_tasks.append(task)
        return task


class TelegramHarness:
    """Drives real Updates through Application.process_update and awaits handlers."""

    def __init__(
        self,
        app: TrackedApplication,
        api: FakeTelegramRequest,
        agent: ScriptedAgent,
        gateway: Gateway,
    ) -> None:
        self.app = app
        self.api = api
        self.agent = agent
        self.gateway = gateway
        self._update_id = 0
        self._message_id = 0
        self._handler_tasks = app.tracked_handler_tasks

    @staticmethod
    def _user(user_id: int, username: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": user_id, "is_bot": False, "first_name": "Test"}
        if username:
            payload["username"] = username
        return payload

    def _message_payload(
        self,
        text: str,
        *,
        chat_id: int,
        user_id: int,
        username: str | None,
        entities: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._message_id += 1
        payload: dict[str, Any] = {
            "message_id": self._message_id,
            "date": int(time.time()),
            "chat": {"id": chat_id, "type": "private"},
            "from": self._user(user_id, username),
            "text": text,
        }
        if entities:
            payload["entities"] = entities
        return payload

    def text_update(
        self,
        text: str,
        *,
        chat_id: int = DEFAULT_CHAT_ID,
        user_id: int = DEFAULT_USER_ID,
        username: str | None = "tester",
    ) -> Update:
        self._update_id += 1
        payload = {
            "update_id": self._update_id,
            "message": self._message_payload(
                text, chat_id=chat_id, user_id=user_id, username=username
            ),
        }
        return Update.de_json(payload, self.app.bot)

    def command_update(
        self,
        text: str,
        *,
        chat_id: int = DEFAULT_CHAT_ID,
        user_id: int = DEFAULT_USER_ID,
        username: str | None = "tester",
    ) -> Update:
        # CommandHandler only matches when a bot_command entity covers the
        # first token, exactly as real Telegram clients send it.
        first_token = text.split()[0]
        entities = [{"type": "bot_command", "offset": 0, "length": len(first_token)}]
        self._update_id += 1
        payload = {
            "update_id": self._update_id,
            "message": self._message_payload(
                text, chat_id=chat_id, user_id=user_id, username=username, entities=entities
            ),
        }
        return Update.de_json(payload, self.app.bot)

    def callback_update(
        self,
        data: str,
        *,
        chat_id: int = DEFAULT_CHAT_ID,
        user_id: int = DEFAULT_USER_ID,
        username: str | None = "tester",
    ) -> Update:
        self._update_id += 1
        message = self._message_payload(
            "Safety review required", chat_id=chat_id, user_id=user_id, username=username
        )
        payload = {
            "update_id": self._update_id,
            "callback_query": {
                "id": f"cbq-{self._update_id}",
                "from": self._user(user_id, username),
                "chat_instance": "ci-1",
                "data": data,
                "message": message,
            },
        }
        return Update.de_json(payload, self.app.bot)

    async def process(self, update: Update) -> None:
        await self.app.process_update(update)

    async def drain(self, timeout: float = 10.0) -> None:
        """Await all handler tasks (including ones they spawn) to completion."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            pending = [task for task in self._handler_tasks if not task.done()]
            if not pending:
                return
            remaining = deadline - loop.time()
            assert remaining > 0, "handler tasks did not finish before drain() timeout"
            await asyncio.wait(pending, timeout=remaining)

    async def wait_until(self, predicate, timeout: float = 5.0) -> None:
        """Poll until predicate() is truthy; fail the test on timeout."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate():
            assert loop.time() < deadline, "condition not met before wait_until() timeout"
            await asyncio.sleep(0.01)

    async def shutdown(self) -> None:
        await self.app.shutdown()


async def build_harness(
    *,
    agent: ScriptedAgent,
    allowed_users: frozenset[str] | set[str] = frozenset({str(DEFAULT_USER_ID)}),
    pairing_code: str = "codecode",
    rich_messages: bool = False,
) -> TelegramHarness:
    """Build an initialized Application with production handlers and a fake Bot API."""
    api = FakeTelegramRequest()
    app = (
        Application.builder()
        .application_class(TrackedApplication)
        .token("123456:TEST")
        .request(api)
        .get_updates_request(FakeTelegramRequest())
        .build()
    )
    register_handlers(app)

    config = DeepClawConfig(
        telegram=TelegramConfig(
            rich_messages=rich_messages,
            # Deterministic streaming: edit after every chunk.
            streaming=TelegramStreamingConfig(edit_interval=0.0, buffer_threshold=1),
        )
    )
    gateway = Gateway(
        agent=agent,
        streaming_config=config.telegram.streaming,
        max_turns=config.max_turns,
        gateway_timeout=config.gateway_timeout,
        gateway_timeout_warning=config.gateway_timeout_warning,
    )
    app.bot_data[CONFIG_KEY] = config
    app.bot_data[ALLOWED_USERS_KEY] = set(allowed_users)
    app.bot_data[THREAD_IDS_KEY] = {}
    app.bot_data[PAIRING_CODE_KEY] = pairing_code
    app.bot_data[GATEWAY_KEY] = gateway

    await app.initialize()
    return TelegramHarness(app, api, agent, gateway)
