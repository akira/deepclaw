"""Telegram bot that wires DeepAgents to python-telegram-bot.

Uses Telegram chat_id as LangGraph thread_id for conversation persistence.
Streams agent responses by progressively editing a single Telegram message.
"""

import logging
import os
import time
import uuid

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
THREAD_IDS_KEY = "thread_ids"
ALLOWED_USERS_KEY = "allowed_users"
STREAM_EDIT_INTERVAL = 1.0  # minimum seconds between message edits
STREAM_CHAR_THRESHOLD = 100  # characters accumulated before forcing an edit
CURSOR_INDICATOR = "▌"
THINKING_MESSAGE = "Thinking..."
REJECTION_MESSAGE = "Sorry, you are not authorized to use this bot."


def _parse_allowed_users() -> set[str]:
    """Parse DEEPCLAW_ALLOWED_USERS env var into a set of user IDs/usernames."""
    raw = os.environ.get("DEEPCLAW_ALLOWED_USERS", "").strip()
    if not raw:
        return set()
    return {u.strip() for u in raw.split(",") if u.strip()}


def is_user_allowed(update: Update, allowed_users: set[str]) -> bool:
    """Check if the effective user is in the allowlist.

    Returns True if the allowlist is empty (open mode) or the user matches.
    """
    if not allowed_users:
        return True
    user = update.effective_user
    if user is None:
        return False
    if str(user.id) in allowed_users:
        return True
    if user.username and user.username in allowed_users:
        return True
    logger.info(f"Rejected user id={user.id} username={user.username}")
    return False


def chunk_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split text into chunks that fit within Telegram's message size limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
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


def get_thread_id(context: ContextTypes.DEFAULT_TYPE, chat_id: str) -> str:
    """Return the current thread_id for a chat, defaulting to the chat_id itself."""
    thread_ids: dict[str, str] = context.bot_data.setdefault(THREAD_IDS_KEY, {})
    return thread_ids.get(chat_id, chat_id)


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new — start a fresh conversation thread."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    new_thread = str(uuid.uuid4())
    context.bot_data.setdefault(THREAD_IDS_KEY, {})[chat_id] = new_thread
    logger.info(f"New thread for chat {chat_id}: {new_thread}")
    await update.message.reply_text(f"Started a new conversation thread.\nThread ID: {new_thread}")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — list available commands."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    allowed_users = context.bot_data.get(ALLOWED_USERS_KEY, set())
    text = (
        "Available commands:\n"
        "/new — Start a fresh conversation thread\n"
        "/status — Show current thread ID and model info\n"
        "/help — Show this help message\n"
        "/start — Welcome message"
    )
    if allowed_users:
        text += "\n\nAccess control is active. Only approved users can interact with this bot."
    await update.message.reply_text(text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show current thread ID and model info."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    thread_id = get_thread_id(context, chat_id)
    model = os.environ.get("DEEPCLAW_MODEL", "not set")
    allowed_users = context.bot_data.get(ALLOWED_USERS_KEY, set())
    allowlist_status = f"active ({len(allowed_users)} users)" if allowed_users else "inactive (open mode)"
    text = f"Chat ID: {chat_id}\nThread ID: {thread_id}\nModel: {model}\nAllowlist: {allowlist_status}"
    await update.message.reply_text(text)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — Telegram's default start command."""
    await update.message.reply_text(
        "Welcome to DeepClaw! Send me a message and I'll respond using AI.\n"
        "Type /help to see available commands."
    )


async def _edit_stream_message(message, text: str) -> None:
    """Edit a Telegram message, ignoring errors when content is unchanged."""
    try:
        await message.edit_text(text)
    except BadRequest as exc:
        if "Message is not modified" not in str(exc):
            raise


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle an incoming Telegram message by streaming the agent response."""
    if not update.message or not update.message.text:
        return

    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    logger.info(f"Received message from chat {chat_id}: {user_text[:80]}")

    await update.effective_chat.send_action(ChatAction.TYPING)

    agent = context.bot_data["agent"]
    thread_id = get_thread_id(context, chat_id)
    config = {"configurable": {"thread_id": thread_id}}

    stream_msg = await update.message.reply_text(THINKING_MESSAGE)
    accumulated = ""
    last_edit_time = time.monotonic()
    chars_since_edit = 0

    try:
        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": user_text}]},
            config=config,
            version="v2",
        ):
            if event["event"] != "on_chat_model_stream":
                continue
            chunk = event["data"].get("chunk")
            if chunk is None:
                continue
            content = chunk.content if hasattr(chunk, "content") else ""
            if not content or not isinstance(content, str):
                continue

            accumulated += content
            chars_since_edit += len(content)
            now = time.monotonic()
            elapsed = now - last_edit_time

            if elapsed >= STREAM_EDIT_INTERVAL or chars_since_edit >= STREAM_CHAR_THRESHOLD:
                display = accumulated + CURSOR_INDICATOR
                if len(display) <= TELEGRAM_MESSAGE_LIMIT:
                    await _edit_stream_message(stream_msg, display)
                last_edit_time = time.monotonic()
                chars_since_edit = 0
    except Exception:
        logger.exception("Agent streaming failed")
        accumulated = accumulated or "Sorry, something went wrong processing your message."

    response_text = accumulated if accumulated else "(no response)"

    # Final edit: send complete text (possibly chunked)
    chunks = chunk_message(response_text)
    await _edit_stream_message(stream_msg, chunks[0])
    for extra_chunk in chunks[1:]:
        await update.message.reply_text(extra_chunk)


async def post_init(application: Application) -> None:
    """Create the DeepAgents agent after the application is initialized."""
    checkpointer = application.bot_data["checkpointer"]
    await checkpointer.__aenter__()

    model = os.environ.get("DEEPCLAW_MODEL")
    backend = LocalShellBackend(virtual_mode=False, inherit_env=True)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        checkpointer=checkpointer,
    )
    application.bot_data["agent"] = agent

    allowed_users = _parse_allowed_users()
    application.bot_data[ALLOWED_USERS_KEY] = allowed_users
    if allowed_users:
        logger.info(f"Allowlist active with {len(allowed_users)} users")
    else:
        logger.info("Allowlist inactive — open mode")

    logger.info("DeepAgents agent initialized")


async def post_shutdown(application: Application) -> None:
    """Clean up the checkpointer on shutdown."""
    checkpointer = application.bot_data.get("checkpointer")
    if checkpointer:
        await checkpointer.__aexit__(None, None, None)


def main() -> None:
    """Entry point: start the Telegram bot with long-polling."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN environment variable is required")

    db_path = os.path.expanduser("~/.deepagents/checkpoints.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    checkpointer = AsyncSqliteSaver.from_conn_string(db_path)

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.bot_data["checkpointer"] = checkpointer

    application.add_handler(CommandHandler("new", cmd_new))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting DeepClaw Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
