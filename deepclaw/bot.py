"""Telegram bot that wires DeepAgents to python-telegram-bot.

Prototype phase 0.1: single-script blocking request/response loop.
Uses Telegram chat_id as LangGraph thread_id for conversation persistence.
"""

import logging
import os
import uuid

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from telegram import Update
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


def get_agent_text(result: dict) -> str:
    """Extract the final text response from an agent invocation result."""
    last_message = result["messages"][-1]
    if isinstance(last_message, AIMessage):
        return last_message.text or ""
    return str(last_message.content)


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
    chat_id = str(update.effective_chat.id)
    new_thread = str(uuid.uuid4())
    context.bot_data.setdefault(THREAD_IDS_KEY, {})[chat_id] = new_thread
    logger.info(f"New thread for chat {chat_id}: {new_thread}")
    await update.message.reply_text(f"Started a new conversation thread.\nThread ID: {new_thread}")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — list available commands."""
    text = (
        "Available commands:\n"
        "/new — Start a fresh conversation thread\n"
        "/status — Show current thread ID and model info\n"
        "/help — Show this help message\n"
        "/start — Welcome message"
    )
    await update.message.reply_text(text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — show current thread ID and model info."""
    chat_id = str(update.effective_chat.id)
    thread_id = get_thread_id(context, chat_id)
    model = os.environ.get("DEEPCLAW_MODEL", "not set")
    text = f"Chat ID: {chat_id}\nThread ID: {thread_id}\nModel: {model}"
    await update.message.reply_text(text)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — Telegram's default start command."""
    await update.message.reply_text(
        "Welcome to DeepClaw! Send me a message and I'll respond using AI.\n"
        "Type /help to see available commands."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle an incoming Telegram message by passing it to the DeepAgents agent."""
    if not update.message or not update.message.text:
        return

    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    logger.info(f"Received message from chat {chat_id}: {user_text[:80]}")

    agent = context.bot_data["agent"]
    thread_id = get_thread_id(context, chat_id)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_text}]},
            config=config,
        )
        response_text = get_agent_text(result)
    except Exception:
        logger.exception("Agent invocation failed")
        response_text = "Sorry, something went wrong processing your message."

    if not response_text:
        response_text = "(no response)"

    for chunk in chunk_message(response_text):
        await update.message.reply_text(chunk)


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
