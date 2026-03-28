"""Telegram bot that wires DeepAgents to python-telegram-bot.

Uses Telegram chat_id as LangGraph thread_id for conversation persistence.
Streams agent responses by progressively editing a single Telegram message.
"""

import logging
import os
import sys
import time
import uuid
from pathlib import Path

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

from deepclaw.config import load_config
from deepclaw.safety import check_command, format_warning
from deepclaw.scheduler import (
    Scheduler,
    add_job,
    list_jobs,
    load_jobs,
    parse_cron_add,
    remove_job,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
THREAD_IDS_KEY = "thread_ids"
ALLOWED_USERS_KEY = "allowed_users"
CONFIG_KEY = "deepclaw_config"
CURSOR_INDICATOR = "▌"
THINKING_MESSAGE = "Thinking..."
REJECTION_MESSAGE = "Sorry, you are not authorized to use this bot."
SCHEDULER_KEY = "scheduler"
JOBS_PATH_KEY = "jobs_path"


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
        "/safety_test <cmd> — Check a command for dangerous patterns\n"
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
    model = context.bot_data[CONFIG_KEY].model or "not set"
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


async def cmd_cron(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cron -- list all cron jobs."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    jobs_path = context.bot_data.get(JOBS_PATH_KEY)
    jobs = list_jobs(jobs_path) if jobs_path else []
    if not jobs:
        await update.message.reply_text("No cron jobs configured.")
        return
    lines = []
    for job in jobs:
        status = "enabled" if job.enabled else "disabled"
        lines.append(f"- {job.id[:8]} | {job.cron_expr} | {status} | {job.name or job.prompt[:40]}")
    await update.message.reply_text("Cron jobs:\n" + "\n".join(lines))


async def cmd_cron_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cron_add <cron_expr> | <prompt>."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    raw = update.message.text
    # Strip the /cron_add command prefix
    prefix = "/cron_add"
    if raw.startswith(prefix):
        raw = raw[len(prefix):].strip()
    if not raw:
        await update.message.reply_text("Usage: /cron_add <cron_expr> | <prompt>")
        return
    try:
        cron_expr, prompt = parse_cron_add(raw)
    except ValueError as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    chat_id = str(update.effective_chat.id)
    jobs_path = context.bot_data.get(JOBS_PATH_KEY)
    job = add_job(
        name=prompt[:50],
        cron_expr=cron_expr,
        prompt=prompt,
        delivery={"channel": "telegram", "chat_id": chat_id},
        path=jobs_path,
    )
    await update.message.reply_text(f"Cron job added: {job.id[:8]}\nSchedule: {cron_expr}\nPrompt: {prompt}")


async def cmd_cron_rm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cron_rm <job_id_prefix>."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    raw = update.message.text
    prefix = "/cron_rm"
    if raw.startswith(prefix):
        raw = raw[len(prefix):].strip()
    if not raw:
        await update.message.reply_text("Usage: /cron_rm <job_id_prefix>")
        return
    id_prefix = raw.strip()
    jobs_path = context.bot_data.get(JOBS_PATH_KEY)
    jobs = load_jobs(jobs_path) if jobs_path else []
    matches = [j for j in jobs if j.id.startswith(id_prefix)]
    if len(matches) == 0:
        await update.message.reply_text(f"No job found matching prefix: {id_prefix}")
        return
    if len(matches) > 1:
        await update.message.reply_text(f"Ambiguous prefix: {id_prefix} matches {len(matches)} jobs. Be more specific.")
        return
    removed = remove_job(matches[0].id, jobs_path)
    if removed:
        await update.message.reply_text(f"Removed cron job: {matches[0].id[:8]}")
    else:
        await update.message.reply_text("Failed to remove job.")


async def cmd_doctor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /doctor — run diagnostic checks and report results."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    from deepclaw.doctor import format_report, run_checks

    config = context.bot_data[CONFIG_KEY]
    checks = await run_checks(config)
    report = format_report(checks)
    for chunk in chunk_message(report):
        await update.message.reply_text(chunk)


async def cmd_safety_test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /safety_test <command> — check a command for dangerous patterns."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    raw = update.message.text
    prefix = "/safety_test"
    if raw.startswith(prefix):
        raw = raw[len(prefix):].strip()
    if not raw:
        await update.message.reply_text("Usage: /safety_test <command>")
        return
    matches = check_command(raw)
    if not matches:
        await update.message.reply_text(f"No dangerous patterns detected in: {raw}")
        return
    await update.message.reply_text(format_warning(raw, matches))


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
    streaming_cfg = context.bot_data[CONFIG_KEY].telegram.streaming

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

            if elapsed >= streaming_cfg.edit_interval or chars_since_edit >= streaming_cfg.buffer_threshold:
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
    config = application.bot_data[CONFIG_KEY]

    checkpointer = application.bot_data["checkpointer"]
    await checkpointer.__aenter__()

    backend = LocalShellBackend(virtual_mode=False, inherit_env=True)

    agent = create_deep_agent(
        model=config.model or None,
        backend=backend,
        checkpointer=checkpointer,
    )
    application.bot_data["agent"] = agent

    allowed_users = set(config.telegram.allowed_users)
    application.bot_data[ALLOWED_USERS_KEY] = allowed_users
    if allowed_users:
        logger.info(f"Allowlist active with {len(allowed_users)} users")
    else:
        logger.info("Allowlist inactive — open mode")

    jobs_path = application.bot_data.get(JOBS_PATH_KEY)
    if jobs_path:
        scheduler = Scheduler(
            jobs_path=jobs_path,
            agent=agent,
            checkpointer=checkpointer,
            bot=application.bot,
        )
        application.bot_data[SCHEDULER_KEY] = scheduler
        await scheduler.start()

    logger.info("DeepAgents agent initialized")


async def post_shutdown(application: Application) -> None:
    """Clean up the scheduler and checkpointer on shutdown."""
    scheduler = application.bot_data.get(SCHEDULER_KEY)
    if scheduler:
        await scheduler.stop()
    checkpointer = application.bot_data.get("checkpointer")
    if checkpointer:
        await checkpointer.__aexit__(None, None, None)


def _handle_doctor_command() -> None:
    """Handle 'deepclaw doctor' CLI command."""
    import asyncio

    from deepclaw.doctor import format_report, run_checks

    config = load_config()
    checks = asyncio.run(run_checks(config))
    print(format_report(checks))


def _handle_service_command(args: list[str]) -> None:
    """Handle 'deepclaw service <subcommand>' CLI commands."""
    from deepclaw.service import (
        detect_platform,
        install_service,
        service_status,
        uninstall_service,
    )

    plat = detect_platform()

    if not args:
        print("Usage: deepclaw service {install|uninstall|status}")
        raise SystemExit(1)

    subcommand = args[0]
    if subcommand == "install":
        print(install_service(plat))
    elif subcommand == "uninstall":
        print(uninstall_service(plat))
    elif subcommand == "status":
        print(service_status(plat))
    else:
        print(f"Unknown service subcommand: {subcommand}")
        print("Usage: deepclaw service {install|uninstall|status}")
        raise SystemExit(1)


def main() -> None:
    """Entry point: start the Telegram bot with long-polling."""
    args = sys.argv[1:]
    if args and args[0] == "service":
        _handle_service_command(args[1:])
        return
    if args and args[0] == "doctor":
        _handle_doctor_command()
        return

    config = load_config()

    token = config.telegram.bot_token
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required (set via env var, .env, or config.yaml)")

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
    jobs_path = Path(os.path.expanduser("~/.deepclaw/cron/jobs.json"))
    jobs_path.parent.mkdir(parents=True, exist_ok=True)

    application.bot_data["checkpointer"] = checkpointer
    application.bot_data[CONFIG_KEY] = config
    application.bot_data[JOBS_PATH_KEY] = jobs_path

    application.add_handler(CommandHandler("new", cmd_new))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("cron", cmd_cron))
    application.add_handler(CommandHandler("cron_add", cmd_cron_add))
    application.add_handler(CommandHandler("cron_rm", cmd_cron_rm))
    application.add_handler(CommandHandler("safety_test", cmd_safety_test))
    application.add_handler(CommandHandler("doctor", cmd_doctor))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting DeepClaw Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
