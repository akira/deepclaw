"""Telegram bot channel for DeepClaw.

Uses Telegram chat_id as LangGraph thread_id for conversation persistence.
Streams agent responses by progressively editing a single Telegram message.
"""

import logging
import time
import uuid

from langchain_core.messages import ToolMessage
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

from deepclaw.agent import create_agent, create_checkpointer
from deepclaw.auth import (
    REJECTION_MESSAGE,
    is_user_allowed,
    load_allowed_users,
    save_allowed_users,
)
from deepclaw.safety import check_command, format_warning
from deepclaw.scheduler import (
    Scheduler,
    add_job,
    list_jobs,
    load_jobs,
    parse_cron_add,
    remove_job,
)

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
THREAD_IDS_KEY = "thread_ids"
ALLOWED_USERS_KEY = "allowed_users"
PAIRING_CODE_KEY = "pairing_code"
CONFIG_KEY = "deepclaw_config"
CURSOR_INDICATOR = "\u258c"
THINKING_MESSAGE = "Thinking..."
SCHEDULER_KEY = "scheduler"
JOBS_PATH_KEY = "jobs_path"


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
    """Handle /new -- start a fresh conversation thread."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    new_thread = str(uuid.uuid4())
    context.bot_data.setdefault(THREAD_IDS_KEY, {})[chat_id] = new_thread
    logger.info(f"New thread for chat {chat_id}: {new_thread}")
    await update.message.reply_text(f"Started a new conversation thread.\nThread ID: {new_thread}")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help -- list available commands."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    allowed_users = context.bot_data.get(ALLOWED_USERS_KEY, set())
    text = (
        "Available commands:\n"
        "/new \u2014 Start a fresh conversation thread\n"
        "/status \u2014 Show current thread ID and model info\n"
        "/safety_test <cmd> \u2014 Check a command for dangerous patterns\n"
        "/cron \u2014 List scheduled jobs\n"
        "/cron_add <expr> | <prompt> \u2014 Add a scheduled job\n"
        "/cron_rm <id_prefix> \u2014 Remove a scheduled job\n"
        "/doctor \u2014 Run system diagnostics\n"
        "/help \u2014 Show this help message"
    )
    text += f"\n\n{len(allowed_users)} paired user(s)."
    await update.message.reply_text(text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status -- show current thread ID and model info."""
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
    """Handle /start -- Telegram's default start command."""
    allowed_users = context.bot_data.get(ALLOWED_USERS_KEY, set())
    if is_user_allowed(update, allowed_users):
        await update.message.reply_text(
            "Welcome to DeepClaw! Send me a message and I'll respond using AI.\n"
            "Type /help to see available commands."
        )
    else:
        await update.message.reply_text(
            "Welcome to DeepClaw! This bot requires pairing.\n"
            "Send /pair <code> with the pairing code shown in the server terminal."
        )


async def cmd_pair(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /pair <code> -- pair a new user with the bot."""
    user = update.effective_user
    if user is None:
        return

    # Already paired?
    allowed_users = context.bot_data.get(ALLOWED_USERS_KEY, set())
    if is_user_allowed(update, allowed_users):
        await update.message.reply_text("You are already paired with this bot.")
        return

    # Check the pairing code
    raw = update.message.text
    parts = raw.split(maxsplit=1)
    provided_code = parts[1].strip() if len(parts) > 1 else ""
    expected_code = context.bot_data.get(PAIRING_CODE_KEY, "")

    if not provided_code:
        await update.message.reply_text("Usage: /pair <code>\nCheck the server terminal for the pairing code.")
        return

    if provided_code != expected_code:
        logger.warning(f"Failed pairing attempt from user id={user.id} username={user.username}")
        await update.message.reply_text("Invalid pairing code. Check the server terminal and try again.")
        return

    # Pair the user
    user_id = str(user.id)
    allowed_users.add(user_id)
    context.bot_data[ALLOWED_USERS_KEY] = allowed_users
    save_allowed_users(allowed_users)

    # Generate a new pairing code so the old one can't be reused
    new_code = uuid.uuid4().hex[:8]
    context.bot_data[PAIRING_CODE_KEY] = new_code

    username_str = f" (@{user.username})" if user.username else ""
    logger.info(f"Paired user id={user_id}{username_str}")
    await update.message.reply_text(
        f"Paired successfully! You now have full access to DeepClaw.\n"
        f"Type /help to see available commands."
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
    """Handle /doctor -- run diagnostic checks and report results."""
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
    """Handle /safety_test <command> -- check a command for dangerous patterns."""
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
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_text}]},
            config=config,
            stream_mode="messages",
        ):
            # astream with stream_mode="messages" yields (message, metadata) tuples
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                continue
            message_obj, _metadata = chunk

            # Log tool results from ToolMessages
            if isinstance(message_obj, ToolMessage):
                tool_name = getattr(message_obj, "name", "unknown")
                content = message_obj.content
                preview = str(content)[:200] if content else "(empty)"
                logger.info(f"Tool result [{tool_name}]: {preview}")
                continue

            # Only process AI messages with content_blocks
            if not hasattr(message_obj, "content_blocks"):
                continue

            for block in message_obj.content_blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")

                if block_type in ("tool_call", "tool_call_chunk"):
                    tool_name = block.get("name")
                    if tool_name:
                        tool_args = block.get("args", {})
                        args_preview = str(tool_args)[:200] if tool_args else ""
                        logger.info(f"Tool call [{tool_name}]: {args_preview}")
                        tool_line = f"\n\U0001f527 {tool_name}\n"
                        accumulated += tool_line
                        chars_since_edit += len(tool_line)
                elif block_type == "text":
                    text = block.get("text", "")
                    if not text:
                        continue
                    accumulated += text
                    chars_since_edit += len(text)
                else:
                    continue

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

    checkpointer_cm = application.bot_data["checkpointer"]
    checkpointer = await checkpointer_cm.__aenter__()
    application.bot_data["checkpointer_cm"] = checkpointer_cm
    application.bot_data["checkpointer_resolved"] = checkpointer

    agent = create_agent(config, checkpointer)
    application.bot_data["agent"] = agent

    # Load allowed users: config + persisted file
    allowed_users = set(config.telegram.allowed_users)
    allowed_users.update(load_allowed_users())
    application.bot_data[ALLOWED_USERS_KEY] = allowed_users

    # Generate pairing code for new user onboarding
    pairing_code = uuid.uuid4().hex[:8]
    application.bot_data[PAIRING_CODE_KEY] = pairing_code

    if allowed_users:
        logger.info(f"Allowlist active with {len(allowed_users)} paired users")
    else:
        logger.info("No users paired yet")

    logger.info(f"Pairing code: {pairing_code}")
    logger.info("Send /pair %s to your bot in Telegram to pair", pairing_code)

    jobs_path = application.bot_data.get(JOBS_PATH_KEY)
    if jobs_path:
        scheduler = Scheduler(
            jobs_path=jobs_path,
            agent=agent,
            checkpointer=application.bot_data["checkpointer_resolved"],
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
    checkpointer_cm = application.bot_data.get("checkpointer_cm")
    if checkpointer_cm:
        await checkpointer_cm.__aexit__(None, None, None)


def run_telegram(config) -> None:
    """Build and run the Telegram bot with long-polling."""
    from pathlib import Path

    token = config.telegram.bot_token
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required (set via env var, .env, or config.yaml)")

    checkpointer = create_checkpointer()

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    jobs_path = Path(__import__("os").path.expanduser("~/.deepclaw/cron/jobs.json"))
    jobs_path.parent.mkdir(parents=True, exist_ok=True)

    application.bot_data["checkpointer"] = checkpointer
    application.bot_data[CONFIG_KEY] = config
    application.bot_data[JOBS_PATH_KEY] = jobs_path

    application.add_handler(CommandHandler("pair", cmd_pair))
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
