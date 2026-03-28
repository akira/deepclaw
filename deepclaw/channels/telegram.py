"""Telegram bot channel for DeepClaw.

Uses Telegram chat_id as LangGraph thread_id for conversation persistence.
Streams agent responses by progressively editing a single Telegram message.
"""

import logging
import uuid

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
from deepclaw.channels.base import Channel, IncomingMessage
from deepclaw.gateway import Gateway, chunk_message
from deepclaw.heartbeat import HeartbeatRunner
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
SCHEDULER_KEY = "scheduler"
HEARTBEAT_KEY = "heartbeat_runner"
JOBS_PATH_KEY = "jobs_path"
GATEWAY_KEY = "gateway"

# Map of chat_id -> dict of msg_id -> telegram Message object
# Used to translate string message IDs back to editable message objects.
_STREAM_MESSAGES: dict[str, dict[str, object]] = {}


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
    logger.info("New thread for chat %s: %s", chat_id, new_thread)
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
    allowlist_status = (
        f"active ({len(allowed_users)} users)" if allowed_users else "inactive (open mode)"
    )
    text = (
        f"Chat ID: {chat_id}\nThread ID: {thread_id}\nModel: {model}\nAllowlist: {allowlist_status}"
    )
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
        await update.message.reply_text(
            "Usage: /pair <code>\nCheck the server terminal for the pairing code."
        )
        return

    if provided_code != expected_code:
        logger.warning("Failed pairing attempt from user id=%s username=%s", user.id, user.username)
        await update.message.reply_text(
            "Invalid pairing code. Check the server terminal and try again."
        )
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
    logger.info("Paired user id=%s%s", user_id, username_str)
    await update.message.reply_text(
        "Paired successfully! You now have full access to DeepClaw.\n"
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
        raw = raw[len(prefix) :].strip()
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
    await update.message.reply_text(
        f"Cron job added: {job.id[:8]}\nSchedule: {cron_expr}\nPrompt: {prompt}"
    )


async def cmd_cron_rm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cron_rm <job_id_prefix>."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    raw = update.message.text
    prefix = "/cron_rm"
    if raw.startswith(prefix):
        raw = raw[len(prefix) :].strip()
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
        await update.message.reply_text(
            f"Ambiguous prefix: {id_prefix} matches {len(matches)} jobs. Be more specific."
        )
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
    for chunk in chunk_message(report, TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(chunk)


async def cmd_safety_test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /safety_test <command> -- check a command for dangerous patterns."""
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    raw = update.message.text
    prefix = "/safety_test"
    if raw.startswith(prefix):
        raw = raw[len(prefix) :].strip()
    if not raw:
        await update.message.reply_text("Usage: /safety_test <command>")
        return
    matches = check_command(raw)
    if not matches:
        await update.message.reply_text(f"No dangerous patterns detected in: {raw}")
        return
    await update.message.reply_text(format_warning(raw, matches))


class TelegramBotChannel(Channel):
    """Long-lived channel adapter using the Bot API directly.

    Used by the scheduler and any code that needs to send messages
    without an active Update context (proactive messaging).
    """

    def __init__(self, bot):
        self._bot = bot

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def supports_edit(self) -> bool:
        return True

    @property
    def message_limit(self) -> int:
        return TELEGRAM_MESSAGE_LIMIT

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, chat_id: str, text: str) -> str:
        """Send a message to a chat. Returns message_id as string."""
        msg = await self._bot.send_message(chat_id=int(chat_id), text=text)
        msg_id = str(msg.message_id)
        _STREAM_MESSAGES.setdefault(chat_id, {})[msg_id] = msg
        return msg_id

    async def edit_message(self, chat_id: str, message_id: str, text: str) -> None:
        """Edit a previously sent message."""
        msg = _STREAM_MESSAGES.get(chat_id, {}).get(message_id)
        if msg is not None:
            try:
                await msg.edit_text(text)
            except BadRequest as exc:
                if "Message is not modified" not in str(exc):
                    raise
        else:
            try:
                await self._bot.edit_message_text(
                    text=text, chat_id=int(chat_id), message_id=int(message_id)
                )
            except BadRequest as exc:
                if "Message is not modified" not in str(exc):
                    raise

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator."""
        await self._bot.send_chat_action(chat_id=int(chat_id), action=ChatAction.TYPING)


class TelegramChannel(Channel):
    """Per-message channel adapter wrapping an Update context.

    Used during the gateway streaming flow where we have an active
    Update and want to reply_text rather than send_message.
    """

    def __init__(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self._update = update
        self._context = context

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def supports_edit(self) -> bool:
        return True

    @property
    def message_limit(self) -> int:
        return TELEGRAM_MESSAGE_LIMIT

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, chat_id: str, text: str) -> str:
        """Send a text message via reply_text. Returns a string message_id."""
        msg = await self._update.message.reply_text(text)
        msg_id = str(msg.message_id)
        _STREAM_MESSAGES.setdefault(chat_id, {})[msg_id] = msg
        return msg_id

    async def edit_message(self, chat_id: str, message_id: str, text: str) -> None:
        """Edit a previously sent message."""
        msg = _STREAM_MESSAGES.get(chat_id, {}).get(message_id)
        if msg is None:
            return
        try:
            await msg.edit_text(text)
        except BadRequest as exc:
            if "Message is not modified" not in str(exc):
                raise

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator."""
        await self._update.effective_chat.send_action(ChatAction.TYPING)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle an incoming Telegram message by streaming the agent response."""
    if not update.message or not update.message.text:
        return

    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    chat_id = str(update.effective_chat.id)
    user_text = update.message.text
    user = update.effective_user
    thread_id = get_thread_id(context, chat_id)

    incoming = IncomingMessage(
        text=user_text,
        chat_id=chat_id,
        user_id=str(user.id) if user else "",
        username=user.username if user else None,
        source="telegram",
    )

    channel = TelegramChannel(update, context)
    gateway: Gateway = context.bot_data[GATEWAY_KEY]

    try:
        await gateway.handle_message(channel, incoming, thread_id)
    finally:
        # Clean up stored message objects for this chat
        _STREAM_MESSAGES.pop(chat_id, None)


async def post_init(application: Application) -> None:
    """Create the DeepAgents agent after the application is initialized."""
    config = application.bot_data[CONFIG_KEY]

    checkpointer_cm = application.bot_data["checkpointer"]
    checkpointer = await checkpointer_cm.__aenter__()
    application.bot_data["checkpointer_cm"] = checkpointer_cm
    application.bot_data["checkpointer_resolved"] = checkpointer

    agent = create_agent(config, checkpointer)
    application.bot_data["agent"] = agent

    # Create the shared gateway
    gateway = Gateway(agent=agent, streaming_config=config.telegram.streaming)
    application.bot_data[GATEWAY_KEY] = gateway

    # Load allowed users: config + persisted file
    allowed_users = set(config.telegram.allowed_users)
    allowed_users.update(load_allowed_users())
    application.bot_data[ALLOWED_USERS_KEY] = allowed_users

    # Generate pairing code for new user onboarding
    pairing_code = uuid.uuid4().hex[:8]
    application.bot_data[PAIRING_CODE_KEY] = pairing_code

    if allowed_users:
        logger.info("Allowlist active with %d paired users", len(allowed_users))
    else:
        logger.info("No users paired yet")

    logger.info("Pairing code: %s", pairing_code)
    logger.info("Send /pair %s to your bot in Telegram to pair", pairing_code)

    # Create a long-lived channel for proactive messaging (cron delivery, etc.)
    bot_channel = TelegramBotChannel(application.bot)
    application.bot_data["telegram_channel"] = bot_channel

    jobs_path = application.bot_data.get(JOBS_PATH_KEY)
    if jobs_path:
        scheduler = Scheduler(
            jobs_path=jobs_path,
            agent=agent,
            checkpointer=application.bot_data["checkpointer_resolved"],
            channels={"telegram": bot_channel},
        )
        application.bot_data[SCHEDULER_KEY] = scheduler
        await scheduler.start()

    # Start heartbeat runner
    heartbeat_runner = HeartbeatRunner(
        config=config.heartbeat,
        agent=agent,
        channels={"telegram": bot_channel},
    )
    application.bot_data[HEARTBEAT_KEY] = heartbeat_runner
    await heartbeat_runner.start()

    logger.info("DeepAgents agent initialized")


async def post_shutdown(application: Application) -> None:
    """Clean up the heartbeat, scheduler, and checkpointer on shutdown."""
    heartbeat_runner = application.bot_data.get(HEARTBEAT_KEY)
    if heartbeat_runner:
        await heartbeat_runner.stop()

    scheduler = application.bot_data.get(SCHEDULER_KEY)
    if scheduler:
        await scheduler.stop()

    checkpointer_cm = application.bot_data.get("checkpointer_cm")
    if checkpointer_cm:
        await checkpointer_cm.__aexit__(None, None, None)


def run_telegram(config) -> None:
    """Build and run the Telegram bot with long-polling."""
    from deepclaw.config import CONFIG_DIR

    token = config.telegram.bot_token
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required (set via env var, .env, or config.yaml)")

    checkpointer = create_checkpointer()

    application = (
        Application.builder().token(token).post_init(post_init).post_shutdown(post_shutdown).build()
    )
    jobs_path = CONFIG_DIR / "cron" / "jobs.json"
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
