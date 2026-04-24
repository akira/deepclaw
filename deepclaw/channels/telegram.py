"""Telegram bot channel for DeepClaw.

Uses Telegram chat_id as LangGraph thread_id for conversation persistence.
Streams agent responses by progressively editing a single Telegram message.
"""

import logging
import mimetypes
import time
import uuid
from dataclasses import replace
from pathlib import Path

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

from deepclaw import agent as agent_module
from deepclaw.agent import create_agent, create_checkpointer
from deepclaw.auth import (
    REJECTION_MESSAGE,
    get_thread_state,
    is_user_allowed,
    load_allowed_users,
    load_thread_ids,
    save_allowed_users,
    save_thread_ids,
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
from deepclaw.tools.skills import (
    skill_create,
    skill_delete,
    skill_install,
    skill_view,
    skills_audit,
    skills_check_resolvable,
    skills_list,
    skills_search_remote,
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
LAST_MESSAGE_KEY = "last_user_message"
MODEL_OVERRIDE_KEY = "model_override"

# Bot start time for /uptime
_BOT_START_TIME: float = time.time()
_UPLOADS_DIR = Path("~/.deepclaw/uploads").expanduser()
_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# Workspace root for /memory and /soul
_WORKSPACE_ROOT = Path("~/.deepclaw/workspace").expanduser()

# Map of chat_id -> dict of msg_id -> telegram Message object
# Used to translate string message IDs back to editable message objects.
_STREAM_MESSAGES: dict[str, dict[str, object]] = {}


def _looks_like_supported_image(document) -> bool:
    """Return True if a Telegram document looks like a supported image."""
    mime_type = (getattr(document, "mime_type", "") or "").lower()
    if mime_type.startswith("image/"):
        return True

    filename = getattr(document, "file_name", "") or ""
    return Path(filename).suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES


async def _download_media_file(update: Update) -> tuple[str | None, str | None]:
    """Download an incoming Telegram photo or image document."""
    message = update.message
    if message is None:
        return None, "No message found."

    chat_id = str(update.effective_chat.id) if update.effective_chat else "unknown"
    file_obj = None
    suffix = ".bin"

    if message.photo:
        photo = message.photo[-1]
        file_obj = await photo.get_file()
        suffix = ".jpg"
    elif message.document:
        document = message.document
        if not _looks_like_supported_image(document):
            return (
                None,
                "I can analyze image uploads, but this file type is not supported yet. "
                "Please send a photo or an image file (png, jpg, jpeg, webp, gif).",
            )
        file_obj = await document.get_file()
        guessed = Path(document.file_name or "").suffix.lower()
        if guessed in _SUPPORTED_IMAGE_SUFFIXES:
            suffix = guessed
        else:
            suffix = mimetypes.guess_extension(document.mime_type or "") or ".img"
    else:
        return None, None

    target_dir = _UPLOADS_DIR / "telegram" / chat_id
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"upload_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{suffix}"
    await file_obj.download_to_drive(custom_path=str(path))
    return str(path), None


async def _build_incoming_text(update: Update) -> tuple[str | None, str | None]:
    """Build the text sent to the agent, including downloaded media paths when present."""
    message = update.message
    if message is None:
        return None, None

    text = (message.text or message.caption or "").strip()
    media_path, media_error = await _download_media_file(update)
    if media_error:
        return None, media_error

    if media_path:
        prompt = text or "The user sent an image."
        prompt += (
            f"\n\nAttached image saved at local path: {media_path}\n"
            "If the image matters for answering, use vision_analyze on that path."
        )
        return prompt, None

    if text:
        return text, None

    return None, None


def authorize_chat(update: Update) -> bool:
    """Return True if the chat is authorized for bot interaction.

    Currently restricts to private (DM) chats only. Group support
    can be added here later via allowlisted chat IDs.
    """
    chat = update.effective_chat
    return chat is not None and chat.type == "private"


def get_thread_id(context: ContextTypes.DEFAULT_TYPE, chat_id: str) -> str:
    """Return the current thread_id for a chat, defaulting to the chat_id itself."""
    thread_ids = context.bot_data.setdefault(THREAD_IDS_KEY, {})
    state = get_thread_state(thread_ids, chat_id)
    return state["current_thread_id"]


def _rotate_thread(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: str,
    *,
    reason: str,
    pending_summary_text: str | None = None,
    summary_artifact_path: str | None = None,
    raw_history_artifact_paths: list[str] | None = None,
) -> str:
    """Rotate a chat onto a fresh thread and persist compaction metadata."""
    thread_ids = context.bot_data.setdefault(THREAD_IDS_KEY, {})
    state = get_thread_state(thread_ids, chat_id)
    old_thread = state["current_thread_id"]
    new_thread = str(uuid.uuid4())
    state["current_thread_id"] = new_thread
    state["parent_thread_id"] = old_thread
    state["pending_summary_text"] = pending_summary_text
    state["summary_artifact_path"] = summary_artifact_path
    state["raw_history_artifact_paths"] = list(raw_history_artifact_paths or [])
    state["last_compacted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["last_compaction_reason"] = reason
    save_thread_ids(thread_ids)
    return new_thread


def _parse_skills_command(raw_text: str) -> tuple[str, str]:
    """Parse `/skills ...` into a subcommand and argument string."""
    parts = (raw_text or "").strip().split(maxsplit=2)
    if len(parts) == 1:
        return "browse", ""
    if len(parts) == 2:
        return parts[1].lower(), ""
    return parts[1].lower(), parts[2].strip()


def _format_skills_list() -> str:
    """Return a user-facing summary of installed local skills."""
    result = skills_list()
    skills = result.get("skills", [])
    if not skills:
        return "No local skills installed yet."

    lines = [f"Installed skills ({result['count']}):"]
    for skill in skills[:20]:
        description = skill.get("description") or "(no description)"
        lines.append(f"- {skill['name']}: {description}")
    if len(skills) > 20:
        lines.append(f"...and {len(skills) - 20} more")
    return "\n".join(lines)


def _format_remote_skills(results: dict) -> str:
    """Return a user-facing summary of remote skills search results."""
    skills = results.get("skills", [])
    query = (results.get("query") or "").strip()
    if not skills:
        if query:
            return f"No skills.sh results for: {query}"
        return "No skills.sh results available right now."

    title = f"skills.sh results for: {query}" if query else "skills.sh popular results:"
    lines = [title]
    for skill in skills[:10]:
        installs = skill.get("installs")
        installs_suffix = f" ({installs:,} installs)" if isinstance(installs, int) else ""
        lines.append(f"- {skill['name']} — {skill['repo']}{installs_suffix}\n  {skill['url']}")
    if len(skills) > 10:
        lines.append(f"...and {len(skills) - 10} more")
    return "\n".join(lines)


def _format_skills_audit(results: dict) -> str:
    """Return a concise user-facing summary of a local skills audit."""
    lines = [
        "Skill audit",
        f"- skills checked: {results.get('count', 0)}",
        f"- duplicate descriptions: {results.get('duplicate_descriptions_count', 0)}",
        f"- missing required sections: {results.get('skills_missing_required_sections_count', 0)}",
    ]
    duplicate_descriptions = results.get("duplicate_descriptions", [])
    if duplicate_descriptions:
        first = duplicate_descriptions[0]
        lines.append(f"- first duplicate: {first['description']} ({', '.join(first['skills'])})")
    missing = [entry for entry in results.get("skills", []) if entry.get("missing_sections")]
    if missing:
        first_missing = missing[0]
        lines.append(
            f"- first missing-sections skill: {first_missing['name']} ({', '.join(first_missing['missing_sections'])})"
        )
    return "\n".join(lines)


def _format_skills_resolvable(results: dict) -> str:
    """Return a concise user-facing summary of skill resolvability."""
    total = results.get("count", 0)
    loadable = results.get("loadable_count", 0)
    lines = [
        "Resolvable skills",
        f"- loadable: {loadable} / {total}",
        f"- unresolvable: {results.get('unresolvable_count', 0)}",
    ]
    if results.get("unresolvable"):
        first = results["unresolvable"][0]
        lines.append(f"- first issue: {first['name']} — {first['reason']}")
    return "\n".join(lines)


async def cmd_skills(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /skills subcommands for local skill management."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    subcommand, args = _parse_skills_command(update.message.text or "/skills")

    if subcommand in {"", "browse", "list"}:
        await update.message.reply_text(_format_skills_list())
        return

    if subcommand in {"search", "remote"}:
        query = args.strip()
        result = skills_search_remote(query)
        if "error" in result:
            await update.message.reply_text(result["error"])
            return
        await update.message.reply_text(_format_remote_skills(result))
        return

    if subcommand == "audit":
        await update.message.reply_text(_format_skills_audit(skills_audit()))
        return

    if subcommand in {"resolvable", "check"}:
        await update.message.reply_text(_format_skills_resolvable(skills_check_resolvable()))
        return

    if subcommand == "view":
        if not args:
            await update.message.reply_text("Usage: /skills view <name>")
            return
        result = skill_view(args)
        if "error" in result:
            await update.message.reply_text(result["error"])
            return
        header = f"Skill: {result['name']}\nPath: {result['path']}\n"
        text = header + "\n" + result["content"]
        for chunk in chunk_message(text, TELEGRAM_MESSAGE_LIMIT):
            await update.message.reply_text(chunk)
        return

    if subcommand == "create":
        if "|" not in args:
            await update.message.reply_text("Usage: /skills create <name> | <description>")
            return
        name, description = [part.strip() for part in args.split("|", maxsplit=1)]
        if not name or not description:
            await update.message.reply_text("Usage: /skills create <name> | <description>")
            return
        result = skill_create(name, description)
        if "error" in result:
            await update.message.reply_text(result["error"])
            return
        await update.message.reply_text(
            f"Created skill: {result['name']}\nPath: {result['path']}\nDescription: {result['description']}"
        )
        return

    if subcommand == "install":
        if not args:
            await update.message.reply_text(
                "Usage: /skills install <source_path_or_url> or /skills install <source_path_or_url> | <name>"
            )
            return
        if "|" in args:
            source_path, name = [part.strip() for part in args.split("|", maxsplit=1)]
            dest_name = name or None
        else:
            source_path = args.strip()
            dest_name = None
        result = skill_install(source_path, name=dest_name)
        if "error" in result:
            await update.message.reply_text(result["error"])
            return
        await update.message.reply_text(
            f"Installed skill: {result['name']}\nPath: {result['path']}\nSource: {result['source_path']}"
        )
        return

    if subcommand in {"delete", "remove", "rm"}:
        if not args:
            await update.message.reply_text("Usage: /skills delete <name>")
            return
        result = skill_delete(args)
        if "error" in result:
            await update.message.reply_text(result["error"])
            return
        await update.message.reply_text(
            f"Deleted skill: {result['name']}\nRemoved: {result['path']}"
        )
        return

    await update.message.reply_text(
        "Usage: /skills [browse|search <query>|remote [query]|audit|resolvable|view <name>|create <name> | <description>|install <path_or_url> [| <name>]|delete <name>]"
    )


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new -- start a fresh conversation thread."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    new_thread = _rotate_thread(context, chat_id, reason="manual-new")
    logger.info("New thread for chat %s: %s", chat_id, new_thread)
    await update.message.reply_text(f"Started a new conversation thread.\nThread ID: {new_thread}")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help -- list available commands."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    allowed_users = context.bot_data.get(ALLOWED_USERS_KEY, set())
    text = (
        "Available commands:\n"
        "/new \u2014 Start a fresh conversation thread\n"
        "/clear \u2014 Clear conversation (alias for /new)\n"
        "/compact \u2014 Compact the current thread into a fresh one with a handoff summary\n"
        "/retry \u2014 Re-send the last message\n"
        "/model [name] — View or set the active model\n"
        "/skills [subcommand] — Browse/view/create/install/delete/audit local skills\n"
        "/memory — Show MEMORY.md\n"
        "/soul — Show SOUL.md\n"
        "/uptime \u2014 Show bot uptime\n"
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
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    thread_ids = context.bot_data.setdefault(THREAD_IDS_KEY, {})
    state = get_thread_state(thread_ids, chat_id)
    thread_id = state["current_thread_id"]
    model = context.bot_data[CONFIG_KEY].model or "not set"
    allowed_users = context.bot_data.get(ALLOWED_USERS_KEY, set())
    allowlist_status = (
        f"active ({len(allowed_users)} users)" if allowed_users else "inactive (open mode)"
    )
    text = (
        f"Chat ID: {chat_id}\nThread ID: {thread_id}\nModel: {model}\nAllowlist: {allowlist_status}"
    )
    if state.get("parent_thread_id"):
        text += f"\nParent Thread: {state['parent_thread_id']}"
    if state.get("last_compaction_reason"):
        text += f"\nLast Compaction: {state['last_compaction_reason']}"
    if state.get("last_compacted_at"):
        text += f"\nLast Compacted At: {state['last_compacted_at']}"
    if state.get("summary_artifact_path"):
        text += f"\nSummary Artifact: {state['summary_artifact_path']}"
    raw_paths = state.get("raw_history_artifact_paths") or []
    if raw_paths:
        text += "\nRaw History Artifacts:"
        for path in raw_paths:
            text += f"\n- {path}"
    await update.message.reply_text(text)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start -- Telegram's default start command."""
    if not authorize_chat(update):
        return
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
    if not authorize_chat(update):
        return
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


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear -- start a fresh conversation thread."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    new_thread = _rotate_thread(context, chat_id, reason="manual-clear")
    context.bot_data.setdefault(LAST_MESSAGE_KEY, {}).pop(chat_id, None)
    logger.info("Cleared thread for chat %s: %s", chat_id, new_thread)
    await update.message.reply_text("Conversation cleared.")


async def cmd_compact(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /compact -- summarize/offload current thread and rotate to a fresh one."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    thread_id = get_thread_id(context, chat_id)
    gateway: Gateway | None = context.bot_data.get(GATEWAY_KEY)
    if gateway is None:
        await update.message.reply_text("Gateway is not initialized yet.")
        return
    result = await gateway.compact_thread(chat_id, thread_id, reason="manual")
    if result is None:
        await update.message.reply_text("Not enough thread history to compact yet.")
        return
    context.bot_data.setdefault(LAST_MESSAGE_KEY, {}).pop(chat_id, None)
    raw_paths = "\n".join(f"- {p}" for p in result.raw_history_artifact_paths) or "- (none)"
    await update.message.reply_text(
        "Compacted current thread into a fresh conversation.\n"
        f"Old thread: {result.old_thread_id}\n"
        f"New thread: {result.new_thread_id}\n"
        f"Summary artifact: {result.summary_artifact_path or '(none)'}\n"
        f"Raw history artifacts:\n{raw_paths}"
    )


KNOWN_PROVIDERS = ("anthropic", "openai", "google", "groq", "mistral", "bedrock", "vertex")


def _validate_model(model: str) -> str | None:
    """Validate a model string of the form 'provider:model-name'.

    Returns an error message string if invalid, or None if valid.
    """
    if ":" not in model:
        providers = ", ".join(KNOWN_PROVIDERS)
        return f"Invalid format. Expected provider:model-name (e.g. anthropic:claude-sonnet-4-6).\nKnown providers: {providers}"
    provider, _, model_name = model.partition(":")
    if not model_name:
        return "Model name cannot be empty after the provider prefix."
    if provider not in KNOWN_PROVIDERS:
        providers = ", ".join(KNOWN_PROVIDERS)
        return f"Unknown provider '{provider}'. Known providers: {providers}"
    return None


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model [name] -- view or set the active model."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    raw = update.message.text
    parts = raw.split(maxsplit=1)
    model_arg = parts[1].strip() if len(parts) > 1 else ""
    if model_arg:
        error = _validate_model(model_arg)
        if error:
            await update.message.reply_text(f"Invalid model: {error}")
            return
        # Build new agent and gateway first — don't touch bot_data until
        # everything succeeds so we never leave state partially updated.
        config = context.bot_data[CONFIG_KEY]
        new_config = replace(config, model=model_arg)
        checkpointer = context.bot_data["checkpointer_resolved"]
        try:
            new_agent = create_agent(new_config, checkpointer)
        except Exception:
            logger.exception("Failed to create agent for model %s", model_arg)
            await update.message.reply_text(
                f"Failed to switch to model {model_arg} — agent creation failed. Keeping current model."
            )
            return
        new_gateway = Gateway(
            agent=new_agent,
            streaming_config=new_config.telegram.streaming,
            checkpointer=checkpointer,
            thread_state_store=context.bot_data.setdefault(THREAD_IDS_KEY, {}),
            persist_thread_state=save_thread_ids,
            model_name=new_config.model,
        )

        # Commit atomically — all-or-nothing from here
        context.bot_data[MODEL_OVERRIDE_KEY] = model_arg
        context.bot_data[CONFIG_KEY] = new_config
        context.bot_data["agent"] = new_agent
        context.bot_data[GATEWAY_KEY] = new_gateway

        scheduler = context.bot_data.get(SCHEDULER_KEY)
        if scheduler is not None:
            scheduler.update_agent(new_agent)

        heartbeat = context.bot_data.get(HEARTBEAT_KEY)
        if heartbeat is not None:
            heartbeat.update_agent(new_agent)

        logger.info("Agent reloaded with model: %s", model_arg)

        await update.message.reply_text(
            f"Model switched to: {model_arg}\nAgent reloaded — use /compact or /clear before continuing in a fresh thread."
        )
    else:
        override = context.bot_data.get(MODEL_OVERRIDE_KEY)
        config_model = context.bot_data[CONFIG_KEY].model or "not set"
        if override:
            await update.message.reply_text(
                f"Active model override: {override}\nConfig default: {config_model}"
            )
        else:
            await update.message.reply_text(f"Current model: {config_model} (no override set)")


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /memory -- display MEMORY.md from the workspace."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    memory_path = agent_module.MEMORY_FILE
    if not memory_path.is_file():
        await update.message.reply_text("No memory file found.")
        return
    text = memory_path.read_text(encoding="utf-8").strip() or "(empty)"
    for chunk in chunk_message(text, TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(chunk)


async def cmd_soul(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /soul -- display SOUL.md from the workspace."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    soul_path = agent_module.SOUL_FILE
    if not soul_path.is_file():
        await update.message.reply_text("No SOUL.md found.")
        return
    text = soul_path.read_text(encoding="utf-8").strip() or "(empty)"
    for chunk in chunk_message(text, TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(chunk)


async def cmd_retry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /retry -- re-send the last user message."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    last_messages: dict[str, str] = context.bot_data.get(LAST_MESSAGE_KEY, {})
    last_text = last_messages.get(chat_id)
    if not last_text:
        await update.message.reply_text("No previous message to retry.")
        return
    thread_id = get_thread_id(context, chat_id)
    incoming = IncomingMessage(
        text=last_text,
        chat_id=chat_id,
        user_id=str(update.effective_user.id) if update.effective_user else "",
        username=update.effective_user.username if update.effective_user else None,
        source="telegram",
    )
    channel = TelegramChannel(update, context)
    gateway: Gateway = context.bot_data[GATEWAY_KEY]
    try:
        await gateway.handle_message(channel, incoming, thread_id)
    finally:
        _STREAM_MESSAGES.pop(chat_id, None)


async def cmd_uptime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /uptime -- show how long the bot has been running."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    elapsed = int(time.time() - _BOT_START_TIME)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    await update.message.reply_text(f"Uptime: {hours}h {minutes}m {seconds}s")


async def cmd_cron(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cron -- list all cron jobs."""
    if not authorize_chat(update):
        return
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
    if not authorize_chat(update):
        return
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
    if not authorize_chat(update):
        return
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
    if not authorize_chat(update):
        return
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
    if not authorize_chat(update):
        return
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
        """Send a message to a chat. Returns the first message_id as string."""
        first_msg_id: str | None = None
        for chunk in chunk_message(text, TELEGRAM_MESSAGE_LIMIT):
            msg = await self._bot.send_message(chat_id=int(chat_id), text=chunk)
            msg_id = str(msg.message_id)
            if first_msg_id is None:
                first_msg_id = msg_id
            _STREAM_MESSAGES.setdefault(chat_id, {})[msg_id] = msg
        return first_msg_id or ""

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
    if not update.message:
        return

    if not authorize_chat(update):
        return

    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    chat_id = str(update.effective_chat.id)
    user = update.effective_user
    thread_id = get_thread_id(context, chat_id)
    user_text, media_error = await _build_incoming_text(update)
    if media_error:
        await update.message.reply_text(media_error)
        return
    if not user_text:
        return

    # Track last message per chat for /retry
    context.bot_data.setdefault(LAST_MESSAGE_KEY, {})[chat_id] = user_text

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

    # Restore thread metadata from disk so /new survives bot restarts
    application.bot_data[THREAD_IDS_KEY] = load_thread_ids()

    # Create the shared gateway
    gateway = Gateway(
        agent=agent,
        streaming_config=config.telegram.streaming,
        checkpointer=checkpointer,
        thread_state_store=application.bot_data[THREAD_IDS_KEY],
        persist_thread_state=save_thread_ids,
        model_name=config.model,
    )
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

    # Register commands in Telegram so they appear in the / dropdown menu
    from telegram import BotCommand

    await application.bot.set_my_commands(
        [
            BotCommand("new", "Start a fresh conversation thread"),
            BotCommand("clear", "Clear conversation (alias for /new)"),
            BotCommand("compact", "Compact current thread into a fresh one"),
            BotCommand("retry", "Re-send the last message"),
            BotCommand("model", "View or set the active model"),
            BotCommand("skills", "Browse or manage local skills"),
            BotCommand("memory", "Show MEMORY.md"),
            BotCommand("soul", "Show SOUL.md"),
            BotCommand("uptime", "Show bot uptime"),
            BotCommand("status", "Show thread ID and model info"),
            BotCommand("cron", "List scheduled jobs"),
            BotCommand("cron_add", "Add a scheduled job"),
            BotCommand("cron_rm", "Remove a scheduled job"),
            BotCommand("doctor", "Run system diagnostics"),
            BotCommand("help", "Show all commands"),
        ]
    )

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
    application.add_handler(CommandHandler("clear", cmd_clear))
    application.add_handler(CommandHandler("compact", cmd_compact))
    application.add_handler(CommandHandler("retry", cmd_retry))
    application.add_handler(CommandHandler("model", cmd_model))
    application.add_handler(CommandHandler("skills", cmd_skills))
    application.add_handler(CommandHandler("memory", cmd_memory))
    application.add_handler(CommandHandler("soul", cmd_soul))
    application.add_handler(CommandHandler("uptime", cmd_uptime))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("cron", cmd_cron))
    application.add_handler(CommandHandler("cron_add", cmd_cron_add))
    application.add_handler(CommandHandler("cron_rm", cmd_cron_rm))
    application.add_handler(CommandHandler("safety_test", cmd_safety_test))
    application.add_handler(CommandHandler("doctor", cmd_doctor))
    application.add_handler(
        MessageHandler((filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND, handle_message)
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting DeepClaw Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
