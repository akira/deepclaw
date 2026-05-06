"""Telegram bot channel for DeepClaw.

Uses Telegram chat_id as LangGraph thread_id for conversation persistence.
Streams agent responses by progressively editing a single Telegram message.
"""

import asyncio
import contextlib
import logging
import mimetypes
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from deepclaw import agent as agent_module
from deepclaw.agent import create_agent, create_checkpointer
from deepclaw.approval_state import aadd_thread_approved_keys
from deepclaw.auth import (
    REJECTION_MESSAGE,
    is_user_allowed,
    load_allowed_users,
    load_thread_ids,
    save_allowed_users,
    save_thread_ids,
)
from deepclaw.channels.base import Channel, IncomingMessage
from deepclaw.config import DeepClawConfig
from deepclaw.context_report import build_context_report
from deepclaw.gateway import Gateway, chunk_message
from deepclaw.heartbeat import HeartbeatRunner
from deepclaw.safety import check_command, format_warning, redact_secrets
from deepclaw.scheduler import (
    Scheduler,
    add_job,
    list_jobs,
    load_jobs,
    parse_cron_add,
    remove_job,
)
from deepclaw.sessions import (
    find_similar_sessions_for_chat,
    get_most_recent_session_for_chat,
    list_sessions_for_chat,
    session_belongs_to_chat,
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
PENDING_APPROVALS_KEY = "pending_approvals"
ACTIVE_RUNS_KEY = "active_runs"

# Bot start time for /uptime
_BOT_START_TIME: float = time.time()
_UPLOADS_DIR = Path("~/.deepclaw/uploads").expanduser()
_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# Workspace root for /memory and /soul
_WORKSPACE_ROOT = Path("~/.deepclaw/workspace").expanduser()

# Map of chat_id -> dict of msg_id -> telegram Message object
# Used to translate string message IDs back to editable message objects.
_STREAM_MESSAGES: dict[str, dict[str, object]] = {}


def _telegram_media_kind(path: str) -> str:
    """Return the Telegram send_* method kind for a local media path."""
    suffix = Path(path).suffix.lower()
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = (mime_type or "").lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"} or mime_type.startswith("image/"):
        return "photo"
    if suffix in {".mp4", ".mov", ".webm", ".mkv"} or mime_type.startswith("video/"):
        return "video"
    if suffix in {".ogg"}:
        return "voice"
    if suffix in {".mp3", ".wav", ".m4a", ".flac"} or mime_type.startswith("audio/"):
        return "audio"
    return "document"


@contextlib.contextmanager
def _open_media_file(path: str):
    """Open a media file in binary mode for Telegram upload."""
    with Path(path).open("rb") as handle:
        yield handle


async def _telegram_send_media_via_bot(bot, chat_id: int, path: str, caption: str | None = None):
    """Send a local file through the Telegram Bot API using the right media method."""
    kind = _telegram_media_kind(path)
    with _open_media_file(path) as handle:
        if kind == "photo":
            return await bot.send_photo(chat_id=chat_id, photo=handle, caption=caption)
        if kind == "video":
            return await bot.send_video(chat_id=chat_id, video=handle, caption=caption)
        if kind == "voice":
            return await bot.send_voice(chat_id=chat_id, voice=handle, caption=caption)
        if kind == "audio":
            return await bot.send_audio(chat_id=chat_id, audio=handle, caption=caption)
        return await bot.send_document(chat_id=chat_id, document=handle, caption=caption)


async def _telegram_send_media_via_message(message, path: str, caption: str | None = None):
    """Send a local file via a Telegram message reply using the right media method."""
    kind = _telegram_media_kind(path)
    with _open_media_file(path) as handle:
        if kind == "photo":
            return await message.reply_photo(photo=handle, caption=caption)
        if kind == "video":
            return await message.reply_video(video=handle, caption=caption)
        if kind == "voice":
            return await message.reply_voice(voice=handle, caption=caption)
        if kind == "audio":
            return await message.reply_audio(audio=handle, caption=caption)
        return await message.reply_document(document=handle, caption=caption)


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
    """Return the current thread_id for a chat, auto-healing missing mappings."""
    thread_ids: dict[str, str] = context.bot_data.setdefault(THREAD_IDS_KEY, {})
    thread_id = thread_ids.get(chat_id)
    if thread_id:
        return thread_id

    thread_id = str(uuid.uuid4())
    thread_ids[chat_id] = thread_id
    save_thread_ids(thread_ids)
    logger.warning(
        "Missing thread ID mapping for chat %s; generated replacement thread %s",
        chat_id,
        thread_id,
    )
    return thread_id


def _pending_approvals(context: ContextTypes.DEFAULT_TYPE) -> dict[str, dict]:
    """Return the mutable pending-approval map stored in bot_data."""
    return context.bot_data.setdefault(PENDING_APPROVALS_KEY, {})


def _active_runs(context: ContextTypes.DEFAULT_TYPE) -> dict[str, asyncio.Task]:
    """Return the mutable per-chat active-run map stored in bot_data."""
    return context.bot_data.setdefault(ACTIVE_RUNS_KEY, {})


def _get_active_run(context: ContextTypes.DEFAULT_TYPE, chat_id: str) -> asyncio.Task | None:
    """Return the active task for a chat, pruning finished tasks."""
    task = _active_runs(context).get(chat_id)
    if task is not None and task.done():
        _active_runs(context).pop(chat_id, None)
        return None
    return task


async def _cancel_active_run(context: ContextTypes.DEFAULT_TYPE, chat_id: str) -> bool:
    """Cancel the current in-flight task for a chat, if any."""
    task = _get_active_run(context, chat_id)
    if task is None:
        return False
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    _active_runs(context).pop(chat_id, None)
    return True


def _begin_active_run(context: ContextTypes.DEFAULT_TYPE, chat_id: str) -> bool:
    """Claim the per-chat active-run slot for the current task."""
    current_task = asyncio.current_task()
    if current_task is None:
        return False
    existing = _get_active_run(context, chat_id)
    if existing is not None and existing is not current_task:
        return False
    _active_runs(context)[chat_id] = current_task
    return True


def _finish_active_run(context: ContextTypes.DEFAULT_TYPE, chat_id: str) -> None:
    """Release the per-chat active-run slot if owned by the current task."""
    current_task = asyncio.current_task()
    active_runs = _active_runs(context)
    if current_task is not None and active_runs.get(chat_id) is current_task:
        active_runs.pop(chat_id, None)


def _active_run_text() -> str:
    """Return the user-visible message when a task is already running."""
    return "A task is already running. Use /stop to interrupt it before sending something new."


def _pending_approval_text() -> str:
    """Return the user-visible message for an unresolved safety review."""
    return "A safety approval is pending. Use /approve, /approve session, or /deny <reason> to respond."


def _pending_approval_markup(pending_id: str) -> InlineKeyboardMarkup:
    """Return Telegram inline buttons for safety approval."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Approve once", callback_data=f"safety:approve_once:{pending_id}"
                ),
                InlineKeyboardButton(
                    "Approve session", callback_data=f"safety:approve_session:{pending_id}"
                ),
                InlineKeyboardButton("Deny", callback_data=f"safety:deny:{pending_id}"),
            ]
        ]
    )


def _parse_approve_scope(raw_text: str) -> str:
    """Extract approval scope from `/approve ...`."""
    args = (raw_text or "").split(maxsplit=1)
    if len(args) < 2:
        return "once"
    scope = args[1].strip().lower()
    if scope in {"session", "sess"}:
        return "session"
    return "once"


def _parse_deny_reason(raw_text: str) -> str:
    """Extract an optional reason from `/deny ...`."""
    parts = (raw_text or "").split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


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
        f"- missing env declarations: {results.get('skills_missing_required_env_declarations_count', 0)}",
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
    missing_env = [
        entry for entry in results.get("skills", []) if entry.get("undeclared_required_env_vars")
    ]
    if missing_env:
        first_env = missing_env[0]
        lines.append(
            "- first missing-env skill: "
            f"{first_env['name']} ({', '.join(first_env['undeclared_required_env_vars'])})"
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
    await _cancel_active_run(context, chat_id)
    new_thread = str(uuid.uuid4())
    thread_ids = context.bot_data.setdefault(THREAD_IDS_KEY, {})
    thread_ids[chat_id] = new_thread
    save_thread_ids(thread_ids)
    _pending_approvals(context).pop(chat_id, None)
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
        "/new — Start a fresh conversation thread\n"
        "/clear — Clear conversation (alias for /new)\n"
        "/retry — Re-send the last message\n"
        "/stop — Stop the current in-flight task\n"
        "/approve — Approve a pending safety-reviewed command\n"
        "/deny [reason] — Reject a pending safety-reviewed command\n"
        "/model [name] — View or set the active model\n"
        "/skills [subcommand] — Browse/view/create/install/delete/audit local skills\n"
        "/memory — Show MEMORY.md\n"
        "/soul — Show SOUL.md\n"
        "/uptime — Show bot uptime\n"
        "/status — Show current thread ID and model info\n"
        "/queue — Show active task state and recent tool activity\n"
        "/sessions — List recent saved sessions for this chat\n"
        "/resume [thread_id] — Resume the most recent or specified session\n"
        "/context — Show full context breakdown and token estimates\n"
        "/safety_test <cmd> — Check a command for dangerous patterns\n"
        "/cron — List scheduled jobs\n"
        "/cron_add <expr> | <prompt> — Add a scheduled job\n"
        "/cron_rm <id_prefix> \u2014 Remove a scheduled job\n"
        "/doctor \u2014 Run system diagnostics\n"
        "/help \u2014 Show this help message"
    )
    text += f"\n\n{len(allowed_users)} paired user(s)."
    await update.message.reply_text(text)


def _status_header(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> tuple[str, str, DeepClawConfig]:
    """Build the common status header and return (header, thread_id, config)."""
    chat_id = str(update.effective_chat.id)
    thread_id = get_thread_id(context, chat_id)
    config = context.bot_data[CONFIG_KEY]
    allowed_users = context.bot_data.get(ALLOWED_USERS_KEY, set())
    allowlist_status = (
        f"active ({len(allowed_users)} users)" if allowed_users else "inactive (open mode)"
    )
    header = (
        f"Chat ID: {chat_id}\n"
        f"Thread ID: {thread_id}\n"
        f"Model: {config.model or 'not set'}\n"
        f"Allowlist: {allowlist_status}"
    )
    return header, thread_id, config


def _format_elapsed(seconds: float | None) -> str:
    """Format a small elapsed-time string for /queue."""
    if seconds is None:
        return "unknown"
    total = max(int(seconds), 0)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _build_queue_report(context: ContextTypes.DEFAULT_TYPE, chat_id: str, gateway: Gateway) -> str:
    """Build a user-facing /queue report for the current chat."""
    active_run = _get_active_run(context, chat_id)
    pending = _pending_approvals(context).get(chat_id)
    snapshot = (
        gateway.get_queue_snapshot(chat_id) if hasattr(gateway, "get_queue_snapshot") else None
    )
    lines = ["📋 Queue"]

    if snapshot is None and active_run is None and pending is None:
        lines.append("- State: idle")
        lines.append("- No active task or recent tool activity for this chat.")
        return "\n".join(lines)

    status = "running" if active_run is not None else "idle"
    if pending:
        status = "awaiting_approval"
    if snapshot and snapshot.get("status"):
        status = str(snapshot["status"])
    lines.append(f"- State: {status}")
    lines.append(f"- Active run: {'yes' if active_run is not None else 'no'}")

    if snapshot:
        request_preview = snapshot.get("user_text_preview")
        if request_preview:
            lines.append(f"- Request: {request_preview}")
        started_at = snapshot.get("started_at")
        updated_at = snapshot.get("updated_at")
        if isinstance(started_at, (int, float)):
            lines.append(f"- Started: {_format_elapsed(time.time() - started_at)} ago")
        if isinstance(updated_at, (int, float)):
            lines.append(f"- Updated: {_format_elapsed(time.time() - updated_at)} ago")

        progress_lines = snapshot.get("progress_lines") or []
        if progress_lines:
            lines.append("")
            lines.append("Recent tool activity:")
            lines.extend(progress_lines)

        final_preview = snapshot.get("final_text_preview")
        if final_preview and active_run is None:
            lines.append("")
            lines.append(f"Last response: {final_preview}")

    approval = pending or (snapshot.get("pending_approval") if snapshot else None)
    if approval:
        lines.append("")
        lines.append("Pending approval:")
        if approval.get("tool"):
            lines.append(f"- Tool: {approval['tool']}")
        if approval.get("warning"):
            lines.append(f"- Warning: {approval['warning']}")
        if approval.get("command"):
            lines.append(f"- Command: {approval['command']}")
        lines.append("- Action: /approve to continue or /deny <reason> to reject")

    return redact_secrets("\n".join(lines))


def _format_relative_time(iso_timestamp: str | None) -> str:
    """Format an ISO timestamp as relative time for Telegram output."""
    if not iso_timestamp:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        now = datetime.now(tz=dt.tzinfo)
        seconds = max(0, int((now - dt).total_seconds()))
    except (TypeError, ValueError):
        return iso_timestamp

    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"


def _truncate_session_prompt(prompt: str | None, limit: int = 72) -> str:
    """Return a compact one-line prompt preview for a session list."""
    if not prompt:
        return "(no prompt preview)"
    compact = " ".join(prompt.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _format_sessions_text(sessions: list[dict[str, Any]], *, current_thread_id: str | None) -> str:
    """Render a Telegram-friendly sessions list."""
    lines = ["Recent sessions for this chat:"]
    for index, session in enumerate(sessions, start=1):
        thread_id = str(session["thread_id"])
        current_marker = " (current)" if thread_id == current_thread_id else ""
        prompt = _truncate_session_prompt(session.get("initial_prompt"))
        updated = _format_relative_time(session.get("updated_at"))
        created = _format_relative_time(session.get("created_at"))
        message_count = int(session.get("message_count", 0))
        lines.append(f"{index}. `{thread_id}`{current_marker} — {prompt}")
        lines.append(f"   Updated: {updated} | Created: {created} | Msgs: {message_count}")
    return "\n".join(lines)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status -- show the current thread and active model."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    header, _thread_id, _config = _status_header(update, context)
    await update.message.reply_text(header)


async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /sessions -- list recent saved sessions for the current chat."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    raw = update.message.text or ""
    parts = raw.split(maxsplit=1)
    limit = 10
    if len(parts) > 1:
        limit_arg = parts[1].strip()
        if not limit_arg.isdigit() or int(limit_arg) < 1:
            await update.message.reply_text("Usage: /sessions [limit]\nExample: /sessions 25")
            return
        limit = int(limit_arg)

    chat_id = str(update.effective_chat.id)
    current_thread_id = context.bot_data.get(THREAD_IDS_KEY, {}).get(chat_id)
    sessions = await asyncio.to_thread(list_sessions_for_chat, chat_id, limit=limit)
    if not sessions:
        await update.message.reply_text("No saved sessions found for this chat yet.")
        return

    text = _format_sessions_text(sessions, current_thread_id=current_thread_id)
    for chunk in chunk_message(text, TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(chunk)


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /resume [thread_id] -- switch the current chat back to a saved session."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    chat_id = str(update.effective_chat.id)
    raw = update.message.text or ""
    parts = raw.split(maxsplit=1)
    requested_thread = parts[1].strip() if len(parts) > 1 else ""

    if requested_thread:
        belongs = await asyncio.to_thread(session_belongs_to_chat, requested_thread, chat_id)
        if not belongs:
            similar = await asyncio.to_thread(
                find_similar_sessions_for_chat, requested_thread, chat_id
            )
            message = f"Session '{requested_thread}' was not found for this chat."
            if similar:
                message += f" Did you mean: {', '.join(similar)}?"
            await update.message.reply_text(message)
            return
        target_thread = requested_thread
    else:
        most_recent = await asyncio.to_thread(get_most_recent_session_for_chat, chat_id)
        if most_recent is None:
            await update.message.reply_text("No saved sessions found for this chat yet.")
            return
        target_thread = str(most_recent["thread_id"])

    await _cancel_active_run(context, chat_id)
    thread_ids = context.bot_data.setdefault(THREAD_IDS_KEY, {})
    thread_ids[chat_id] = target_thread
    save_thread_ids(thread_ids)
    context.bot_data.setdefault(LAST_MESSAGE_KEY, {}).pop(chat_id, None)
    _pending_approvals(context).pop(chat_id, None)
    logger.info("Resumed thread for chat %s: %s", chat_id, target_thread)
    await update.message.reply_text(
        f"Resumed session {target_thread}. Send a message to continue it."
    )


async def cmd_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /context -- show a full context breakdown for the current thread."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    header, thread_id, config = _status_header(update, context)
    report = build_context_report(config, thread_id)
    text = f"{header}\n\n{report}"
    for chunk in chunk_message(text, TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(chunk)


async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /queue -- show active-task and recent tool progress for the current chat."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return

    header, _thread_id, _config = _status_header(update, context)
    gateway: Gateway = context.bot_data[GATEWAY_KEY]
    chat_id = str(update.effective_chat.id)
    report = _build_queue_report(context, chat_id, gateway)
    text = f"{header}\n\n{report}"
    for chunk in chunk_message(text, TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(chunk)


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
    await _cancel_active_run(context, chat_id)
    new_thread = str(uuid.uuid4())
    thread_ids = context.bot_data.setdefault(THREAD_IDS_KEY, {})
    thread_ids[chat_id] = new_thread
    save_thread_ids(thread_ids)
    context.bot_data.setdefault(LAST_MESSAGE_KEY, {}).pop(chat_id, None)
    _pending_approvals(context).pop(chat_id, None)
    logger.info("Cleared thread for chat %s: %s", chat_id, new_thread)
    await update.message.reply_text("Conversation cleared.")


KNOWN_PROVIDERS = (
    "anthropic",
    "openai",
    "google",
    "groq",
    "mistral",
    "bedrock",
    "vertex",
    "nvidia",
    "deepinfra",
    "baseten",
)


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
            max_turns=new_config.max_turns,
            gateway_timeout=new_config.gateway_timeout,
            gateway_timeout_warning=new_config.gateway_timeout_warning,
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
            f"Model switched to: {model_arg}\nAgent reloaded — use /clear to start a fresh thread."
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
    if _pending_approvals(context).get(chat_id):
        await update.message.reply_text(_pending_approval_text())
        return
    if not _begin_active_run(context, chat_id):
        await update.message.reply_text(_active_run_text())
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
        pending = await gateway.handle_message(channel, incoming, thread_id)
    except asyncio.CancelledError:
        logger.info("Cancelled active retry for chat %s", chat_id)
        return
    finally:
        _finish_active_run(context, chat_id)
        _STREAM_MESSAGES.pop(chat_id, None)
    if pending:
        _pending_approvals(context)[chat_id] = pending
    else:
        _pending_approvals(context).pop(chat_id, None)


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stop -- cancel the current in-flight chat task."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    approvals = _pending_approvals(context)
    pending = approvals.get(chat_id)
    stopped = await _cancel_active_run(context, chat_id)
    _STREAM_MESSAGES.pop(chat_id, None)

    if pending is not None and not stopped:
        gateway: Gateway = context.bot_data[GATEWAY_KEY]
        channel = TelegramChannel(update, context)
        try:
            await gateway.resume_interrupt(
                channel,
                chat_id=chat_id,
                thread_id=pending["thread_id"],
                decision={"type": "reject", "message": "Stopped by user."},
            )
        finally:
            approvals.pop(chat_id, None)
        await update.message.reply_text("Stopped the current task.")
        return

    if pending is not None:
        approvals.pop(chat_id, None)

    if not stopped:
        await update.message.reply_text("No active task to stop.")
        return
    await update.message.reply_text("Stopped the current task.")


async def _resume_pending_interrupt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: str,
    pending: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    """Resume a pending interrupt while tracking it as the chat's active run."""
    if not _begin_active_run(context, chat_id):
        if update.message is not None:
            await update.message.reply_text(_active_run_text())
        elif update.callback_query and update.callback_query.message is not None:
            await update.callback_query.message.reply_text(_active_run_text())
        return False

    channel = TelegramChannel(update, context)
    gateway: Gateway = context.bot_data[GATEWAY_KEY]
    try:
        if decision.get("type") == "approve" and decision.get("scope") == "session":
            await aadd_thread_approved_keys(
                pending["thread_id"],
                pending.get("approval_keys", []),
            )
        next_pending = await gateway.resume_interrupt(
            channel,
            chat_id=chat_id,
            thread_id=pending["thread_id"],
            decision=decision,
        )
        approvals = _pending_approvals(context)
        if next_pending:
            approvals[chat_id] = next_pending
            pending_text = next_pending.get("message") or "Safety review required."
            reply_kwargs = {"reply_markup": _pending_approval_markup(next_pending["id"])}
            if update.message is not None:
                await update.message.reply_text(pending_text, **reply_kwargs)
            elif update.callback_query and update.callback_query.message is not None:
                await update.callback_query.message.reply_text(pending_text, **reply_kwargs)
        else:
            approvals.pop(chat_id, None)
    except asyncio.CancelledError:
        logger.info("Cancelled resumed interrupt for chat %s", chat_id)
        return False
    finally:
        _finish_active_run(context, chat_id)
        _STREAM_MESSAGES.pop(chat_id, None)
    return True


async def cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /approve -- resume a pending safety-reviewed command."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    pending = _pending_approvals(context).get(chat_id)
    if not pending:
        await update.message.reply_text("No pending safety approval for this chat.")
        return
    scope = _parse_approve_scope(update.message.text)
    await _resume_pending_interrupt(
        update,
        context,
        chat_id=chat_id,
        pending=pending,
        decision={"type": "approve", "scope": scope},
    )


async def cmd_deny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /deny [reason] -- reject a pending safety-reviewed command."""
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    pending = _pending_approvals(context).get(chat_id)
    if not pending:
        await update.message.reply_text("No pending safety approval for this chat.")
        return
    reason = _parse_deny_reason(update.message.text)
    decision = {"type": "reject"}
    if reason:
        decision["message"] = reason
    await _resume_pending_interrupt(
        update,
        context,
        chat_id=chat_id,
        pending=pending,
        decision=decision,
    )


async def cmd_approval_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline safety approval buttons."""
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.reply_text(REJECTION_MESSAGE)
        return
    chat_id = str(update.effective_chat.id)
    pending = _pending_approvals(context).get(chat_id)
    if not pending:
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.reply_text("No pending safety approval for this chat.")
        return

    parts = (query.data or "").split(":", maxsplit=2)
    if len(parts) != 3:
        return
    _prefix, action, pending_id = parts
    if pending.get("id") != pending_id:
        if query.message is not None:
            with contextlib.suppress(Exception):
                await query.message.reply_text(
                    "That approval button is stale. Please use the latest safety review prompt."
                )
        return

    if action == "approve_once":
        decision: dict[str, Any] = {"type": "approve", "scope": "once"}
    elif action == "approve_session":
        decision = {"type": "approve", "scope": "session"}
    elif action == "deny":
        decision = {"type": "reject", "message": "Command rejected via inline deny button"}
    else:
        return

    resumed = await _resume_pending_interrupt(
        update,
        context,
        chat_id=chat_id,
        pending=pending,
        decision=decision,
    )
    if resumed and query.message is not None:
        with contextlib.suppress(Exception):
            await query.message.edit_reply_markup(reply_markup=None)


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

    async def send_media(self, chat_id: str, path: str, caption: str | None = None) -> str:
        """Send native Telegram media to a chat."""
        msg = await _telegram_send_media_via_bot(self._bot, int(chat_id), path, caption)
        msg_id = str(msg.message_id)
        _STREAM_MESSAGES.setdefault(chat_id, {})[msg_id] = msg
        return msg_id

    async def send_voice(self, chat_id: str, path: str, caption: str | None = None) -> str:
        """Send audio as a Telegram voice message (round bubble)."""
        with _open_media_file(path) as handle:
            msg = await self._bot.send_voice(chat_id=int(chat_id), voice=handle, caption=caption)
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
                error_text = str(exc)
                logger.warning(
                    "Telegram bot-channel edit_message failed for chat %s msg %s: %s | text=%r",
                    chat_id,
                    message_id,
                    error_text,
                    text[:300],
                )
                if "Message is not modified" not in error_text:
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
        if self._update.message is not None:
            msg = await self._update.message.reply_text(text)
        elif (
            self._update.callback_query is not None
            and self._update.callback_query.message is not None
        ):
            msg = await self._update.callback_query.message.reply_text(text)
        else:
            raise RuntimeError("No Telegram message context available for send()")
        msg_id = str(msg.message_id)
        _STREAM_MESSAGES.setdefault(chat_id, {})[msg_id] = msg
        return msg_id

    async def send_media(self, chat_id: str, path: str, caption: str | None = None) -> str:
        """Send native Telegram media as a reply in the current context."""
        if self._update.message is not None:
            msg = await _telegram_send_media_via_message(self._update.message, path, caption)
        elif (
            self._update.callback_query is not None
            and self._update.callback_query.message is not None
        ):
            msg = await _telegram_send_media_via_message(
                self._update.callback_query.message, path, caption
            )
        else:
            raise RuntimeError("No Telegram message context available for send_media()")
        msg_id = str(msg.message_id)
        _STREAM_MESSAGES.setdefault(chat_id, {})[msg_id] = msg
        return msg_id

    async def send_voice(self, chat_id: str, path: str, caption: str | None = None) -> str:
        """Send audio as a Telegram voice message (round bubble)."""
        message = self._update.message
        if message is None and self._update.callback_query is not None:
            message = self._update.callback_query.message
        if message is None:
            raise RuntimeError("No Telegram message context available for send_voice()")
        with _open_media_file(path) as handle:
            msg = await message.reply_voice(voice=handle, caption=caption)
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
            error_text = str(exc)
            logger.warning(
                "Telegram edit_message failed for chat %s msg %s: %s | text=%r",
                chat_id,
                message_id,
                error_text,
                text[:300],
            )
            if "Message is not modified" not in error_text:
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
    if not _begin_active_run(context, chat_id):
        await update.message.reply_text(_active_run_text())
        return
    try:
        thread_id = get_thread_id(context, chat_id)
        user_text, media_error = await _build_incoming_text(update)
        if media_error:
            await update.message.reply_text(media_error)
            return
        if not user_text:
            return

        if _pending_approvals(context).get(chat_id):
            await update.message.reply_text(_pending_approval_text())
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

        pending = await gateway.handle_message(channel, incoming, thread_id)
        if pending:
            _pending_approvals(context)[chat_id] = pending
            await update.message.reply_text(
                pending.get("message") or "Safety review required.",
                reply_markup=_pending_approval_markup(pending["id"]),
            )
        else:
            _pending_approvals(context).pop(chat_id, None)
    except asyncio.CancelledError:
        logger.info("Cancelled active message run for chat %s", chat_id)
        return
    finally:
        _finish_active_run(context, chat_id)
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
    gateway = Gateway(
        agent=agent,
        streaming_config=config.telegram.streaming,
        max_turns=config.max_turns,
        gateway_timeout=config.gateway_timeout,
        gateway_timeout_warning=config.gateway_timeout_warning,
    )
    application.bot_data[GATEWAY_KEY] = gateway

    # Load allowed users: config + persisted file
    allowed_users = set(config.telegram.allowed_users)
    allowed_users.update(load_allowed_users())
    application.bot_data[ALLOWED_USERS_KEY] = allowed_users

    # Restore thread IDs from disk so /new survives bot restarts
    application.bot_data[THREAD_IDS_KEY] = load_thread_ids()

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
            max_turns=config.max_turns,
            run_timeout=config.gateway_timeout,
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
            BotCommand("retry", "Re-send the last message"),
            BotCommand("stop", "Stop the current in-flight task"),
            BotCommand("approve", "Approve a pending safety-reviewed command"),
            BotCommand("deny", "Reject a pending safety-reviewed command"),
            BotCommand("model", "View or set the active model"),
            BotCommand("skills", "Browse or manage local skills"),
            BotCommand("memory", "Show MEMORY.md"),
            BotCommand("soul", "Show SOUL.md"),
            BotCommand("uptime", "Show bot uptime"),
            BotCommand("status", "Show thread ID and model info"),
            BotCommand("queue", "Show active task and recent tool activity"),
            BotCommand("sessions", "List recent saved sessions"),
            BotCommand("resume", "Resume the latest or specified session"),
            BotCommand("context", "Show full context breakdown"),
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
    application.add_handler(CommandHandler("retry", cmd_retry, block=False))
    application.add_handler(CommandHandler("stop", cmd_stop, block=False))
    application.add_handler(CommandHandler("approve", cmd_approve, block=False))
    application.add_handler(CommandHandler("deny", cmd_deny, block=False))
    application.add_handler(CommandHandler("model", cmd_model))
    application.add_handler(CommandHandler("skills", cmd_skills))
    application.add_handler(CommandHandler("memory", cmd_memory))
    application.add_handler(CommandHandler("soul", cmd_soul))
    application.add_handler(CommandHandler("uptime", cmd_uptime))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("queue", cmd_queue))
    application.add_handler(CommandHandler("sessions", cmd_sessions))
    application.add_handler(CommandHandler("resume", cmd_resume))
    application.add_handler(CommandHandler("context", cmd_context))
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("cron", cmd_cron))
    application.add_handler(CommandHandler("cron_add", cmd_cron_add))
    application.add_handler(CommandHandler("cron_rm", cmd_cron_rm))
    application.add_handler(CommandHandler("safety_test", cmd_safety_test))
    application.add_handler(CommandHandler("doctor", cmd_doctor))
    application.add_handler(
        CallbackQueryHandler(cmd_approval_callback, pattern=r"^safety:", block=False)
    )
    application.add_handler(
        MessageHandler(
            (filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND, handle_message, block=False
        )
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message, block=False)
    )

    logger.info("Starting DeepClaw Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
