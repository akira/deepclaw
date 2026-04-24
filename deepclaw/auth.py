"""User pairing and access control for DeepClaw."""

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ALLOWED_USERS_FILE = "~/.deepclaw/allowed_users.json"
THREAD_IDS_FILE = "~/.deepclaw/thread_ids.json"
REJECTION_MESSAGE = "You are not authorized to use this bot. Send /pair <code> to pair."


ThreadState = dict[str, Any]


def load_allowed_users() -> set[str]:
    """Load allowed users from the persistent JSON file."""
    path = Path(os.path.expanduser(ALLOWED_USERS_FILE))
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception:
        logger.warning("Failed to read %s, starting with empty allowlist", path)
        return set()


def save_allowed_users(users: set[str]) -> None:
    """Save allowed users to the persistent JSON file."""
    path = Path(os.path.expanduser(ALLOWED_USERS_FILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(users), indent=2))


def normalize_thread_state(value: str | dict[str, Any] | None, *, chat_id: str) -> ThreadState:
    """Return a normalized thread-state record.

    Backward-compatible with the legacy ``chat_id -> thread_id`` string mapping.
    """
    if isinstance(value, str) and value:
        current_thread_id = value
    elif isinstance(value, dict) and isinstance(value.get("current_thread_id"), str):
        current_thread_id = value["current_thread_id"]
    else:
        current_thread_id = chat_id

    state: ThreadState = {
        "current_thread_id": current_thread_id,
        "parent_thread_id": None,
        "summary_artifact_path": None,
        "raw_history_artifact_paths": [],
        "pending_summary_text": None,
        "last_compacted_at": None,
        "last_compaction_reason": None,
    }
    if isinstance(value, dict):
        state.update({k: deepcopy(v) for k, v in value.items() if k in state})
        if not isinstance(state.get("raw_history_artifact_paths"), list):
            state["raw_history_artifact_paths"] = []
    return state


def load_thread_ids() -> dict[str, ThreadState]:
    """Load per-chat thread metadata from the persistent JSON file."""
    path = Path(os.path.expanduser(THREAD_IDS_FILE))
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return {}
        return {
            str(chat_id): normalize_thread_state(value, chat_id=str(chat_id))
            for chat_id, value in data.items()
        }
    except Exception:
        logger.warning("Failed to read %s, starting with empty thread IDs", path)
        return {}


def save_thread_ids(thread_ids: dict[str, ThreadState]) -> None:
    """Save per-chat thread metadata to the persistent JSON file."""
    path = Path(os.path.expanduser(THREAD_IDS_FILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        str(chat_id): normalize_thread_state(value, chat_id=str(chat_id))
        for chat_id, value in thread_ids.items()
    }
    path.write_text(json.dumps(normalized, indent=2))


def get_thread_state(thread_ids: dict[str, ThreadState], chat_id: str) -> ThreadState:
    """Return mutable thread metadata for a chat, initializing defaults if absent."""
    state = normalize_thread_state(thread_ids.get(chat_id), chat_id=chat_id)
    thread_ids[chat_id] = state
    return state


def is_user_allowed(update, allowed_users: set[str]) -> bool:
    """Check if the effective user is in the allowlist.

    Always rejects if no users are paired (no open mode).
    """
    if not allowed_users:
        return False
    user = update.effective_user
    if user is None:
        return False
    if str(user.id) in allowed_users:
        return True
    if user.username and user.username in allowed_users:
        return True
    logger.info("Rejected user id=%s username=%s", user.id, user.username)
    return False
