"""User pairing, thread rotation metadata, and access control for DeepClaw."""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

ALLOWED_USERS_FILE = "~/.deepclaw/allowed_users.json"
THREAD_IDS_FILE = "~/.deepclaw/thread_ids.json"
REJECTION_MESSAGE = "You are not authorized to use this bot. Send /pair <code> to pair."


class ThreadState(TypedDict):
    current_thread_id: str
    parent_thread_id: str | None
    pending_summary_text: str | None
    summary_artifact_path: str | None
    checkpoint_artifact_path: str | None
    raw_history_artifact_paths: list[str]
    last_compacted_at: str | None
    last_compaction_reason: str | None


_EMPTY_THREAD_STATE: ThreadState = {
    "current_thread_id": "",
    "parent_thread_id": None,
    "pending_summary_text": None,
    "summary_artifact_path": None,
    "checkpoint_artifact_path": None,
    "raw_history_artifact_paths": [],
    "last_compacted_at": None,
    "last_compaction_reason": None,
}


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


def normalize_thread_state(value: Any, *, chat_id: str) -> ThreadState:
    """Upgrade legacy thread-id values into the structured rotation state."""
    state = deepcopy(_EMPTY_THREAD_STATE)
    state["current_thread_id"] = chat_id

    if isinstance(value, str) and value.strip():
        state["current_thread_id"] = value.strip()
        return state

    if not isinstance(value, dict):
        return state

    current_thread_id = value.get("current_thread_id")
    if isinstance(current_thread_id, str) and current_thread_id.strip():
        state["current_thread_id"] = current_thread_id.strip()

    parent_thread_id = value.get("parent_thread_id")
    if isinstance(parent_thread_id, str) and parent_thread_id.strip():
        state["parent_thread_id"] = parent_thread_id.strip()

    for key in (
        "pending_summary_text",
        "summary_artifact_path",
        "checkpoint_artifact_path",
        "last_compacted_at",
        "last_compaction_reason",
    ):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            state[key] = item

    raw_paths = value.get("raw_history_artifact_paths")
    if isinstance(raw_paths, list):
        state["raw_history_artifact_paths"] = [
            path.strip() for path in raw_paths if isinstance(path, str) and path.strip()
        ]

    return state


def get_thread_state(store: dict[str, Any], chat_id: str) -> ThreadState:
    """Return the normalized thread state for a chat, seeding defaults when missing."""
    state = normalize_thread_state(store.get(chat_id), chat_id=chat_id)
    store[chat_id] = state
    return state


def load_thread_ids() -> dict[str, ThreadState]:
    """Load per-chat thread rotation state from the persistent JSON file."""
    path = Path(os.path.expanduser(THREAD_IDS_FILE))
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        logger.warning("Failed to read %s, starting with empty thread IDs", path)
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(chat_id): normalize_thread_state(value, chat_id=str(chat_id))
        for chat_id, value in data.items()
    }


def save_thread_ids(thread_ids: dict[str, Any]) -> None:
    """Save per-chat thread rotation state to the persistent JSON file."""
    normalized = {
        str(chat_id): normalize_thread_state(value, chat_id=str(chat_id))
        for chat_id, value in thread_ids.items()
    }
    path = Path(os.path.expanduser(THREAD_IDS_FILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True))


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
