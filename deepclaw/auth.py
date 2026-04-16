"""User pairing and access control for DeepClaw."""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_USERS_FILE = "~/.deepclaw/allowed_users.json"
THREAD_IDS_FILE = "~/.deepclaw/thread_ids.json"
REJECTION_MESSAGE = "You are not authorized to use this bot. Send /pair <code> to pair."


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


def load_thread_ids() -> dict[str, str]:
    """Load per-chat thread IDs from the persistent JSON file."""
    path = Path(os.path.expanduser(THREAD_IDS_FILE))
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Failed to read %s, starting with empty thread IDs", path)
        return {}


def save_thread_ids(thread_ids: dict[str, str]) -> None:
    """Save per-chat thread IDs to the persistent JSON file."""
    path = Path(os.path.expanduser(THREAD_IDS_FILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thread_ids, indent=2))


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
