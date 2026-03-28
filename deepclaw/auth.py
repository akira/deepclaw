"""User pairing and access control for DeepClaw."""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_USERS_FILE = "~/.deepclaw/allowed_users.json"
REJECTION_MESSAGE = "You are not authorized to use this bot. Send /pair <code> to pair."


def load_allowed_users() -> set[str]:
    """Load allowed users from the persistent JSON file."""
    path = Path(os.path.expanduser(ALLOWED_USERS_FILE))
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception:
        logger.warning(f"Failed to read {path}, starting with empty allowlist")
        return set()


def save_allowed_users(users: set[str]) -> None:
    """Save allowed users to the persistent JSON file."""
    path = Path(os.path.expanduser(ALLOWED_USERS_FILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(users), indent=2))


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
    logger.info(f"Rejected user id={user.id} username={user.username}")
    return False
