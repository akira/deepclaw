"""Backwards-compatibility shim -- re-exports from refactored modules."""

# Re-export auth
from deepclaw.auth import (  # noqa: F401
    ALLOWED_USERS_FILE,
    REJECTION_MESSAGE,
    is_user_allowed,
)

# Re-export telegram channel
from deepclaw.channels.telegram import (  # noqa: F401
    ALLOWED_USERS_KEY,
    CONFIG_KEY,
    JOBS_PATH_KEY,
    PAIRING_CODE_KEY,
    SCHEDULER_KEY,
    TELEGRAM_MESSAGE_LIMIT,
    THREAD_IDS_KEY,
    cmd_cron,
    cmd_cron_add,
    cmd_cron_rm,
    cmd_doctor,
    cmd_help,
    cmd_new,
    cmd_pair,
    cmd_safety_test,
    cmd_start,
    cmd_status,
    get_thread_id,
    handle_message,
    post_init,
    post_shutdown,
)

# Re-export CLI entry point
from deepclaw.cli import (  # noqa: F401
    _handle_doctor_command,
    _handle_service_command,
    main,
)

# Re-export gateway symbols (moved from telegram.py in refactor)
from deepclaw.gateway import (  # noqa: F401
    CURSOR_INDICATOR,
    THINKING_MESSAGE,
    chunk_message,
)
