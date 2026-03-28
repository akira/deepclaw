"""Backwards-compatibility shim -- re-exports from refactored modules."""

# Re-export auth
from deepclaw.auth import (  # noqa: F401
    REJECTION_MESSAGE,
    is_user_allowed,
)

# Re-export telegram channel
from deepclaw.channels.telegram import (  # noqa: F401
    ALLOWED_USERS_KEY,
    CONFIG_KEY,
    CURSOR_INDICATOR,
    JOBS_PATH_KEY,
    PAIRING_CODE_KEY,
    SCHEDULER_KEY,
    TELEGRAM_MESSAGE_LIMIT,
    THINKING_MESSAGE,
    THREAD_IDS_KEY,
    _edit_stream_message,
    chunk_message,
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

# Re-export auth file helpers with old names
from deepclaw.auth import (  # noqa: F401
    ALLOWED_USERS_FILE,
    load_allowed_users as _load_allowed_users_file,
    save_allowed_users as _save_allowed_users_file,
)

# Re-export CLI entry point
from deepclaw.cli import (  # noqa: F401
    _handle_doctor_command,
    _handle_service_command,
    main,
)
