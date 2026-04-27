"""Configuration loader for DeepClaw.

Loads config from three layers with descending precedence:
  1. Shell environment variables
  2. ~/.deepclaw/.env file
  3. ~/.deepclaw/config.yaml file
  4. Dataclass defaults
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("~/.deepclaw").expanduser()
ENV_FILE = CONFIG_DIR / ".env"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
CHECKPOINTER_DB_PATH = Path("~/.deepagents/checkpoints.db").expanduser()

# Env var names matching the original bot.py conventions
ENV_TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
ENV_DEEPCLAW_MODEL = "DEEPCLAW_MODEL"
ENV_DEEPCLAW_ALLOWED_USERS = "DEEPCLAW_ALLOWED_USERS"
ENV_COMMAND_TIMEOUT = "DEEPCLAW_COMMAND_TIMEOUT"


@dataclass
class TelegramStreamingConfig:
    enabled: bool = True
    edit_interval: float = 1.0
    buffer_threshold: int = 100


@dataclass
class TelegramConfig:
    bot_token: str = ""
    allowed_users: list[str] = field(default_factory=list)
    streaming: TelegramStreamingConfig = field(default_factory=TelegramStreamingConfig)


@dataclass
class HeartbeatConfig:
    enabled: bool = False
    interval_minutes: int = 30
    quiet_hours_start: int | None = None  # 0-23, e.g. 23 for 11 PM
    quiet_hours_end: int | None = None  # 0-23, e.g. 8 for 8 AM
    timezone: str = "UTC"
    max_failures: int = 3
    notify_chat_id: str = ""


@dataclass
class TerminalConfig:
    env_passthrough: list[str] = field(default_factory=list)


@dataclass
class DeepClawConfig:
    model: str = "anthropic:claude-sonnet-4-6-20250514"
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    terminal: TerminalConfig = field(default_factory=TerminalConfig)
    workspace_root: str = "~/.deepclaw/workspace"
    command_timeout: int = 300  # seconds, default 5 minutes


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict, handling comments, blanks, and quoted values."""
    env_vars: dict[str, str] = {}
    if not path.is_file():
        return env_vars
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read env file: %s", path)
        return env_vars

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Handle lines prefixed with "export "
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        eq_pos = stripped.find("=")
        if eq_pos == -1:
            continue
        key = stripped[:eq_pos].strip()
        value = stripped[eq_pos + 1 :].strip()
        # Strip matching surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env_vars[key] = value
    return env_vars


def _resolve(env_name: str, dot_env: dict[str, str], yaml_value: Any) -> str | None:
    """Return the highest-precedence non-empty value across env, .env, and yaml."""
    shell_val = os.environ.get(env_name)
    if shell_val:
        return shell_val

    dot_val = dot_env.get(env_name)
    if dot_val:
        return dot_val

    if yaml_value is None:
        return None

    yaml_str = str(yaml_value)
    if yaml_str:
        return yaml_str

    return None


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML dict from disk, returning {} for missing/invalid/non-dict content."""
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    except Exception:
        logger.warning("Could not parse config file: %s", path, exc_info=True)
        return {}

    if isinstance(loaded, dict):
        return loaded
    return {}


DEFAULT_CONFIG_YAML = """\
# DeepClaw configuration
# Docs: https://github.com/akira/deepclaw#configuration

# Model — use provider:model format (default: anthropic:claude-sonnet-4-6)
# model: "openai:gpt-4o"

# Shell command timeout in seconds (default: 300 = 5 minutes)
# command_timeout: 300

# LangSmith tracing — add these to ~/.deepclaw/.env (not this file):
#   LANGSMITH_API_KEY=lsv2_...
#   LANGSMITH_TRACING=true
#   LANGSMITH_PROJECT=deepclaw

# Telegram settings
# telegram:
#   allowed_users:
#     - "123456789"      # your Telegram user ID
#   streaming:
#     edit_interval: 1.0
#     buffer_threshold: 100

# Heartbeat — periodic monitoring (disabled by default)
# heartbeat:
#   enabled: true
#   interval_minutes: 30
#   notify_chat_id: "123456789"  # check /status for your chat ID
#   quiet_hours_start: 23
#   quiet_hours_end: 8
#   timezone: "America/Los_Angeles"
#
# Terminal child-process env passthrough for blocked DeepClaw-managed secrets.
# Use this when a trusted skill or workflow needs one of DeepClaw's own API keys
# available inside shell commands.
# terminal:
#   env_passthrough:
#     - LANGSMITH_API_KEY
#     - LANGCHAIN_API_KEY
"""


def _seed_config() -> None:
    """Create default config.yaml if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
        logger.info("Seeded default config.yaml at %s", CONFIG_FILE)


def load_config() -> DeepClawConfig:
    """Load and merge configuration from all sources, returning a DeepClawConfig."""
    _seed_config()
    dot_env = _parse_env_file(ENV_FILE)

    for key, value in dot_env.items():
        if key not in os.environ:
            os.environ[key] = value

    yaml_data = _load_yaml(CONFIG_FILE)
    yaml_telegram = yaml_data.get("telegram", {}) or {}
    yaml_streaming = yaml_telegram.get("streaming", {}) or {}
    yaml_workspace = yaml_data.get("workspace", {}) or {}
    yaml_heartbeat = yaml_data.get("heartbeat", {}) or {}
    yaml_terminal = yaml_data.get("terminal", {}) or {}

    streaming = TelegramStreamingConfig(
        enabled=_to_bool(yaml_streaming.get("enabled"), TelegramStreamingConfig.enabled),
        edit_interval=_to_float(
            yaml_streaming.get("edit_interval"), TelegramStreamingConfig.edit_interval
        ),
        buffer_threshold=_to_int(
            yaml_streaming.get("buffer_threshold"), TelegramStreamingConfig.buffer_threshold
        ),
    )

    bot_token = _resolve(ENV_TELEGRAM_BOT_TOKEN, dot_env, yaml_telegram.get("bot_token")) or ""

    env_allowed_raw = _resolve(ENV_DEEPCLAW_ALLOWED_USERS, dot_env, None)
    if env_allowed_raw is not None:
        allowed_users = _to_str_list(env_allowed_raw)
    else:
        allowed_users = _to_str_list(yaml_telegram.get("allowed_users", []))

    telegram = TelegramConfig(
        bot_token=bot_token,
        allowed_users=allowed_users,
        streaming=streaming,
    )

    model = _resolve(ENV_DEEPCLAW_MODEL, dot_env, yaml_data.get("model")) or ""

    workspace_root = str(yaml_workspace.get("root", DeepClawConfig.workspace_root))

    heartbeat = HeartbeatConfig(
        enabled=_to_bool(yaml_heartbeat.get("enabled"), HeartbeatConfig.enabled),
        interval_minutes=_to_int(
            yaml_heartbeat.get("interval_minutes"), HeartbeatConfig.interval_minutes
        ),
        quiet_hours_start=yaml_heartbeat.get("quiet_hours_start"),
        quiet_hours_end=yaml_heartbeat.get("quiet_hours_end"),
        timezone=str(yaml_heartbeat.get("timezone", HeartbeatConfig.timezone)),
        max_failures=_to_int(yaml_heartbeat.get("max_failures"), HeartbeatConfig.max_failures),
        notify_chat_id=str(yaml_heartbeat.get("notify_chat_id", "")),
    )

    terminal = TerminalConfig(
        env_passthrough=_to_str_list(yaml_terminal.get("env_passthrough", []))
    )

    command_timeout = _to_int(
        _resolve(ENV_COMMAND_TIMEOUT, dot_env, yaml_data.get("command_timeout")),
        DeepClawConfig.command_timeout,
    )

    config = DeepClawConfig(
        model=model,
        telegram=telegram,
        heartbeat=heartbeat,
        terminal=terminal,
        workspace_root=workspace_root,
        command_timeout=command_timeout,
    )

    logger.info(
        "Config loaded: model=%s, allowed_users=%d",
        config.model,
        len(config.telegram.allowed_users),
    )
    return config
