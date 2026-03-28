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

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("~/.deepclaw").expanduser()
ENV_FILE = CONFIG_DIR / ".env"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Env var names matching the original bot.py conventions
ENV_TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
ENV_DEEPCLAW_MODEL = "DEEPCLAW_MODEL"
ENV_DEEPCLAW_ALLOWED_USERS = "DEEPCLAW_ALLOWED_USERS"


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
class DeepClawConfig:
    model: str = ""
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    workspace_root: str = "~/.deepclaw/workspace"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict, handling comments, blanks, and quoted values."""
    env_vars: dict[str, str] = {}
    if not path.is_file():
        return env_vars
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning(f"Could not read env file: {path}")
        return env_vars

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eq_pos = line.find("=")
        if eq_pos == -1:
            continue
        key = line[:eq_pos].strip()
        value = line[eq_pos + 1 :].strip()
        # Strip matching surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env_vars[key] = value
    return env_vars


def _resolve(env_name: str, dot_env: dict[str, str], yaml_value: str | None) -> str | None:
    """Return the highest-precedence non-empty value across the three layers."""
    # Shell env takes priority
    shell_val = os.environ.get(env_name)
    if shell_val is not None and shell_val != "":
        return shell_val
    # .env file next
    dot_val = dot_env.get(env_name)
    if dot_val is not None and dot_val != "":
        return dot_val
    # config.yaml value last
    if yaml_value is not None and yaml_value != "":
        return str(yaml_value)
    return None


def load_config() -> DeepClawConfig:
    """Load and merge configuration from all sources, returning a DeepClawConfig."""
    # Ensure config directory exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load .env
    dot_env = _parse_env_file(ENV_FILE)

    # Load config.yaml
    yaml_data: dict = {}
    if CONFIG_FILE.is_file():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded
        except Exception:
            logger.warning(f"Could not parse config file: {CONFIG_FILE}", exc_info=True)

    # Extract yaml sections
    yaml_telegram: dict = yaml_data.get("telegram", {}) or {}
    yaml_streaming: dict = yaml_telegram.get("streaming", {}) or {}
    yaml_workspace: dict = yaml_data.get("workspace", {}) or {}

    # Build streaming config
    streaming = TelegramStreamingConfig(
        enabled=yaml_streaming.get("enabled", TelegramStreamingConfig.enabled),
        edit_interval=float(yaml_streaming.get("edit_interval", TelegramStreamingConfig.edit_interval)),
        buffer_threshold=int(yaml_streaming.get("buffer_threshold", TelegramStreamingConfig.buffer_threshold)),
    )

    # Resolve bot_token
    yaml_bot_token = yaml_telegram.get("bot_token")
    bot_token = _resolve(ENV_TELEGRAM_BOT_TOKEN, dot_env, yaml_bot_token) or ""

    # Resolve allowed_users — env var is comma-separated, yaml is a list
    yaml_allowed: list = yaml_telegram.get("allowed_users", []) or []
    yaml_allowed_str = [str(u) for u in yaml_allowed]

    env_allowed_raw = _resolve(ENV_DEEPCLAW_ALLOWED_USERS, dot_env, None)
    if env_allowed_raw is not None:
        allowed_users = [u.strip() for u in env_allowed_raw.split(",") if u.strip()]
    elif yaml_allowed_str:
        allowed_users = yaml_allowed_str
    else:
        allowed_users = []

    telegram = TelegramConfig(
        bot_token=bot_token,
        allowed_users=allowed_users,
        streaming=streaming,
    )

    # Resolve model
    yaml_model = yaml_data.get("model")
    model = _resolve(ENV_DEEPCLAW_MODEL, dot_env, yaml_model) or ""

    # Workspace root
    workspace_root = yaml_workspace.get("root", DeepClawConfig.workspace_root)

    config = DeepClawConfig(
        model=model,
        telegram=telegram,
        workspace_root=str(workspace_root),
    )

    logger.info(f"Config loaded: model={config.model}, allowed_users={len(config.telegram.allowed_users)}")
    return config
