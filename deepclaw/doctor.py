"""Diagnostics for DeepClaw — checks system health and reports issues."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from deepclaw.config import CHECKPOINTER_DB_PATH, CONFIG_DIR, CONFIG_FILE, ENV_FILE, DeepClawConfig
from deepclaw.service import detect_platform, get_service_path
from deepclaw.tools.skills import skills_audit, skills_check_resolvable

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"

MARKER_OK = "\u2713"
MARKER_WARN = "\u26a0"
MARKER_FAIL = "\u2717"

CRON_DIR = CONFIG_DIR / "cron"


CheckStatus = Literal["ok", "warn", "fail"]


@dataclass
class Check:
    name: str
    status: CheckStatus
    message: str


def check_config_dir() -> Check:
    """Check whether ~/.deepclaw/ exists."""
    if CONFIG_DIR.is_dir():
        return Check("Config directory", STATUS_OK, f"{CONFIG_DIR} exists")
    return Check("Config directory", STATUS_FAIL, f"{CONFIG_DIR} not found")


def check_config_file() -> Check:
    """Check whether config.yaml is present and parseable."""
    if not CONFIG_FILE.is_file():
        return Check("Config file", STATUS_WARN, f"{CONFIG_FILE} not found")
    try:
        import yaml

        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return Check("Config file", STATUS_WARN, f"{CONFIG_FILE} is empty or not a mapping")
        return Check("Config file", STATUS_OK, f"{CONFIG_FILE} is valid")
    except Exception as exc:
        return Check("Config file", STATUS_FAIL, f"{CONFIG_FILE} parse error: {exc}")


def check_env_file() -> Check:
    """Check whether .env is present."""
    if ENV_FILE.is_file():
        return Check("Env file", STATUS_OK, f"{ENV_FILE} exists")
    return Check("Env file", STATUS_WARN, f"{ENV_FILE} not found")


def check_telegram_token(config: DeepClawConfig) -> Check:
    """Check whether bot_token is set (non-empty)."""
    if config.telegram.bot_token:
        return Check("Telegram token", STATUS_OK, "bot_token is set")
    return Check("Telegram token", STATUS_FAIL, "bot_token is not set")


def check_llm_api_key() -> Check:
    """Check whether ANTHROPIC_API_KEY or OPENAI_API_KEY is set in environment."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return Check("LLM API key", STATUS_OK, "ANTHROPIC_API_KEY is set")
    if os.environ.get("OPENAI_API_KEY"):
        return Check("LLM API key", STATUS_OK, "OPENAI_API_KEY is set")
    return Check("LLM API key", STATUS_FAIL, "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set")


def check_workspace(config: DeepClawConfig) -> Check:
    """Check whether the workspace root directory exists."""
    workspace = Path(config.workspace_root).expanduser()
    if workspace.is_dir():
        return Check("Workspace", STATUS_OK, f"{workspace} exists")
    return Check("Workspace", STATUS_WARN, f"{workspace} not found")


def check_terminal_compression(config: DeepClawConfig) -> Check:
    """Check whether configured terminal compression backend is available."""
    mode = str(config.terminal.compression or "none").strip().lower()
    if mode == "none":
        return Check("Terminal compression", STATUS_OK, "disabled")
    if mode == "rtk":
        if shutil.which("rtk"):
            return Check("Terminal compression", STATUS_OK, "rtk is available")
        return Check(
            "Terminal compression",
            STATUS_FAIL,
            "rtk compression enabled but `rtk` is not installed",
        )
    return Check("Terminal compression", STATUS_FAIL, f"unknown compression mode: {mode}")


def check_checkpointer_path() -> Check:
    """Check whether the checkpoints.db parent directory exists."""
    parent = CHECKPOINTER_DB_PATH.parent
    if parent.is_dir():
        return Check("Checkpointer path", STATUS_OK, f"{parent} exists")
    return Check("Checkpointer path", STATUS_WARN, f"{parent} not found")


def check_service_installed() -> Check:
    """Check whether the service file is present."""
    plat = detect_platform()
    service_path = get_service_path(plat)
    if service_path.exists():
        return Check("Service installed", STATUS_OK, f"{service_path} exists")
    return Check("Service installed", STATUS_WARN, f"{service_path} not found")


def check_cron_dir() -> Check:
    """Check whether ~/.deepclaw/cron/ exists."""
    if CRON_DIR.is_dir():
        return Check("Cron directory", STATUS_OK, f"{CRON_DIR} exists")
    return Check("Cron directory", STATUS_WARN, f"{CRON_DIR} not found")


def check_skills_health() -> Check:
    """Check whether local skills are loadable and follow baseline skill conventions."""
    resolvable = skills_check_resolvable()
    audit = skills_audit()

    if resolvable.get("unresolvable_count", 0):
        first = (resolvable.get("unresolvable") or [{}])[0]
        return Check(
            "Skills health",
            STATUS_FAIL,
            f"{resolvable['unresolvable_count']} unresolvable skills; first issue: {first.get('name', 'unknown')} — {first.get('reason', 'unknown reason')}",
        )

    duplicate_count = audit.get("duplicate_descriptions_count", 0)
    missing_sections_count = audit.get("skills_missing_required_sections_count", 0)
    if duplicate_count or missing_sections_count:
        return Check(
            "Skills health",
            STATUS_WARN,
            f"{audit.get('count', 0)} skills checked; duplicate descriptions: {duplicate_count}; missing required sections: {missing_sections_count}",
        )

    return Check(
        "Skills health",
        STATUS_OK,
        f"{audit.get('count', 0)} skills checked; all loadable and baseline sections present",
    )


async def run_checks(config: DeepClawConfig) -> list[Check]:
    """Run all diagnostic checks and return the results."""
    return [
        check_config_dir(),
        check_config_file(),
        check_env_file(),
        check_telegram_token(config),
        check_llm_api_key(),
        check_workspace(config),
        check_terminal_compression(config),
        check_checkpointer_path(),
        check_service_installed(),
        check_cron_dir(),
        check_skills_health(),
    ]


def format_report(checks: list[Check]) -> str:
    """Format checks as a readable report with markers and a summary line."""
    markers = {
        STATUS_OK: MARKER_OK,
        STATUS_WARN: MARKER_WARN,
        STATUS_FAIL: MARKER_FAIL,
    }

    lines: list[str] = ["DeepClaw Doctor", "=" * 40]
    for check in checks:
        marker = markers.get(check.status, "?")
        lines.append(f"  {marker} {check.name}: {check.message}")

    ok_count = sum(1 for c in checks if c.status == STATUS_OK)
    warn_count = sum(1 for c in checks if c.status == STATUS_WARN)
    fail_count = sum(1 for c in checks if c.status == STATUS_FAIL)
    lines.append("=" * 40)
    lines.append(f"Summary: {ok_count} ok, {warn_count} warnings, {fail_count} failures")

    return "\n".join(lines)
