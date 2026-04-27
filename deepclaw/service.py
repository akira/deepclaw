"""Service lifecycle management for running DeepClaw as a daemon.

Generates and manages systemd (Linux) and launchd (macOS) service definitions.
Environment variables from ~/.deepclaw/.env are inlined into the service file
at install time so the daemon has access to API keys and tokens.
"""

import os
import platform
import shutil
import sys
from pathlib import Path

from deepclaw.config import ENV_FILE, _parse_env_file

PLIST_LABEL = "com.deepclaw.bot"
SYSTEMD_UNIT_NAME = "deepclaw.service"
LOGS_DIR = Path("~/.deepclaw/logs").expanduser()


def detect_platform() -> str:
    """Return 'macos' or 'linux' based on the current OS."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return "linux"


def get_service_path(plat: str) -> Path:
    """Return the expected service file path for the given platform."""
    if plat == "macos":
        return Path(f"~/Library/LaunchAgents/{PLIST_LABEL}.plist").expanduser()
    return Path(f"~/.config/systemd/user/{SYSTEMD_UNIT_NAME}").expanduser()


def _resolve_deepclaw_command() -> str:
    """Find the deepclaw executable path, falling back to 'uv run deepclaw'."""
    path = shutil.which("deepclaw")
    if path:
        return path
    return "uv run deepclaw"


_SERVICE_ENV_KEYS = frozenset(
    {
        "TELEGRAM_BOT_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "OPENAI_API_KEY",
        "DEEPINFRA_API_TOKEN",
        "DEEPINFRA_API_KEY",
        "TAVILY_API_KEY",
        "DEEPCLAW_MODEL",
        "DEEPCLAW_ALLOWED_USERS",
    }
)


def _collect_env_vars() -> dict[str, str]:
    """Collect environment variables for the macOS service from ~/.deepclaw/.env.

    Only includes vars from the known set (_SERVICE_ENV_KEYS). Reads .env first,
    then overlays from shell env (shell takes precedence). Used for macOS launchd
    which has no EnvironmentFile equivalent.
    """
    env_vars: dict[str, str] = {}

    # Load from .env file — only known keys
    dot_env = _parse_env_file(ENV_FILE)
    for key in _SERVICE_ENV_KEYS:
        value = dot_env.get(key, "")
        if value:
            env_vars[key] = value

    # Overlay from shell — only known keys, shell wins
    for key in _SERVICE_ENV_KEYS:
        shell_val = os.environ.get(key, "")
        if shell_val:
            env_vars[key] = shell_val

    return env_vars


def generate_service_file(plat: str) -> str:
    """Generate the service file content for the given platform."""
    home = str(Path.home())
    command = _resolve_deepclaw_command()
    stdout_log = str(LOGS_DIR / "stdout.log")
    stderr_log = str(LOGS_DIR / "stderr.log")
    env_vars = _collect_env_vars()

    if plat == "macos":
        return _generate_launchd_plist(command, home, stdout_log, stderr_log, env_vars)
    return _generate_systemd_unit(command, home, env_vars)


def _xml_escape(value: str) -> str:
    """Escape a string for safe inclusion in XML/plist."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _generate_launchd_plist(
    command: str,
    working_dir: str,
    stdout_log: str,
    stderr_log: str,
    env_vars: dict[str, str],
) -> str:
    # Split command into program + arguments for ProgramArguments
    parts = command.split()
    program_args = "\n".join(f"        <string>{p}</string>" for p in parts)

    # Build EnvironmentVariables dict
    env_block = ""
    if env_vars:
        env_entries = "\n".join(
            f"        <key>{_xml_escape(k)}</key>\n        <string>{_xml_escape(v)}</string>"
            for k, v in sorted(env_vars.items())
        )
        env_block = f"""    <key>EnvironmentVariables</key>
    <dict>
{env_entries}
    </dict>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{program_args}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>StandardOutPath</key>
    <string>{stdout_log}</string>
    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>
{env_block}
</dict>
</plist>
"""


def _systemd_escape(value: str) -> str:
    """Escape a value for systemd Environment= lines."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _generate_systemd_unit(command: str, working_dir: str, env_vars: dict[str, str]) -> str:
    all_env = {"PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin"}
    all_env.update(env_vars)

    env_lines = "\n".join(
        f"Environment={k}={_systemd_escape(v)}" for k, v in sorted(all_env.items())
    )

    return f"""[Unit]
Description=DeepClaw Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={command}
WorkingDirectory={working_dir}
Restart=on-failure
RestartSec=5
{env_lines}

[Install]
WantedBy=default.target
"""


def install_service(plat: str) -> str:
    """Write the service file and return instructions for the user."""
    service_path = get_service_path(plat)
    service_path.parent.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    content = generate_service_file(plat)
    service_path.write_text(content, encoding="utf-8")
    service_path.chmod(0o600)

    if plat == "macos":
        return (
            f"Service file written to: {service_path}\n"
            f"Logs directory: {LOGS_DIR}\n"
            "\nTo enable and start the service, run:\n"
            f"  launchctl load {service_path}\n"
            "\nTo stop the service:\n"
            f"  launchctl unload {service_path}"
        )
    return (
        f"Service file written to: {service_path}\n"
        f"Logs directory: {LOGS_DIR}\n"
        "\nTo enable and start the service, run:\n"
        "  systemctl --user daemon-reload\n"
        "  systemctl --user enable deepclaw\n"
        "  systemctl --user start deepclaw\n"
        "\nTo stop the service:\n"
        "  systemctl --user stop deepclaw"
    )


def uninstall_service(plat: str) -> str:
    """Return instructions for uninstalling the service."""
    service_path = get_service_path(plat)

    if plat == "macos":
        return (
            f"Service file: {service_path}\n"
            "\nTo uninstall, run:\n"
            f"  launchctl unload {service_path}\n"
            f"  rm {service_path}"
        )
    return (
        f"Service file: {service_path}\n"
        "\nTo uninstall, run:\n"
        "  systemctl --user stop deepclaw\n"
        "  systemctl --user disable deepclaw\n"
        f"  rm {service_path}\n"
        "  systemctl --user daemon-reload"
    )


def service_status(plat: str) -> str:
    """Check if the service file exists and report status."""
    service_path = get_service_path(plat)
    if service_path.exists():
        return f"Service file installed at: {service_path}"
    return f"Service file not found at: {service_path}"
