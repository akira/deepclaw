"""CLI entry point and subcommand routing for DeepClaw."""

import logging
import sys

from deepclaw.config import load_config

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def _handle_doctor_command() -> None:
    """Handle 'deepclaw doctor' CLI command."""
    import asyncio

    from deepclaw.doctor import format_report, run_checks

    config = load_config()
    checks = asyncio.run(run_checks(config))
    print(format_report(checks))  # noqa: T201


def _handle_service_command(args: list[str]) -> None:
    """Handle 'deepclaw service <subcommand>' CLI commands."""
    from deepclaw.service import (
        detect_platform,
        install_service,
        service_status,
        uninstall_service,
    )

    plat = detect_platform()

    if not args:
        print("Usage: deepclaw service {install|uninstall|status}")  # noqa: T201
        raise SystemExit(1)

    subcommand = args[0]
    if subcommand == "install":
        print(install_service(plat))  # noqa: T201
    elif subcommand == "uninstall":
        print(uninstall_service(plat))  # noqa: T201
    elif subcommand == "status":
        print(service_status(plat))  # noqa: T201
    else:
        print(f"Unknown service subcommand: {subcommand}")  # noqa: T201
        print("Usage: deepclaw service {install|uninstall|status}")  # noqa: T201
        raise SystemExit(1)


def _handle_login_command(args: list[str]) -> None:
    """Handle 'deepclaw login [provider]'.

    Providers:
      (none)   — Claude Pro/Max OAuth PKCE flow (default)
      copilot  — GitHub Copilot device code flow
      codex    — Import tokens from ~/.codex/auth.json (written by Codex CLI)
    """
    provider = args[0].lower() if args else "claude"

    if provider == "copilot":
        _handle_login_copilot()
    elif provider == "codex":
        _handle_login_codex()
    elif provider == "claude":
        _handle_login_claude()
    else:
        print(f"Unknown login provider: {provider}")  # noqa: T201
        print("Usage: deepclaw login [claude|copilot|codex]")  # noqa: T201
        raise SystemExit(1)


def _handle_login_claude() -> None:
    from deepclaw.oauth import login, resolve_token

    token, is_oauth = resolve_token()
    if token and is_oauth:
        print("Already authenticated with OAuth credentials.")  # noqa: T201
        answer = input("Re-authenticate? [y/N] ").strip().lower()
        if answer != "y":
            return

    result = login()
    if result:
        print("Login successful. DeepClaw will use your Claude subscription.")  # noqa: T201
    else:
        print("Login failed.")  # noqa: T201
        raise SystemExit(1)


def _handle_login_copilot() -> None:
    """Device code OAuth flow to obtain a GitHub gho_* token for Copilot."""
    from deepclaw.codex_auth import copilot_device_code_login, resolve_copilot_token

    # Show existing token if present
    token, source = resolve_copilot_token()
    if token:
        print(f"Already authenticated with GitHub Copilot (token from {source}).")  # noqa: T201
        answer = input("Re-authenticate? [y/N] ").strip().lower()
        if answer != "y":
            return

    print("Authenticating with GitHub Copilot via device code flow...")  # noqa: T201
    token = copilot_device_code_login()
    if token:
        # Store in GH_TOKEN so the current process and subprocesses pick it up,
        # and advise the user to persist it in their .env file.
        import os

        os.environ["GH_TOKEN"] = token
        print("Login successful!")  # noqa: T201
        print(  # noqa: T201
            f"Add the following to ~/.deepclaw/.env to persist across restarts:\n  GH_TOKEN={token}"
        )
    else:
        print("Login failed.")  # noqa: T201
        raise SystemExit(1)


def _handle_login_codex() -> None:
    """Import Codex tokens from ~/.codex/auth.json (written by the Codex CLI)."""
    from deepclaw.codex_auth import (
        _import_codex_cli_tokens,
        _save_codex_tokens,
        resolve_codex_token,
    )

    print("Importing Codex credentials from ~/.codex/auth.json ...")  # noqa: T201
    cli_tokens = _import_codex_cli_tokens()
    if not cli_tokens:
        print(  # noqa: T201
            "No valid Codex tokens found in ~/.codex/auth.json.\n"
            "Run `codex` in your terminal to log in via the Codex CLI, then re-run "
            "`deepclaw login codex`."
        )
        raise SystemExit(1)

    _save_codex_tokens(cli_tokens["access_token"], cli_tokens["refresh_token"])
    print("Codex credentials imported successfully.")  # noqa: T201

    # Verify resolve works end-to-end
    try:
        resolve_codex_token()
        print("Token verified. Set `model: codex:<model-name>` in config.yaml to use Codex.")  # noqa: T201
    except ValueError as exc:
        print(f"Warning: token import succeeded but verification failed: {exc}")  # noqa: T201


def _handle_logout_command() -> None:
    """Handle 'deepclaw logout' — remove saved OAuth credentials."""
    from deepclaw.oauth import _OAUTH_FILE

    if _OAUTH_FILE.is_file():
        _OAUTH_FILE.unlink()
        print("OAuth credentials removed.")  # noqa: T201
    else:
        print("No OAuth credentials found.")  # noqa: T201


def main() -> None:
    """Entry point: start the Telegram bot with long-polling."""
    args = sys.argv[1:]
    if args and args[0] == "service":
        _handle_service_command(args[1:])
        return
    if args and args[0] == "doctor":
        _handle_doctor_command()
        return
    if args and args[0] == "login":
        _handle_login_command(args[1:])
        return
    if args and args[0] == "logout":
        _handle_logout_command()
        return

    config = load_config()

    from deepclaw.channels.telegram import run_telegram

    run_telegram(config)


if __name__ == "__main__":
    main()
