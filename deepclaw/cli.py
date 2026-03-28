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


def main() -> None:
    """Entry point: start the Telegram bot with long-polling."""
    args = sys.argv[1:]
    if args and args[0] == "service":
        _handle_service_command(args[1:])
        return
    if args and args[0] == "doctor":
        _handle_doctor_command()
        return

    config = load_config()

    from deepclaw.channels.telegram import run_telegram

    run_telegram(config)


if __name__ == "__main__":
    main()
