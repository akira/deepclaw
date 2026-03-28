"""Tests for deepclaw.doctor module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from deepclaw.config import DeepClawConfig, TelegramConfig
from deepclaw.doctor import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_WARN,
    Check,
    check_checkpointer_path,
    check_config_dir,
    check_config_file,
    check_cron_dir,
    check_env_file,
    check_llm_api_key,
    check_service_installed,
    check_telegram_token,
    check_workspace,
    format_report,
    run_checks,
)


# --- helpers ---


def _make_config(bot_token: str = "", workspace_root: str = "/tmp/ws") -> DeepClawConfig:
    return DeepClawConfig(
        model="test-model",
        telegram=TelegramConfig(bot_token=bot_token),
        workspace_root=workspace_root,
    )


# --- tests ---


class TestCheckConfigDir:
    def test_exists(self, tmp_path):
        with patch("deepclaw.doctor.CONFIG_DIR", tmp_path):
            result = check_config_dir()
        assert result.status == STATUS_OK

    def test_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with patch("deepclaw.doctor.CONFIG_DIR", missing):
            result = check_config_dir()
        assert result.status == STATUS_FAIL


class TestCheckConfigFile:
    def test_valid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test\n")
        with patch("deepclaw.doctor.CONFIG_FILE", config_file):
            result = check_config_file()
        assert result.status == STATUS_OK

    def test_missing_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        with patch("deepclaw.doctor.CONFIG_FILE", config_file):
            result = check_config_file()
        assert result.status == STATUS_WARN

    def test_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(": : : not valid yaml [[[")
        with patch("deepclaw.doctor.CONFIG_FILE", config_file):
            result = check_config_file()
        # Could be warn or fail depending on how yaml handles it
        assert result.status in (STATUS_WARN, STATUS_FAIL)

    def test_empty_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        with patch("deepclaw.doctor.CONFIG_FILE", config_file):
            result = check_config_file()
        assert result.status == STATUS_WARN


class TestCheckEnvFile:
    def test_exists(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value\n")
        with patch("deepclaw.doctor.ENV_FILE", env_file):
            result = check_env_file()
        assert result.status == STATUS_OK

    def test_missing(self, tmp_path):
        env_file = tmp_path / ".env"
        with patch("deepclaw.doctor.ENV_FILE", env_file):
            result = check_env_file()
        assert result.status == STATUS_WARN


class TestCheckTelegramToken:
    def test_token_set(self):
        config = _make_config(bot_token="abc123")
        result = check_telegram_token(config)
        assert result.status == STATUS_OK

    def test_token_empty(self):
        config = _make_config(bot_token="")
        result = check_telegram_token(config)
        assert result.status == STATUS_FAIL


class TestCheckLlmApiKey:
    def test_anthropic_key_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = check_llm_api_key()
        assert result.status == STATUS_OK
        assert "ANTHROPIC_API_KEY" in result.message

    def test_openai_key_set(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = check_llm_api_key()
        assert result.status == STATUS_OK
        assert "OPENAI_API_KEY" in result.message

    def test_no_key_set(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = check_llm_api_key()
        assert result.status == STATUS_FAIL


class TestCheckWorkspace:
    def test_exists(self, tmp_path):
        config = _make_config(workspace_root=str(tmp_path))
        result = check_workspace(config)
        assert result.status == STATUS_OK

    def test_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        config = _make_config(workspace_root=str(missing))
        result = check_workspace(config)
        assert result.status == STATUS_WARN


class TestCheckCheckpointerPath:
    def test_exists(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        with patch("deepclaw.doctor.CHECKPOINTER_DB_PATH", db_path):
            result = check_checkpointer_path()
        assert result.status == STATUS_OK

    def test_missing(self, tmp_path):
        db_path = tmp_path / "nonexistent" / "checkpoints.db"
        with patch("deepclaw.doctor.CHECKPOINTER_DB_PATH", db_path):
            result = check_checkpointer_path()
        assert result.status == STATUS_WARN


class TestCheckServiceInstalled:
    def test_exists(self, tmp_path):
        service_file = tmp_path / "test.service"
        service_file.write_text("dummy")
        with patch("deepclaw.doctor.get_service_path", return_value=service_file):
            result = check_service_installed()
        assert result.status == STATUS_OK

    def test_missing(self, tmp_path):
        service_file = tmp_path / "nonexistent.service"
        with patch("deepclaw.doctor.get_service_path", return_value=service_file):
            result = check_service_installed()
        assert result.status == STATUS_WARN


class TestCheckCronDir:
    def test_exists(self, tmp_path):
        with patch("deepclaw.doctor.CRON_DIR", tmp_path):
            result = check_cron_dir()
        assert result.status == STATUS_OK

    def test_missing(self, tmp_path):
        missing = tmp_path / "nonexistent"
        with patch("deepclaw.doctor.CRON_DIR", missing):
            result = check_cron_dir()
        assert result.status == STATUS_WARN


class TestFormatReport:
    def test_markers_present(self):
        checks = [
            Check("A", STATUS_OK, "good"),
            Check("B", STATUS_WARN, "meh"),
            Check("C", STATUS_FAIL, "bad"),
        ]
        report = format_report(checks)
        assert "\u2713" in report
        assert "\u26a0" in report
        assert "\u2717" in report

    def test_summary_line(self):
        checks = [
            Check("A", STATUS_OK, "good"),
            Check("B", STATUS_OK, "also good"),
            Check("C", STATUS_WARN, "meh"),
        ]
        report = format_report(checks)
        assert "2 ok" in report
        assert "1 warnings" in report
        assert "0 failures" in report

    def test_empty_checks(self):
        report = format_report([])
        assert "0 ok" in report
        assert "0 warnings" in report
        assert "0 failures" in report

    def test_header(self):
        report = format_report([])
        assert "DeepClaw Doctor" in report


class TestRunChecks:
    @pytest.mark.asyncio
    async def test_returns_list_of_checks(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        config = _make_config(bot_token="token123", workspace_root=str(tmp_path))
        with (
            patch("deepclaw.doctor.CONFIG_DIR", tmp_path),
            patch("deepclaw.doctor.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("deepclaw.doctor.ENV_FILE", tmp_path / ".env"),
            patch("deepclaw.doctor.CHECKPOINTER_DB_PATH", tmp_path / "checkpoints.db"),
            patch("deepclaw.doctor.CRON_DIR", tmp_path / "cron"),
            patch("deepclaw.doctor.get_service_path", return_value=tmp_path / "svc"),
        ):
            checks = await run_checks(config)

        assert isinstance(checks, list)
        assert len(checks) == 9
        assert all(isinstance(c, Check) for c in checks)

    @pytest.mark.asyncio
    async def test_all_statuses_are_valid(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = _make_config()
        with (
            patch("deepclaw.doctor.CONFIG_DIR", tmp_path),
            patch("deepclaw.doctor.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("deepclaw.doctor.ENV_FILE", tmp_path / ".env"),
            patch("deepclaw.doctor.CHECKPOINTER_DB_PATH", tmp_path / "checkpoints.db"),
            patch("deepclaw.doctor.CRON_DIR", tmp_path / "cron"),
            patch("deepclaw.doctor.get_service_path", return_value=tmp_path / "svc"),
        ):
            checks = await run_checks(config)

        valid = {STATUS_OK, STATUS_WARN, STATUS_FAIL}
        for check in checks:
            assert check.status in valid, f"{check.name} has invalid status: {check.status}"
