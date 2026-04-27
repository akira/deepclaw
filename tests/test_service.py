"""Tests for deepclaw.service module."""

import os
from unittest.mock import patch

from deepclaw.service import (
    _collect_env_vars,
    detect_platform,
    generate_service_file,
    get_service_path,
    install_service,
    service_status,
    uninstall_service,
)

# --- helpers ---


def _mock_platform(plat: str):
    """Return a patch that makes detect_platform() return the given value."""
    system = "Darwin" if plat == "macos" else "Linux"
    return patch("deepclaw.service.platform.system", return_value=system)


# --- tests ---


class TestDetectPlatform:
    def test_returns_macos_on_darwin(self):
        with _mock_platform("macos"):
            assert detect_platform() == "macos"

    def test_returns_linux_on_linux(self):
        with _mock_platform("linux"):
            assert detect_platform() == "linux"

    def test_returns_valid_value(self):
        result = detect_platform()
        assert result in ("macos", "linux")


class TestGetServicePath:
    def test_macos_path(self):
        path = get_service_path("macos")
        assert path.name == "com.deepclaw.bot.plist"
        assert "LaunchAgents" in str(path)

    def test_linux_path(self):
        path = get_service_path("linux")
        assert path.name == "deepclaw.service"
        assert "systemd/user" in str(path)


class TestGenerateServiceFile:
    def test_macos_plist_content(self):
        content = generate_service_file("macos")
        assert "<?xml" in content
        assert "com.deepclaw.bot" in content
        assert "<key>RunAtLoad</key>" in content
        assert "<true/>" in content
        assert "<key>KeepAlive</key>" in content
        assert "<key>StandardOutPath</key>" in content
        assert "<key>StandardErrorPath</key>" in content
        assert "deepclaw" in content.lower()

    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token", "ANTHROPIC_API_KEY": "sk-test"})
    def test_macos_plist_inlines_env_vars(self):
        content = generate_service_file("macos")
        assert "<key>EnvironmentVariables</key>" in content
        assert "<key>TELEGRAM_BOT_TOKEN</key>" in content
        assert "<string>test-token</string>" in content
        assert "<key>ANTHROPIC_API_KEY</key>" in content
        assert "<string>sk-test</string>" in content

    def test_macos_plist_no_env_block_when_empty(self):
        with patch("deepclaw.service._collect_env_vars", return_value={}):
            content = generate_service_file("macos")
        assert "EnvironmentVariables" not in content

    def test_linux_systemd_content(self):
        content = generate_service_file("linux")
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content
        assert "Restart=on-failure" in content
        assert "RestartSec=5" in content
        assert "Environment=" in content
        assert "PATH=" in content
        assert "ExecStart=" in content
        assert "deepclaw" in content.lower()

    @patch.dict(
        os.environ,
        {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "ANTHROPIC_API_KEY": "sk-test",
            "DEEPINFRA_API_TOKEN": "di-test",
            "DEEPINFRA_API_KEY": "di-key",
        },
    )
    def test_linux_systemd_inlines_env_vars(self):
        content = generate_service_file("linux")
        assert 'Environment=TELEGRAM_BOT_TOKEN="test-token"' in content
        assert 'Environment=ANTHROPIC_API_KEY="sk-test"' in content
        assert 'Environment=DEEPINFRA_API_TOKEN="di-test"' in content
        assert 'Environment=DEEPINFRA_API_KEY="di-key"' in content

    def test_linux_systemd_no_unknown_vars(self):
        with patch.dict(os.environ, {"RANDOM_SECRET": "should-not-appear"}):
            content = generate_service_file("linux")
        assert "RANDOM_SECRET" not in content


class TestCollectEnvVars:
    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "from-shell", "RANDOM_VAR": "ignored"})
    def test_only_known_keys_from_shell(self):
        with patch("deepclaw.service._parse_env_file", return_value={}):
            result = _collect_env_vars()
        assert "TELEGRAM_BOT_TOKEN" in result
        assert result["TELEGRAM_BOT_TOKEN"] == "from-shell"
        assert "RANDOM_VAR" not in result

    def test_reads_from_env_file(self):
        fake_env = {"ANTHROPIC_API_KEY": "from-file", "UNKNOWN_KEY": "ignored"}
        with (
            patch("deepclaw.service._parse_env_file", return_value=fake_env),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = _collect_env_vars()
        assert result.get("ANTHROPIC_API_KEY") == "from-file"
        assert "UNKNOWN_KEY" not in result

    @patch.dict(os.environ, {"TAVILY_API_KEY": "shell-wins"})
    def test_shell_overrides_env_file(self):
        fake_env = {"TAVILY_API_KEY": "from-file"}
        with patch("deepclaw.service._parse_env_file", return_value=fake_env):
            result = _collect_env_vars()
        assert result["TAVILY_API_KEY"] == "shell-wins"

    def test_empty_when_nothing_set(self):
        with (
            patch("deepclaw.service._parse_env_file", return_value={}),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = _collect_env_vars()
        assert result == {}


class TestInstallService:
    def test_writes_file_macos(self, tmp_path):
        plist_path = tmp_path / "com.deepclaw.bot.plist"
        logs_dir = tmp_path / "logs"

        with (
            patch("deepclaw.service.get_service_path", return_value=plist_path),
            patch("deepclaw.service.LOGS_DIR", logs_dir),
        ):
            result = install_service("macos")

        assert plist_path.exists()
        content = plist_path.read_text()
        assert "com.deepclaw.bot" in content
        assert "launchctl load" in result
        assert logs_dir.exists()

    def test_writes_file_linux(self, tmp_path):
        unit_path = tmp_path / "deepclaw.service"
        logs_dir = tmp_path / "logs"

        with (
            patch("deepclaw.service.get_service_path", return_value=unit_path),
            patch("deepclaw.service.LOGS_DIR", logs_dir),
        ):
            result = install_service("linux")

        assert unit_path.exists()
        content = unit_path.read_text()
        assert "[Service]" in content
        assert "systemctl --user" in result
        assert logs_dir.exists()


class TestUninstallService:
    def test_macos_returns_correct_path(self):
        result = uninstall_service("macos")
        path = get_service_path("macos")
        assert str(path) in result
        assert "launchctl unload" in result
        assert f"rm {path}" in result

    def test_linux_returns_correct_path(self):
        result = uninstall_service("linux")
        path = get_service_path("linux")
        assert str(path) in result
        assert "systemctl --user stop" in result
        assert "systemctl --user disable" in result
        assert f"rm {path}" in result


class TestServiceStatus:
    def test_file_exists(self, tmp_path):
        service_file = tmp_path / "test.service"
        service_file.write_text("dummy")

        with patch("deepclaw.service.get_service_path", return_value=service_file):
            result = service_status("linux")

        assert "installed" in result.lower()
        assert str(service_file) in result

    def test_file_missing(self, tmp_path):
        service_file = tmp_path / "nonexistent.service"

        with patch("deepclaw.service.get_service_path", return_value=service_file):
            result = service_status("linux")

        assert "not found" in result.lower()
        assert str(service_file) in result
