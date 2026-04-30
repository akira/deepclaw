"""Tests for deepclaw.config module."""

import pytest

from deepclaw.config import (
    ENV_COMMAND_TIMEOUT,
    _parse_env_file,
    _resolve,
    load_config,
)

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    """Redirect CONFIG_DIR, ENV_FILE, and CONFIG_FILE to a temp directory."""
    import deepclaw.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(cfg_mod, "CONFIG_FILE", tmp_path / "config.yaml")
    return tmp_path


def _write_yaml(config_dir, content: str):
    (config_dir / "config.yaml").write_text(content, encoding="utf-8")


def _write_env(config_dir, content: str):
    (config_dir / ".env").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# _parse_env_file
# ---------------------------------------------------------------------------


class TestParseEnvFile:
    def test_missing_file(self, tmp_path):
        assert _parse_env_file(tmp_path / "nonexistent") == {}

    def test_basic_key_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"FOO": "bar"}

    def test_comments_and_blank_lines(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=val\n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"KEY": "val"}

    def test_double_quoted_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('KEY="hello world"\n', encoding="utf-8")
        assert _parse_env_file(env_file) == {"KEY": "hello world"}

    def test_single_quoted_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY='hello world'\n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"KEY": "hello world"}

    def test_value_with_equals_sign(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=a=b=c\n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"KEY": "a=b=c"}

    def test_line_without_equals_is_skipped(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("NOEQUALS\nGOOD=yes\n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"GOOD": "yes"}

    def test_whitespace_around_key_and_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("  KEY  =  val  \n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"KEY": "val"}

    def test_empty_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=\n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"KEY": ""}

    def test_export_prefix_stripped(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("export FOO=bar\nexport BAZ='qux'\n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"FOO": "bar", "BAZ": "qux"}

    def test_export_prefix_mixed_with_normal(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("export A=1\nB=2\n# comment\nexport C=3\n", encoding="utf-8")
        assert _parse_env_file(env_file) == {"A": "1", "B": "2", "C": "3"}


# ---------------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def test_shell_env_wins(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "from_env")
        result = _resolve("TEST_VAR", {"TEST_VAR": "from_dot"}, "from_yaml")
        assert result == "from_env"

    def test_dot_env_wins_over_yaml(self, monkeypatch):
        monkeypatch.delenv("TEST_VAR", raising=False)
        result = _resolve("TEST_VAR", {"TEST_VAR": "from_dot"}, "from_yaml")
        assert result == "from_dot"

    def test_yaml_used_as_fallback(self, monkeypatch):
        monkeypatch.delenv("TEST_VAR", raising=False)
        result = _resolve("TEST_VAR", {}, "from_yaml")
        assert result == "from_yaml"

    def test_returns_none_when_all_empty(self, monkeypatch):
        monkeypatch.delenv("TEST_VAR", raising=False)
        assert _resolve("TEST_VAR", {}, None) is None

    def test_empty_string_shell_env_skipped(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "")
        result = _resolve("TEST_VAR", {"TEST_VAR": "from_dot"}, None)
        assert result == "from_dot"


# ---------------------------------------------------------------------------
# load_config — defaults
# ---------------------------------------------------------------------------


class TestLoadConfigDefaults:
    def test_default_values_no_files(self, config_dir, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        cfg = load_config()

        assert cfg.model == ""
        assert cfg.telegram.bot_token == ""
        assert cfg.telegram.allowed_users == []
        assert cfg.telegram.streaming.enabled is True
        assert cfg.telegram.streaming.edit_interval == 1.0
        assert cfg.telegram.streaming.buffer_threshold == 100
        assert cfg.terminal.compression == "none"
        assert cfg.generation.temperature is None
        assert cfg.generation.max_tokens is None
        assert cfg.generation.top_p is None
        assert cfg.generation.repetition_penalty is None
        assert cfg.terminal.env_passthrough == []
        assert cfg.workspace_root == "~/.deepclaw/workspace"


# ---------------------------------------------------------------------------
# load_config — from config.yaml
# ---------------------------------------------------------------------------


class TestLoadConfigYaml:
    def test_reads_yaml(self, config_dir, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        _write_yaml(
            config_dir,
            """\
model: gpt-4o
telegram:
  bot_token: yaml-token
  allowed_users:
    - "111"
    - alice
  streaming:
    enabled: false
    edit_interval: 2.5
    buffer_threshold: 200
generation:
  temperature: 0.3
  max_tokens: 4096
  top_p: 0.9
  repetition_penalty: 1.05
workspace:
  root: /tmp/ws
terminal:
  compression: rtk
  env_passthrough:
    - LANGSMITH_API_KEY
    - CUSTOM_TOKEN
""",
        )

        cfg = load_config()

        assert cfg.model == "gpt-4o"
        assert cfg.telegram.bot_token == "yaml-token"
        assert cfg.telegram.allowed_users == ["111", "alice"]
        assert cfg.telegram.streaming.enabled is False
        assert cfg.telegram.streaming.edit_interval == 2.5
        assert cfg.telegram.streaming.buffer_threshold == 200
        assert cfg.terminal.compression == "rtk"
        assert cfg.generation.temperature == pytest.approx(0.3)
        assert cfg.generation.max_tokens == 4096
        assert cfg.generation.top_p == pytest.approx(0.9)
        assert cfg.generation.repetition_penalty == pytest.approx(1.05)
        assert cfg.terminal.env_passthrough == ["LANGSMITH_API_KEY", "CUSTOM_TOKEN"]
        assert cfg.terminal.cron_approval_allowlist == []
        assert cfg.workspace_root == "/tmp/ws"

    def test_cron_approval_allowlist_loads_from_yaml(self, config_dir, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        _write_yaml(
            config_dir,
            """terminal:
  cron_approval_allowlist:
    - dangerous:code_injection
    - env_passthrough:LANGSMITH_API_KEY
""",
        )

        cfg = load_config()

        assert cfg.terminal.cron_approval_allowlist == [
            "dangerous:code_injection",
            "env_passthrough:LANGSMITH_API_KEY",
        ]

    def test_invalid_terminal_compression_raises(self, config_dir, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        _write_yaml(config_dir, "terminal:\n  compression: gzip\n")

        with pytest.raises(ValueError, match="terminal\\.compression"):
            load_config()

    def test_malformed_yaml_returns_defaults(self, config_dir, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        _write_yaml(config_dir, ":::bad yaml{{{\n")

        cfg = load_config()

        assert cfg.model == ""
        assert cfg.telegram.bot_token == ""

    def test_yaml_with_non_dict_content(self, config_dir, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        _write_yaml(config_dir, "- just\n- a\n- list\n")

        cfg = load_config()
        assert cfg.model == ""


# ---------------------------------------------------------------------------
# load_config — .env file
# ---------------------------------------------------------------------------


class TestLoadConfigDotEnv:
    def test_dot_env_overrides_yaml(self, config_dir, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        _write_yaml(config_dir, "model: yaml-model\ntelegram:\n  bot_token: yaml-token\n")
        _write_env(config_dir, "DEEPCLAW_MODEL=dotenv-model\nTELEGRAM_BOT_TOKEN=dotenv-token\n")

        cfg = load_config()

        assert cfg.model == "dotenv-model"
        assert cfg.telegram.bot_token == "dotenv-token"


# ---------------------------------------------------------------------------
# load_config — env var override
# ---------------------------------------------------------------------------


class TestLoadConfigEnvOverride:
    def test_env_var_overrides_all(self, config_dir, monkeypatch):
        _write_yaml(config_dir, "model: yaml-model\ntelegram:\n  bot_token: yaml-token\n")
        _write_env(config_dir, "DEEPCLAW_MODEL=dotenv-model\nTELEGRAM_BOT_TOKEN=dotenv-token\n")

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
        monkeypatch.setenv("DEEPCLAW_MODEL", "env-model")

        cfg = load_config()

        assert cfg.model == "env-model"
        assert cfg.telegram.bot_token == "env-token"


# ---------------------------------------------------------------------------
# Three-layer precedence end-to-end
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_full_precedence_chain(self, config_dir, monkeypatch):
        """env > .env > yaml > defaults — each layer visible where higher layer is absent."""
        _write_yaml(
            config_dir,
            "model: yaml-model\ntelegram:\n  bot_token: yaml-token\n  allowed_users:\n    - yaml-user\n",
        )
        _write_env(config_dir, "TELEGRAM_BOT_TOKEN=dotenv-token\n")
        monkeypatch.setenv("DEEPCLAW_MODEL", "env-model")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        cfg = load_config()

        # model: env var wins over yaml
        assert cfg.model == "env-model"
        # bot_token: .env wins over yaml (no shell env set)
        assert cfg.telegram.bot_token == "dotenv-token"
        # allowed_users: yaml used (no env var or .env for this)
        assert cfg.telegram.allowed_users == ["yaml-user"]


# ---------------------------------------------------------------------------
# Allowed users parsing
# ---------------------------------------------------------------------------


class TestAllowedUsersParsing:
    def test_comma_separated_from_env(self, config_dir, monkeypatch):
        monkeypatch.setenv("DEEPCLAW_ALLOWED_USERS", "111,alice, 222 ")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)

        cfg = load_config()
        assert cfg.telegram.allowed_users == ["111", "alice", "222"]

    def test_list_from_yaml(self, config_dir, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        _write_yaml(config_dir, "telegram:\n  allowed_users:\n    - 100\n    - bob\n")

        cfg = load_config()
        assert cfg.telegram.allowed_users == ["100", "bob"]

    def test_comma_separated_from_dot_env(self, config_dir, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        _write_env(config_dir, "DEEPCLAW_ALLOWED_USERS=aaa,bbb\n")

        cfg = load_config()
        assert cfg.telegram.allowed_users == ["aaa", "bbb"]


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


class TestDirectoryCreation:
    def test_config_dir_created(self, tmp_path, monkeypatch):
        import deepclaw.config as cfg_mod

        new_dir = tmp_path / "nonexistent_subdir" / ".deepclaw"
        monkeypatch.setattr(cfg_mod, "CONFIG_DIR", new_dir)
        monkeypatch.setattr(cfg_mod, "ENV_FILE", new_dir / ".env")
        monkeypatch.setattr(cfg_mod, "CONFIG_FILE", new_dir / "config.yaml")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("DEEPCLAW_MODEL", raising=False)
        monkeypatch.delenv("DEEPCLAW_ALLOWED_USERS", raising=False)

        assert not new_dir.exists()
        load_config()
        assert new_dir.exists()


class TestCommandTimeout:
    def test_command_timeout_from_yaml(self, config_dir, monkeypatch):
        monkeypatch.delenv(ENV_COMMAND_TIMEOUT, raising=False)
        _write_yaml(config_dir, "command_timeout: 123\n")
        cfg = load_config()
        assert cfg.command_timeout == 123

    def test_command_timeout_env_overrides_yaml(self, config_dir, monkeypatch):
        _write_yaml(config_dir, "command_timeout: 123\n")
        monkeypatch.setenv(ENV_COMMAND_TIMEOUT, "456")
        cfg = load_config()
        assert cfg.command_timeout == 456

    def test_command_timeout_invalid_falls_back_to_default(self, config_dir, monkeypatch):
        _write_yaml(config_dir, "command_timeout: nope\n")
        monkeypatch.delenv(ENV_COMMAND_TIMEOUT, raising=False)
        cfg = load_config()
        assert cfg.command_timeout == 300


class TestGenerationParsing:
    def test_invalid_generation_values_are_ignored(self, config_dir, monkeypatch):
        monkeypatch.delenv(ENV_COMMAND_TIMEOUT, raising=False)
        _write_yaml(
            config_dir,
            """\
generation:
  temperature: nope
  max_tokens: bad
  top_p: no-thanks
  repetition_penalty: still-no
""",
        )

        cfg = load_config()

        assert cfg.generation.temperature is None
        assert cfg.generation.max_tokens is None
        assert cfg.generation.top_p is None
        assert cfg.generation.repetition_penalty is None
