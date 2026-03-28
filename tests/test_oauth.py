"""Tests for deepclaw.oauth module."""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepclaw.oauth import (
    _generate_pkce,
    _is_token_expired,
    _read_claude_code_credentials,
    _read_deepclaw_credentials,
    is_oauth_token,
    resolve_token,
)


# ---------------------------------------------------------------------------
# is_oauth_token
# ---------------------------------------------------------------------------


class TestIsOauthToken:
    def test_regular_api_key(self):
        assert is_oauth_token("sk-ant-api03-abc123") is False

    def test_oauth_token(self):
        assert is_oauth_token("sk-ant-oat01-abc123") is True

    def test_jwt_token(self):
        assert is_oauth_token("eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.abc") is True

    def test_empty(self):
        assert is_oauth_token("") is False

    def test_generic_key(self):
        assert is_oauth_token("some-other-key-format") is True


# ---------------------------------------------------------------------------
# _generate_pkce
# ---------------------------------------------------------------------------


class TestGeneratePkce:
    def test_returns_verifier_and_challenge(self):
        verifier, challenge = _generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)
        assert len(verifier) > 20
        assert len(challenge) > 20

    def test_unique_each_call(self):
        v1, c1 = _generate_pkce()
        v2, c2 = _generate_pkce()
        assert v1 != v2
        assert c1 != c2

    def test_no_padding(self):
        verifier, challenge = _generate_pkce()
        assert "=" not in verifier
        assert "=" not in challenge


# ---------------------------------------------------------------------------
# _is_token_expired
# ---------------------------------------------------------------------------


class TestIsTokenExpired:
    def test_not_expired(self):
        future_ms = int(time.time() * 1000) + 3_600_000  # 1 hour from now
        assert _is_token_expired({"expiresAt": future_ms}) is False

    def test_expired(self):
        past_ms = int(time.time() * 1000) - 60_000  # 1 minute ago
        assert _is_token_expired({"expiresAt": past_ms}) is True

    def test_within_refresh_buffer(self):
        # 1 minute from now but within 2-minute refresh buffer
        almost_ms = int(time.time() * 1000) + 60_000
        assert _is_token_expired({"expiresAt": almost_ms}) is True

    def test_no_expiry(self):
        assert _is_token_expired({"expiresAt": 0}) is False
        assert _is_token_expired({}) is False


# ---------------------------------------------------------------------------
# _read_deepclaw_credentials
# ---------------------------------------------------------------------------


class TestReadDeepclawCredentials:
    def test_reads_valid_file(self, tmp_path):
        creds = {"accessToken": "sk-ant-oat01-test", "refreshToken": "rt", "expiresAt": 999}
        cred_file = tmp_path / ".anthropic_oauth.json"
        cred_file.write_text(json.dumps(creds))

        with patch("deepclaw.oauth._OAUTH_FILE", cred_file):
            result = _read_deepclaw_credentials()
        assert result is not None
        assert result["accessToken"] == "sk-ant-oat01-test"

    def test_returns_none_if_missing(self, tmp_path):
        with patch("deepclaw.oauth._OAUTH_FILE", tmp_path / "nonexistent.json"):
            assert _read_deepclaw_credentials() is None

    def test_returns_none_if_empty_token(self, tmp_path):
        cred_file = tmp_path / ".anthropic_oauth.json"
        cred_file.write_text(json.dumps({"accessToken": "", "refreshToken": "rt"}))

        with patch("deepclaw.oauth._OAUTH_FILE", cred_file):
            assert _read_deepclaw_credentials() is None


# ---------------------------------------------------------------------------
# _read_claude_code_credentials
# ---------------------------------------------------------------------------


class TestReadClaudeCodeCredentials:
    def test_reads_valid_file(self, tmp_path):
        data = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-cc",
                "refreshToken": "rt-cc",
                "expiresAt": 999,
            }
        }
        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps(data))

        with patch("deepclaw.oauth._CLAUDE_CODE_CREDS", cred_file):
            result = _read_claude_code_credentials()
        assert result is not None
        assert result["accessToken"] == "sk-ant-oat01-cc"

    def test_returns_none_if_missing(self, tmp_path):
        with patch("deepclaw.oauth._CLAUDE_CODE_CREDS", tmp_path / "nonexistent.json"):
            assert _read_claude_code_credentials() is None

    def test_returns_none_if_no_oauth_section(self, tmp_path):
        cred_file = tmp_path / ".credentials.json"
        cred_file.write_text(json.dumps({"other": "data"}))

        with patch("deepclaw.oauth._CLAUDE_CODE_CREDS", cred_file):
            assert _read_claude_code_credentials() is None


# ---------------------------------------------------------------------------
# resolve_token — priority chain
# ---------------------------------------------------------------------------


class TestResolveToken:
    def test_anthropic_token_env_first(self):
        with patch.dict(os.environ, {"ANTHROPIC_TOKEN": "sk-ant-oat01-env", "ANTHROPIC_API_KEY": "sk-ant-api03-key"}):
            token, is_oauth = resolve_token()
        assert token == "sk-ant-oat01-env"
        assert is_oauth is True

    def test_deepclaw_creds_second(self, tmp_path):
        creds = {
            "accessToken": "sk-ant-oat01-dc",
            "refreshToken": "rt",
            "expiresAt": int(time.time() * 1000) + 3_600_000,
        }
        cred_file = tmp_path / ".anthropic_oauth.json"
        cred_file.write_text(json.dumps(creds))

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("deepclaw.oauth._OAUTH_FILE", cred_file),
            patch("deepclaw.oauth._CLAUDE_CODE_CREDS", tmp_path / "nonexistent"),
        ):
            # Remove env vars
            os.environ.pop("ANTHROPIC_TOKEN", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            token, is_oauth = resolve_token()
        assert token == "sk-ant-oat01-dc"
        assert is_oauth is True

    def test_claude_code_creds_third(self, tmp_path):
        data = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-cc",
                "refreshToken": "rt",
                "expiresAt": int(time.time() * 1000) + 3_600_000,
            }
        }
        cc_file = tmp_path / ".credentials.json"
        cc_file.write_text(json.dumps(data))

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("deepclaw.oauth._OAUTH_FILE", tmp_path / "nonexistent"),
            patch("deepclaw.oauth._CLAUDE_CODE_CREDS", cc_file),
        ):
            os.environ.pop("ANTHROPIC_TOKEN", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            token, is_oauth = resolve_token()
        assert token == "sk-ant-oat01-cc"
        assert is_oauth is True

    def test_api_key_last(self):
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-api03-key"}),
            patch("deepclaw.oauth._read_deepclaw_credentials", return_value=None),
            patch("deepclaw.oauth._read_claude_code_credentials", return_value=None),
        ):
            os.environ.pop("ANTHROPIC_TOKEN", None)
            token, is_oauth = resolve_token()
        assert token == "sk-ant-api03-key"
        assert is_oauth is False

    def test_returns_none_if_nothing(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("deepclaw.oauth._read_deepclaw_credentials", return_value=None),
            patch("deepclaw.oauth._read_claude_code_credentials", return_value=None),
        ):
            os.environ.pop("ANTHROPIC_TOKEN", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            token, is_oauth = resolve_token()
        assert token is None
        assert is_oauth is False
