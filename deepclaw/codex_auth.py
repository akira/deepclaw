"""Codex and GitHub Copilot authentication for DeepClaw.

Supports two providers:

GitHub Copilot
  Token resolution order (matches Copilot CLI behaviour):
    1. COPILOT_GITHUB_TOKEN env var
    2. GH_TOKEN env var
    3. GITHUB_TOKEN env var
    4. gh auth token  (CLI fallback)

  Interactive login: GitHub OAuth device code flow → gho_* token

OpenAI Codex
  Tokens stored at ~/.deepclaw/codex_auth.json (0600).
  After every refresh, tokens are written back to ~/.codex/auth.json so
  that the Codex CLI and VS Code extension stay in sync and don't hit a
  refresh_token_reused error.

  Interactive login: not implemented here — use `codex` CLI to log in,
  then DeepClaw will import the tokens from ~/.codex/auth.json.

Usage in agent.py:
  resolve_copilot_token()     → (github_token, source)
  resolve_codex_token()       → access_token string
  copilot_request_headers()   → dict of Copilot API headers
"""

from __future__ import annotations

import base64
import fcntl
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from deepclaw.config import CONFIG_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GitHub OAuth — same client ID as the Copilot CLI / opencode
COPILOT_OAUTH_CLIENT_ID = "Ov23li8tweQw6odWQebz"
_COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
_CLASSIC_PAT_PREFIX = "ghp_"
_SUPPORTED_PREFIXES = ("gho_", "github_pat_", "ghu_")

# OpenAI Codex OAuth
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120  # refresh 2 min before expiry

# Storage
_CODEX_AUTH_FILE = CONFIG_DIR / "codex_auth.json"
_REQUEST_TIMEOUT = 15

# Device code polling
_POLL_INTERVAL_DEFAULT = 5  # seconds
_POLL_SAFETY_MARGIN = 3  # seconds extra per poll


# ---------------------------------------------------------------------------
# GitHub Copilot — token resolution
# ---------------------------------------------------------------------------


def validate_copilot_token(token: str) -> tuple[bool, str]:
    """Return (valid, message). Classic PATs (ghp_*) are rejected."""
    token = token.strip()
    if not token:
        return False, "Empty token"
    if token.startswith(_CLASSIC_PAT_PREFIX):
        return (
            False,
            "Classic Personal Access Tokens (ghp_*) are not supported by the "
            "Copilot API. Use `deepclaw login copilot` or `gh auth login` to "
            "obtain an OAuth token (gho_*).",
        )
    return True, "OK"


def resolve_copilot_token() -> tuple[str, str]:
    """Resolve a GitHub token suitable for the Copilot API.

    Returns (token, source). Raises ValueError if only a classic PAT is found.
    Returns ("", "") if no token is available.
    """
    for env_var in _COPILOT_ENV_VARS:
        val = os.getenv(env_var, "").strip()
        if val:
            valid, msg = validate_copilot_token(val)
            if not valid:
                logger.warning("Token from %s is not usable: %s", env_var, msg)
                continue
            return val, env_var

    token = _try_gh_cli_token()
    if token:
        valid, msg = validate_copilot_token(token)
        if not valid:
            raise ValueError(f"Token from `gh auth token` is a classic PAT. {msg}")
        return token, "gh auth token"

    return "", ""


def _gh_cli_candidates() -> list[str]:
    candidates: list[str] = []
    resolved = shutil.which("gh")
    if resolved:
        candidates.append(resolved)
    for path in (
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        str(Path.home() / ".local" / "bin" / "gh"),
    ):
        if path not in candidates and os.path.isfile(path) and os.access(path, os.X_OK):
            candidates.append(path)
    return candidates


def _try_gh_cli_token() -> str | None:
    """Call `gh auth token`, stripping GH_TOKEN/GITHUB_TOKEN from env so gh
    reads from its own credential store rather than echoing the env var."""
    hostname = os.getenv("COPILOT_GH_HOST", "").strip()
    clean_env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "GH_TOKEN")}
    for gh_path in _gh_cli_candidates():
        cmd = [gh_path, "auth", "token"]
        if hostname:
            cmd += ["--hostname", hostname]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5, env=clean_env, check=False
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("gh CLI lookup failed (%s): %s", gh_path, exc)
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


# ---------------------------------------------------------------------------
# Copilot API headers
# ---------------------------------------------------------------------------


def copilot_request_headers(
    *, is_agent_turn: bool = True, is_vision: bool = False
) -> dict[str, str]:
    """Standard headers required by the GitHub Copilot API."""
    headers: dict[str, str] = {
        "Editor-Version": "vscode/1.104.1",
        "User-Agent": "DeepClaw/1.0",
        "Copilot-Integration-Id": "vscode-chat",
        "Openai-Intent": "conversation-edits",
        "x-initiator": "agent" if is_agent_turn else "user",
    }
    if is_vision:
        headers["Copilot-Vision-Request"] = "true"
    return headers


# ---------------------------------------------------------------------------
# GitHub Device Code OAuth flow (Copilot login)
# ---------------------------------------------------------------------------


def copilot_device_code_login(
    *,
    host: str = "github.com",
    timeout_seconds: float = 300,
) -> str | None:
    """Run the GitHub OAuth device code flow (RFC 8628) to obtain a gho_* token.

    Prints instructions for the user and polls until authorization completes.
    Returns the access token on success, or None on failure/cancellation.
    """
    domain = host.rstrip("/")
    device_code_url = f"https://{domain}/login/device/code"
    access_token_url = f"https://{domain}/login/oauth/access_token"

    data = urllib.parse.urlencode(
        {"client_id": COPILOT_OAUTH_CLIENT_ID, "scope": "read:user"}
    ).encode()
    req = urllib.request.Request(
        device_code_url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "DeepClaw/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            device_data = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"  ✗ Failed to start device authorization: {exc}")  # noqa: T201
        return None

    verification_uri = device_data.get("verification_uri", f"https://{domain}/login/device")
    user_code = device_data.get("user_code", "")
    device_code = device_data.get("device_code", "")
    interval = max(device_data.get("interval", _POLL_INTERVAL_DEFAULT), 1)

    if not device_code or not user_code:
        print("  ✗ GitHub did not return a device code.")  # noqa: T201
        return None

    print()  # noqa: T201
    print(f"  Open this URL in your browser: {verification_uri}")  # noqa: T201
    print(f"  Enter this code:               {user_code}")  # noqa: T201
    print()  # noqa: T201
    print("  Waiting for authorization...", end="", flush=True)  # noqa: T201

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(interval + _POLL_SAFETY_MARGIN)

        poll_data = urllib.parse.urlencode(
            {
                "client_id": COPILOT_OAUTH_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }
        ).encode()
        poll_req = urllib.request.Request(
            access_token_url,
            data=poll_data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "DeepClaw/1.0",
            },
        )
        try:
            with urllib.request.urlopen(poll_req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception:
            print(".", end="", flush=True)  # noqa: T201
            continue

        if result.get("access_token"):
            print(" ✓")  # noqa: T201
            return result["access_token"]

        error = result.get("error", "")
        if error == "authorization_pending":
            print(".", end="", flush=True)  # noqa: T201
        elif error == "slow_down":
            server_interval = result.get("interval")
            interval = (
                int(server_interval)
                if isinstance(server_interval, (int, float)) and server_interval > 0
                else interval + 5
            )
            print(".", end="", flush=True)  # noqa: T201
        elif error == "expired_token":
            print("\n  ✗ Device code expired. Please try again.")  # noqa: T201
            return None
        elif error == "access_denied":
            print("\n  ✗ Authorization was denied.")  # noqa: T201
            return None
        elif error:
            print(f"\n  ✗ Authorization failed: {error}")  # noqa: T201
            return None

    print("\n  ✗ Timed out waiting for authorization.")  # noqa: T201
    return None


# ---------------------------------------------------------------------------
# OpenAI Codex — JWT helpers
# ---------------------------------------------------------------------------


def _decode_jwt_exp(token: str) -> int | None:
    """Return the `exp` claim (Unix seconds) from a JWT, or None on error."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        padding = 4 - len(parts[1]) % 4
        padded = parts[1] + "=" * (padding % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


def _codex_token_is_expiring(
    access_token: str, skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
) -> bool:
    """Return True if the Codex access token expires within skew_seconds."""
    exp = _decode_jwt_exp(access_token)
    if exp is None:
        return False
    return time.time() >= (exp - skew_seconds)


# ---------------------------------------------------------------------------
# OpenAI Codex — token storage
# ---------------------------------------------------------------------------


def _codex_auth_file() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return _CODEX_AUTH_FILE


def _load_codex_tokens() -> dict[str, Any] | None:
    """Read tokens from ~/.deepclaw/codex_auth.json. Returns None if absent/invalid."""
    path = _codex_auth_file()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("access_token"), str) and isinstance(data.get("refresh_token"), str):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_codex_tokens(access_token: str, refresh_token: str) -> None:
    """Save Codex tokens to ~/.deepclaw/codex_auth.json (0600)."""
    path = _codex_auth_file()
    data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "saved_at": int(time.time()),
    }

    # Write atomically to avoid readers observing empty/partial files.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2) + "\n")
    tmp_path.chmod(0o600)
    os.replace(tmp_path, path)


def _write_codex_cli_tokens(access_token: str, refresh_token: str) -> None:
    """Write refreshed tokens back to ~/.codex/auth.json.

    OpenAI refresh tokens are single-use and rotate on every refresh. If we
    don't write the new pair back, the Codex CLI / VS Code extension will fail
    with refresh_token_reused on its next attempt.
    """
    codex_home = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    auth_path = Path(codex_home).expanduser() / "auth.json"
    try:
        existing: dict[str, Any] = {}
        if auth_path.is_file():
            existing = json.loads(auth_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {}
        tokens = existing.get("tokens")
        if not isinstance(tokens, dict):
            tokens = {}
        tokens["access_token"] = access_token
        tokens["refresh_token"] = refresh_token
        existing["tokens"] = tokens
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        auth_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        auth_path.chmod(0o600)
        logger.debug("Wrote refreshed Codex tokens back to %s", auth_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to write Codex tokens to %s: %s", auth_path, exc)


def _import_codex_cli_tokens() -> dict[str, Any] | None:
    """Try to read tokens from ~/.codex/auth.json (written by the Codex CLI).

    Returns the tokens dict if valid, else None.
    Does NOT write anything to the shared file.
    """
    codex_home = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    auth_path = Path(codex_home).expanduser() / "auth.json"
    if not auth_path.is_file():
        return None
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return None
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if not access_token or not refresh_token:
            return None
        # Do not filter on access_token expiry here; resolve_codex_token()
        # will refresh automatically when needed.
        return dict(tokens)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# OpenAI Codex — token refresh
# ---------------------------------------------------------------------------


def _refresh_codex_tokens(refresh_token: str) -> tuple[str, str]:
    """Refresh Codex OAuth tokens via https://auth.openai.com/oauth/token.

    Returns (new_access_token, new_refresh_token).
    Raises ValueError on failure (including refresh_token_reused).
    """
    data = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CODEX_OAUTH_CLIENT_ID,
        }
    ).encode()
    req = urllib.request.Request(
        CODEX_OAUTH_TOKEN_URL,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "DeepClaw/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode()
            err = json.loads(body)
            code = err.get("error", "")
            desc = err.get("error_description") or err.get("message") or ""
            if code == "refresh_token_reused":
                raise ValueError(
                    "Codex refresh token was already consumed by another client "
                    "(Codex CLI or VS Code extension). Run `codex` in your terminal "
                    "to generate fresh tokens, then run `deepclaw login codex` to "
                    "re-import them."
                ) from exc
            raise ValueError(f"Codex token refresh failed ({code}): {desc or body}") from exc
        except ValueError:
            raise
        except Exception:
            raise ValueError(f"Codex token refresh failed: {exc.code} {body}") from exc
    except Exception as exc:
        raise ValueError(f"Codex token refresh failed: {exc}") from exc

    new_access = payload.get("access_token")
    if not new_access:
        raise ValueError("Codex token refresh response missing access_token")

    new_refresh = payload.get("refresh_token") or refresh_token
    return str(new_access), str(new_refresh)


# ---------------------------------------------------------------------------
# OpenAI Codex — main entry point
# ---------------------------------------------------------------------------


def resolve_codex_token() -> str:
    """Resolve a valid Codex access token, refreshing if needed.

    Resolution order:
      1. ~/.deepclaw/codex_auth.json  (DeepClaw's own store)
      2. ~/.codex/auth.json           (import from Codex CLI if store is empty)

    Auto-refreshes when the JWT is within CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    of expiry. On refresh, writes updated tokens to both stores.

    Raises ValueError if no tokens are available (user needs to log in via
    the Codex CLI or run `deepclaw login codex`).
    """
    tokens = _load_codex_tokens()

    # If nothing in our store, try to import from the Codex CLI
    if tokens is None:
        cli_tokens = _import_codex_cli_tokens()
        if cli_tokens:
            logger.info("Importing Codex tokens from ~/.codex/auth.json")
            _save_codex_tokens(cli_tokens["access_token"], cli_tokens["refresh_token"])
            tokens = cli_tokens
        else:
            raise ValueError(
                "No Codex credentials found. Run `codex` in your terminal to log in, "
                "then restart DeepClaw — it will import your credentials automatically."
            )

    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    if _codex_token_is_expiring(access_token):
        logger.info("Codex access token expiring soon — refreshing")
        try:
            new_access, new_refresh = _refresh_codex_tokens(refresh_token)
            _save_codex_tokens(new_access, new_refresh)
            _write_codex_cli_tokens(new_access, new_refresh)
            access_token = new_access
            logger.info("Codex token refreshed successfully")
        except ValueError as exc:
            logger.warning("Codex token refresh failed: %s", exc)
            # Fall through and try the existing token; if it's truly expired the
            # API call will fail with a clear error

    return access_token
