"""OAuth PKCE authentication for Claude Pro/Max subscriptions.

Implements the same OAuth flow as Claude Code, allowing DeepClaw to use
a Claude subscription instead of (or in addition to) an API key.

Credential priority:
  1. ANTHROPIC_TOKEN env var (OAuth token)
  2. DeepClaw OAuth credentials (~/.deepclaw/.anthropic_oauth.json)
  3. Claude Code credentials (~/.claude/.credentials.json)
  4. ANTHROPIC_API_KEY env var (regular API key)
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import time
import urllib.request
import webbrowser
from pathlib import Path

from deepclaw.config import CONFIG_DIR

logger = logging.getLogger(__name__)

_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_OAUTH_SCOPES = "org:create_api_key user:profile user:inference"
_CLAUDE_CODE_VERSION = "1.0.33"

_OAUTH_FILE = CONFIG_DIR / ".anthropic_oauth.json"
_CLAUDE_CODE_CREDS = Path.home() / ".claude" / ".credentials.json"

_TOKEN_REFRESH_BUFFER_MS = 120_000  # Refresh 2 minutes before expiry
_REQUEST_TIMEOUT = 15

OAUTH_HEADERS = {
    "anthropic-beta": ",".join([
        "interleaved-thinking-2025-05-14",
        "claude-code-20250219",
        "oauth-2025-04-20",
    ]),
    "user-agent": f"claude-cli/{_CLAUDE_CODE_VERSION} (external, cli)",
    "x-app": "cli",
}


def is_oauth_token(key: str) -> bool:
    """Check if the key is an OAuth token (not a regular API key)."""
    if not key:
        return False
    if key.startswith("sk-ant-api"):
        return False
    return True


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


# ---------------------------------------------------------------------------
# Credential storage
# ---------------------------------------------------------------------------


def _save_credentials(access_token: str, refresh_token: str, expires_at_ms: int) -> None:
    """Save OAuth credentials to ~/.deepclaw/.anthropic_oauth.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at_ms,
    }
    _OAUTH_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _OAUTH_FILE.chmod(0o600)
    logger.info("Saved OAuth credentials to %s", _OAUTH_FILE)


def _read_deepclaw_credentials() -> dict | None:
    """Read DeepClaw OAuth credentials."""
    if not _OAUTH_FILE.is_file():
        return None
    try:
        data = json.loads(_OAUTH_FILE.read_text(encoding="utf-8"))
        if data.get("accessToken"):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _read_claude_code_credentials() -> dict | None:
    """Read Claude Code OAuth credentials from ~/.claude/.credentials.json."""
    if not _CLAUDE_CODE_CREDS.is_file():
        return None
    try:
        data = json.loads(_CLAUDE_CODE_CREDS.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth", {})
        if oauth.get("accessToken"):
            return oauth
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _is_token_expired(creds: dict) -> bool:
    """Check if credentials are expired (with refresh buffer)."""
    expires_at = creds.get("expiresAt", 0)
    if not expires_at:
        return False  # No expiry = assume valid
    now_ms = int(time.time() * 1000)
    return now_ms >= (expires_at - _TOKEN_REFRESH_BUFFER_MS)


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------


def _refresh_token(creds: dict, save_fn) -> str | None:
    """Refresh an expired OAuth token. Returns new access token or None."""
    refresh = creds.get("refreshToken")
    if not refresh:
        return None

    data = json.dumps({
        "grant_type": "refresh_token",
        "refresh_token": refresh,
        "client_id": _OAUTH_CLIENT_ID,
    }).encode()

    req = urllib.request.Request(
        _OAUTH_TOKEN_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"claude-cli/{_CLAUDE_CODE_VERSION} (external, cli)",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            result = json.loads(resp.read().decode())
    except Exception:
        logger.warning("OAuth token refresh failed", exc_info=True)
        return None

    new_access = result.get("access_token", "")
    new_refresh = result.get("refresh_token", refresh)
    expires_in = result.get("expires_in", 3600)
    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)

    if new_access:
        save_fn(new_access, new_refresh, expires_at_ms)
        return new_access
    return None


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def resolve_token() -> tuple[str | None, bool]:
    """Resolve the best available Anthropic token.

    Returns (token, is_oauth). Falls through the priority chain:
      1. ANTHROPIC_TOKEN env var
      2. DeepClaw OAuth credentials
      3. Claude Code credentials
      4. ANTHROPIC_API_KEY env var
    """
    # 1. ANTHROPIC_TOKEN env var
    env_token = os.environ.get("ANTHROPIC_TOKEN", "").strip()
    if env_token:
        return env_token, is_oauth_token(env_token)

    # 2. DeepClaw OAuth credentials
    dc_creds = _read_deepclaw_credentials()
    if dc_creds:
        if _is_token_expired(dc_creds):
            refreshed = _refresh_token(dc_creds, _save_credentials)
            if refreshed:
                return refreshed, True
            logger.warning("DeepClaw OAuth token expired and refresh failed")
        else:
            return dc_creds["accessToken"], True

    # 3. Claude Code credentials
    cc_creds = _read_claude_code_credentials()
    if cc_creds:
        if _is_token_expired(cc_creds):
            # Refresh and save to DeepClaw's file (don't modify Claude Code's)
            refreshed = _refresh_token(cc_creds, _save_credentials)
            if refreshed:
                return refreshed, True
            logger.warning("Claude Code OAuth token expired and refresh failed")
        else:
            return cc_creds["accessToken"], True

    # 4. ANTHROPIC_API_KEY env var
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key, is_oauth_token(api_key)

    return None, False


# ---------------------------------------------------------------------------
# Interactive OAuth login
# ---------------------------------------------------------------------------


def login() -> str | None:
    """Run the OAuth PKCE login flow interactively.

    Opens a browser for authorization, prompts for the code, exchanges it
    for tokens, and saves them. Returns the access token or None on failure.
    """
    verifier, challenge = _generate_pkce()

    params = (
        f"code=true"
        f"&client_id={_OAUTH_CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={urllib.request.quote(_OAUTH_REDIRECT_URI, safe='')}"
        f"&scope={urllib.request.quote(_OAUTH_SCOPES, safe='')}"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=S256"
        f"&state={verifier}"
    )
    auth_url = f"{_OAUTH_AUTHORIZE_URL}?{params}"

    print("Opening browser for Claude authorization...")  # noqa: T201
    print(f"If the browser doesn't open, visit:\n{auth_url}\n")  # noqa: T201
    webbrowser.open(auth_url)

    print("After authorizing, you'll see a page with a code.")  # noqa: T201
    auth_input = input("Paste the authorization code here: ").strip()
    if not auth_input:
        print("No code provided, aborting.")  # noqa: T201
        return None

    # Parse code#state format
    splits = auth_input.split("#")
    code = splits[0]
    state = splits[1] if len(splits) > 1 else ""

    # Exchange code for tokens
    exchange_data = json.dumps({
        "grant_type": "authorization_code",
        "client_id": _OAUTH_CLIENT_ID,
        "code": code,
        "state": state,
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "code_verifier": verifier,
    }).encode()

    req = urllib.request.Request(
        _OAUTH_TOKEN_URL,
        data=exchange_data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"claude-cli/{_CLAUDE_CODE_VERSION} (external, cli)",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"Authorization failed: {e.code} {body}")  # noqa: T201
        return None
    except Exception as e:
        print(f"Authorization failed: {e}")  # noqa: T201
        return None

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 3600)

    if not access_token:
        print("No access token in response.")  # noqa: T201
        return None

    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    _save_credentials(access_token, refresh_token, expires_at_ms)

    print("Successfully authenticated with Claude.")  # noqa: T201
    return access_token
