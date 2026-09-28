"""OpenAI Codex subscription OAuth integration for DeepClaw.

This module deliberately delegates OAuth storage, refresh, and the Codex
Responses API behavior to langchain-openai.  DeepClaw owns only its separate
credential-store location and non-secret status presentation.
"""

from __future__ import annotations

import getpass
import secrets
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from deepclaw.config import DeepClawConfig

OPENAI_CODEX_PROVIDER = "openai_codex"
CHATGPT_AUTH_STORE_PATH = Path("~/.deepclaw/auth/chatgpt-auth.json").expanduser()
_LOGIN_COMMAND = "deepclaw auth login openai_codex --paste"


class OpenAICodexError(RuntimeError):
    """Base class for safe OpenAI Codex integration errors."""


class OpenAICodexDependencyError(OpenAICodexError):
    """The installed langchain-openai package cannot provide Codex OAuth."""


class OpenAICodexAuthError(OpenAICodexError):
    """A safe, actionable OAuth credential-store error."""


def load_openai_codex_symbols():
    """Lazily load the langchain-openai OAuth and Codex implementation.

    These are intentionally imported here so non-Codex DeepClaw installs do
    not need to import the private upstream OAuth implementation at startup.
    """
    try:
        oauth_module = import_module("langchain_openai.chatgpt_oauth")
        codex_module = import_module("langchain_openai.chat_models.codex")
        return (
            oauth_module._FileChatGPTOAuthTokenProvider,
            oauth_module.login_chatgpt,
            codex_module._ChatOpenAICodex,
        )
    except (ImportError, AttributeError) as exc:
        msg = (
            "OpenAI Codex OAuth requires langchain-openai==1.3.2 or a compatible "
            "version exposing chatgpt_oauth and _ChatOpenAICodex. Run `uv sync`."
        )
        raise OpenAICodexDependencyError(msg) from exc


def load_openai_codex_paste_primitives():
    """Load pinned upstream OAuth primitives without starting a callback server."""
    try:
        oauth_module = import_module("langchain_openai.chatgpt_oauth")
        for symbol in (
            "_generate_pkce_pair",
            "_build_authorize_url",
            "_post_form",
            "_token_from_response",
            "_FileChatGPTOAuthTokenProvider",
            "CHATGPT_CLIENT_ID",
            "CHATGPT_TOKEN_URL",
            "DEFAULT_REDIRECT_HOST",
            "DEFAULT_REDIRECT_PORT",
            "DEFAULT_REDIRECT_PATH",
            "DEFAULT_SCOPE",
        ):
            getattr(oauth_module, symbol)
        return oauth_module
    except (ImportError, AttributeError):
        msg = (
            "Pasted OpenAI Codex OAuth requires langchain-openai==1.3.2 "
            "with chatgpt_oauth helpers. Run `uv sync`."
        )
        raise OpenAICodexDependencyError(msg) from None


def _auth_error() -> OpenAICodexAuthError:
    """Return a non-secret error for every runtime credential failure."""
    return OpenAICodexAuthError(
        "OpenAI Codex OAuth credentials are unavailable, corrupt, expired, or could not be "
        f"refreshed. Run `{_LOGIN_COMMAND}` to sign in again."
    )


def _logout_error() -> OpenAICodexAuthError:
    """Return safe recovery guidance when the local credential store cannot be removed."""
    return OpenAICodexAuthError(
        "OpenAI Codex logout failed because DeepClaw could not remove its local OAuth "
        "credentials. Check file permissions and try again."
    )


class _DeepClawTokenProvider:
    """Translate upstream store errors while delegating all token mechanics.

    The upstream provider continues to own its locks, refresh flow, and atomic
    writes. This proxy only prevents its implementation-specific recovery
    guidance from escaping through an eventual model invocation or stream.
    """

    def __init__(self, provider: Any):
        self._provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def get_token(self) -> Any:
        try:
            return self._provider.get_token()
        except Exception:
            raise _auth_error() from None

    async def aget_token(self) -> Any:
        try:
            return await self._provider.aget_token()
        except Exception:
            raise _auth_error() from None

    def get_access_token(self) -> str:
        try:
            return self._provider.get_access_token()
        except Exception:
            raise _auth_error() from None

    async def aget_access_token(self) -> str:
        try:
            return await self._provider.aget_access_token()
        except Exception:
            raise _auth_error() from None


def _new_token_provider():
    provider_class, _, _ = load_openai_codex_symbols()
    return provider_class(path=CHATGPT_AUTH_STORE_PATH)


def _safe_token_details(token: Any) -> dict[str, str | None]:
    """Return only explicitly non-secret fields from an upstream token."""
    expires_at = getattr(token, "expires_at", None)
    return {
        "account_id": getattr(token, "account_id", None),
        "plan_type": getattr(token, "plan_type", None),
        "user_id": getattr(token, "user_id", None),
        "expires_at": str(expires_at) if expires_at is not None else None,
    }


def get_openai_codex_auth_status() -> dict[str, str | bool | None]:
    """Read/refresh DeepClaw's store and return non-secret account status only."""
    if not CHATGPT_AUTH_STORE_PATH.is_file():
        return {"logged_in": False, "state": "missing"}

    try:
        token = _new_token_provider().get_token()
    except FileNotFoundError:
        return {"logged_in": False, "state": "missing"}
    except OpenAICodexDependencyError:
        # Preserve the pinned-version recovery guidance from the private-symbol
        # import instead of treating an incompatible installation as bad OAuth.
        raise
    except Exception:
        # Never surface upstream exception text: it may contain transport data
        # or implementation details. The state still distinguishes recovery.
        raise _auth_error() from None

    return {"logged_in": True, "state": "valid", **_safe_token_details(token)}


def login_openai_codex() -> None:
    """Run langchain-openai's browser OAuth flow using DeepClaw's own store."""
    _, login_chatgpt, _ = load_openai_codex_symbols()
    CHATGPT_AUTH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    login_chatgpt(store_path=CHATGPT_AUTH_STORE_PATH)


def _pasted_authorization_code(pasted: str, *, redirect_uri: str, state: str) -> str:
    """Validate a pasted localhost callback (or query) before any token request."""
    pasted = pasted.strip()
    if not pasted:
        raise OpenAICodexAuthError("No OAuth callback supplied. Paste the full redirect URL.")

    if pasted.startswith("?") or "://" in pasted:
        if pasted.startswith("?"):
            query = pasted[1:]
            if "#" in query:
                raise OpenAICodexAuthError("OAuth callback must not contain a fragment.")
        else:
            try:
                parsed = urlsplit(pasted)
                expected = urlsplit(redirect_uri)
                if (
                    parsed.scheme != expected.scheme
                    or parsed.netloc != expected.netloc
                    or parsed.path != expected.path
                    or parsed.username is not None
                    or parsed.password is not None
                    or "#" in pasted
                ):
                    raise OpenAICodexAuthError(
                        "OAuth redirect origin or path is not the expected localhost callback."
                    )
            except ValueError:
                raise OpenAICodexAuthError("Invalid OAuth redirect URL.") from None
            query = parsed.query
        try:
            pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True)
        except ValueError:
            raise OpenAICodexAuthError("Malformed OAuth callback query.") from None
        values: dict[str, str] = {}
        for key, value in pairs:
            # OpenAI also includes fields such as `scope` in real callbacks.
            # Like upstream's listener, use only the OAuth fields we need;
            # unlike upstream, reject ambiguous duplicates of those fields.
            if key not in {"code", "state", "error", "error_description"}:
                continue
            if key in values:
                raise OpenAICodexAuthError("Duplicate OAuth callback parameter.")
            values[key] = value
        if not values.get("state") or not secrets.compare_digest(values["state"], state):
            raise OpenAICodexAuthError(
                "OAuth callback state mismatch. Restart login and try again."
            )
        if "error" in values or "error_description" in values:
            raise OpenAICodexAuthError(
                "Authorization was denied or failed. Restart login and try again."
            )
        if not values.get("code"):
            raise OpenAICodexAuthError("OAuth callback has no authorization code. Restart login.")
        return values["code"]

    # Only direct, trusted terminal input may bypass state validation; PKCE still binds the code.
    if any(char.isspace() or char in "/?:#&=%" for char in pasted):
        raise OpenAICodexAuthError(
            "Invalid authorization code. Paste the full redirect URL instead."
        )
    return pasted


def login_openai_codex_paste() -> None:
    """Exchange a pasted browser redirect; never bind a server or open a browser."""
    oauth = load_openai_codex_paste_primitives()
    redirect_uri = (
        f"http://{oauth.DEFAULT_REDIRECT_HOST}:{oauth.DEFAULT_REDIRECT_PORT}"
        f"{oauth.DEFAULT_REDIRECT_PATH}"
    )
    try:
        state = secrets.token_urlsafe(32)
        verifier, challenge = oauth._generate_pkce_pair()
        authorize_url = oauth._build_authorize_url(
            client_id=oauth.CHATGPT_CLIENT_ID,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=challenge,
            scope=oauth.DEFAULT_SCOPE,
        )
    except Exception:
        raise OpenAICodexAuthError(
            "Could not start OAuth login. Run `uv sync` and try again."
        ) from None

    print(f"\nOpen this authorization URL in your local browser:\n  {authorize_url}\n")  # noqa: T201
    print(  # noqa: T201
        "The localhost page may fail to load: that is expected on a remote server. "
        "Copy its full address-bar URL and paste it below. A ?code=...&state=... query "
        "also works. Full URL/query is preferred: a raw code cannot be state-checked "
        "(PKCE still applies). Do not share the redirect or code."
    )
    try:
        # An authorization code is a short-lived secret. Refuse getpass's
        # non-TTY echo fallback rather than exposing the pasted URL in a log.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            pasted = getpass.getpass("Paste redirect URL, query, or raw code (hidden): ")
    except getpass.GetPassWarning:
        raise OpenAICodexAuthError(
            "Paste login requires an interactive terminal so the code is not echoed."
        ) from None
    except (EOFError, KeyboardInterrupt):
        raise OpenAICodexAuthError("Login cancelled. Run the login command again.") from None
    code = _pasted_authorization_code(pasted, redirect_uri=redirect_uri, state=state)
    try:
        response = oauth._post_form(
            oauth.CHATGPT_TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": oauth.CHATGPT_CLIENT_ID,
                "code_verifier": verifier,
            },
        )
        token = oauth._token_from_response(response)
    except Exception as exc:
        # Upstream includes the raw provider response in its exception text.
        # Never echo it: OAuth errors can contain user-controlled data.
        if str(exc).startswith(
            f"OAuth request to {oauth.CHATGPT_TOKEN_URL} failed with status 401:"
        ):
            raise OpenAICodexAuthError(
                "OpenAI rejected the OAuth token exchange (401). Start a fresh --paste login; "
                "open that invocation's authorization URL and paste its full callback into "
                "the same waiting terminal promptly. If a fresh attempt still fails, "
                "the upstream OAuth flow or account access may be incompatible."
            ) from None
        raise OpenAICodexAuthError(
            "OAuth token exchange failed. Start a fresh login and check network access."
        ) from None
    try:
        provider = oauth._FileChatGPTOAuthTokenProvider(
            path=CHATGPT_AUTH_STORE_PATH, client_id=oauth.CHATGPT_CLIENT_ID
        )
        provider.save(token)
    except Exception:
        raise OpenAICodexAuthError(
            "OAuth credential save failed. Check ~/.deepclaw/auth/ permissions."
        ) from None


def logout_openai_codex() -> bool:
    """Remove only DeepClaw's OAuth store, never the Codex CLI store."""
    try:
        CHATGPT_AUTH_STORE_PATH.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        raise _logout_error() from None
    return True


def resolve_openai_codex_model(config: DeepClawConfig):
    """Resolve ``openai_codex:<model>`` to the upstream Codex chat model."""
    model_spec = (config.model or "").strip()
    provider, separator, model_name = model_spec.partition(":")
    if separator == "" or provider != OPENAI_CODEX_PROVIDER:
        return model_spec
    if not model_name.strip():
        msg = "OpenAI Codex model name cannot be empty"
        raise ValueError(msg)

    generation = config.generation
    if generation.repetition_penalty is not None:
        msg = (
            "generation.repetition_penalty is unsupported for openai_codex:* models: "
            "the Codex Responses API does not support repetition_penalty. Remove "
            "generation.repetition_penalty or use a provider that supports it."
        )
        raise ValueError(msg)
    if generation.max_tokens is not None:
        msg = (
            "generation.max_tokens is unsupported for openai_codex:* models: "
            "the Codex Responses API rejects max_output_tokens. Remove "
            "generation.max_tokens or use a provider that supports it."
        )
        raise ValueError(msg)
    if generation.temperature is not None:
        msg = (
            "generation.temperature is unsupported for openai_codex:* models: "
            "the pinned Codex client silently drops temperature from Responses API requests. "
            "Remove generation.temperature or use a provider that supports it."
        )
        raise ValueError(msg)

    provider_class, _, chat_model_class = load_openai_codex_symbols()
    token_provider = _DeepClawTokenProvider(provider_class(path=CHATGPT_AUTH_STORE_PATH))
    kwargs: dict[str, Any] = {
        "model": model_name,
        "token_provider": token_provider,
        "originator": "deepclaw",
    }
    if generation.temperature is not None:
        kwargs["temperature"] = generation.temperature
    if generation.top_p is not None:
        kwargs["top_p"] = generation.top_p

    # _ChatOpenAICodex itself forces Responses API, store=False, and streaming.
    return chat_model_class(**kwargs)
