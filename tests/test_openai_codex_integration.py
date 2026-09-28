"""Focused tests for the OpenAI Codex OAuth integration."""

import asyncio
import traceback
from unittest.mock import MagicMock

import pytest

from deepclaw import agent as agent_mod
from deepclaw import cli
from deepclaw.config import DeepClawConfig
from deepclaw.doctor import STATUS_FAIL, STATUS_OK, check_llm_api_key
from deepclaw.integrations import openai_codex as codex
from deepclaw.integrations import resolve_provider_model


class FakeToken:
    account_id = "account-123"
    plan_type = "plus"
    user_id = "user-456"
    expires_at = "2030-01-01T00:00:00Z"
    access_token = "secret-access-token"
    refresh_token = "secret-refresh-token"
    id_token = "secret-id-token"


class FakeProvider:
    token = FakeToken()

    def __init__(self, *, path):
        self.path = path

    def get_token(self):
        return self.token

    async def aget_token(self):
        return self.token

    def get_access_token(self):
        return self.token.access_token

    async def aget_access_token(self):
        return self.token.access_token


class FakeChatCodex:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _symbols(login=None):
    if login is None:
        login = MagicMock()
    return FakeProvider, login, FakeChatCodex


def test_non_codex_model_is_left_as_a_string():
    assert (
        codex.resolve_openai_codex_model(DeepClawConfig(model="openai:gpt-4o")) == "openai:gpt-4o"
    )


def test_codex_model_selection_and_generation_handoff(monkeypatch, tmp_path):
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", tmp_path / "chatgpt-auth.json")
    monkeypatch.setattr(codex, "load_openai_codex_symbols", _symbols)
    config = DeepClawConfig(model="openai_codex:gpt-5.3-codex")
    config.generation.top_p = 0.8

    model = codex.resolve_openai_codex_model(config)

    assert isinstance(model, FakeChatCodex)
    assert model.kwargs == {
        "model": "gpt-5.3-codex",
        "token_provider": model.kwargs["token_provider"],
        "originator": "deepclaw",
        "top_p": 0.8,
    }
    assert model.kwargs["token_provider"].path == tmp_path / "chatgpt-auth.json"
    assert not {"max_completion_tokens", "repetition_penalty"} & model.kwargs.keys()
    assert not {"streaming", "store", "api_key", "base_url"} & model.kwargs.keys()


def test_codex_model_requires_name():
    with pytest.raises(ValueError, match="model name cannot be empty"):
        codex.resolve_openai_codex_model(DeepClawConfig(model="openai_codex:"))


def test_actual_codex_model_rejects_unsupported_repetition_penalty(monkeypatch):
    from langchain_openai.chat_models.codex import _ChatOpenAICodex

    config = DeepClawConfig(model="openai_codex:gpt-5.3-codex")
    config.generation.repetition_penalty = 1.1
    monkeypatch.setattr(
        codex,
        "load_openai_codex_symbols",
        lambda: (_ for _ in ()).throw(AssertionError("model construction should not run")),
    )

    with pytest.raises(ValueError, match="generation.repetition_penalty") as exc_info:
        codex.resolve_openai_codex_model(config)

    assert "openai_codex:*" in str(exc_info.value)
    assert _ChatOpenAICodex is not None


def test_actual_codex_model_drops_temperature_and_resolver_rejects_it(monkeypatch, tmp_path):
    from langchain_openai.chat_models.codex import _ChatOpenAICodex

    upstream_model = _ChatOpenAICodex(
        model="gpt-5.3-codex",
        token_provider=FakeProvider(path=tmp_path / "chatgpt-auth.json"),
        temperature=0.2,
    )
    assert upstream_model.temperature is None
    assert "temperature" not in upstream_model._default_params

    config = DeepClawConfig(model="openai_codex:gpt-5.3-codex")
    config.generation.temperature = 0.2
    monkeypatch.setattr(
        codex,
        "load_openai_codex_symbols",
        lambda: (_ for _ in ()).throw(AssertionError("model construction should not run")),
    )

    with pytest.raises(ValueError, match="generation.temperature") as exc_info:
        codex.resolve_openai_codex_model(config)

    message = str(exc_info.value)
    assert "openai_codex:*" in message
    assert "silently drops temperature" in message


def test_codex_model_rejects_unsupported_max_tokens_before_construction(monkeypatch):
    config = DeepClawConfig(model="openai_codex:gpt-5.3-codex")
    config.generation.max_tokens = 1234
    monkeypatch.setattr(
        codex,
        "load_openai_codex_symbols",
        lambda: (_ for _ in ()).throw(AssertionError("model construction should not run")),
    )

    with pytest.raises(ValueError, match="generation.max_tokens") as exc_info:
        codex.resolve_openai_codex_model(config)

    message = str(exc_info.value)
    assert "openai_codex:*" in message
    assert "max_output_tokens" in message


def test_missing_dependency_is_actionable(monkeypatch):
    monkeypatch.setattr(
        codex, "import_module", lambda _name: (_ for _ in ()).throw(ImportError("missing"))
    )
    with pytest.raises(
        codex.OpenAICodexDependencyError, match="langchain-openai==1.3.2"
    ) as exc_info:
        codex.load_openai_codex_symbols()
    assert isinstance(exc_info.value.__cause__, ImportError)


@pytest.mark.parametrize(
    "upstream_error",
    [
        FileNotFoundError("Run `login_chatgpt()` first: secret-token"),
        RuntimeError("invalid JSON; run `login_chatgpt()`: secret-token"),
        RuntimeError("refresh rejected; run `login_chatgpt()`: secret-token"),
    ],
)
def test_actual_codex_model_translates_runtime_store_errors(monkeypatch, tmp_path, upstream_error):
    from langchain_openai.chat_models.codex import _ChatOpenAICodex

    class BrokenProvider(FakeProvider):
        def get_token(self):
            raise upstream_error

        async def aget_token(self):
            raise upstream_error

        def get_access_token(self):
            raise upstream_error

        async def aget_access_token(self):
            raise upstream_error

    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", tmp_path / "chatgpt-auth.json")
    monkeypatch.setattr(
        codex, "load_openai_codex_symbols", lambda: (BrokenProvider, None, _ChatOpenAICodex)
    )
    model = codex.resolve_openai_codex_model(DeepClawConfig(model="openai_codex:gpt-5.3-codex"))

    with pytest.raises(codex.OpenAICodexAuthError) as exc_info:
        model._codex_headers_sync()
    public_error = str(exc_info.value)
    formatted_traceback = "".join(traceback.format_exception(exc_info.value))
    assert "deepclaw auth login openai_codex" in public_error
    assert "login_chatgpt" not in public_error
    assert "secret-token" not in public_error
    assert "secret-token" not in formatted_traceback
    assert exc_info.value.__cause__ is None


def test_model_token_provider_forwards_all_token_methods(monkeypatch, tmp_path):
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", tmp_path / "chatgpt-auth.json")
    monkeypatch.setattr(codex, "load_openai_codex_symbols", _symbols)
    provider = codex.resolve_openai_codex_model(
        DeepClawConfig(model="openai_codex:gpt-5.3-codex")
    ).kwargs["token_provider"]

    assert provider.get_token() is FakeProvider.token
    assert asyncio.run(provider.aget_token()) is FakeProvider.token
    assert provider.get_access_token() == "secret-access-token"
    assert asyncio.run(provider.aget_access_token()) == "secret-access-token"


@pytest.mark.parametrize(
    ("method", "is_async"),
    [
        ("get_token", False),
        ("aget_token", True),
        ("get_access_token", False),
        ("aget_access_token", True),
    ],
)
def test_model_token_provider_suppresses_upstream_exception_chains(method, is_async):
    secret_sentinel = "model-token-secret-sentinel"

    class BrokenProvider:
        def get_token(self):
            raise RuntimeError(secret_sentinel)

        async def aget_token(self):
            raise RuntimeError(secret_sentinel)

        def get_access_token(self):
            raise RuntimeError(secret_sentinel)

        async def aget_access_token(self):
            raise RuntimeError(secret_sentinel)

    provider = codex._DeepClawTokenProvider(BrokenProvider())
    with pytest.raises(codex.OpenAICodexAuthError) as exc_info:
        result = getattr(provider, method)()
        if is_async:
            asyncio.run(result)

    public_error = str(exc_info.value)
    formatted_traceback = "".join(traceback.format_exception(exc_info.value))
    assert secret_sentinel not in public_error
    assert secret_sentinel not in formatted_traceback
    assert exc_info.value.__cause__ is None


def test_status_missing_store(monkeypatch, tmp_path):
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", tmp_path / "missing.json")
    assert codex.get_openai_codex_auth_status() == {"logged_in": False, "state": "missing"}


def test_status_maps_malformed_or_refresh_failure_without_secrets(monkeypatch, tmp_path):
    path = tmp_path / "chatgpt-auth.json"
    path.write_text("not json")
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", path)

    class BrokenProvider(FakeProvider):
        def get_token(self):
            raise RuntimeError("refresh rejected: secret-refresh-token")

    monkeypatch.setattr(codex, "load_openai_codex_symbols", lambda: (BrokenProvider, None, None))
    with pytest.raises(codex.OpenAICodexAuthError) as exc_info:
        codex.get_openai_codex_auth_status()
    public_error = str(exc_info.value)
    formatted_traceback = "".join(traceback.format_exception(exc_info.value))
    assert "deepclaw auth login openai_codex" in public_error
    assert "secret-refresh-token" not in public_error
    assert "secret-refresh-token" not in formatted_traceback
    assert exc_info.value.__cause__ is None


def test_status_preserves_dependency_error_from_private_symbol_import(monkeypatch, tmp_path):
    path = tmp_path / "chatgpt-auth.json"
    path.write_text("{}")
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", path)
    monkeypatch.setattr(
        codex, "import_module", lambda _name: (_ for _ in ()).throw(ImportError("missing symbol"))
    )

    with pytest.raises(codex.OpenAICodexDependencyError, match="Run `uv sync`") as exc_info:
        codex.get_openai_codex_auth_status()

    assert isinstance(exc_info.value.__cause__, ImportError)


def test_status_returns_only_nonsecret_fields(monkeypatch, tmp_path):
    path = tmp_path / "chatgpt-auth.json"
    path.write_text("{}")
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", path)
    monkeypatch.setattr(codex, "load_openai_codex_symbols", _symbols)
    status = codex.get_openai_codex_auth_status()
    assert status["logged_in"] is True
    assert status["account_id"] == "account-123"
    assert not {"access_token", "refresh_token", "id_token"} & status.keys()


def test_login_uses_browser_helper_and_logout_only_removes_deepclaw_store(monkeypatch, tmp_path):
    path = tmp_path / "auth" / "chatgpt-auth.json"
    other_store = tmp_path / ".codex" / "auth.json"
    other_store.parent.mkdir()
    other_store.write_text("keep")
    path.parent.mkdir()
    path.write_text("remove")
    login = MagicMock()
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", path)
    monkeypatch.setattr(codex, "load_openai_codex_symbols", lambda: _symbols(login))

    codex.login_openai_codex()
    login.assert_called_once_with(store_path=path)
    assert codex.logout_openai_codex() is True
    assert not path.exists()
    assert other_store.read_text() == "keep"
    assert codex.logout_openai_codex() is False


@pytest.fixture
def paste_flow(monkeypatch, tmp_path, capsys):
    from urllib.parse import parse_qs, urlsplit

    path = tmp_path / "auth" / "chatgpt-auth.json"
    other = tmp_path / ".codex" / "auth.json"
    other.parent.mkdir()
    other.write_text("keep")
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", path)
    oauth = codex.import_module("langchain_openai.chatgpt_oauth")
    pkce = MagicMock(return_value=("secret-verifier", "challenge"))
    build = MagicMock(side_effect=oauth._build_authorize_url)
    post = MagicMock(
        return_value={
            "access_token": "secret-access",
            "refresh_token": "secret-refresh",
            "expires_in": 3600,
        }
    )
    token = MagicMock(return_value=FakeToken())
    saved = []

    class Store:
        def __init__(self, *, path, client_id):
            assert path == tmp_path / "auth" / "chatgpt-auth.json"
            assert client_id == oauth.CHATGPT_CLIENT_ID

        def save(self, value):
            saved.append(value)

    monkeypatch.setattr(oauth, "_generate_pkce_pair", pkce)
    monkeypatch.setattr(oauth, "_build_authorize_url", build)
    monkeypatch.setattr(oauth, "_post_form", post)
    monkeypatch.setattr(oauth, "_token_from_response", token)
    monkeypatch.setattr(oauth, "_FileChatGPTOAuthTokenProvider", Store)
    monkeypatch.setattr(
        codex.secrets, "token_urlsafe", lambda size: "secret-state" if size == 32 else None
    )

    def run(pasted):
        monkeypatch.setattr(codex.getpass, "getpass", lambda prompt: pasted)
        codex.login_openai_codex_paste()
        output = capsys.readouterr().out
        assert "secret-verifier" not in output
        assert pasted not in output
        assert "secret-access" not in output
        assert "secret-refresh" not in output
        assert other.read_text() == "keep"
        url = output.split("  https://auth.openai.com/oauth/authorize?", 1)[1].splitlines()[0]
        params = parse_qs(urlsplit("https://auth.openai.com/oauth/authorize?" + url).query)
        return params

    return run, post, saved, path, build, token


@pytest.mark.parametrize("kind", ["url", "query", "raw"])
def test_paste_login_exchanges_and_saves_only_own_store(paste_flow, kind):
    run, post, saved, path, build, token = paste_flow
    query = "code=secret-code&state=secret-state"
    pasted = (
        "http://localhost:1455/auth/callback?" + query
        if kind == "url"
        else "?" + query
        if kind == "query"
        else "secret-code"
    )
    params = run(pasted)
    assert params["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert params["state"] == ["secret-state"]
    assert params["code_challenge"] == ["challenge"]
    assert (
        build.call_args.kwargs["scope"]
        == codex.import_module("langchain_openai.chatgpt_oauth").DEFAULT_SCOPE
    )
    post.assert_called_once_with(
        codex.import_module("langchain_openai.chatgpt_oauth").CHATGPT_TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "code": "secret-code",
            "redirect_uri": "http://localhost:1455/auth/callback",
            "client_id": codex.import_module("langchain_openai.chatgpt_oauth").CHATGPT_CLIENT_ID,
            "code_verifier": "secret-verifier",
        },
    )
    token.assert_called_once_with(post.return_value)
    assert saved == [token.return_value]
    assert not path.exists()  # fake provider captures save; it does not write


@pytest.mark.parametrize("kind", ["url", "query"])
def test_paste_login_accepts_openai_scope_parameter(paste_flow, kind):
    run, post, saved, *_ = paste_flow
    query = "code=secret-code&scope=openid+profile+email+offline_access&state=secret-state"
    pasted = "http://localhost:1455/auth/callback?" + query if kind == "url" else "?" + query
    run(pasted)
    assert post.call_args.args[1]["code"] == "secret-code"
    assert len(saved) == 1


@pytest.mark.parametrize(
    "pasted",
    [
        "",
        "?code=secret-code&state=bad",
        "?code=secret-code",
        "?error=denied&state=secret-state",
        "?code=secret-code&state=secret-state&code=other",
        "?code=secret-code&state=secret-state&state=other",
        "?code=&state=secret-state",
        "?code=secret-code&state=secret-state&error=denied",
        "?code=secret-code&state=secret-state#fragment",
        "https://localhost:1455/auth/callback?code=secret-code&state=secret-state",
        "http://127.0.0.1:1455/auth/callback?code=secret-code&state=secret-state",
        "http://user@localhost:1455/auth/callback?code=secret-code&state=secret-state",
        "http://localhost:1456/auth/callback?code=secret-code&state=secret-state",
        "http://localhost:1455/wrong?code=secret-code&state=secret-state",
        "http://localhost:1455/auth/callback?code=secret-code&state=secret-state#fragment",
    ],
)
def test_paste_validation_fails_closed_without_exchange(paste_flow, pasted):
    run, post, saved, *_ = paste_flow
    with pytest.raises(codex.OpenAICodexAuthError) as exc:
        run(pasted)
    assert "secret-code" not in "".join(traceback.format_exception(exc.value))
    assert exc.value.__cause__ is None
    post.assert_not_called()
    assert not saved


def test_paste_upstream_failure_does_not_leak(paste_flow, capsys):
    run, post, saved, *_ = paste_flow
    post.side_effect = RuntimeError("secret-code secret-verifier secret-access")
    pasted = "?code=secret-code&state=secret-state"
    with pytest.raises(codex.OpenAICodexAuthError) as exc:
        run(pasted)
    assert "secret-code" not in "".join(traceback.format_exception(exc.value))
    assert exc.value.__cause__ is None
    assert not saved
    assert "secret-code" not in capsys.readouterr().out


def test_paste_401_explains_fresh_same_session_and_hides_provider_body(paste_flow):
    run, post, saved, *_ = paste_flow
    post.side_effect = RuntimeError(
        "OAuth request to https://auth.openai.com/oauth/token failed with status 401: "
        "{'error': {'code': 'token_expired', 'message': 'secret-code secret-access'}}"
    )
    pasted = "?code=secret-code&state=secret-state"
    with pytest.raises(codex.OpenAICodexAuthError) as exc:
        run(pasted)
    assert "401" in str(exc.value)
    assert "same waiting terminal" in str(exc.value)
    assert "secret-code" not in "".join(traceback.format_exception(exc.value))
    assert "secret-access" not in "".join(traceback.format_exception(exc.value))
    assert exc.value.__cause__ is None
    assert not saved


def test_paste_save_error_does_not_claim_exchange_failure(paste_flow, monkeypatch):
    run, post, saved, _, _, _ = paste_flow
    oauth = codex.import_module("langchain_openai.chatgpt_oauth")
    monkeypatch.setattr(
        oauth,
        "_FileChatGPTOAuthTokenProvider",
        MagicMock(side_effect=OSError("secret-access")),
    )
    with pytest.raises(codex.OpenAICodexAuthError, match="credential save failed") as exc:
        run("?code=secret-code&state=secret-state")
    post.assert_called_once()
    assert "secret-access" not in "".join(traceback.format_exception(exc.value))
    assert exc.value.__cause__ is None
    assert not saved


def test_paste_rejects_getpass_non_tty_echo_fallback(paste_flow, monkeypatch):
    _, post, saved, *_ = paste_flow

    def fallback(_prompt):
        import warnings

        warnings.warn("No terminal input", codex.getpass.GetPassWarning, stacklevel=2)
        return "secret-code"

    monkeypatch.setattr(codex.getpass, "getpass", fallback)
    with pytest.raises(codex.OpenAICodexAuthError, match="interactive terminal"):
        codex.login_openai_codex_paste()
    post.assert_not_called()
    assert not saved


def test_paste_real_upstream_storage_without_network(monkeypatch, tmp_path, capsys):
    oauth = codex.import_module("langchain_openai.chatgpt_oauth")
    path = tmp_path / "auth" / "chatgpt-auth.json"
    other = tmp_path / ".codex" / "auth.json"
    other.parent.mkdir()
    other.write_text("keep")
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", path)
    monkeypatch.setattr(codex.secrets, "token_urlsafe", lambda _size: "known-state")
    post = MagicMock(
        return_value={
            "access_token": "real-test-access",
            "refresh_token": "real-test-refresh",
            "expires_in": 3600,
        }
    )
    monkeypatch.setattr(oauth, "_post_form", post)
    monkeypatch.setattr(
        codex.getpass,
        "getpass",
        lambda _prompt: "http://localhost:1455/auth/callback?code=test-code&state=known-state",
    )

    codex.login_openai_codex_paste()

    assert post.call_count == 1
    assert path.is_file()
    assert (
        oauth._FileChatGPTOAuthTokenProvider(path=path).get_token().access_token
        == "real-test-access"
    )
    assert other.read_text() == "keep"
    output = capsys.readouterr().out
    assert "test-code" not in output
    assert "real-test-access" not in output


def test_cli_paste_failure_is_actionable_and_secret_free(monkeypatch, capsys):
    def fail():
        raise codex.OpenAICodexAuthError(
            "OAuth callback state mismatch. Restart login and try again."
        )

    monkeypatch.setattr(codex, "login_openai_codex_paste", fail)
    with pytest.raises(SystemExit, match="1"):
        cli._handle_auth_command(["login", "openai_codex", "--paste"])
    assert "state mismatch" in capsys.readouterr().out


def test_logout_maps_store_unlink_errors_without_leaking_details(monkeypatch, tmp_path):
    path = tmp_path / "auth" / "chatgpt-auth.json"
    path.parent.mkdir()
    path.write_text("keep")
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", path)

    def fail_unlink(self, *args, **kwargs):
        assert self == path
        raise PermissionError("secret-token and sensitive OS details")

    monkeypatch.setattr(type(path), "unlink", fail_unlink)

    with pytest.raises(codex.OpenAICodexAuthError) as exc_info:
        codex.logout_openai_codex()

    message = str(exc_info.value)
    assert "Check file permissions" in message
    assert "secret-token" not in message
    assert "sensitive OS details" not in message
    assert path.exists()


def test_cli_logout_failure_is_safe_and_exits_one(monkeypatch, capsys):
    error = codex.OpenAICodexAuthError(
        "OpenAI Codex logout failed because DeepClaw could not remove its local OAuth credentials. "
        "Check file permissions and try again."
    )
    monkeypatch.setattr(codex, "logout_openai_codex", lambda: (_ for _ in ()).throw(error))

    with pytest.raises(SystemExit, match="1"):
        cli._handle_auth_command(["logout", "openai_codex"])

    output = capsys.readouterr().out
    assert "Check file permissions" in output
    assert "secret-token" not in output


def test_cli_logout_success_requires_service_restart(monkeypatch, capsys):
    monkeypatch.setattr(codex, "logout_openai_codex", lambda: True)

    cli._handle_auth_command(["logout", "openai_codex"])

    output = capsys.readouterr().out
    assert "local logout complete" in output
    assert "Restart the DeepClaw service" in output
    assert "is logged out" not in output


def test_dispatcher_routes_codex_before_fallback(monkeypatch):
    resolved_model = object()
    monkeypatch.setattr(
        "deepclaw.integrations.resolve_openai_codex_model", lambda _config: resolved_model
    )
    assert (
        resolve_provider_model(DeepClawConfig(model="openai_codex:gpt-5.3-codex")) is resolved_model
    )


def test_create_agent_receives_codex_model_instance_from_dispatcher(monkeypatch, tmp_path):
    captured = {}

    class FakeShellBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute(self, command, *, timeout=None):
            return None

    class FakeFilesystemBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCompositeBackend:
        def __init__(self, *, default, routes):
            self.default = default
            self.routes = routes

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return "agent"

    monkeypatch.setattr(codex, "load_openai_codex_symbols", _symbols)
    monkeypatch.setattr(codex, "CHATGPT_AUTH_STORE_PATH", tmp_path / "chatgpt-auth.json")
    monkeypatch.setattr(agent_mod, "_load_soul", lambda: "soul")
    monkeypatch.setattr(agent_mod, "_setup_memory", lambda: ["/memory/AGENTS.md"])
    monkeypatch.setattr(agent_mod, "_setup_skills", lambda: ["/skills"])
    monkeypatch.setattr(agent_mod, "discover_tools", list)
    monkeypatch.setattr(agent_mod, "DeepClawLocalShellBackend", FakeShellBackend)
    monkeypatch.setattr(agent_mod, "FilesystemBackend", FakeFilesystemBackend)
    monkeypatch.setattr(agent_mod, "CompositeBackend", FakeCompositeBackend)
    monkeypatch.setattr(agent_mod, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(agent_mod, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        agent_mod,
        "create_summarization_tool_middleware",
        lambda model, backend: ("compact-tool", model, backend),
        raising=False,
    )

    result = agent_mod.create_agent(
        DeepClawConfig(model="openai_codex:gpt-5.3-codex", workspace_root=str(tmp_path)),
        checkpointer="checkpointer",
    )

    assert result == "agent"
    assert isinstance(captured["model"], FakeChatCodex)


def test_cli_status_never_prints_tokens(monkeypatch, capsys):
    monkeypatch.setattr(
        codex,
        "get_openai_codex_auth_status",
        lambda: {
            "logged_in": True,
            "account_id": "account-123",
            "plan_type": "plus",
            "user_id": "user-456",
            "expires_at": "2030-01-01",
            "access_token": "must-not-print",
        },
    )
    cli._handle_auth_command(["status", "openai_codex"])
    output = capsys.readouterr().out
    assert "must-not-print" not in output
    assert "account-123" in output


def test_cli_login_failure_and_unknown_auth_command(monkeypatch, capsys):
    monkeypatch.setattr(
        codex, "login_openai_codex", lambda: (_ for _ in ()).throw(RuntimeError("nope"))
    )
    with pytest.raises(SystemExit, match="1"):
        cli._handle_auth_command(["login", "openai_codex"])
    assert "login failed" in capsys.readouterr().out.lower()
    with pytest.raises(SystemExit, match="1"):
        cli._handle_auth_command(["wat", "openai_codex"])


def test_cli_selects_browser_or_paste_login(monkeypatch, capsys):
    browser_login = MagicMock()
    paste_login = MagicMock()
    monkeypatch.setattr(codex, "login_openai_codex", browser_login)
    monkeypatch.setattr(codex, "login_openai_codex_paste", paste_login)

    cli._handle_auth_command(["login", "openai_codex"])
    browser_login.assert_called_once_with()
    paste_login.assert_not_called()

    cli._handle_auth_command(["login", "openai_codex", "--paste"])
    paste_login.assert_called_once_with()
    assert "login completed" in capsys.readouterr().out.lower()


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["login"],
        ["login", "openai_codex", "--browser"],
        ["login", "openai_codex", "--device"],
        ["login", "openai_codex", "--paste", "extra"],
        ["status", "openai_codex", "--paste"],
        ["logout", "openai_codex", "--paste"],
        ["login", "other"],
    ],
)
def test_cli_auth_usage_rejects_non_exact_forms(capsys, args):
    with pytest.raises(SystemExit, match="1"):
        cli._handle_auth_command(args)

    output = capsys.readouterr().out
    assert "Usage: deepclaw auth login openai_codex [--paste]" in output


@pytest.mark.parametrize("action", ["login", "status"])
def test_cli_preserves_codex_dependency_error(monkeypatch, capsys, action):
    dependency_error = codex.OpenAICodexDependencyError(
        "OpenAI Codex OAuth requires langchain-openai==1.3.2. Run `uv sync`."
    )
    if action == "login":
        monkeypatch.setattr(
            codex, "login_openai_codex", lambda: (_ for _ in ()).throw(dependency_error)
        )
    else:
        monkeypatch.setattr(
            codex,
            "get_openai_codex_auth_status",
            lambda: (_ for _ in ()).throw(dependency_error),
        )

    with pytest.raises(SystemExit, match="1"):
        cli._handle_auth_command([action, "openai_codex"])
    output = capsys.readouterr().out
    assert "langchain-openai==1.3.2" in output
    assert "secret-token" not in output


def test_doctor_codex_auth_branches(monkeypatch):
    config = DeepClawConfig(model="openai_codex:gpt-5.3-codex")
    monkeypatch.setattr(codex, "get_openai_codex_auth_status", lambda: {"logged_in": False})
    missing = check_llm_api_key(config)
    assert missing.status == STATUS_FAIL
    assert "deepclaw auth login openai_codex --paste" in missing.message

    monkeypatch.setattr(
        codex,
        "get_openai_codex_auth_status",
        lambda: {"logged_in": True, "account_id": "account-123", "expires_at": "2030-01-01"},
    )
    valid = check_llm_api_key(config)
    assert valid.status == STATUS_OK
    assert "account-123" in valid.message

    monkeypatch.setattr(
        codex,
        "get_openai_codex_auth_status",
        lambda: (_ for _ in ()).throw(codex.OpenAICodexAuthError("bad")),
    )
    failed = check_llm_api_key(config)
    assert failed.status == STATUS_FAIL

    dependency_error = codex.OpenAICodexDependencyError(
        "OpenAI Codex OAuth requires langchain-openai==1.3.2. Run `uv sync`."
    )
    monkeypatch.setattr(
        codex,
        "get_openai_codex_auth_status",
        lambda: (_ for _ in ()).throw(dependency_error),
    )
    dependency_failed = check_llm_api_key(config)
    assert dependency_failed.status == STATUS_FAIL
    assert "langchain-openai==1.3.2" in dependency_failed.message
