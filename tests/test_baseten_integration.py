"""Tests for Baseten integration and agent wiring."""

from unittest.mock import MagicMock

import pytest

from deepclaw import agent as agent_mod
from deepclaw.config import DeepClawConfig
from deepclaw.integrations import resolve_provider_model
from deepclaw.integrations.baseten import BASETEN_PROVIDER, resolve_baseten_model


class TestResolveBasetenModel:
    def test_returns_plain_model_string_for_non_baseten(self):
        config = DeepClawConfig(model="anthropic:claude-sonnet-4-6")

        resolved = resolve_baseten_model(config)

        assert resolved == "anthropic:claude-sonnet-4-6"

    def test_builds_chat_baseten_model_with_model_slug(self, monkeypatch):
        captured = {}

        class FakeChatBaseten:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        config = DeepClawConfig(model="baseten:moonshotai/Kimi-K2-Instruct-0905")
        config.generation.temperature = 0.25
        config.generation.max_tokens = 2048
        config.generation.top_p = 0.8
        config.generation.repetition_penalty = 1.1

        resolved = resolve_baseten_model(config)

        assert type(resolved).__name__ == "FakeChatBaseten"
        assert captured == {
            "model": "moonshotai/Kimi-K2-Instruct-0905",
            "streaming": True,
            "disable_streaming": False,
            "stream_usage": True,
            "temperature": 0.25,
            "max_tokens": 2048,
            "top_p": 0.8,
            "model_kwargs": {"repetition_penalty": 1.1},
        }

    def test_builds_chat_baseten_model_with_dedicated_url(self, monkeypatch):
        captured = {}

        class FakeChatBaseten:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "deepclaw.integrations.baseten.load_chat_baseten_class", lambda: FakeChatBaseten
        )
        config = DeepClawConfig(
            model="baseten:https://model-123.api.baseten.co/environments/production/sync/v1"
        )

        resolved = resolve_baseten_model(config)

        assert type(resolved).__name__ == "FakeChatBaseten"
        assert captured["model_url"] == (
            "https://model-123.api.baseten.co/environments/production/sync/v1"
        )
        assert "model" not in captured

    def test_provider_dispatcher_routes_baseten(self, monkeypatch):
        fake_model = object()
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_baseten_model", lambda config: fake_model
        )
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_deepinfra_model", lambda config: config.model.strip()
        )

        resolved = resolve_provider_model(
            DeepClawConfig(model="baseten:moonshotai/Kimi-K2-Instruct-0905")
        )

        assert resolved is fake_model

    def test_provider_dispatcher_falls_back_to_raw_model(self, monkeypatch):
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_baseten_model", lambda config: config.model.strip()
        )
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_deepinfra_model", lambda config: config.model.strip()
        )

        resolved = resolve_provider_model(DeepClawConfig(model="anthropic:claude-sonnet-4-6"))

        assert resolved == "anthropic:claude-sonnet-4-6"

    def test_baseten_requires_model_name(self):
        config = DeepClawConfig(model=f"{BASETEN_PROVIDER}:")

        with pytest.raises(ValueError, match="Baseten model name cannot be empty"):
            resolve_baseten_model(config)

    def test_baseten_raises_helpful_error_when_dependency_missing(self, monkeypatch):
        monkeypatch.setattr(
            "deepclaw.integrations.baseten.import_module",
            lambda _name: (_ for _ in ()).throw(ImportError("missing langchain_baseten")),
        )

        with pytest.raises(RuntimeError, match="Baseten models require langchain-baseten"):
            resolve_baseten_model(DeepClawConfig(model="baseten:moonshotai/Kimi-K2.6"))


class TestCreateAgentBaseten:
    def test_create_agent_passes_resolved_baseten_model_instance(self, tmp_path, monkeypatch):
        captured = {}
        fake_model = MagicMock(name="baseten-model")

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

        monkeypatch.setattr(agent_mod, "_setup_auth", lambda: None)
        monkeypatch.setattr(agent_mod, "_load_soul", lambda: "soul")
        monkeypatch.setattr(agent_mod, "_setup_memory", lambda: ["/memory/AGENTS.md"])
        monkeypatch.setattr(agent_mod, "_setup_skills", lambda: ["/skills"])
        monkeypatch.setattr(agent_mod, "discover_tools", list)
        monkeypatch.setattr(agent_mod, "DeepClawLocalShellBackend", FakeShellBackend)
        monkeypatch.setattr(agent_mod, "FilesystemBackend", FakeFilesystemBackend)
        monkeypatch.setattr(agent_mod, "CompositeBackend", FakeCompositeBackend)
        monkeypatch.setattr(agent_mod, "RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr(agent_mod, "create_deep_agent", fake_create_deep_agent)
        monkeypatch.setattr(agent_mod, "resolve_provider_model", lambda config: fake_model)
        monkeypatch.setattr(
            agent_mod,
            "_create_deepclaw_summarization_tool_middleware",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            agent_mod,
            "_patched_deepagents_summarization_factory",
            lambda: __import__("contextlib").nullcontext(),
        )

        config = DeepClawConfig(
            model="baseten:moonshotai/Kimi-K2-Instruct-0905", workspace_root=str(tmp_path)
        )
        result = agent_mod.create_agent(config, checkpointer="checkpointer")

        assert result == "agent"
        assert captured["model"] is fake_model
