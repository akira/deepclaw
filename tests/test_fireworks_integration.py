"""Tests for Fireworks AI integration and agent wiring."""

from unittest.mock import MagicMock

import pytest

from deepclaw import agent as agent_mod
from deepclaw.config import DeepClawConfig
from deepclaw.integrations import resolve_provider_model
from deepclaw.integrations.fireworks import FIREWORKS_PROVIDER, resolve_fireworks_model


class TestResolveFireworksModel:
    def test_returns_plain_model_string_for_non_fireworks(self):
        config = DeepClawConfig(model="anthropic:claude-sonnet-4-6")

        resolved = resolve_fireworks_model(config)

        assert resolved == "anthropic:claude-sonnet-4-6"

    def test_builds_chat_fireworks_model_with_generation_config(self, monkeypatch):
        captured = {}

        class FakeChatFireworks:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "deepclaw.integrations.fireworks.load_chat_fireworks_class", lambda: FakeChatFireworks
        )
        config = DeepClawConfig(model="fireworks:accounts/fireworks/models/llama-v3p1-70b-instruct")
        config.generation.temperature = 0.25
        config.generation.max_tokens = 2048
        config.generation.top_p = 0.8
        config.generation.repetition_penalty = 1.1

        resolved = resolve_fireworks_model(config)

        assert type(resolved).__name__ == "FakeChatFireworks"
        assert captured == {
            "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "streaming": True,
            "temperature": 0.25,
            "max_tokens": 2048,
            "top_p": 0.8,
            "model_kwargs": {"repetition_penalty": 1.1},
        }

    def test_provider_dispatcher_routes_fireworks(self, monkeypatch):
        fake_model = object()
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_fireworks_model", lambda config: fake_model
        )
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_baseten_model", lambda config: config.model.strip()
        )
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_deepinfra_model", lambda config: config.model.strip()
        )

        resolved = resolve_provider_model(
            DeepClawConfig(model="fireworks:accounts/fireworks/models/llama-v3p1-70b-instruct")
        )

        assert resolved is fake_model

    def test_provider_dispatcher_falls_back_to_raw_model(self, monkeypatch):
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_fireworks_model", lambda config: config.model.strip()
        )
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_baseten_model", lambda config: config.model.strip()
        )
        monkeypatch.setattr(
            "deepclaw.integrations.resolve_deepinfra_model", lambda config: config.model.strip()
        )

        resolved = resolve_provider_model(DeepClawConfig(model="anthropic:claude-sonnet-4-6"))

        assert resolved == "anthropic:claude-sonnet-4-6"

    def test_fireworks_requires_model_name(self):
        config = DeepClawConfig(model=f"{FIREWORKS_PROVIDER}:")

        with pytest.raises(ValueError, match="Fireworks model name cannot be empty"):
            resolve_fireworks_model(config)

    def test_fireworks_raises_helpful_error_when_dependency_missing(self, monkeypatch):
        monkeypatch.setattr(
            "deepclaw.integrations.fireworks.import_module",
            lambda _name: (_ for _ in ()).throw(ImportError("missing langchain_community")),
        )

        with pytest.raises(RuntimeError, match="Fireworks models require langchain-community"):
            resolve_fireworks_model(
                DeepClawConfig(model="fireworks:accounts/fireworks/models/llama-v3p1-70b-instruct")
            )


class TestCreateAgentFireworks:
    def test_create_agent_passes_resolved_fireworks_model_instance(self, tmp_path, monkeypatch):
        captured = {}
        fake_model = MagicMock(name="fireworks-model")

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
        monkeypatch.setattr(agent_mod, "resolve_provider_model", lambda config: fake_model)

        config = DeepClawConfig(
            model="fireworks:accounts/fireworks/models/llama-v3p1-70b-instruct",
            workspace_root=str(tmp_path),
        )
        result = agent_mod.create_agent(config, checkpointer="checkpointer")

        assert result == "agent"
        assert captured["model"] is fake_model
