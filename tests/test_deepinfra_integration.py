"""Tests for DeepInfra integration and agent wiring."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from deepclaw import agent as agent_mod
from deepclaw.config import DeepClawConfig
from deepclaw.integrations.deepinfra import DEEPINFRA_PROVIDER, resolve_deepinfra_model


class TestResolveAgentModel:
    def test_returns_plain_model_string_for_non_deepinfra(self):
        config = DeepClawConfig(model="anthropic:claude-sonnet-4-6")

        resolved = resolve_deepinfra_model(config)

        assert resolved == "anthropic:claude-sonnet-4-6"

    def test_builds_chat_deepinfra_model_with_generation_config(self, monkeypatch):
        captured = {}

        class FakeChatDeepInfra:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def _create_message_dicts(self, messages, stop):
                return [{"content": msg.content} for msg in messages], {"stop": stop}

        monkeypatch.setattr(
            "deepclaw.integrations.deepinfra.load_chat_deepinfra_class", lambda: FakeChatDeepInfra
        )
        config = DeepClawConfig(model="deepinfra:deepseek-ai/DeepSeek-V3")
        config.generation.temperature = 0.25
        config.generation.max_tokens = 2048
        config.generation.top_p = 0.8
        config.generation.repetition_penalty = 1.1

        resolved = resolve_deepinfra_model(config)

        assert type(resolved).__name__ == "FakeChatDeepInfra"
        assert captured == {
            "model": "deepseek-ai/DeepSeek-V3",
            "streaming": False,
            "disable_streaming": True,
            "temperature": 0.25,
            "max_tokens": 2048,
            "top_p": 0.8,
            "model_kwargs": {"repetition_penalty": 1.1},
        }

    def test_resolved_model_leaves_message_content_unchanged(self, monkeypatch):
        class FakeChatDeepInfra:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def _create_message_dicts(self, messages, stop):
                return [{"content": msg.content} for msg in messages], {"stop": stop}

        monkeypatch.setattr(
            "deepclaw.integrations.deepinfra.load_chat_deepinfra_class", lambda: FakeChatDeepInfra
        )
        model = resolve_deepinfra_model(DeepClawConfig(model="deepinfra:foo/bar"))

        content = [
            {"type": "reasoning", "summary": []},
            {"type": "text", "text": "assistant text"},
        ]
        ai = AIMessage(content=content)
        payload, _params = model._create_message_dicts([ai], stop=None)

        assert payload == [{"content": content}]

    def test_deepinfra_requires_model_name(self):
        config = DeepClawConfig(model=f"{DEEPINFRA_PROVIDER}:")

        with pytest.raises(ValueError, match="DeepInfra model name cannot be empty"):
            resolve_deepinfra_model(config)


class TestCreateAgentDeepInfra:
    def test_create_agent_passes_resolved_deepinfra_model_instance(self, tmp_path, monkeypatch):
        captured = {}
        fake_model = MagicMock(name="deepinfra-model")

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
            agent_mod, "_append_summarization_middleware", lambda *args, **kwargs: None
        )

        config = DeepClawConfig(
            model="deepinfra:meta-llama/Llama-3.3-70B-Instruct", workspace_root=str(tmp_path)
        )
        result = agent_mod.create_agent(config, checkpointer="checkpointer")

        assert result == "agent"
        assert captured["model"] is fake_model
