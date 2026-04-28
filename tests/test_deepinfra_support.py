"""Tests for DeepInfra support helpers."""

import pytest
from langchain_core.messages import AIMessage

from deepclaw.config import DeepClawConfig
from deepclaw.integrations.deepinfra import DEEPINFRA_PROVIDER, resolve_deepinfra_model


class TestDeepInfraHelpers:
    def test_resolve_deepinfra_model_requires_model_name(self):
        config = DeepClawConfig(model=f"{DEEPINFRA_PROVIDER}:")

        with pytest.raises(ValueError, match="DeepInfra model name cannot be empty"):
            resolve_deepinfra_model(config)

    def test_resolve_deepinfra_model_builds_wrapped_chat_model(self, monkeypatch):
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
