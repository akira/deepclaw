"""Tests for Headroom model wrapping helpers."""

from __future__ import annotations

import sys
from types import ModuleType

from deepclaw.config import HeadroomConfig
from deepclaw.headroom import wrap_model_with_headroom


class TestWrapModelWithHeadroom:
    def test_returns_original_model_when_disabled(self):
        model = object()

        wrapped = wrap_model_with_headroom(model, HeadroomConfig(enabled=False))

        assert wrapped is model

    def test_wraps_resolved_model_when_enabled(self, monkeypatch):
        original_model = "openai:gpt-5"
        resolved_model = object()
        wrapped_model = object()

        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: resolved_model

        headroom_integrations = ModuleType("headroom.integrations")

        def fake_headroom_chat_model(model):
            assert model is resolved_model
            return wrapped_model

        headroom_integrations.HeadroomChatModel = fake_headroom_chat_model

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)

        wrapped = wrap_model_with_headroom(original_model, HeadroomConfig(enabled=True))

        assert wrapped is wrapped_model
