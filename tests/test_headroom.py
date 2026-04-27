"""Tests for Headroom model wrapping helpers."""

from __future__ import annotations

import logging
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

        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: resolved_model

        headroom_integrations = ModuleType("headroom.integrations")

        class FakeHeadroomChatModel:
            def __init__(self, model):
                self.wrapped_model = model

        headroom_integrations.HeadroomChatModel = FakeHeadroomChatModel

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)

        wrapped = wrap_model_with_headroom(original_model, HeadroomConfig(enabled=True))

        assert wrapped.wrapped_model is resolved_model

    def test_logs_savings_for_each_optimization(self, monkeypatch, caplog):
        resolved_model = object()

        deepagents_models = ModuleType("deepagents._models")
        deepagents_models.resolve_model = lambda model: resolved_model

        headroom_integrations = ModuleType("headroom.integrations")

        class FakeHeadroomChatModel:
            def __init__(self, model):
                self.wrapped_model = model

            def _optimize_messages(self, messages):
                metrics = type(
                    "Metrics",
                    (),
                    {
                        "tokens_before": 100,
                        "tokens_after": 40,
                        "tokens_saved": 60,
                        "savings_percent": 60.0,
                        "transforms_applied": ["router:smart_crusher"],
                    },
                )()
                return messages, metrics

        headroom_integrations.HeadroomChatModel = FakeHeadroomChatModel

        monkeypatch.setitem(sys.modules, "deepagents._models", deepagents_models)
        monkeypatch.setitem(sys.modules, "headroom.integrations", headroom_integrations)

        wrapped = wrap_model_with_headroom("openai:gpt-5", HeadroomConfig(enabled=True))

        with caplog.at_level(logging.INFO):
            optimized_messages, metrics = wrapped._optimize_messages(["hello"])

        assert optimized_messages == ["hello"]
        assert metrics.tokens_saved == 60
        assert "Headroom savings: 100 -> 40 tokens (60 saved, 60.0%)" in caplog.text
