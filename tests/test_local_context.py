"""Tests for DeepClaw local context middleware."""

from types import SimpleNamespace
from unittest.mock import Mock

from deepclaw.local_context import LocalContextMiddleware


class FakeBackend:
    def __init__(self, output: str):
        self.output = output
        self.calls = []

    def execute(self, command: str, *, timeout: int | None = None):
        self.calls.append((command, timeout))
        return SimpleNamespace(output=self.output, exit_code=0)


def test_before_agent_caches_detected_context():
    backend = FakeBackend("## Local Context\n\n**Current Directory**: `/tmp/demo`")
    middleware = LocalContextMiddleware(backend)

    result = middleware.before_agent({"messages": []}, Mock())

    assert result == {"local_context": "## Local Context\n\n**Current Directory**: `/tmp/demo`"}
    assert len(backend.calls) == 1


def test_wrap_model_call_appends_context_to_system_prompt():
    middleware = LocalContextMiddleware(FakeBackend(""))
    request = Mock()
    request.system_prompt = "base"
    request.state = {"local_context": "## Local Context\n\n**Git**: Current branch `main`"}
    overridden = Mock()
    request.override.return_value = overridden
    handler = Mock(return_value="response")

    result = middleware.wrap_model_call(request, handler)

    assert result == "response"
    request.override.assert_called_once()
    assert "base" in request.override.call_args.kwargs["system_prompt"]
    assert "Current branch `main`" in request.override.call_args.kwargs["system_prompt"]
    handler.assert_called_once_with(overridden)
