"""Tests for structured working state and continuity checkpoint helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from deepclaw import agent as agent_mod
from deepclaw import runtime_hygiene
from deepclaw.state_continuity import (
    REFERENCE_ONLY_NOTE,
    ContinuityMiddleware,
    add_attempt,
    add_blocker,
    add_decision,
    add_relevant_file,
    build_continuity_checkpoint,
    create_working_state,
    ensure_working_state,
)


class TestContinuityCheckpointProjection:
    def test_checkpoint_is_projection_with_reference_only_semantics(self):
        working_state = create_working_state(thread_id="chat-1", invocation_id="run-1")
        working_state["goal"] = "Ship PR 3"
        working_state["next_action"] = "Run the focused state tests"
        add_decision(
            working_state,
            summary="Use a structured working_state plus checkpoint projection",
            source="implementation",
        )
        add_attempt(
            working_state,
            action="tool:pytest",
            status="completed",
            details="pytest tests/test_state_continuity.py",
        )
        add_blocker(working_state, summary="Need to confirm middleware state merging")
        add_relevant_file(working_state, "/tmp/project/deepclaw/state_continuity.py")
        working_state["verification"]["status"] = "pending"
        working_state["verification"]["pending_checks"] = ["ruff", "pytest"]
        working_state["reference_notes"] = ["Prior summaries may be stale after new tool output."]
        working_state["artifact_refs"] = [
            {
                "path": "/tmp/runtime/artifact.txt",
                "kind": "tool-output",
                "label": "tool-output",
            }
        ]

        checkpoint = build_continuity_checkpoint(working_state)

        assert checkpoint["version"] == "v1"
        assert checkpoint["current_goal"] == "Ship PR 3"
        assert checkpoint["next_action"] == "Run the focused state tests"
        assert checkpoint["active_blockers"] == ["Need to confirm middleware state merging"]
        assert checkpoint["pending_verification"] == ["ruff", "pytest"]
        assert checkpoint["recent_decisions"] == [
            "Use a structured working_state plus checkpoint projection"
        ]
        assert checkpoint["recent_attempts"] == ["tool:pytest"]
        assert checkpoint["reference_summary"]["semantics"] == "historical_reference_only"
        assert checkpoint["reference_summary"]["note"] == REFERENCE_ONLY_NOTE
        assert checkpoint["reference_summary"]["items"] == [
            "Prior summaries may be stale after new tool output."
        ]


class TestRuntimeStateIntegration:
    def test_bind_runtime_state_seeds_working_state_and_tracks_artifacts(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(runtime_hygiene, "ARTIFACTS_DIR", tmp_path / "artifacts")

        with runtime_hygiene.bind_runtime_state("thread-42") as state:
            assert state.working_state["thread_id"] == "thread-42"
            assert state.working_state["invocation_id"] == state.invocation_id
            record = runtime_hygiene.write_text_artifact(
                "artifact body",
                category="tool-results",
                kind="execute",
            )

        assert record.path.exists()
        assert state.working_state["artifact_refs"][0]["path"] == str(record.path)
        assert state.working_state["artifact_refs"][0]["kind"] == "execute"


class TestContinuityMiddleware:
    def test_middleware_updates_working_state_and_checkpoint_from_messages(self):
        middleware = ContinuityMiddleware()
        state = {
            "messages": [
                HumanMessage(content="Implement the state continuity model"),
                AIMessage(content="Next action: wire the middleware and run tests."),
            ]
        }

        updates = middleware.after_model(state, runtime=MagicMock())
        working_state = updates["working_state"]
        checkpoint = updates["continuity_checkpoint"]

        assert working_state["goal"] == "Implement the state continuity model"
        assert checkpoint["current_goal"] == "Implement the state continuity model"
        assert checkpoint["next_action"] == "Next action: wire the middleware and run tests."
        assert checkpoint["reference_summary"]["semantics"] == "historical_reference_only"

    def test_ensure_working_state_can_upgrade_plain_state_dict(self):
        state = {"messages": []}

        working_state = ensure_working_state(state)

        assert state["working_state"] is working_state
        assert working_state["verification"]["status"] == "unknown"


class TestCreateAgentContinuityRegistration:
    def test_create_agent_registers_continuity_middleware(self, tmp_path, monkeypatch):
        captured = {}

        monkeypatch.setattr(agent_mod, "_setup_auth", lambda: None)
        monkeypatch.setattr(agent_mod, "_setup_memory", lambda: ["/tmp/AGENTS.md"])
        monkeypatch.setattr(agent_mod, "_setup_skills", lambda: ["/tmp/skills"])
        monkeypatch.setattr(agent_mod, "_load_soul", lambda: "SOUL")
        monkeypatch.setattr(agent_mod, "discover_tools", list)

        def _fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return "agent"

        monkeypatch.setattr(agent_mod, "create_deep_agent", _fake_create_deep_agent)

        config = SimpleNamespace(
            model="anthropic:test",
            command_timeout=30,
            workspace_root=str(tmp_path),
        )

        result = agent_mod.create_agent(config, checkpointer=object())

        assert result == "agent"
        assert any(
            isinstance(middleware, ContinuityMiddleware) for middleware in captured["middleware"]
        )
