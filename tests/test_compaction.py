from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepclaw.auth import get_thread_state, load_thread_ids, normalize_thread_state, save_thread_ids
from deepclaw.compaction import (
    _build_reference_summary,
    _format_messages_markdown,
    _should_auto_compact,
    compact_thread,
    estimate_thread_tokens,
    get_auto_compaction_decision,
)


def test_normalize_thread_state_accepts_legacy_string():
    state = normalize_thread_state("thread-123", chat_id="42")
    assert state["current_thread_id"] == "thread-123"
    assert state["parent_thread_id"] is None
    assert state["pending_summary_text"] is None
    assert state["raw_history_artifact_paths"] == []


def test_get_thread_state_defaults_to_chat_id():
    store = {}
    state = get_thread_state(store, "42")
    assert state["current_thread_id"] == "42"
    assert store["42"]["current_thread_id"] == "42"


def test_save_and_load_thread_ids_round_trip(tmp_path, monkeypatch):
    target = tmp_path / "thread_ids.json"
    monkeypatch.setattr("deepclaw.auth.THREAD_IDS_FILE", str(target))
    save_thread_ids(
        {
            "42": {
                "current_thread_id": "thread-2",
                "parent_thread_id": "thread-1",
                "summary_artifact_path": "/tmp/summary.md",
                "raw_history_artifact_paths": ["/tmp/raw.md"],
                "pending_summary_text": "summary",
                "last_compacted_at": "2026-04-24T00:00:00Z",
                "last_compaction_reason": "manual",
            }
        }
    )
    loaded = load_thread_ids()
    assert loaded["42"]["current_thread_id"] == "thread-2"
    assert loaded["42"]["parent_thread_id"] == "thread-1"
    assert loaded["42"]["raw_history_artifact_paths"] == ["/tmp/raw.md"]


def test_should_auto_compact_uses_message_threshold():
    assert _should_auto_compact([object()] * 40) is False
    assert _should_auto_compact([object()] * 41) is True


def test_estimate_thread_tokens_counts_message_content():
    messages = [
        SystemMessage(content="system rule"),
        HumanMessage(content="hello"),
        ToolMessage(content="very long tool output" * 20, tool_call_id="call-1", name="read_file"),
    ]
    assert estimate_thread_tokens(messages) > 100


def test_estimate_thread_tokens_counts_ai_tool_calls():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "/tmp/demo.txt", "padding": "x" * 4000},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
    ]
    assert estimate_thread_tokens(messages) > 500


def test_estimate_thread_tokens_deduplicates_tool_calls_from_additional_kwargs():
    raw_tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": "/tmp/demo.txt", "padding": "x" * 4000}),
            },
        }
    ]
    normalized_tool_calls = [
        {
            "name": "read_file",
            "args": {"path": "/tmp/demo.txt", "padding": "x" * 4000},
            "id": "call-1",
            "type": "tool_call",
        }
    ]
    duplicated = AIMessage(
        content=[
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "read_file",
                "input": {"path": "/tmp/demo.txt", "padding": "x" * 4000},
            }
        ],
        tool_calls=normalized_tool_calls,
        additional_kwargs={"tool_calls": raw_tool_calls, "stop_reason": "tool_use"},
    )
    without_duplicate_payloads = AIMessage(
        content="",
        tool_calls=normalized_tool_calls,
        additional_kwargs={"stop_reason": "tool_use"},
    )
    duplicated_tokens = estimate_thread_tokens([duplicated])
    baseline_tokens = estimate_thread_tokens([without_duplicate_payloads])
    assert duplicated_tokens >= baseline_tokens
    assert duplicated_tokens - baseline_tokens < 80


def test_auto_compaction_decision_uses_token_budget(monkeypatch):
    monkeypatch.setattr("deepclaw.compaction.AUTO_COMPACT_MIN_TOKEN_BUDGET", 30)
    monkeypatch.setattr("deepclaw.compaction.AUTO_COMPACT_MAX_TOKEN_BUDGET", 30)
    messages = [HumanMessage(content="x" * 400)]
    decision = get_auto_compaction_decision(
        messages,
        model_name="anthropic:claude-sonnet-4-6-20250514",
        threshold=99,
    )
    assert decision.should_compact is True
    assert decision.reason == "token-budget"
    assert decision.token_budget == 30


def test_auto_compaction_decision_budget_varies_by_model_window():
    messages = [HumanMessage(content="short")]
    small = get_auto_compaction_decision(messages, model_name="openai:gpt-4o", threshold=99)
    large = get_auto_compaction_decision(messages, model_name="google:gemini-2.5-pro", threshold=99)
    assert small.token_budget < large.token_budget


def test_format_messages_markdown_renders_roles_and_content():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="hello"),
        ToolMessage(content="result", tool_call_id="call-1", name="read_file"),
        AIMessage(content="done"),
    ]
    rendered = _format_messages_markdown(messages)
    assert "## Message 1 — system" in rendered
    assert "hello" in rendered
    assert "tool_name: read_file" in rendered
    assert "done" in rendered


def test_build_reference_summary_includes_recent_state_and_artifacts():
    messages = [
        SystemMessage(content="ignore me"),
        HumanMessage(content="Please fix the failing login test"),
        ToolMessage(content="traceback", tool_call_id="call-1", name="pytest"),
        AIMessage(content="I found the auth bug and updated the test."),
    ]
    result = _build_reference_summary(
        messages,
        raw_history_path="/tmp/raw.md",
        reason="threshold",
        thread_id="thread-old",
    )
    assert "REFERENCE ONLY" in result
    assert "Please fix the failing login test" in result
    assert "I found the auth bug" in result
    assert "/tmp/raw.md" in result
    assert "pytest" in result


class _FakeCheckpointTuple:
    def __init__(self, messages):
        self.checkpoint = {"channel_values": {"messages": messages}}


class _FakeCheckpointer:
    def __init__(self, messages):
        self._messages = messages

    async def aget_tuple(self, _config):
        return _FakeCheckpointTuple(self._messages)


def test_compact_thread_writes_raw_and_summary_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("deepclaw.compaction.COMPACTION_ROOT", tmp_path)
    checkpointer = _FakeCheckpointer(
        [
            SystemMessage(content="system"),
            HumanMessage(content="Please fix the failing login test"),
            ToolMessage(content="traceback", tool_call_id="call-1", name="pytest"),
            AIMessage(content="I found the auth bug and updated the test."),
        ]
    )

    result = asyncio.run(
        compact_thread(
            checkpointer,
            thread_id="thread-old",
            chat_id="42",
            reason="manual",
        )
    )

    assert result is not None
    assert result.summary_artifact_path is not None
    assert len(result.raw_history_artifact_paths) == 1

    raw_path = result.raw_history_artifact_paths[0]
    summary_path = result.summary_artifact_path
    assert "raw.md" in raw_path
    assert "summary.md" in summary_path
    assert tmp_path.as_posix() in raw_path
    assert tmp_path.as_posix() in summary_path

    raw_text = Path(raw_path).read_text(encoding="utf-8")
    summary_text = Path(summary_path).read_text(encoding="utf-8")
    assert "DeepClaw Compacted Thread History" in raw_text
    assert "Please fix the failing login test" in raw_text
    assert "REFERENCE ONLY" in summary_text
    assert "raw_history_artifact" in summary_text
