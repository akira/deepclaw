from __future__ import annotations

import asyncio
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepclaw.auth import get_thread_state, load_thread_ids, normalize_thread_state, save_thread_ids
from deepclaw.compaction import (
    build_reference_handoff_summary,
    compact_thread,
    get_thread_snapshot,
)
from deepclaw.state_continuity import build_continuity_checkpoint, create_working_state


class _FakeCheckpointTuple:
    def __init__(self, messages, working_state, continuity_checkpoint):
        self.checkpoint = {
            "channel_values": {
                "messages": messages,
                "working_state": working_state,
                "continuity_checkpoint": continuity_checkpoint,
            }
        }


class _FakeCheckpointer:
    def __init__(self, messages, working_state, continuity_checkpoint):
        self._tuple = _FakeCheckpointTuple(messages, working_state, continuity_checkpoint)

    async def aget_tuple(self, _config):
        return self._tuple


def test_normalize_thread_state_accepts_legacy_string():
    state = normalize_thread_state("thread-123", chat_id="42")
    assert state["current_thread_id"] == "thread-123"
    assert state["parent_thread_id"] is None
    assert state["pending_summary_text"] is None
    assert state["raw_history_artifact_paths"] == []


def test_get_thread_state_seeds_structured_defaults():
    store = {}
    state = get_thread_state(store, "42")
    assert state["current_thread_id"] == "42"
    assert store["42"]["current_thread_id"] == "42"
    assert state["checkpoint_artifact_path"] is None


def test_save_and_load_thread_ids_round_trip(tmp_path, monkeypatch):
    target = tmp_path / "thread_ids.json"
    monkeypatch.setattr("deepclaw.auth.THREAD_IDS_FILE", str(target))
    save_thread_ids(
        {
            "42": {
                "current_thread_id": "thread-2",
                "parent_thread_id": "thread-1",
                "summary_artifact_path": "/tmp/summary.md",
                "checkpoint_artifact_path": "/tmp/checkpoint.json",
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
    assert loaded["42"]["checkpoint_artifact_path"] == "/tmp/checkpoint.json"
    assert loaded["42"]["raw_history_artifact_paths"] == ["/tmp/raw.md"]


def test_get_thread_snapshot_rebuilds_checkpoint_when_missing():
    working_state = create_working_state(thread_id="thread-old", invocation_id="inv-1")
    working_state["goal"] = "Fix login"
    checkpointer = _FakeCheckpointer(
        [HumanMessage(content="Fix login")],
        working_state,
        continuity_checkpoint=None,
    )

    snapshot = asyncio.run(get_thread_snapshot(checkpointer, "thread-old"))

    assert snapshot.working_state["goal"] == "Fix login"
    assert snapshot.continuity_checkpoint["current_goal"] == "Fix login"


def test_build_reference_handoff_summary_is_reference_only():
    checkpoint = build_continuity_checkpoint(create_working_state(thread_id="thread-old"))
    text = build_reference_handoff_summary(
        continuity_checkpoint=checkpoint,
        thread_id="thread-old",
        reason="manual",
        raw_history_path="/tmp/raw.md",
        checkpoint_path="/tmp/checkpoint.json",
        message_count=4,
    )
    assert "REFERENCE ONLY" in text
    assert "/tmp/checkpoint.json" in text
    assert "/tmp/raw.md" in text
    assert "read_file" in text


def test_compact_thread_writes_checkpoint_raw_history_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("deepclaw.compaction.COMPACTION_ROOT", tmp_path)
    working_state = create_working_state(thread_id="thread-old", invocation_id="inv-1")
    working_state["goal"] = "Fix the failing login test"
    working_state["next_action"] = "Run pytest after updating auth handling"
    working_state["relevant_files"] = ["deepclaw/auth.py"]
    working_state["verification"]["status"] = "pending"
    working_state["verification"]["pending_checks"] = ["pytest"]
    working_state["artifact_refs"] = [
        {"path": "/tmp/tool.txt", "kind": "tool-output", "label": "tool-output"}
    ]
    working_state["reference_notes"] = ["Prior summaries may be stale after new tool output."]
    checkpoint = build_continuity_checkpoint(working_state)
    checkpointer = _FakeCheckpointer(
        [
            SystemMessage(content="system"),
            HumanMessage(content="Please fix the failing login test"),
            ToolMessage(content="traceback", tool_call_id="call-1", name="pytest"),
            AIMessage(content="I found the auth bug and updated the test."),
        ],
        working_state,
        checkpoint,
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
    assert result.summary_artifact_path.endswith("-handoff.md")
    assert result.checkpoint_artifact_path.endswith("-checkpoint.json")
    assert len(result.raw_history_artifact_paths) == 1

    raw_text = Path(result.raw_history_artifact_paths[0]).read_text(encoding="utf-8")
    summary_text = Path(result.summary_artifact_path).read_text(encoding="utf-8")
    checkpoint_payload = json.loads(
        Path(result.checkpoint_artifact_path).read_text(encoding="utf-8")
    )

    assert "Thread History Artifact" in raw_text
    assert "Please fix the failing login test" in raw_text
    assert "REFERENCE ONLY" in summary_text
    assert (
        checkpoint_payload["continuity_checkpoint"]["current_goal"] == "Fix the failing login test"
    )
    assert (
        checkpoint_payload["working_state"]["next_action"]
        == "Run pytest after updating auth handling"
    )
