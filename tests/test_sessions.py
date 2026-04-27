"""Tests for DeepClaw session listing/resume helpers."""

import json
import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from deepclaw import sessions as sessions_mod


def _init_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        )
        """
    )
    conn.commit()
    conn.close()


def _insert_checkpoint(
    path: Path,
    *,
    thread_id: str,
    checkpoint_id: str,
    updated_at: str,
    prompt: str,
    chat_id: str = "1",
    channel: str = "telegram",
    extra_messages: int = 0,
) -> None:
    serde = JsonPlusSerializer()
    messages = [HumanMessage(content=prompt), AIMessage(content="ack")]
    for idx in range(extra_messages):
        messages.append(AIMessage(content=f"extra-{idx}"))
    payload = {"channel_values": {"messages": messages}}
    type_str, checkpoint_blob = serde.dumps_typed(payload)
    metadata = json.dumps(
        {
            "chat_id": chat_id,
            "channel": channel,
            "updated_at": updated_at,
        }
    ).encode()

    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?)",
        (thread_id, checkpoint_id, type_str, checkpoint_blob, metadata),
    )
    conn.commit()
    conn.close()


def test_list_sessions_for_chat_filters_and_summarizes(monkeypatch, tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _init_db(db_path)
    monkeypatch.setattr(sessions_mod, "CHECKPOINTER_DB_PATH", db_path)

    _insert_checkpoint(
        db_path,
        thread_id="thread-new",
        checkpoint_id="cp-003",
        updated_at="2026-04-27T05:00:00+00:00",
        prompt="Newest prompt",
        extra_messages=2,
    )
    _insert_checkpoint(
        db_path,
        thread_id="thread-old",
        checkpoint_id="cp-002",
        updated_at="2026-04-27T03:00:00+00:00",
        prompt="Older prompt",
    )
    _insert_checkpoint(
        db_path,
        thread_id="thread-other-chat",
        checkpoint_id="cp-004",
        updated_at="2026-04-27T06:00:00+00:00",
        prompt="Should be hidden",
        chat_id="999",
    )

    sessions = sessions_mod.list_sessions_for_chat("1", limit=10)

    assert [session["thread_id"] for session in sessions] == ["thread-new", "thread-old"]
    assert sessions[0]["initial_prompt"] == "Newest prompt"
    assert sessions[0]["message_count"] == 4
    assert sessions[0]["checkpoint_count"] == 1
    assert sessions[0]["updated_at"] == "2026-04-27T05:00:00+00:00"


def test_get_most_recent_session_for_chat_returns_latest(monkeypatch, tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _init_db(db_path)
    monkeypatch.setattr(sessions_mod, "CHECKPOINTER_DB_PATH", db_path)

    _insert_checkpoint(
        db_path,
        thread_id="thread-a",
        checkpoint_id="cp-001",
        updated_at="2026-04-27T01:00:00+00:00",
        prompt="First",
    )
    _insert_checkpoint(
        db_path,
        thread_id="thread-b",
        checkpoint_id="cp-002",
        updated_at="2026-04-27T02:00:00+00:00",
        prompt="Second",
    )

    session = sessions_mod.get_most_recent_session_for_chat("1")

    assert session is not None
    assert session["thread_id"] == "thread-b"
    assert session["initial_prompt"] == "Second"


def test_session_belongs_to_chat_and_similar_matches_are_scoped(monkeypatch, tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _init_db(db_path)
    monkeypatch.setattr(sessions_mod, "CHECKPOINTER_DB_PATH", db_path)

    _insert_checkpoint(
        db_path,
        thread_id="thread-alpha",
        checkpoint_id="cp-001",
        updated_at="2026-04-27T01:00:00+00:00",
        prompt="Alpha",
    )
    _insert_checkpoint(
        db_path,
        thread_id="thread-alpine",
        checkpoint_id="cp-002",
        updated_at="2026-04-27T02:00:00+00:00",
        prompt="Alpine",
    )
    _insert_checkpoint(
        db_path,
        thread_id="thread-alien",
        checkpoint_id="cp-003",
        updated_at="2026-04-27T03:00:00+00:00",
        prompt="Alien",
        chat_id="999",
    )

    assert sessions_mod.session_belongs_to_chat("thread-alpha", "1") is True
    assert sessions_mod.session_belongs_to_chat("thread-alien", "1") is False
    assert sessions_mod.find_similar_sessions_for_chat("thread-al", "1") == [
        "thread-alpha",
        "thread-alpine",
    ]


def test_list_sessions_summary_stays_scoped_to_chat_and_channel(monkeypatch, tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _init_db(db_path)
    monkeypatch.setattr(sessions_mod, "CHECKPOINTER_DB_PATH", db_path)

    _insert_checkpoint(
        db_path,
        thread_id="shared-thread",
        checkpoint_id="cp-001",
        updated_at="2026-04-27T01:00:00+00:00",
        prompt="Prompt owned by chat 1",
        chat_id="1",
        channel="telegram",
    )
    _insert_checkpoint(
        db_path,
        thread_id="shared-thread",
        checkpoint_id="cp-999",
        updated_at="2026-04-27T09:00:00+00:00",
        prompt="Leaked prompt from another chat",
        chat_id="999",
        channel="telegram",
    )
    _insert_checkpoint(
        db_path,
        thread_id="shared-thread",
        checkpoint_id="cp-998",
        updated_at="2026-04-27T08:00:00+00:00",
        prompt="Leaked prompt from another channel",
        chat_id="1",
        channel="cron",
    )

    sessions = sessions_mod.list_sessions_for_chat("1", channel="telegram", limit=10)

    assert sessions == []


def test_session_belongs_to_chat_requires_exclusive_owner(monkeypatch, tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _init_db(db_path)
    monkeypatch.setattr(sessions_mod, "CHECKPOINTER_DB_PATH", db_path)

    _insert_checkpoint(
        db_path,
        thread_id="shared-thread",
        checkpoint_id="cp-001",
        updated_at="2026-04-27T01:00:00+00:00",
        prompt="Chat 1",
        chat_id="1",
        channel="telegram",
    )
    _insert_checkpoint(
        db_path,
        thread_id="shared-thread",
        checkpoint_id="cp-002",
        updated_at="2026-04-27T02:00:00+00:00",
        prompt="Chat 999",
        chat_id="999",
        channel="telegram",
    )

    assert sessions_mod.session_belongs_to_chat("shared-thread", "1") is False
    assert sessions_mod.find_similar_sessions_for_chat("shared", "1") == []
    assert sessions_mod.list_sessions_for_chat("1") == []


def test_invalid_json_metadata_is_ignored(monkeypatch, tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _init_db(db_path)
    monkeypatch.setattr(sessions_mod, "CHECKPOINTER_DB_PATH", db_path)

    _insert_checkpoint(
        db_path,
        thread_id="thread-valid",
        checkpoint_id="cp-001",
        updated_at="2026-04-27T01:00:00+00:00",
        prompt="Valid prompt",
    )

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?)",
        ("thread-bad", "cp-002", "json", b"{}", b"not-json"),
    )
    conn.commit()
    conn.close()

    sessions = sessions_mod.list_sessions_for_chat("1", limit=10)

    assert [session["thread_id"] for session in sessions] == ["thread-valid"]
    assert sessions_mod.session_belongs_to_chat("thread-valid", "1") is True
    assert sessions_mod.find_similar_sessions_for_chat("thread", "1") == ["thread-valid"]


def test_malformed_metadata_contaminates_thread_ownership(monkeypatch, tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _init_db(db_path)
    monkeypatch.setattr(sessions_mod, "CHECKPOINTER_DB_PATH", db_path)

    _insert_checkpoint(
        db_path,
        thread_id="victim-thread",
        checkpoint_id="cp-001",
        updated_at="2026-04-27T01:00:00+00:00",
        prompt="Owned by chat 1",
    )

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?)",
        ("victim-thread", "cp-002", "json", b"{}", b"not-json"),
    )
    conn.commit()
    conn.close()

    assert sessions_mod.session_belongs_to_chat("victim-thread", "1") is False
    assert sessions_mod.list_sessions_for_chat("1") == []
    assert sessions_mod.find_similar_sessions_for_chat("victim", "1") == []


def test_missing_db_path_does_not_create_file(monkeypatch, tmp_path):
    db_path = tmp_path / "missing.db"
    monkeypatch.setattr(sessions_mod, "CHECKPOINTER_DB_PATH", db_path)

    assert not db_path.exists()
    assert sessions_mod.list_sessions_for_chat("1") == []
    assert sessions_mod.session_belongs_to_chat("thread-1", "1") is False
    assert sessions_mod.find_similar_sessions_for_chat("thread", "1") == []
    assert not db_path.exists()


def test_numeric_chat_id_metadata_is_supported(monkeypatch, tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _init_db(db_path)
    monkeypatch.setattr(sessions_mod, "CHECKPOINTER_DB_PATH", db_path)

    serde = JsonPlusSerializer()
    type_str, checkpoint_blob = serde.dumps_typed(
        {"channel_values": {"messages": [HumanMessage(content="Numeric owner prompt")]}}
    )
    metadata = json.dumps(
        {"chat_id": 1, "channel": "telegram", "updated_at": "2026-04-27T01:00:00+00:00"}
    ).encode()

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?)",
        ("numeric-thread", "cp-001", type_str, checkpoint_blob, metadata),
    )
    conn.commit()
    conn.close()

    sessions = sessions_mod.list_sessions_for_chat("1")

    assert [session["thread_id"] for session in sessions] == ["numeric-thread"]
    assert sessions_mod.session_belongs_to_chat("numeric-thread", "1") is True
    assert sessions_mod.find_similar_sessions_for_chat("numeric", "1") == ["numeric-thread"]
