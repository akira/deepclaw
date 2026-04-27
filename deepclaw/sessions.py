"""Session listing and resume helpers for DeepClaw Telegram threads."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from deepclaw.config import CHECKPOINTER_DB_PATH


class SessionInfo(TypedDict):
    """Displayable summary for a resumable chat session."""

    thread_id: str
    updated_at: str | None
    created_at: NotRequired[str | None]
    latest_checkpoint_id: NotRequired[str | None]
    checkpoint_count: int
    message_count: int
    initial_prompt: str | None


@lru_cache(maxsize=1)
def _serializer():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer()


def _db_path() -> Path:
    return CHECKPOINTER_DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{_db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1", (table,)
    ).fetchone()
    return row is not None


def _owner_filter_sql(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return (
        f"json_valid({prefix}metadata) "
        f"AND CAST(json_extract({prefix}metadata, '$.chat_id') AS TEXT) = ? "
        f"AND CAST(json_extract({prefix}metadata, '$.channel') AS TEXT) = ?"
    )


def _eligible_threads_cte() -> str:
    owner_filter = _owner_filter_sql("checkpoints")
    return f"""
        eligible_threads AS (
            SELECT thread_id
            FROM checkpoints
            GROUP BY thread_id
            HAVING SUM(CASE WHEN {owner_filter} THEN 1 ELSE 0 END) = COUNT(*)
               AND COUNT(*) > 0
        )
    """


def list_sessions_for_chat(
    chat_id: str, *, channel: str = "telegram", limit: int = 10
) -> list[SessionInfo]:
    """Return recent sessions for one chat, newest first."""
    limit = max(1, limit)
    if not _db_path().exists():
        return []

    with _connect() as conn:
        if not _table_exists(conn, "checkpoints"):
            return []

        owner_filter = _owner_filter_sql("checkpoints")
        rows = conn.execute(
            f"""
            WITH
            {_eligible_threads_cte()},
            scoped AS (
                SELECT checkpoints.rowid, checkpoints.thread_id, checkpoints.checkpoint_id, checkpoints.metadata
                FROM checkpoints
                JOIN eligible_threads ON eligible_threads.thread_id = checkpoints.thread_id
                WHERE {owner_filter}
            ),
            grouped AS (
                SELECT
                    thread_id,
                    MAX(json_extract(metadata, '$.updated_at')) AS updated_at,
                    MIN(json_extract(metadata, '$.updated_at')) AS created_at,
                    MAX(rowid) AS latest_rowid,
                    COUNT(*) AS checkpoint_count
                FROM scoped
                GROUP BY thread_id
            )
            SELECT
                grouped.thread_id,
                grouped.updated_at,
                grouped.created_at,
                grouped.checkpoint_count,
                scoped.checkpoint_id AS latest_checkpoint_id,
                grouped.latest_rowid
            FROM grouped
            JOIN scoped ON scoped.rowid = grouped.latest_rowid
            ORDER BY
                CASE WHEN grouped.updated_at IS NULL THEN 1 ELSE 0 END,
                grouped.updated_at DESC,
                grouped.latest_rowid DESC
            LIMIT ?
            """,
            (chat_id, channel, chat_id, channel, limit),
        ).fetchall()

        sessions: list[SessionInfo] = [
            SessionInfo(
                thread_id=cast(str, row["thread_id"]),
                updated_at=cast(str | None, row["updated_at"]),
                created_at=cast(str | None, row["created_at"]),
                latest_checkpoint_id=cast(str | None, row["latest_checkpoint_id"]),
                checkpoint_count=int(row["checkpoint_count"] or 0),
                message_count=0,
                initial_prompt=None,
            )
            for row in rows
        ]
        _populate_checkpoint_summaries(conn, sessions, chat_id=chat_id, channel=channel)
        return sessions


def get_most_recent_session_for_chat(
    chat_id: str, *, channel: str = "telegram"
) -> SessionInfo | None:
    """Return the most recent session for one chat, if any."""
    sessions = list_sessions_for_chat(chat_id, channel=channel, limit=1)
    return sessions[0] if sessions else None


def session_belongs_to_chat(thread_id: str, chat_id: str, *, channel: str = "telegram") -> bool:
    """Return whether every checkpoint for the thread belongs to the given chat/channel."""
    if not _db_path().exists():
        return False

    with _connect() as conn:
        if not _table_exists(conn, "checkpoints"):
            return False

        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN {_owner_filter_sql("checkpoints")} THEN 1 ELSE 0 END) AS owned_rows
            FROM checkpoints
            WHERE thread_id = ?
            """,
            (chat_id, channel, thread_id),
        ).fetchone()
        total_rows = int(row["total_rows"] or 0)
        owned_rows = int(row["owned_rows"] or 0)
        return total_rows > 0 and owned_rows == total_rows


def find_similar_sessions_for_chat(
    prefix: str, chat_id: str, *, channel: str = "telegram", limit: int = 3
) -> list[str]:
    """Return matching thread IDs in the same chat sharing the given prefix."""
    limit = max(1, limit)
    if not _db_path().exists():
        return []

    with _connect() as conn:
        if not _table_exists(conn, "checkpoints"):
            return []

        rows = conn.execute(
            f"""
            WITH {_eligible_threads_cte()}
            SELECT eligible_threads.thread_id
            FROM eligible_threads
            WHERE eligible_threads.thread_id LIKE ?
            ORDER BY eligible_threads.thread_id
            LIMIT ?
            """,
            (chat_id, channel, prefix + "%", limit),
        ).fetchall()
        return [cast(str, row["thread_id"]) for row in rows]


def _populate_checkpoint_summaries(
    conn: sqlite3.Connection,
    sessions: list[SessionInfo],
    *,
    chat_id: str,
    channel: str,
) -> None:
    if not sessions:
        return

    thread_ids = [session["thread_id"] for session in sessions]
    placeholders = ",".join("?" for _ in thread_ids)
    owner_filter = _owner_filter_sql("checkpoints")
    rows = conn.execute(
        f"""
        SELECT thread_id, type, checkpoint
        FROM (
            SELECT
                checkpoints.rowid,
                checkpoints.thread_id,
                checkpoints.type,
                checkpoints.checkpoint,
                ROW_NUMBER() OVER (PARTITION BY checkpoints.thread_id ORDER BY checkpoints.rowid DESC) AS rn
            FROM checkpoints
            WHERE checkpoints.thread_id IN ({placeholders})
              AND {owner_filter}
        )
        WHERE rn = 1
        """,
        (*thread_ids, chat_id, channel),
    ).fetchall()

    summaries = {
        cast(str, row["thread_id"]): _summarize_checkpoint_blob(
            cast(str | None, row["type"]), cast(bytes | None, row["checkpoint"])
        )
        for row in rows
    }
    for session in sessions:
        summary = summaries.get(session["thread_id"], {"message_count": 0, "initial_prompt": None})
        session["message_count"] = int(summary["message_count"])
        session["initial_prompt"] = cast(str | None, summary["initial_prompt"])


def _summarize_checkpoint_blob(
    type_str: str | None, checkpoint_blob: bytes | None
) -> dict[str, object]:
    if not type_str or not checkpoint_blob:
        return {"message_count": 0, "initial_prompt": None}
    try:
        payload = _serializer().loads_typed((type_str, checkpoint_blob))
    except Exception:
        return {"message_count": 0, "initial_prompt": None}
    messages = _checkpoint_messages(payload)
    return {
        "message_count": len(messages),
        "initial_prompt": _initial_prompt_from_messages(messages),
    }


def _checkpoint_messages(data: object) -> list[object]:
    if not isinstance(data, dict):
        return []
    channel_values = cast(dict[str, object], data).get("channel_values")
    if not isinstance(channel_values, dict):
        return []
    messages = cast(dict[str, object], channel_values).get("messages")
    if not isinstance(messages, list):
        return []
    return cast(list[object], messages)


def _initial_prompt_from_messages(messages: list[object]) -> str | None:
    for msg in messages:
        if getattr(msg, "type", None) == "human":
            return _coerce_prompt_text(getattr(msg, "content", None))
    return None


def _coerce_prompt_text(content: object) -> str | None:
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = cast(dict[str, object], part).get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(part))
        joined = " ".join(parts).strip()
        return joined or None
    if content is None:
        return None
    text = str(content).strip()
    return text or None
