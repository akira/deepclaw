"""Persistent session-approval store, decoupled from LangGraph checkpoints.

Earlier versions wrote approval state into the latest checkpoint's
`channel_values`, but `approval_state` is not declared as a graph channel —
the next graph step produces a new checkpoint from the declared schema and
silently drops the injected key, making session approvals non-functional
across tool calls.

This module owns its own SQLite table keyed by thread_id so approvals
survive any number of subsequent graph steps.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

import aiosqlite

from deepclaw.config import CONFIG_DIR

APPROVAL_STATE_DB_PATH = CONFIG_DIR / "approval_state.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_approvals (
    thread_id TEXT NOT NULL,
    approved_key TEXT NOT NULL,
    PRIMARY KEY (thread_id, approved_key)
);
"""

_init_lock = asyncio.Lock()
_initialized_paths: set[Path] = set()


def _db_path() -> Path:
    return APPROVAL_STATE_DB_PATH


async def _ensure_schema(db_path: Path) -> None:
    if db_path in _initialized_paths:
        return
    async with _init_lock:
        if db_path in _initialized_paths:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(db_path) as conn:
            await conn.executescript(_SCHEMA)
            await conn.commit()
        _initialized_paths.add(db_path)


async def aget_thread_approved_keys(thread_id: str) -> set[str]:
    """Return the set of session-approved warning keys for a thread."""
    if not thread_id:
        return set()
    db_path = _db_path()
    await _ensure_schema(db_path)
    async with (
        aiosqlite.connect(db_path) as conn,
        conn.execute(
            "SELECT approved_key FROM thread_approvals WHERE thread_id = ?",
            (thread_id,),
        ) as cursor,
    ):
        rows = await cursor.fetchall()
    return {row[0] for row in rows}


async def aadd_thread_approved_keys(thread_id: str, approved_keys: Iterable[str]) -> bool:
    """Persist session-approved warning keys for a thread (idempotent)."""
    normalized = [str(key) for key in approved_keys if str(key)]
    if not thread_id or not normalized:
        return False
    db_path = _db_path()
    await _ensure_schema(db_path)
    async with aiosqlite.connect(db_path) as conn:
        await conn.executemany(
            "INSERT OR IGNORE INTO thread_approvals (thread_id, approved_key) VALUES (?, ?)",
            [(thread_id, key) for key in normalized],
        )
        await conn.commit()
    return True
