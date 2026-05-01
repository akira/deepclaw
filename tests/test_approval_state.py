from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepclaw.approval_state import aadd_thread_approved_keys, aget_thread_approved_keys


@pytest.fixture()
def temp_checkpointer(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "checkpoints.db"
    monkeypatch.setattr(
        "deepclaw.approval_state.create_checkpointer",
        lambda: AsyncSqliteSaver.from_conn_string(str(db_path)),
    )
    return db_path


async def _seed_thread(thread_id: str, db_path: Path) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        checkpoint = empty_checkpoint()
        await checkpointer.aput(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
            checkpoint,
            {},
            checkpoint["channel_versions"],
        )


@pytest.mark.asyncio
async def test_add_and_get_thread_approved_keys(temp_checkpointer: Path):
    await _seed_thread("thread-1", temp_checkpointer)

    updated = await aadd_thread_approved_keys(
        "thread-1",
        ["dangerous:code_injection", "intent:inline_python"],
    )

    approved = await aget_thread_approved_keys("thread-1")

    assert updated is True
    assert approved == {"dangerous:code_injection", "intent:inline_python"}


@pytest.mark.asyncio
async def test_add_thread_approved_keys_merges_existing_values(temp_checkpointer: Path):
    await _seed_thread("thread-2", temp_checkpointer)
    await aadd_thread_approved_keys("thread-2", ["dangerous:code_injection"])

    updated = await aadd_thread_approved_keys(
        "thread-2",
        ["dangerous:code_injection", "intent:bash"],
    )

    approved = await aget_thread_approved_keys("thread-2")

    assert updated is True
    assert approved == {"dangerous:code_injection", "intent:bash"}


@pytest.mark.asyncio
async def test_add_thread_approved_keys_returns_false_for_missing_thread(temp_checkpointer: Path):
    updated = await aadd_thread_approved_keys("missing-thread", ["dangerous:code_injection"])

    assert updated is False
    assert await aget_thread_approved_keys("missing-thread") == set()
