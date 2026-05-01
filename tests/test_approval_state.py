from __future__ import annotations

from pathlib import Path

import pytest

from deepclaw import approval_state
from deepclaw.approval_state import aadd_thread_approved_keys, aget_thread_approved_keys


@pytest.fixture()
def temp_db(monkeypatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "approval_state.db"
    monkeypatch.setattr(approval_state, "APPROVAL_STATE_DB_PATH", db_path)
    monkeypatch.setattr(approval_state, "_initialized_paths", set())
    return db_path


@pytest.mark.asyncio
async def test_add_and_get_thread_approved_keys(temp_db: Path):
    updated = await aadd_thread_approved_keys(
        "thread-1",
        ["dangerous:code_injection", "intent:inline_python"],
    )

    approved = await aget_thread_approved_keys("thread-1")

    assert updated is True
    assert approved == {"dangerous:code_injection", "intent:inline_python"}


@pytest.mark.asyncio
async def test_add_thread_approved_keys_merges_existing_values(temp_db: Path):
    await aadd_thread_approved_keys("thread-2", ["dangerous:code_injection"])

    updated = await aadd_thread_approved_keys(
        "thread-2",
        ["dangerous:code_injection", "intent:bash"],
    )

    approved = await aget_thread_approved_keys("thread-2")

    assert updated is True
    assert approved == {"dangerous:code_injection", "intent:bash"}


@pytest.mark.asyncio
async def test_threads_are_isolated(temp_db: Path):
    await aadd_thread_approved_keys("thread-a", ["dangerous:code_injection"])
    await aadd_thread_approved_keys("thread-b", ["intent:bash"])

    assert await aget_thread_approved_keys("thread-a") == {"dangerous:code_injection"}
    assert await aget_thread_approved_keys("thread-b") == {"intent:bash"}


@pytest.mark.asyncio
async def test_empty_inputs_are_no_ops(temp_db: Path):
    assert await aadd_thread_approved_keys("", ["dangerous:code_injection"]) is False
    assert await aadd_thread_approved_keys("thread-x", []) is False
    assert await aget_thread_approved_keys("") == set()
    assert await aget_thread_approved_keys("thread-never-stored") == set()


@pytest.mark.asyncio
async def test_approvals_survive_simulated_graph_step(temp_db: Path):
    """Regression: approvals must not be wiped by graph state updates.

    The previous implementation injected approval state into a LangGraph
    checkpoint's channel_values. Because `approval_state` was not a declared
    graph channel, the next graph step produced a new checkpoint that dropped
    the injected key. This test mimics that scenario by writing approvals,
    performing arbitrary unrelated checkpoint-style activity, and asserting
    the approvals remain.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class _State(TypedDict):
        messages: list

    async def _step(state):
        return {"messages": state.get("messages", []) + ["stepped"]}

    await aadd_thread_approved_keys("thread-graph", ["dangerous:code_injection"])
    assert await aget_thread_approved_keys("thread-graph") == {"dangerous:code_injection"}

    graph_db = temp_db.parent / "graph_checkpoints.db"
    async with AsyncSqliteSaver.from_conn_string(str(graph_db)) as ck:
        graph = (
            StateGraph(_State)
            .add_node("s", _step)
            .add_edge(START, "s")
            .add_edge("s", END)
            .compile(checkpointer=ck)
        )
        cfg = {"configurable": {"thread_id": "thread-graph"}}
        await graph.ainvoke({"messages": ["init"]}, cfg)
        await graph.ainvoke({"messages": ["second"]}, cfg)

    assert await aget_thread_approved_keys("thread-graph") == {"dangerous:code_injection"}
