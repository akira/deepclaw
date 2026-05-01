"""Helpers for storing session approval state in the LangGraph thread checkpoint."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langgraph.checkpoint.base import copy_checkpoint, create_checkpoint

APPROVAL_STATE_KEY = "approval_state"


def create_checkpointer():
    """Return the project checkpointer context manager.

    Wrapped here so tests can monkeypatch approval-state persistence without
    importing the full agent module during module import.
    """
    from deepclaw.agent import create_checkpointer as agent_create_checkpointer

    return agent_create_checkpointer()


def _normalize_approved_keys(value: Any) -> list[str]:
    """Return a normalized list of approved warning keys from checkpoint state."""
    if not isinstance(value, dict):
        return []
    approved_keys = value.get("approved_keys")
    if not isinstance(approved_keys, list):
        return []
    return [str(item) for item in approved_keys if str(item)]


async def aget_thread_approved_keys(thread_id: str) -> set[str]:
    """Load session-approved warning keys for a thread from the latest checkpoint."""
    if not thread_id:
        return set()

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    async with create_checkpointer() as checkpointer:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            return set()
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        if not isinstance(channel_values, dict):
            return set()
        return set(_normalize_approved_keys(channel_values.get(APPROVAL_STATE_KEY)))


async def aadd_thread_approved_keys(thread_id: str, approved_keys: Iterable[str]) -> bool:
    """Persist session-approved warning keys onto the latest checkpoint for a thread."""
    normalized_keys = [str(key) for key in approved_keys if str(key)]
    if not thread_id or not normalized_keys:
        return False

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    async with create_checkpointer() as checkpointer:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            return False

        checkpoint = copy_checkpoint(checkpoint_tuple.checkpoint)
        channel_values = dict(checkpoint.get("channel_values", {}))
        existing = set(_normalize_approved_keys(channel_values.get(APPROVAL_STATE_KEY)))
        existing.update(normalized_keys)
        channel_values[APPROVAL_STATE_KEY] = {
            "scope": "session",
            "approved_keys": sorted(existing),
        }
        checkpoint["channel_values"] = channel_values

        updated_checkpoint = create_checkpoint(checkpoint, None, 0)
        await checkpointer.aput(
            checkpoint_tuple.config,
            updated_checkpoint,
            checkpoint_tuple.metadata,
            checkpoint_tuple.checkpoint["channel_versions"],
        )
        return True
