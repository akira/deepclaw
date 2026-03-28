"""Agent factory for DeepClaw."""

import os

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepclaw.middleware import SafetyMiddleware
from deepclaw.safety import scrub_env


def create_checkpointer():
    """Create and return an async SQLite checkpointer context manager."""
    db_path = os.path.expanduser("~/.deepagents/checkpoints.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(db_path)


def create_agent(config, checkpointer):
    """Create a DeepAgents agent with the given config and checkpointer."""
    backend = LocalShellBackend(virtual_mode=False, env=scrub_env())
    middleware = []
    if SafetyMiddleware is not None:
        middleware.append(SafetyMiddleware())
    return create_deep_agent(
        model=config.model or None,
        backend=backend,
        checkpointer=checkpointer,
        middleware=middleware,
    )
