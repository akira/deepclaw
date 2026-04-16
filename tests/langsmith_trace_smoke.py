#!/usr/bin/env python3
"""Minimal LangSmith tracing smoke test.

Usage:
  python tests/langsmith_trace_smoke.py

This script:
1) Prints sanitized env diagnostics (never prints full API keys).
2) Runs a tiny @traceable function.
3) Calls a lightweight LangSmith API endpoint via Client.list_projects().
4) Exits non-zero if tracing preconditions are not met or API call fails.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def _masked(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "<set>"
    return f"{value[:4]}...{value[-4:]}"


def _as_bool_str(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def print_env_diagnostics() -> dict[str, Any]:
    env = {
        "LANGSMITH_API_KEY": _masked(os.getenv("LANGSMITH_API_KEY")),
        "LANGCHAIN_API_KEY": _masked(os.getenv("LANGCHAIN_API_KEY")),
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
        "LANGSMITH_TRACING_V2": os.getenv("LANGSMITH_TRACING_V2"),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT"),
        "DEEPAGENTS_LANGSMITH_PROJECT": os.getenv("DEEPAGENTS_LANGSMITH_PROJECT"),
        "LANGSMITH_ENDPOINT": os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT"),
        "LANGSMITH_WORKSPACE_ID": os.getenv("LANGSMITH_WORKSPACE_ID"),
    }

    print("== LangSmith env diagnostics ==")
    for k, v in env.items():
        print(f"{k}={v}")

    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    tracing_enabled = any(
        _as_bool_str(os.getenv(name))
        for name in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2")
    )

    print(f"\nDerived: has_api_key={has_key}, tracing_enabled={tracing_enabled}")
    return {"has_key": has_key, "tracing_enabled": tracing_enabled}


def main() -> int:
    status = print_env_diagnostics()

    try:
        from langsmith import Client, traceable
        from langsmith.run_helpers import get_current_run_tree
    except Exception as e:  # pragma: no cover
        print(f"\nERROR: Failed to import langsmith SDK: {e}")
        return 2

    if not status["has_key"]:
        print("\nERROR: Missing LANGSMITH_API_KEY (or LANGCHAIN_API_KEY).")
        return 3

    if not status["tracing_enabled"]:
        print(
            "\nERROR: Tracing not enabled. Set one of: "
            "LANGSMITH_TRACING=true, LANGSMITH_TRACING_V2=true, or LANGCHAIN_TRACING_V2=true"
        )
        return 4

    @traceable(name="langsmith-smoke-add")
    def add(a: int, b: int) -> int:
        run_tree = get_current_run_tree()
        print(f"Inside traced function, run_tree_present={run_tree is not None}")
        return a + b

    print("\n== Running traced function ==")
    result = add(2, 3)
    print(f"Result: {result}")

    print("\n== Checking LangSmith API connectivity ==")
    try:
        client = Client()
        projects = list(client.list_projects(limit=1))
        print(f"API call OK. Retrieved {len(projects)} project record(s).")
    except Exception as e:
        print(f"ERROR: LangSmith API call failed: {e}")
        return 5

    print("\nPASS: LangSmith smoke test succeeded.")
    print("If traces are still not visible, verify project/workspace selection in LangSmith UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
