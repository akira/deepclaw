"""Tests for evals/langsmith_pairwise.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).resolve().parents[1] / "evals" / "langsmith_pairwise.py"
    spec = importlib.util.spec_from_file_location("langsmith_pairwise", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_run(run_id: str, outputs: dict):
    return SimpleNamespace(id=run_id, outputs=outputs)


def _fake_example(outputs: dict):
    return SimpleNamespace(outputs=outputs)


def test_pairwise_prefers_overall_pass_fail_before_tool_use_tiebreakers():
    module = _load_module()

    left = _fake_run(
        "run-a",
        {
            "tool_calls_seen": True,
            "tool_names": ["read_file"],
            "first_pass_tool_calls_seen": True,
            "retried": False,
            "attempts": 1,
        },
    )
    right = _fake_run(
        "run-b",
        {
            "tool_calls_seen": True,
            "tool_names": ["execute"],
            "first_pass_tool_calls_seen": True,
            "retried": False,
            "attempts": 1,
        },
    )
    example = _fake_example(
        {
            "requires_tool_call": True,
            "expected_tool_names": ["execute"],
            "must_succeed_first_pass": True,
        }
    )

    result = module.pairwise_tool_use_preference([left, right], example)

    assert result["scores"] == {"run-a": 0, "run-b": 1}
    assert "'overall_pass': False" in result["comment"]
    assert "'overall_pass': True" in result["comment"]


def test_pairwise_ties_when_both_have_same_pass_fail_and_tool_features():
    module = _load_module()

    left = _fake_run(
        "run-a",
        {
            "tool_calls_seen": False,
            "tool_names": [],
            "first_pass_tool_calls_seen": False,
            "retried": False,
            "attempts": 1,
        },
    )
    right = _fake_run(
        "run-b",
        {
            "tool_calls_seen": False,
            "tool_names": [],
            "first_pass_tool_calls_seen": False,
            "retried": False,
            "attempts": 1,
        },
    )
    example = _fake_example({"requires_tool_call": False})

    result = module.pairwise_tool_use_preference([left, right], example)

    assert result["scores"] == {"run-a": 0.5, "run-b": 0.5}
