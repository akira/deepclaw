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


def _run(run_id: str, **outputs):
    return SimpleNamespace(id=run_id, outputs=outputs)


def test_pairwise_prefers_first_pass_tool_use():
    module = _load_module()

    result = module.pairwise_tool_use_preference(
        [
            _run(
                "a",
                first_pass_tool_calls_seen=True,
                tool_calls_seen=True,
                attempts=1,
                tool_names=["execute"],
            ),
            _run(
                "b",
                first_pass_tool_calls_seen=False,
                tool_calls_seen=True,
                attempts=1,
                tool_names=["execute"],
            ),
        ],
        SimpleNamespace(outputs={"expected_tool_names": ["execute"]}),
    )

    assert result == {
        "key": "pairwise_tool_use_preference",
        "scores": {"a": 1, "b": 0},
        "comment": (
            "A preferred: A={'first_pass': True, 'expected_match': True, 'eventual': True, "
            "'rescued': False, 'attempts': 1, 'tool_names': ['execute']} "
            "B={'first_pass': False, 'expected_match': True, 'eventual': True, 'rescued': False, "
            "'attempts': 1, 'tool_names': ['execute']}"
        ),
    }


def test_pairwise_ties_when_ranks_match():
    module = _load_module()

    result = module.pairwise_tool_use_preference(
        [
            _run(
                "a",
                first_pass_tool_calls_seen=True,
                tool_calls_seen=True,
                attempts=1,
                tool_names=["execute"],
            ),
            _run(
                "b",
                first_pass_tool_calls_seen=True,
                tool_calls_seen=True,
                attempts=1,
                tool_names=["execute"],
            ),
        ],
        SimpleNamespace(outputs={"expected_tool_names": ["execute"]}),
    )

    assert result["key"] == "pairwise_tool_use_preference"
    assert result["scores"] == {"a": 0.5, "b": 0.5}
    assert result["comment"].startswith("Tie: A=")


def test_resolve_examples_uses_limit_when_requested():
    module = _load_module()

    class FakeClient:
        def list_examples(self, *, dataset_name, limit):
            assert dataset_name == "deepclaw"
            assert limit == 2
            return [SimpleNamespace(id="1"), SimpleNamespace(id="2")]

    result = module.resolve_examples(
        FakeClient(), dataset_name="deepclaw", example_ids=None, example_limit=2
    )

    assert [example.id for example in result] == ["1", "2"]


def test_resolve_examples_preserves_requested_order():
    module = _load_module()

    class FakeClient:
        def list_examples(self, *, dataset_name, limit):
            assert dataset_name == "deepclaw"
            assert limit == 100
            return [
                SimpleNamespace(id="b"),
                SimpleNamespace(id="a"),
            ]

    result = module.resolve_examples(
        FakeClient(), dataset_name="deepclaw", example_ids=["a", "b"], example_limit=None
    )

    assert [example.id for example in result] == ["a", "b"]


def test_summarize_pairwise_rows_counts_wins_losses_and_ties():
    module = _load_module()

    rows = [
        {"evaluation_results": {"scores": {"a": 1, "b": 0}}},
        {"evaluation_results": {"scores": {"a": 0, "b": 1}}},
        {"evaluation_results": {"scores": {"a": 0.5, "b": 0.5}}},
    ]

    assert module.summarize_pairwise_rows(rows) == {
        "model_a_wins": 1,
        "model_b_wins": 1,
        "ties": 1,
        "total": 3,
    }
