"""Tests for evals/langsmith_regression.py."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "evals" / "langsmith_regression.py"
    spec = importlib.util.spec_from_file_location("langsmith_regression", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluator_expected_tool_names_fails_on_wrong_tool():
    module = _load_module()

    result = module.evaluator_expected_tool_names(
        {"outputs": {"tool_names": ["web_search"]}},
        {"outputs": {"expected_tool_names": ["execute"]}},
    )

    assert result == {
        "key": "expected_tool_names",
        "score": 0,
        "comment": "expected=['execute'] observed=['web_search']",
    }


def test_evaluator_first_pass_succeeds_only_when_required_first_pass_happens():
    module = _load_module()

    result = module.evaluator_first_pass_tool_use(
        {"outputs": {"first_pass_tool_calls_seen": False}},
        {"outputs": {"requires_tool_call": True, "must_succeed_first_pass": True}},
    )

    assert result == {
        "key": "first_pass_tool_use",
        "score": 0,
        "comment": "required=True must_succeed_first_pass=True first_pass_tool_calls_seen=False",
    }


def test_run_eval_supports_dict_rows(monkeypatch):
    module = _load_module()

    def fake_run_case(*, repo_path, user_text, model_name):
        return {
            "tool_calls_seen": True,
            "tool_names": ["execute"],
            "retried": False,
            "attempts": 1,
            "first_pass_tool_calls_seen": True,
            "final_text": f"handled {user_text}",
        }

    class FakeResults:
        experiment_name = "exp-name"
        url = "https://example.com/exp"
        comparison_url = "https://example.com/compare"

        def __iter__(self):
            yield {
                "run": {
                    "outputs": {
                        "tool_calls_seen": True,
                        "tool_names": ["execute"],
                        "retried": False,
                        "attempts": 1,
                        "first_pass_tool_calls_seen": True,
                    }
                },
                "example": {
                    "id": "ex-1",
                    "inputs": {"user_text": "Install ruff"},
                    "outputs": {
                        "requires_tool_call": True,
                        "expected_tool_names": ["execute"],
                        "must_succeed_first_pass": True,
                        "category": "tool-use",
                    },
                },
                "evaluation_results": {
                    "results": [
                        {"key": "tool_call_required", "score": 1},
                        {"key": "expected_tool_names", "score": 1},
                        {"key": "first_pass_tool_use", "score": 1},
                    ]
                },
            }

    def fake_evaluate(target, **_kwargs):
        assert target({"user_text": "Install ruff"})["tool_calls_seen"] is True
        return FakeResults()

    monkeypatch.setattr(module, "run_case", fake_run_case)
    monkeypatch.setattr(module, "evaluate", fake_evaluate)

    result = module.run_eval(
        client=None,
        dataset_name="deepclaw",
        repo_path="/tmp/repo",
        experiment_prefix="deepclaw-post",
        model_name="openai:gpt-5.3-codex",
    )

    assert result["experiment_name"] == "exp-name"
    assert result["summary"] == {
        "tool_call_required": 1.0,
        "expected_tool_names": 1.0,
        "first_pass_tool_use": 1.0,
    }
    assert result["examples"] == [
        {
            "example_id": "ex-1",
            "user_text": "Install ruff",
            "category": "tool-use",
            "metrics": {
                "tool_call_required": 1,
                "expected_tool_names": 1,
                "first_pass_tool_use": 1,
            },
            "tool_calls_seen": True,
            "tool_names": ["execute"],
            "retried": False,
            "attempts": 1,
            "first_pass_tool_calls_seen": True,
        }
    ]


def test_ensure_baseline_worktree_refreshes_existing_path(monkeypatch, tmp_path):
    module = _load_module()
    source_repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    source_repo.mkdir()
    worktree.mkdir()
    calls = []

    def fake_run(cmd, cwd=None, check=None, **_kwargs):
        calls.append((tuple(cmd), Path(cwd) if cwd else None, check))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.ensure_baseline_worktree("origin/main", str(worktree), str(source_repo))

    assert result == str(worktree)
    assert calls == [
        (("git", "fetch", "origin"), source_repo, True),
        (("git", "checkout", "--detach", "origin/main"), worktree, True),
        (("git", "reset", "--hard", "origin/main"), worktree, True),
        (("git", "clean", "-fd"), worktree, True),
    ]
