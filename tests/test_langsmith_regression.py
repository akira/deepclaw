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


def test_evaluator_expected_tool_names_is_null_when_not_specified():
    module = _load_module()

    result = module.evaluator_expected_tool_names(
        {"outputs": {"tool_names": []}},
        {"outputs": {"requires_tool_call": True}},
    )

    assert result == {
        "key": "expected_tool_names",
        "score": None,
        "comment": "no expected tool names specified",
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


def test_evaluator_first_pass_is_null_when_not_required():
    module = _load_module()

    result = module.evaluator_first_pass_tool_use(
        {"outputs": {"first_pass_tool_calls_seen": False}},
        {"outputs": {"requires_tool_call": True}},
    )

    assert result == {
        "key": "first_pass_tool_use",
        "score": None,
        "comment": "required=True must_succeed_first_pass=False first_pass_tool_calls_seen=False",
    }


def test_evaluator_secondary_tool_recovery_scores_recovered_second_pass():
    module = _load_module()

    result = module.evaluator_secondary_tool_recovery(
        {
            "outputs": {
                "tool_calls_seen": True,
                "first_pass_tool_calls_seen": False,
                "retried": True,
            }
        },
        {"outputs": {"requires_tool_call": True, "must_succeed_first_pass": True}},
    )

    assert result == {
        "key": "secondary_tool_recovery",
        "score": 1,
        "comment": (
            "required=True must_succeed_first_pass=True first_pass_tool_calls_seen=False "
            "retried=True tool_calls_seen=True"
        ),
    }


def test_evaluator_secondary_tool_recovery_scores_failed_recovery_when_retry_did_not_help():
    module = _load_module()

    result = module.evaluator_secondary_tool_recovery(
        {
            "outputs": {
                "tool_calls_seen": False,
                "first_pass_tool_calls_seen": False,
                "retried": True,
            }
        },
        {"outputs": {"requires_tool_call": True, "must_succeed_first_pass": True}},
    )

    assert result == {
        "key": "secondary_tool_recovery",
        "score": 0,
        "comment": (
            "required=True must_succeed_first_pass=True first_pass_tool_calls_seen=False "
            "retried=True tool_calls_seen=False"
        ),
    }


def test_evaluator_secondary_tool_recovery_is_null_when_first_pass_already_succeeded():
    module = _load_module()

    result = module.evaluator_secondary_tool_recovery(
        {
            "outputs": {
                "tool_calls_seen": True,
                "first_pass_tool_calls_seen": True,
                "retried": False,
            }
        },
        {"outputs": {"requires_tool_call": True, "must_succeed_first_pass": True}},
    )

    assert result == {
        "key": "secondary_tool_recovery",
        "score": None,
        "comment": (
            "required=True must_succeed_first_pass=True first_pass_tool_calls_seen=True "
            "retried=False tool_calls_seen=True"
        ),
    }


def test_run_case_invokes_repo_worker_file(monkeypatch):
    module = _load_module()
    calls = []

    def fake_run(cmd, capture_output=None, text=None, check=None, **_kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='{"tool_calls_seen": true}\n', stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_case(
        repo_path="/tmp/repo",
        user_text="Install ruff",
        model_name="openai:gpt-5.3-codex",
    )

    assert result == {"tool_calls_seen": True}
    assert calls == [
        [
            module.sys.executable,
            str(module.WORKER_SCRIPT_PATH),
            "--repo",
            "/tmp/repo",
            "--user-text",
            "Install ruff",
            "--model",
            "openai:gpt-5.3-codex",
            "--workspace-env",
            module.WORKSPACE_ENV,
        ]
    ]


def test_run_eval_supports_dict_rows_and_passes_metadata(monkeypatch):
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
                        {"key": "secondary_tool_recovery", "score": None},
                        {"key": "overall_pass_fail", "score": 1},
                    ]
                },
            }

    evaluate_calls = []

    def fake_evaluate(target, **kwargs):
        evaluate_calls.append(kwargs)
        assert target({"user_text": "Install ruff"})["tool_calls_seen"] is True
        return FakeResults()

    monkeypatch.setattr(module, "run_case", fake_run_case)
    monkeypatch.setattr(module, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        module,
        "git_metadata_for_repo",
        lambda repo_path: {"branch": "feat/test", "commit": "abc123"},
    )

    result = module.run_eval(
        client=None,
        dataset_name="deepclaw",
        repo_path="/tmp/repo",
        experiment_prefix="deepclaw-post",
        model_name="openai:gpt-5.3-codex",
        baseline_commit="origin/main",
        run_kind="post",
    )

    assert evaluate_calls == [
        {
            "data": "deepclaw",
            "evaluators": [
                module.evaluator_tool_call,
                module.evaluator_expected_tool_names,
                module.evaluator_first_pass_tool_use,
                module.evaluator_secondary_tool_recovery,
                module.evaluator_overall_pass_fail,
            ],
            "client": None,
            "experiment_prefix": "deepclaw-post",
            "description": "DeepClaw regression eval for /tmp/repo",
            "max_concurrency": 1,
            "blocking": True,
            "metadata": {
                "model": "openai:gpt-5.3-codex",
                "repo_path": "/tmp/repo",
                "run_kind": "post",
                "baseline_commit": "origin/main",
                "target_git_branch": "feat/test",
                "target_git_commit": "abc123",
            },
        }
    ]
    assert result["experiment_name"] == "exp-name"
    assert result["summary"] == {
        "tool_call_required": 1.0,
        "expected_tool_names": 1.0,
        "first_pass_tool_use": 1.0,
        "overall_pass_fail": 1.0,
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
                "secondary_tool_recovery": None,
                "overall_pass_fail": 1,
            },
            "pass_fail": "PASS",
            "failure_reasons": [],
            "tool_calls_seen": True,
            "tool_names": ["execute"],
            "retried": False,
            "attempts": 1,
            "first_pass_tool_calls_seen": True,
            "first_pass_text": None,
        }
    ]


def test_rollup_status_fails_when_required_metric_fails():
    module = _load_module()

    result = module.rollup_pass_fail(
        metrics={
            "tool_call_required": 1,
            "expected_tool_names": 0,
            "first_pass_tool_use": 1,
            "secondary_tool_recovery": None,
        },
        example_outputs={
            "requires_tool_call": True,
            "expected_tool_names": ["execute"],
            "must_succeed_first_pass": True,
        },
    )

    assert result == {
        "pass_fail": "FAIL",
        "failure_reasons": ["wrong tool used"],
    }


def test_rollup_status_passes_no_tool_example_when_no_tool_required():
    module = _load_module()

    result = module.rollup_pass_fail(
        metrics={
            "tool_call_required": 1,
            "expected_tool_names": None,
            "first_pass_tool_use": None,
            "secondary_tool_recovery": None,
        },
        example_outputs={
            "requires_tool_call": False,
        },
    )

    assert result == {
        "pass_fail": "PASS",
        "failure_reasons": [],
    }


def test_evaluator_overall_pass_fail_returns_binary_score_and_reason():
    module = _load_module()

    result = module.evaluator_overall_pass_fail(
        {
            "outputs": {
                "tool_calls_seen": True,
                "tool_names": ["read_file"],
                "retried": False,
                "attempts": 1,
                "first_pass_tool_calls_seen": True,
            }
        },
        {
            "outputs": {
                "requires_tool_call": True,
                "expected_tool_names": ["execute"],
                "must_succeed_first_pass": True,
            }
        },
    )

    assert result == {
        "key": "overall_pass_fail",
        "score": 0,
        "comment": "FAIL: wrong tool used",
    }


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
