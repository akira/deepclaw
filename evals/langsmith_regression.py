#!/usr/bin/env python3
"""Evaluate the LangSmith "deepclaw" regression dataset against DeepClaw revisions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

EVALS_DIR = Path(__file__).resolve().parent
_EXAMPLE_FILTERS_PATH = EVALS_DIR / "example_filters.py"
_EXAMPLE_FILTERS_SPEC = importlib.util.spec_from_file_location(
    "deepclaw_example_filters", _EXAMPLE_FILTERS_PATH
)
if _EXAMPLE_FILTERS_SPEC is None or _EXAMPLE_FILTERS_SPEC.loader is None:
    raise RuntimeError(f"Unable to load example_filters module from {_EXAMPLE_FILTERS_PATH}")
_EXAMPLE_FILTERS_MODULE = importlib.util.module_from_spec(_EXAMPLE_FILTERS_SPEC)
_EXAMPLE_FILTERS_SPEC.loader.exec_module(_EXAMPLE_FILTERS_MODULE)
filter_examples = _EXAMPLE_FILTERS_MODULE.filter_examples

WORKSPACE_ENV = str(Path("~/.deepclaw/.env").expanduser())
DEFAULT_DATASET = "deepclaw"
DEFAULT_MODEL = "openai:gpt-5.3-codex"
DEFAULT_RESULTS_PATH = "/tmp/deepclaw-evals/results.json"
DEFAULT_TRACE_PROJECT = "deepclaw-eval-target"
WORKER_SCRIPT_PATH = Path(__file__).with_name("worker_run_case.py")


def load_env() -> None:
    load_dotenv(WORKSPACE_ENV, override=True)


def get_client() -> Client:
    load_env()
    return Client()


def _get_outputs(obj) -> dict[str, Any]:
    if hasattr(obj, "outputs"):
        return obj.outputs or {}
    if isinstance(obj, dict):
        return obj.get("outputs", {}) or {}
    return {}


def _get_inputs(obj) -> dict[str, Any]:
    if hasattr(obj, "inputs"):
        return obj.inputs or {}
    if isinstance(obj, dict):
        return obj.get("inputs", {}) or {}
    return {}


def _normalize_tool_names(names: Any) -> list[str]:
    if not isinstance(names, list):
        return []
    normalized = []
    for name in names:
        if isinstance(name, str) and name:
            normalized.append(name)
    return normalized


def resolve_examples(
    client: Client,
    *,
    dataset_name: str,
    example_ids: list[str] | None,
    example_limit: int | None,
    categories: list[str] | None,
    eval_modes: list[str] | None,
    behavior_tags: list[str] | None,
    side_effect_risks: list[str] | None,
):
    examples = list(client.list_examples(dataset_name=dataset_name, limit=200))
    if example_ids:
        by_id = {str(example.id): example for example in examples}
        missing = [example_id for example_id in example_ids if example_id not in by_id]
        if missing:
            raise ValueError(f"Example ids not found in dataset {dataset_name!r}: {missing}")
        selected = [by_id[example_id] for example_id in example_ids]
    else:
        selected = examples

    return filter_examples(
        selected,
        categories=categories,
        eval_modes=eval_modes,
        behavior_tags=behavior_tags,
        side_effect_risks=side_effect_risks,
        example_limit=example_limit,
    )


def git_metadata_for_repo(repo_path: str) -> dict[str, str | None]:
    def _run_git(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        value = completed.stdout.strip()
        return value or None

    branch = _run_git("branch", "--show-current")
    if branch is None:
        branch = _run_git("rev-parse", "--short", "HEAD")
    return {
        "branch": branch,
        "commit": _run_git("rev-parse", "HEAD"),
    }


def run_case(
    repo_path: str,
    user_text: str,
    model_name: str,
    trace_project: str,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(WORKER_SCRIPT_PATH),
            "--repo",
            repo_path,
            "--user-text",
            user_text,
            "--model",
            model_name,
            "--workspace-env",
            WORKSPACE_ENV,
            "--trace-project",
            trace_project,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker failed for repo={repo_path} user_text={user_text!r}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    stdout = completed.stdout.strip().splitlines()
    if not stdout:
        raise RuntimeError(f"Worker produced no stdout. stderr={completed.stderr}")
    return json.loads(stdout[-1])


def evaluator_tool_call(run, example):
    outputs = _get_outputs(run)
    expected = _get_outputs(example)
    required = bool(expected.get("requires_tool_call"))
    observed = bool(outputs.get("tool_calls_seen"))
    return {
        "key": "tool_call_required",
        "score": 1 if observed == required else 0,
        "comment": f"required={required} observed={observed}",
    }


def evaluator_expected_tool_names(run, example):
    outputs = _get_outputs(run)
    expected = _get_outputs(example)
    expected_names = _normalize_tool_names(expected.get("expected_tool_names"))
    if not expected_names:
        return {
            "key": "expected_tool_names",
            "score": None,
            "comment": "no expected tool names specified",
        }
    observed_names = _normalize_tool_names(outputs.get("tool_names"))
    matched = all(name in observed_names for name in expected_names)
    return {
        "key": "expected_tool_names",
        "score": 1 if matched else 0,
        "comment": f"expected={expected_names!r} observed={observed_names!r}",
    }


def evaluator_first_pass_tool_use(run, example):
    outputs = _get_outputs(run)
    expected = _get_outputs(example)
    required = bool(expected.get("requires_tool_call"))
    must_succeed_first_pass = bool(expected.get("must_succeed_first_pass"))
    if not required or not must_succeed_first_pass:
        return {
            "key": "first_pass_tool_use",
            "score": None,
            "comment": (
                "required="
                f"{required} must_succeed_first_pass={must_succeed_first_pass} first_pass_tool_calls_seen="
                f"{bool(outputs.get('first_pass_tool_calls_seen'))}"
            ),
        }
    first_pass_tool_calls_seen = bool(outputs.get("first_pass_tool_calls_seen"))
    return {
        "key": "first_pass_tool_use",
        "score": 1 if first_pass_tool_calls_seen else 0,
        "comment": (
            "required="
            f"{required} must_succeed_first_pass={must_succeed_first_pass} first_pass_tool_calls_seen={first_pass_tool_calls_seen}"
        ),
    }


def evaluator_secondary_tool_recovery(run, example):
    outputs = _get_outputs(run)
    expected = _get_outputs(example)
    required = bool(expected.get("requires_tool_call"))
    must_succeed_first_pass = bool(expected.get("must_succeed_first_pass"))
    first_pass_tool_calls_seen = bool(outputs.get("first_pass_tool_calls_seen"))
    retried = bool(outputs.get("retried"))
    eventual_tool_calls_seen = bool(outputs.get("tool_calls_seen"))

    if not required or not must_succeed_first_pass or first_pass_tool_calls_seen:
        return {
            "key": "secondary_tool_recovery",
            "score": None,
            "comment": (
                "required="
                f"{required} must_succeed_first_pass={must_succeed_first_pass} "
                f"first_pass_tool_calls_seen={first_pass_tool_calls_seen} retried={retried} "
                f"tool_calls_seen={eventual_tool_calls_seen}"
            ),
        }

    return {
        "key": "secondary_tool_recovery",
        "score": 1 if retried and eventual_tool_calls_seen else 0,
        "comment": (
            "required="
            f"{required} must_succeed_first_pass={must_succeed_first_pass} "
            f"first_pass_tool_calls_seen={first_pass_tool_calls_seen} retried={retried} "
            f"tool_calls_seen={eventual_tool_calls_seen}"
        ),
    }


def rollup_pass_fail(metrics: dict[str, Any], example_outputs: dict[str, Any]) -> dict[str, Any]:
    """Derive a per-example PASS/FAIL rollup from the metric set."""

    failure_reasons: list[str] = []

    required = bool(example_outputs.get("requires_tool_call"))
    expected_tool_names = _normalize_tool_names(example_outputs.get("expected_tool_names"))
    must_succeed_first_pass = bool(example_outputs.get("must_succeed_first_pass"))

    tool_call_required = metrics.get("tool_call_required")
    if tool_call_required == 0:
        if required:
            failure_reasons.append("missing required tool call")
        else:
            failure_reasons.append("used a tool when no tool call was expected")

    expected_tool_metric = metrics.get("expected_tool_names")
    if expected_tool_names and expected_tool_metric == 0:
        failure_reasons.append("wrong tool used")

    first_pass_metric = metrics.get("first_pass_tool_use")
    if required and must_succeed_first_pass and first_pass_metric == 0:
        failure_reasons.append("missed first-pass requirement")

    return {
        "pass_fail": "FAIL" if failure_reasons else "PASS",
        "failure_reasons": failure_reasons,
    }


def evaluator_overall_pass_fail(run, example):
    """Emit a binary overall pass/fail metric for LangSmith UI display."""
    metrics = {
        "tool_call_required": evaluator_tool_call(run, example).get("score"),
        "expected_tool_names": evaluator_expected_tool_names(run, example).get("score"),
        "first_pass_tool_use": evaluator_first_pass_tool_use(run, example).get("score"),
        "secondary_tool_recovery": evaluator_secondary_tool_recovery(run, example).get("score"),
    }
    example_outputs = _get_outputs(example)
    rollup = rollup_pass_fail(metrics, example_outputs)
    return {
        "key": "overall_pass_fail",
        "score": 1 if rollup["pass_fail"] == "PASS" else 0,
        "comment": "PASS"
        if not rollup["failure_reasons"]
        else f"FAIL: {'; '.join(rollup['failure_reasons'])}",
    }


def run_eval(
    *,
    client: Client,
    dataset_name,
    repo_path: str,
    experiment_prefix: str,
    model_name: str,
    baseline_commit: str,
    run_kind: str,
    trace_project: str,
):
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_case(
            repo_path=repo_path,
            user_text=inputs["user_text"],
            model_name=model_name,
            trace_project=trace_project,
        )

    git_metadata = git_metadata_for_repo(repo_path)
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[
            evaluator_tool_call,
            evaluator_expected_tool_names,
            evaluator_first_pass_tool_use,
            evaluator_secondary_tool_recovery,
            evaluator_overall_pass_fail,
        ],
        client=client,
        experiment_prefix=experiment_prefix,
        description=f"DeepClaw regression eval for {repo_path}",
        max_concurrency=1,
        blocking=True,
        metadata={
            "model": model_name,
            "repo_path": repo_path,
            "run_kind": run_kind,
            "baseline_commit": baseline_commit,
            "target_git_branch": git_metadata.get("branch"),
            "target_git_commit": git_metadata.get("commit"),
        },
    )
    rows = list(results)
    score_buckets: dict[str, list[float]] = defaultdict(list)
    examples = []
    for row in rows:
        if isinstance(row, dict):
            run_obj = row.get("run")
            example_obj = row.get("example")
            evaluation_results = row.get("evaluation_results") or {}
            eval_items = evaluation_results.get("results", [])
        else:
            run_obj = row.run
            example_obj = row.example
            evaluation_results = row.evaluation_results
            eval_items = evaluation_results.results

        metrics = {}
        for item in eval_items:
            if isinstance(item, dict):
                key = item.get("key")
                score = item.get("score")
            else:
                key = item.key
                score = item.score
            metrics[key] = score
            if score is not None:
                score_buckets[key].append(float(score))
        run_outputs = _get_outputs(run_obj)
        example_inputs = _get_inputs(example_obj)
        example_outputs = _get_outputs(example_obj)
        example_id = getattr(example_obj, "id", None) or (
            example_obj.get("id") if isinstance(example_obj, dict) else None
        )
        examples.append(
            {
                "example_id": str(example_id),
                "user_text": example_inputs.get("user_text"),
                "category": example_outputs.get("category"),
                "metrics": metrics,
                **rollup_pass_fail(metrics, example_outputs),
                "tool_calls_seen": run_outputs.get("tool_calls_seen"),
                "tool_names": run_outputs.get("tool_names"),
                "retried": run_outputs.get("retried"),
                "attempts": run_outputs.get("attempts"),
                "first_pass_tool_calls_seen": run_outputs.get("first_pass_tool_calls_seen"),
                "first_pass_text": run_outputs.get("first_pass_text"),
            }
        )
    summary = {
        key: (sum(values) / len(values) if values else None)
        for key, values in score_buckets.items()
    }
    return {
        "experiment_name": results.experiment_name,
        "experiment_url": str(results.url),
        "comparison_url": str(results.comparison_url) if results.comparison_url else None,
        "summary": summary,
        "examples": examples,
    }


def ensure_baseline_worktree(commitish: str, path: str, source_repo: str) -> str:
    worktree = Path(path)
    if worktree.exists():
        subprocess.run(["git", "fetch", "origin"], cwd=source_repo, check=True)
        subprocess.run(["git", "checkout", "--detach", commitish], cwd=worktree, check=True)
        subprocess.run(["git", "reset", "--hard", commitish], cwd=worktree, check=True)
        subprocess.run(["git", "clean", "-fd"], cwd=worktree, check=True)
        return str(worktree)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), commitish],
        cwd=source_repo,
        check=True,
    )
    return str(worktree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--baseline-commit", default="origin/main")
    parser.add_argument("--baseline-worktree", default="/tmp/deepclaw-eval-baseline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trace-project", default=DEFAULT_TRACE_PROJECT)
    parser.add_argument("--example-id", dest="example_ids", action="append")
    parser.add_argument("--example-limit", type=int)
    parser.add_argument("--category", dest="categories", action="append")
    parser.add_argument("--eval-mode", dest="eval_modes", action="append")
    parser.add_argument("--behavior-tag", dest="behavior_tags", action="append")
    parser.add_argument("--side-effect-risk", dest="side_effect_risks", action="append")
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    client = get_client()
    data = resolve_examples(
        client,
        dataset_name=args.dataset,
        example_ids=args.example_ids,
        example_limit=args.example_limit,
        categories=args.categories,
        eval_modes=args.eval_modes,
        behavior_tags=args.behavior_tags,
        side_effect_risks=args.side_effect_risks,
    )
    baseline_repo = ensure_baseline_worktree(
        args.baseline_commit, args.baseline_worktree, args.repo
    )
    baseline = run_eval(
        client=client,
        dataset_name=data,
        repo_path=baseline_repo,
        experiment_prefix="deepclaw-pre",
        model_name=args.model,
        baseline_commit=args.baseline_commit,
        run_kind="baseline",
        trace_project=args.trace_project,
    )
    post = run_eval(
        client=client,
        dataset_name=data,
        repo_path=args.repo,
        experiment_prefix="deepclaw-post",
        model_name=args.model,
        baseline_commit=args.baseline_commit,
        run_kind="post",
        trace_project=args.trace_project,
    )
    result_payload = {"baseline": baseline, "post": post}
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    sys.stdout.write(json.dumps(result_payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
