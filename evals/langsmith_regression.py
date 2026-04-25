#!/usr/bin/env python3
"""Evaluate the LangSmith "deepclaw" regression dataset against DeepClaw revisions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

WORKSPACE_ENV = "/home/ubuntu/.deepclaw/.env"
DEFAULT_DATASET = "deepclaw"
DEFAULT_MODEL = "openai:gpt-5.3-codex"
DEFAULT_RESULTS_PATH = "/tmp/deepclaw-evals/results.json"
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


def run_case(repo_path: str, user_text: str, model_name: str) -> dict[str, Any]:
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
            "score": 1,
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
            "score": 1,
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


def run_eval(
    client: Client, dataset_name: str, repo_path: str, experiment_prefix: str, model_name: str
):
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_case(repo_path=repo_path, user_text=inputs["user_text"], model_name=model_name)

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[
            evaluator_tool_call,
            evaluator_expected_tool_names,
            evaluator_first_pass_tool_use,
        ],
        client=client,
        experiment_prefix=experiment_prefix,
        description=f"DeepClaw regression eval for {repo_path}",
        max_concurrency=1,
        blocking=True,
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
                "tool_calls_seen": run_outputs.get("tool_calls_seen"),
                "tool_names": run_outputs.get("tool_names"),
                "retried": run_outputs.get("retried"),
                "attempts": run_outputs.get("attempts"),
                "first_pass_tool_calls_seen": run_outputs.get("first_pass_tool_calls_seen"),
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
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    client = get_client()
    baseline_repo = ensure_baseline_worktree(
        args.baseline_commit, args.baseline_worktree, args.repo
    )
    baseline = run_eval(
        client=client,
        dataset_name=args.dataset,
        repo_path=baseline_repo,
        experiment_prefix="deepclaw-pre",
        model_name=args.model,
    )
    post = run_eval(
        client=client,
        dataset_name=args.dataset,
        repo_path=args.repo,
        experiment_prefix="deepclaw-post",
        model_name=args.model,
    )
    result_payload = {"baseline": baseline, "post": post}
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    sys.stdout.write(json.dumps(result_payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
