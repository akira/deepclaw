#!/usr/bin/env python3
"""Run LangSmith pairwise comparisons between two DeepClaw experiments/models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate, evaluate_comparative

REPO_ROOT = Path(__file__).resolve().parents[1]
_REGRESSION_PATH = REPO_ROOT / "evals" / "langsmith_regression.py"
_REGRESSION_SPEC = importlib.util.spec_from_file_location(
    "deepclaw_langsmith_regression", _REGRESSION_PATH
)
if _REGRESSION_SPEC is None or _REGRESSION_SPEC.loader is None:
    raise RuntimeError(f"Unable to load regression module from {_REGRESSION_PATH}")
_REGRESSION_MODULE = importlib.util.module_from_spec(_REGRESSION_SPEC)
_REGRESSION_SPEC.loader.exec_module(_REGRESSION_MODULE)

WORKSPACE_ENV = _REGRESSION_MODULE.WORKSPACE_ENV
run_case = _REGRESSION_MODULE.run_case
resolve_examples = _REGRESSION_MODULE.resolve_examples
EvaluatorToolCall = _REGRESSION_MODULE.evaluator_tool_call
EvaluatorExpectedToolNames = _REGRESSION_MODULE.evaluator_expected_tool_names
EvaluatorFirstPassToolUse = _REGRESSION_MODULE.evaluator_first_pass_tool_use
EvaluatorSecondaryToolRecovery = _REGRESSION_MODULE.evaluator_secondary_tool_recovery
EvaluatorFinalAnswerQualityJudge = _REGRESSION_MODULE.evaluator_final_answer_quality_judge
EvaluatorOverallPassFail = _REGRESSION_MODULE.evaluator_overall_pass_fail

DEFAULT_DATASET = "deepclaw"
DEFAULT_REPO = str(REPO_ROOT)
DEFAULT_MODEL_A = "anthropic:claude-haiku-4-5"
DEFAULT_MODEL_B = "openai:gpt-4o-mini"
DEFAULT_RESULTS_PATH = "/tmp/deepclaw-evals/pairwise-results.json"
DEFAULT_TRACE_PROJECT = _REGRESSION_MODULE.DEFAULT_TRACE_PROJECT
_DEFAULT_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"


def _ensure_valid_ca_bundle() -> None:
    for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        configured = os.environ.get(env_var)
        if configured and Path(configured).exists():
            continue
        if Path(_DEFAULT_CA_BUNDLE).exists():
            os.environ[env_var] = _DEFAULT_CA_BUNDLE


def load_env() -> None:
    _ensure_valid_ca_bundle()
    load_dotenv(WORKSPACE_ENV, override=True)
    _ensure_valid_ca_bundle()


def get_client() -> Client:
    load_env()
    return Client()


def _normalize_tool_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def make_target(repo_path: str, model_name: str, trace_project: str):
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_case(
            repo_path=repo_path,
            user_text=inputs["user_text"],
            model_name=model_name,
            trace_project=trace_project,
        )

    return target


def pairwise_tool_use_preference(runs, example):
    if len(runs) != 2:
        raise ValueError(f"Expected exactly 2 runs, got {len(runs)}")

    example_outputs = example.outputs or {} if example else {}
    expected_tools = _normalize_tool_names(example_outputs.get("expected_tool_names"))

    def features(run) -> dict[str, Any]:
        outputs = run.outputs or {}
        observed_tools = _normalize_tool_names(outputs.get("tool_names"))
        expected_match = bool(expected_tools) and all(
            tool_name in observed_tools for tool_name in expected_tools
        )
        overall = EvaluatorOverallPassFail(run, example)
        return {
            "overall_pass": bool(overall.get("score")),
            "overall_comment": overall.get("comment"),
            "first_pass": bool(outputs.get("first_pass_tool_calls_seen")),
            "expected_match": expected_match,
            "eventual": bool(outputs.get("tool_calls_seen")),
            "rescued": bool(outputs.get("retried")) and bool(outputs.get("tool_calls_seen")),
            "attempts": int(outputs.get("attempts") or 0),
            "tool_names": observed_tools,
        }

    left = features(runs[0])
    right = features(runs[1])
    left_rank = (
        1 if left["overall_pass"] else 0,
        1 if left["first_pass"] else 0,
        1 if left["expected_match"] else 0,
        1 if left["eventual"] else 0,
        1 if left["rescued"] else 0,
        -left["attempts"],
    )
    right_rank = (
        1 if right["overall_pass"] else 0,
        1 if right["first_pass"] else 0,
        1 if right["expected_match"] else 0,
        1 if right["eventual"] else 0,
        1 if right["rescued"] else 0,
        -right["attempts"],
    )

    if left_rank > right_rank:
        return {
            "key": "pairwise_tool_use_preference",
            "scores": {runs[0].id: 1, runs[1].id: 0},
            "comment": f"A preferred: A={left} B={right}",
        }
    if right_rank > left_rank:
        return {
            "key": "pairwise_tool_use_preference",
            "scores": {runs[0].id: 0, runs[1].id: 1},
            "comment": f"B preferred: A={left} B={right}",
        }
    return {
        "key": "pairwise_tool_use_preference",
        "scores": {runs[0].id: 0.5, runs[1].id: 0.5},
        "comment": f"Tie: A={left} B={right}",
    }


def summarize_pairwise_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    left_wins = 0
    right_wins = 0
    ties = 0
    total = 0

    for row in rows:
        eval_result = row.get("evaluation_results") or {}
        items = eval_result if isinstance(eval_result, list) else [eval_result]
        if not items:
            continue
        scores = list((items[0].get("scores") or {}).values())
        total += 1
        if scores == [1, 0] or scores == [1.0, 0.0]:
            left_wins += 1
        elif scores == [0, 1] or scores == [0.0, 1.0]:
            right_wins += 1
        else:
            ties += 1

    return {
        "model_a_wins": left_wins,
        "model_b_wins": right_wins,
        "ties": ties,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
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
    scalar_evaluators = [
        EvaluatorToolCall,
        EvaluatorExpectedToolNames,
        EvaluatorFirstPassToolUse,
        EvaluatorSecondaryToolRecovery,
        EvaluatorFinalAnswerQualityJudge,
        EvaluatorOverallPassFail,
    ]

    results_a = evaluate(
        make_target(args.repo, args.model_a, args.trace_project),
        data=data,
        evaluators=scalar_evaluators,
        client=client,
        experiment_prefix="deepclaw-pairwise-a",
        description=f"Source run for pairwise comparison: {args.model_a}",
        metadata={
            "model": args.model_a,
            "pairwise_source": True,
            "dataset": args.dataset,
            "example_ids": args.example_ids,
            "example_limit": args.example_limit,
        },
        max_concurrency=1,
        blocking=True,
    )

    results_b = evaluate(
        make_target(args.repo, args.model_b, args.trace_project),
        data=data,
        evaluators=scalar_evaluators,
        client=client,
        experiment_prefix="deepclaw-pairwise-b",
        description=f"Source run for pairwise comparison: {args.model_b}",
        metadata={
            "model": args.model_b,
            "pairwise_source": True,
            "dataset": args.dataset,
            "example_ids": args.example_ids,
            "example_limit": args.example_limit,
        },
        max_concurrency=1,
        blocking=True,
    )

    comparative = evaluate_comparative(
        (results_a.experiment_id, results_b.experiment_id),
        evaluators=[pairwise_tool_use_preference],
        client=client,
        experiment_prefix="deepclaw-pairwise",
        description="Pairwise comparison of two DeepClaw experiments",
        metadata={
            "dataset": args.dataset,
            "model_a": args.model_a,
            "model_b": args.model_b,
            "example_ids": args.example_ids,
            "example_limit": args.example_limit,
        },
        randomize_order=False,
    )

    rows = list(comparative)
    payload = {
        "dataset": args.dataset,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "experiment_a": {
            "id": str(results_a.experiment_id),
            "name": results_a.experiment_name,
            "url": results_a.url,
        },
        "experiment_b": {
            "id": str(results_b.experiment_id),
            "name": results_b.experiment_name,
            "url": results_b.url,
        },
        "pairwise_summary": summarize_pairwise_rows(rows),
        "rows": rows,
    }

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
