#!/usr/bin/env python3
"""Create and run a LangSmith compaction regression dataset for DeepClaw."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import Client, schemas
from langsmith.evaluation import evaluate

WORKSPACE_ENV = str(Path("~/.deepclaw/.env").expanduser())
DEFAULT_DATASET = "deepclaw-compaction-regressions"
DEFAULT_EXPERIMENT_PREFIX = "deepclaw-compaction"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

COMPACTION_WRAPPER_TEXT = (
    "You are in the middle of a conversation that has been summarized. "
    "The full conversation history has been saved to /conversation_history/. "
    "A condensed summary follows."
)

DATASET_EXAMPLES = [
    {
        "inputs": {
            "case_id": "placeholder_summary_replaced",
            "raw_summary": "Previous conversation was too long to summarize.",
            "messages": [
                {
                    "role": "user",
                    "content": "Please fix context bloat in DeepClaw and keep compact_conversation available.",
                },
                {
                    "role": "assistant",
                    "content": "I switched to DeepAgents middleware and reviewed the trace evidence.",
                },
            ],
        },
        "outputs": {
            "label": "placeholder-summary-fallback",
            "should_fallback": True,
            "expected_active_task_substrings": [
                "context bloat in DeepClaw",
                "compact_conversation",
            ],
            "expected_goal_substrings": [
                "Please fix context bloat in DeepClaw",
            ],
            "required_constraint_substrings": [],
            "forbidden_substrings": [
                "Continue the user's in-progress objective",
                "Continue from the latest user request",
                "Previous conversation was too long to summarize",
            ],
            "judge_rubric": (
                "The normalized summary must preserve the concrete debugging task and avoid generic filler."
            ),
            "category": "fallback",
        },
        "metadata": {
            "source": "unit-test-style synthetic regression",
            "notes": "Covers bad placeholder summary replacement.",
        },
    },
    {
        "inputs": {
            "case_id": "generic_structured_summary_loses_task",
            "raw_summary": "## Active Task\nNone\n\n## Goal\nContinue the user's in-progress objective without losing critical prior context.\n\n## Constraints & Preferences\n- Preserve prior decisions.\n\n## Completed Actions\n- Cloned the repo.\n\n## Active State\n- Repo available locally.\n\n## Next Steps\n- Continue from the latest user request.\n",
            "messages": [
                {
                    "role": "user",
                    "content": "Clone the OpenSWE repo and prepare a detailed adaptation report for DeepClaw.",
                },
                {
                    "role": "assistant",
                    "content": "I cloned the repo and started reading the prompt and system files.",
                },
            ],
        },
        "outputs": {
            "label": "generic-structured-summary-fallback",
            "should_fallback": True,
            "expected_active_task_substrings": [
                "Clone the OpenSWE repo",
                "adaptation report",
            ],
            "expected_goal_substrings": [
                "Clone the OpenSWE repo",
            ],
            "required_constraint_substrings": [],
            "forbidden_substrings": [
                "Continue the user's in-progress objective",
                "Continue from the latest user request",
                "## Active Task\nNone",
            ],
            "judge_rubric": (
                "The normalized summary must reconstruct the concrete repo-analysis task rather than preserve generic placeholders."
            ),
            "category": "fallback",
        },
        "metadata": {
            "source": "unit-test-style synthetic regression",
            "notes": "Covers placeholder sections inside an otherwise structured summary.",
        },
    },
    {
        "inputs": {
            "case_id": "trace_wrapper_summary_must_be_rejected",
            "raw_summary": "## Active Task\nYou are in the middle of a conversation that has been summarized. The full conversation history has been saved to /conversation_history/. A condensed summary follows.\n\n## Goal\nYou are in the middle of a conversation that has been summarized. Continue carefully.\n\n## Constraints & Preferences\n- Preserve key context.\n\n## Completed Actions\n- Read several DeepClaw files.\n\n## Active State\n- Coder subagent is active.\n\n## Next Steps\n- You are in the middle of a conversation that has been summarized.\n",
            "messages": [
                {
                    "role": "user",
                    "content": "Can you examine the deepclaw codebase for any refactoring improvement additions or removals. DO NOT CHANGE ANY CODE. Only come back with findings",
                },
                {
                    "role": "assistant",
                    "content": "I'll inspect the codebase and report findings without modifying files.",
                },
            ],
        },
        "outputs": {
            "label": "trace-wrapper-summary-rejected",
            "should_fallback": True,
            "expected_active_task_substrings": [
                "examine the deepclaw codebase",
            ],
            "expected_goal_substrings": [
                "Only come back with findings",
            ],
            "required_constraint_substrings": [
                "DO NOT CHANGE ANY CODE",
            ],
            "forbidden_substrings": [
                "You are in the middle of a conversation that has been summarized",
                "/conversation_history/",
            ],
            "judge_rubric": (
                "The normalized summary must preserve the explicit read-only constraint and must not surface compaction wrapper boilerplate as the task."
            ),
            "category": "trace-derived",
        },
        "metadata": {
            "source": "observed trace",
            "source_trace_id": "019de3dd-d90c-74c3-bda2-270f258ab4ab",
            "notes": "Observed production failure where compaction wrapper text poisoned task handoff.",
        },
    },
    {
        "inputs": {
            "case_id": "fallback_must_ignore_latest_wrapper_human_message",
            "raw_summary": "Previous conversation was too long to summarize.",
            "messages": [
                {
                    "role": "user",
                    "content": "Can you examine the deepclaw codebase for any refactoring improvement additions or removals. DO NOT CHANGE ANY CODE. Only come back with findings",
                },
                {
                    "role": "assistant",
                    "content": "I'll inspect the codebase and report findings without modifying files.",
                },
                {
                    "role": "user",
                    "content": COMPACTION_WRAPPER_TEXT,
                },
            ],
        },
        "outputs": {
            "label": "fallback-ignores-wrapper-human-message",
            "should_fallback": True,
            "expected_active_task_substrings": [
                "examine the deepclaw codebase",
            ],
            "expected_goal_substrings": [
                "Only come back with findings",
            ],
            "required_constraint_substrings": [
                "DO NOT CHANGE ANY CODE",
            ],
            "forbidden_substrings": [
                "You are in the middle of a conversation that has been summarized",
                "A condensed summary follows",
            ],
            "judge_rubric": (
                "The fallback summary must derive the task from the real user request, not from a synthetic compaction handoff message appended later."
            ),
            "category": "fallback",
        },
        "metadata": {
            "source": "observed failure mode",
            "source_trace_id": "019de3dd-d90c-74c3-bda2-270f258ab4ab",
            "notes": "Directly targets latest-user poisoning in fallback reconstruction.",
        },
    },
    {
        "inputs": {
            "case_id": "useful_none_text_kept",
            "raw_summary": "## Active Task\nAudit the branch and explain whether none of the commits regress compaction.\n\n## Goal\nDetermine whether the compaction fix is safe to merge.\n\n## Constraints & Preferences\n- None beyond preserving the current branch state.\n\n## Completed Actions\n- Compared the current branch against origin/main.\n\n## Active State\n- Branch is fix/compaction-task-handoff.\n\n## Next Steps\n- Summarize findings and prepare the PR description.\n",
            "messages": [
                {
                    "role": "user",
                    "content": "Audit the branch and explain whether none of the commits regress compaction.",
                }
            ],
        },
        "outputs": {
            "label": "useful-none-kept",
            "should_fallback": False,
            "expected_active_task_substrings": [
                "none of the commits regress compaction",
            ],
            "expected_goal_substrings": [
                "compaction fix is safe to merge",
            ],
            "required_constraint_substrings": [],
            "forbidden_substrings": [
                "Continue the user's in-progress objective",
                "Continue from the latest user request",
            ],
            "judge_rubric": (
                "The summary should be preserved because 'none' appears as meaningful natural language, not as a placeholder."
            ),
            "category": "non-regression",
        },
        "metadata": {
            "source": "unit-test-style synthetic regression",
            "notes": "Guards against over-aggressive fallback when 'none' is semantic content.",
        },
    },
]


def load_env() -> None:
    load_dotenv(WORKSPACE_ENV, override=True)


def get_client() -> Client:
    load_env()
    return Client()


def _message_from_dict(payload: dict[str, str]):
    role = str(payload.get("role") or "user").lower()
    content = str(payload.get("content") or "")
    if role in {"assistant", "ai"}:
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


def _get_inputs(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "inputs"):
        return obj.inputs or {}
    if isinstance(obj, dict):
        return obj.get("inputs", {}) or {}
    return {}


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_outputs(obj: Any) -> dict[str, Any]:
    outputs = _get_field(obj, "outputs", {})
    return outputs or {}


def _summary_section_text(summary: str, heading: str) -> str:
    from deepclaw import agent as agent_mod

    return agent_mod._summary_section_text(summary, heading)


def run_case(inputs: dict[str, Any]) -> dict[str, Any]:
    from deepclaw import agent as agent_mod

    raw_summary = str(inputs.get("raw_summary") or "")
    messages = [_message_from_dict(item) for item in inputs.get("messages", [])]
    fallback_summary = agent_mod._build_compaction_fallback_summary(messages)
    normalized_summary = agent_mod._normalize_compaction_summary(raw_summary, messages)

    return {
        "case_id": inputs.get("case_id"),
        "raw_summary": raw_summary,
        "normalized_summary": normalized_summary,
        "used_fallback": normalized_summary.strip() == fallback_summary.strip(),
        "active_task": _summary_section_text(normalized_summary, "Active Task"),
        "goal": _summary_section_text(normalized_summary, "Goal"),
        "constraints": _summary_section_text(normalized_summary, "Constraints & Preferences"),
        "completed_actions": _summary_section_text(normalized_summary, "Completed Actions"),
        "active_state": _summary_section_text(normalized_summary, "Active State"),
        "next_steps": _summary_section_text(normalized_summary, "Next Steps"),
    }


def evaluator_should_fallback(run, example):
    observed = bool(_get_outputs(run).get("used_fallback"))
    expected = bool(_get_outputs(example).get("should_fallback"))
    return {
        "key": "should_fallback",
        "score": 1 if observed == expected else 0,
        "comment": f"expected={expected} observed={observed}",
    }


def _contains_all(text: str, parts: list[str]) -> tuple[bool, list[str]]:
    missing = [part for part in parts if part not in text]
    return (not missing), missing


def evaluator_active_task(run, example):
    observed = str(_get_outputs(run).get("active_task") or "")
    expected_parts = [
        str(part)
        for part in _get_outputs(example).get("expected_active_task_substrings", [])
        if str(part)
    ]
    ok, missing = _contains_all(observed, expected_parts)
    return {
        "key": "active_task_preserved",
        "score": 1 if ok else 0,
        "comment": f"missing={missing} observed={observed!r}",
    }


def evaluator_goal(run, example):
    observed = str(_get_outputs(run).get("goal") or "")
    expected_parts = [
        str(part) for part in _get_outputs(example).get("expected_goal_substrings", []) if str(part)
    ]
    ok, missing = _contains_all(observed, expected_parts)
    return {
        "key": "goal_preserved",
        "score": 1 if ok else 0,
        "comment": f"missing={missing} observed={observed!r}",
    }


def evaluator_constraints(run, example):
    observed = str(_get_outputs(run).get("normalized_summary") or "")
    required = [
        str(part)
        for part in _get_outputs(example).get("required_constraint_substrings", [])
        if str(part)
    ]
    ok, missing = _contains_all(observed, required)
    return {
        "key": "constraints_preserved",
        "score": 1 if ok else 0,
        "comment": f"missing={missing}",
    }


def evaluator_forbidden_substrings(run, example):
    observed = str(_get_outputs(run).get("normalized_summary") or "")
    forbidden = [
        str(part) for part in _get_outputs(example).get("forbidden_substrings", []) if str(part)
    ]
    present = [part for part in forbidden if part in observed]
    return {
        "key": "forbidden_substrings_absent",
        "score": 1 if not present else 0,
        "comment": f"present={present}",
    }


def _judge_client_and_model() -> tuple[object | None, str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError:
        return None, None
    return OpenAI(api_key=api_key), os.getenv("DEEPCLOW_EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def evaluator_llm_judge(run, example):
    client, model = _judge_client_and_model()
    if client is None or model is None:
        return {
            "key": "llm_judge",
            "score": None,
            "comment": "judge unavailable (missing OPENAI_API_KEY or openai package)",
        }

    run_outputs = _get_outputs(run)
    example_inputs = _get_inputs(example)
    example_outputs = _get_outputs(example)
    prompt = (
        "You are grading whether a normalized compaction summary preserved the real task and constraints. "
        "Return strict JSON with keys score:number and comment:string. Use score 1 for pass and 0 for fail.\n\n"
        f"Case ID: {example_inputs.get('case_id')}\n\n"
        f"Original messages:\n{json.dumps(example_inputs.get('messages', []), indent=2)}\n\n"
        f"Raw summary:\n{example_inputs.get('raw_summary', '')}\n\n"
        f"Normalized summary:\n{run_outputs.get('normalized_summary', '')}\n\n"
        f"Rubric:\n{example_outputs.get('judge_rubric', '')}\n\n"
        f"Required task substrings: {example_outputs.get('expected_active_task_substrings', [])}\n"
        f"Required goal substrings: {example_outputs.get('expected_goal_substrings', [])}\n"
        f"Required constraints: {example_outputs.get('required_constraint_substrings', [])}\n"
        f"Forbidden substrings: {example_outputs.get('forbidden_substrings', [])}\n"
    )
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Judge strictly. Return only JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    try:
        score = 1 if float(parsed.get("score", 0)) >= 0.5 else 0
    except Exception:
        score = 0
    return {
        "key": "llm_judge",
        "score": score,
        "comment": str(parsed.get("comment") or "llm judge verdict"),
    }


def evaluator_overall_pass_fail(run, _example):
    outputs = _get_outputs(run)
    normalized_summary = str(outputs.get("normalized_summary") or "")
    if not normalized_summary.strip():
        return {
            "key": "overall_pass_fail",
            "score": 0,
            "comment": "normalized summary was empty",
        }
    return {
        "key": "overall_pass_fail",
        "score": 1,
        "comment": "non-empty normalized summary produced; see other metrics for semantic checks",
    }


def ensure_dataset(client: Client, dataset_name: str) -> dict[str, Any]:
    existing = next(iter(client.list_datasets(dataset_name=dataset_name, limit=1)), None)
    if existing is None:
        dataset = client.create_dataset(
            dataset_name,
            description="Compaction regression cases for DeepClaw summary normalization and fallback behavior.",
            data_type=schemas.DataType.kv,
        )
    else:
        dataset = existing

    existing_examples = list(client.list_examples(dataset_name=dataset_name, limit=200))
    existing_case_ids = {
        str(_get_inputs(example).get("case_id"))
        for example in existing_examples
        if _get_inputs(example).get("case_id") is not None
    }
    new_examples = [
        example
        for example in DATASET_EXAMPLES
        if example["inputs"]["case_id"] not in existing_case_ids
    ]
    if new_examples:
        client.create_examples(dataset_id=dataset.id, examples=new_examples)

    return {
        "dataset_id": str(dataset.id),
        "dataset_name": getattr(dataset, "name", dataset_name),
        "created_examples": len(new_examples),
        "total_examples": len(existing_examples) + len(new_examples),
    }


def run_eval(*, client: Client, dataset_name: str, experiment_prefix: str) -> dict[str, Any]:
    results = evaluate(
        run_case,
        data=dataset_name,
        evaluators=[
            evaluator_should_fallback,
            evaluator_active_task,
            evaluator_goal,
            evaluator_constraints,
            evaluator_forbidden_substrings,
            evaluator_llm_judge,
            evaluator_overall_pass_fail,
        ],
        client=client,
        experiment_prefix=experiment_prefix,
        description="DeepClaw compaction regression eval",
        max_concurrency=1,
        blocking=True,
        metadata={
            "dataset": dataset_name,
            "eval_kind": "compaction_regression",
        },
    )

    rows = []
    metric_totals: dict[str, list[float]] = {}
    for row in results:
        run_outputs = _get_outputs(_get_field(row, "run", {}))
        example_inputs = _get_inputs(_get_field(row, "example", {}))
        metrics = {}
        evaluation_results = _get_field(row, "evaluation_results", {})
        for metric in _get_field(evaluation_results, "results", []) or []:
            key = _get_field(metric, "key")
            score = _get_field(metric, "score")
            metrics[key] = score
            if isinstance(score, (int, float)):
                metric_totals.setdefault(str(key), []).append(float(score))
        rows.append(
            {
                "case_id": example_inputs.get("case_id"),
                "used_fallback": run_outputs.get("used_fallback"),
                "active_task": run_outputs.get("active_task"),
                "goal": run_outputs.get("goal"),
                "metrics": metrics,
            }
        )

    summary = {
        key: round(sum(values) / len(values), 3)
        for key, values in sorted(metric_totals.items())
        if values
    }
    return {
        "experiment_name": getattr(results, "experiment_name", None),
        "url": getattr(results, "url", None),
        "summary": summary,
        "examples": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-dataset")
    create.add_argument("--dataset", default=DEFAULT_DATASET)

    run = subparsers.add_parser("run")
    run.add_argument("--dataset", default=DEFAULT_DATASET)
    run.add_argument("--experiment-prefix", default=DEFAULT_EXPERIMENT_PREFIX)
    run.add_argument("--ensure-dataset", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = get_client()
    if args.command == "create-dataset":
        result = ensure_dataset(client, args.dataset)
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return
    if args.ensure_dataset:
        ensure_dataset(client, args.dataset)
    result = run_eval(
        client=client, dataset_name=args.dataset, experiment_prefix=args.experiment_prefix
    )
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
