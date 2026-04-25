#!/usr/bin/env python3
"""Populate and evaluate the LangSmith "deepclaw" regression dataset.

Uses selected DeepClaw traces as source examples, then runs the current or baseline
DeepClaw code against those prompts and scores whether tool use happened.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client, schemas
from langsmith.evaluation import evaluate

WORKSPACE_ENV = "/home/ubuntu/.deepclaw/.env"
DEFAULT_DATASET = "deepclaw"
DEFAULT_PROJECT = "default"
DEFAULT_MODEL = "openai:gpt-5.3-codex"
DEFAULT_RESULTS_PATH = "/tmp/deepclaw-evals/results.json"
DEFAULT_SEED_TRACES_PATH = str(Path(__file__).with_name("datasets") / "deepclaw_seed_traces.json")

WORKER_TEMPLATE = r"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

repo_path = {repo_path!r}
user_text = {user_text!r}
model_name = {model_name!r}
workspace_env = {workspace_env!r}

home_dir = tempfile.mkdtemp(prefix="deepclaw-eval-home-")
os.environ["HOME"] = home_dir
os.environ.setdefault("LANGSMITH_TRACING", "false")
load_dotenv(workspace_env, override=True)

repo_root = Path(repo_path).resolve()
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))

from deepclaw.agent import create_agent, create_checkpointer
from deepclaw.config import DeepClawConfig
from deepclaw import gateway as gateway_mod
from langchain_core.messages import ToolMessage

_NUDGE_MESSAGE = getattr(
    gateway_mod,
    "_NUDGE_MESSAGE",
    "You described an action but did not call any tools. Please call the appropriate tool now to carry out what you described.",
)
_looks_like_narration = getattr(gateway_mod, "_looks_like_narration", lambda _text: False)
_looks_like_false_completion = getattr(
    gateway_mod, "_looks_like_false_completion", lambda _user, _assistant: False
)
_looks_like_memory_request = getattr(
    gateway_mod, "_looks_like_memory_request", lambda _user, _assistant: False
)


async def run_once(agent, thread_id: str, input_messages: list[dict]):
    tool_calls_seen = False
    tool_names = []
    accumulated = ""
    async for chunk in agent.astream(
        {{"messages": input_messages}},
        config={{"configurable": {{"thread_id": thread_id}}}},
        stream_mode="messages",
    ):
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            continue
        message_obj, _metadata = chunk
        if isinstance(message_obj, ToolMessage):
            continue
        if not hasattr(message_obj, "content_blocks"):
            continue
        for block in message_obj.content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in ("tool_call", "tool_call_chunk"):
                tool_calls_seen = True
                name = block.get("name")
                if name:
                    tool_names.append(name)
            elif block_type == "text":
                accumulated += block.get("text", "")
    return {{
        "tool_calls_seen": tool_calls_seen,
        "tool_names": tool_names,
        "text": accumulated,
    }}


async def main():
    config = DeepClawConfig(model=model_name, workspace_root=str(repo_root))
    async with create_checkpointer() as checkpointer:
        agent = create_agent(config, checkpointer)
        thread_id = f"eval-{{uuid.uuid4()}}"
        first = await run_once(agent, thread_id, [{{"role": "user", "content": user_text}}])
        attempts = 1
        retried = False
        final = first
        if (not first["tool_calls_seen"]) and (
            _looks_like_narration(first["text"])
            or _looks_like_false_completion(user_text, first["text"])
            or _looks_like_memory_request(user_text, first["text"])
        ):
            retried = True
            attempts += 1
            second = await run_once(agent, thread_id, [{{"role": "user", "content": _NUDGE_MESSAGE}}])
            final = {{
                "tool_calls_seen": second["tool_calls_seen"],
                "tool_names": first["tool_names"] + second["tool_names"],
                "text": second["text"],
            }}
        print(json.dumps({{
            "final_text": final["text"],
            "tool_calls_seen": final["tool_calls_seen"] or first["tool_calls_seen"],
            "tool_names": final["tool_names"],
            "attempts": attempts,
            "retried": retried,
            "first_pass_tool_calls_seen": first["tool_calls_seen"],
            "first_pass_text": first["text"],
        }}))

asyncio.run(main())
"""


def load_env() -> None:
    load_dotenv(WORKSPACE_ENV, override=True)


def get_client() -> Client:
    load_env()
    return Client()


def ensure_dataset(client: Client, dataset_name: str) -> schemas.Dataset:
    datasets = list(client.list_datasets(dataset_name=dataset_name, limit=5))
    if datasets:
        return datasets[0]
    return client.create_dataset(
        dataset_name,
        description="DeepClaw regression dataset sourced from real LangSmith traces.",
        data_type=schemas.DataType.kv,
        metadata={"project": "deepclaw", "purpose": "trace-sourced-regression-evals"},
    )


def get_root_run(client: Client, trace_id: str, project_name: str):
    roots = list(
        client.list_runs(project_name=project_name, trace_id=trace_id, is_root=True, limit=5)
    )
    if not roots:
        raise RuntimeError(f"No root run found for trace {trace_id}")
    return roots[0]


def get_trace_stats(client: Client, trace_id: str, project_name: str) -> dict[str, Any]:
    runs = list(client.list_runs(project_name=project_name, trace_id=trace_id, limit=100))
    tool_runs = [run for run in runs if run.run_type == "tool"]
    llm_runs = [run for run in runs if run.run_type == "llm"]
    return {
        "observed_tool_runs": len(tool_runs),
        "observed_tool_names": sorted({run.name for run in tool_runs}),
        "observed_llm_runs": len(llm_runs),
        "observed_run_count": len(runs),
    }


def latest_ai_text(root_run) -> str:
    outputs = root_run.outputs or {}
    messages = outputs.get("messages") or []
    for msg in reversed(messages):
        if msg.get("type") != "ai":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(part for part in parts if part)
    return ""


def load_seed_traces(seed_traces_path: str) -> list[dict[str, Any]]:
    path = Path(seed_traces_path)
    if not path.is_file():
        raise FileNotFoundError(f"Seed traces file not found: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"Seed traces file must contain a JSON list: {path}")
    return loaded


def list_existing_trace_ids(client: Client, dataset_id: str) -> set[str]:
    trace_ids: set[str] = set()
    offset = 0
    page_size = 100
    while True:
        examples = list(client.list_examples(dataset_id=dataset_id, offset=offset, limit=page_size))
        if not examples:
            break
        for example in examples:
            trace_id = (example.metadata or {}).get("source_trace_id")
            if trace_id:
                trace_ids.add(trace_id)
        if len(examples) < page_size:
            break
        offset += page_size
    return trace_ids


def populate_dataset(
    client: Client, dataset_name: str, project_name: str, seed_traces_path: str
) -> tuple[schemas.Dataset, int]:
    dataset = ensure_dataset(client, dataset_name)
    existing_trace_ids = list_existing_trace_ids(client, str(dataset.id))

    new_examples = []
    trace_specs = load_seed_traces(seed_traces_path)
    for spec in trace_specs:
        if spec["trace_id"] in existing_trace_ids:
            continue
        root_run = get_root_run(client, spec["trace_id"], project_name)
        stats = get_trace_stats(client, spec["trace_id"], project_name)
        messages = (root_run.inputs or {}).get("messages") or []
        if not messages:
            continue
        user_text = messages[0].get("content")
        if not user_text:
            continue
        assistant_text = latest_ai_text(root_run)
        new_examples.append(
            {
                "inputs": {
                    "user_text": user_text,
                },
                "outputs": {
                    "requires_tool_call": spec["requires_tool_call"],
                    "label": spec["label"],
                },
                "source_run_id": str(root_run.id),
                "metadata": {
                    "source_trace_id": spec["trace_id"],
                    "label": spec["label"],
                    "notes": spec["notes"],
                    "source_project": project_name,
                    "reference_assistant_text": assistant_text[:1000],
                    **stats,
                },
            }
        )

    if new_examples:
        client.create_examples(dataset_id=dataset.id, examples=new_examples)
    return dataset, len(new_examples)


def run_case(repo_path: str, user_text: str, model_name: str) -> dict[str, Any]:
    worker = WORKER_TEMPLATE.format(
        repo_path=repo_path,
        user_text=user_text,
        model_name=model_name,
        workspace_env=WORKSPACE_ENV,
    )
    completed = subprocess.run(
        [sys.executable, "-c", worker],
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
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    expected = example.outputs if hasattr(example, "outputs") else example.get("outputs", {})
    required = bool((expected or {}).get("requires_tool_call"))
    observed = bool((outputs or {}).get("tool_calls_seen"))
    return {
        "key": "tool_call_required",
        "score": 1 if observed == required else 0,
        "comment": f"required={required} observed={observed}",
    }


def evaluator_retry(run, example):
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    retried = bool((outputs or {}).get("retried"))
    return {
        "key": "retried_after_no_tool",
        "score": 1 if retried else 0,
        "comment": f"retried={retried}",
    }


def run_eval(
    client: Client, dataset_name: str, repo_path: str, experiment_prefix: str, model_name: str
):
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_case(repo_path=repo_path, user_text=inputs["user_text"], model_name=model_name)

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[evaluator_tool_call, evaluator_retry],
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
        run_outputs = getattr(run_obj, "outputs", None) or (
            run_obj.get("outputs", {}) if isinstance(run_obj, dict) else {}
        )
        example_inputs = getattr(example_obj, "inputs", None) or (
            example_obj.get("inputs", {}) if isinstance(example_obj, dict) else {}
        )
        example_id = getattr(example_obj, "id", None) or (
            example_obj.get("id") if isinstance(example_obj, dict) else None
        )
        examples.append(
            {
                "example_id": str(example_id),
                "user_text": example_inputs.get("user_text"),
                "metrics": metrics,
                "tool_calls_seen": run_outputs.get("tool_calls_seen"),
                "tool_names": run_outputs.get("tool_names"),
                "retried": run_outputs.get("retried"),
                "attempts": run_outputs.get("attempts"),
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
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--baseline-commit", default="origin/main")
    parser.add_argument("--baseline-worktree", default="/tmp/deepclaw-eval-baseline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--seed-traces-path", default=DEFAULT_SEED_TRACES_PATH)
    parser.add_argument("--populate-only", action="store_true")
    args = parser.parse_args()

    client = get_client()
    dataset, created = populate_dataset(client, args.dataset, args.project, args.seed_traces_path)
    sys.stdout.write(
        json.dumps(
            {
                "dataset": args.dataset,
                "dataset_id": str(dataset.id),
                "examples_added": created,
            }
        )
        + "\n"
    )
    if args.populate_only:
        return

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
