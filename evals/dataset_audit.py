#!/usr/bin/env python3
"""Audit LangSmith dataset coverage for DeepClaw eval metadata."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any

from dotenv import load_dotenv
from langsmith import Client

WORKSPACE_ENV = "/home/ubuntu/.deepclaw/.env"
DEFAULT_DATASET = "deepclaw"


def load_env() -> None:
    load_dotenv(WORKSPACE_ENV, override=True)


def get_client() -> Client:
    load_env()
    return Client()


def _get_outputs(example: Any) -> dict[str, Any]:
    if hasattr(example, "outputs"):
        return example.outputs or {}
    if isinstance(example, dict):
        return example.get("outputs", {}) or {}
    return {}


def _get_metadata(example: Any) -> dict[str, Any]:
    if hasattr(example, "metadata"):
        return example.metadata or {}
    if isinstance(example, dict):
        return example.get("metadata", {}) or {}
    return {}


def _get_example_id(example: Any) -> str | None:
    if hasattr(example, "id"):
        value = example.id
        return str(value) if value is not None else None
    if isinstance(example, dict):
        value = example.get("id")
        return str(value) if value is not None else None
    return None


def summarize_examples(examples: list[Any]) -> dict[str, Any]:
    requires_counter = Counter()
    category_counts = Counter()
    eval_mode_counts = Counter()
    side_effect_risk_counts = Counter()
    behavior_tag_counts = Counter()
    coverage = Counter()
    missing_provenance = []

    for example in examples:
        outputs = _get_outputs(example)
        metadata = _get_metadata(example)
        example_id = _get_example_id(example)
        label = outputs.get("label")

        requires_counter[str(bool(outputs.get("requires_tool_call"))).lower()] += 1

        category = outputs.get("category")
        if category:
            coverage["category"] += 1
            category_counts[str(category)] += 1

        eval_mode = outputs.get("eval_mode")
        if eval_mode:
            coverage["eval_mode"] += 1
            eval_mode_counts[str(eval_mode)] += 1

        side_effect_risk = outputs.get("side_effect_risk")
        if side_effect_risk:
            coverage["side_effect_risk"] += 1
            side_effect_risk_counts[str(side_effect_risk)] += 1

        behavior_tags = outputs.get("behavior_tags")
        if isinstance(behavior_tags, list) and behavior_tags:
            coverage["behavior_tags"] += 1
            for tag in behavior_tags:
                if isinstance(tag, str) and tag:
                    behavior_tag_counts[tag] += 1

        if outputs.get("expected_tool_names"):
            coverage["expected_tool_names"] += 1
        if outputs.get("must_succeed_first_pass") is not None:
            coverage["must_succeed_first_pass"] += 1

        if metadata.get("source_trace_id") is not None:
            coverage["source_trace_id"] += 1
        if metadata.get("notes") is not None:
            coverage["notes"] += 1
        if metadata.get("reference_assistant_text") is not None:
            coverage["reference_assistant_text"] += 1

        missing_metadata_fields = []
        for field in ["source_trace_id", "reference_assistant_text"]:
            if metadata.get(field) is None:
                missing_metadata_fields.append(field)
        if missing_metadata_fields:
            missing_provenance.append(
                {
                    "example_id": example_id,
                    "label": label,
                    "missing_metadata_fields": missing_metadata_fields,
                }
            )

    return {
        "total_examples": len(examples),
        "requires_tool_call": dict(requires_counter),
        "coverage": {
            key: coverage.get(key, 0)
            for key in [
                "expected_tool_names",
                "must_succeed_first_pass",
                "category",
                "eval_mode",
                "behavior_tags",
                "side_effect_risk",
                "source_trace_id",
                "notes",
                "reference_assistant_text",
            ]
        },
        "category_counts": dict(sorted(category_counts.items())),
        "eval_mode_counts": dict(sorted(eval_mode_counts.items())),
        "side_effect_risk_counts": dict(sorted(side_effect_risk_counts.items())),
        "behavior_tag_counts": dict(sorted(behavior_tag_counts.items())),
        "missing_provenance": missing_provenance,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client = get_client()
    examples = list(client.list_examples(dataset_name=args.dataset, limit=200))
    summary = summarize_examples(examples)
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
