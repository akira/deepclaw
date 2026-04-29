"""Tests for evals/dataset_audit.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).resolve().parents[1] / "evals" / "dataset_audit.py"
    spec = importlib.util.spec_from_file_location("dataset_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _example(example_id: str, *, outputs: dict, metadata: dict | None = None):
    return SimpleNamespace(id=example_id, outputs=outputs, metadata=metadata or {})


def test_summarize_examples_reports_annotation_coverage_and_category_counts():
    module = _load_module()

    examples = [
        _example(
            "ex-1",
            outputs={
                "label": "remember-preference",
                "category": "memory-and-skills",
                "requires_tool_call": True,
                "expected_tool_names": ["edit_file"],
                "must_succeed_first_pass": True,
                "eval_mode": "compliance",
                "behavior_tags": ["memory_persistence"],
                "side_effect_risk": "low",
            },
            metadata={
                "source_trace_id": "trace-1",
                "notes": "good example",
                "reference_assistant_text": "I'll save that now.",
            },
        ),
        _example(
            "ex-2",
            outputs={
                "label": "thanks-no-tool",
                "category": "conversation",
                "requires_tool_call": False,
                "eval_mode": "compliance",
                "behavior_tags": ["no_tool_conversation"],
                "side_effect_risk": "low",
            },
            metadata={
                "notes": "missing trace provenance",
            },
        ),
    ]

    summary = module.summarize_examples(examples)

    assert summary["total_examples"] == 2
    assert summary["requires_tool_call"] == {"true": 1, "false": 1}
    assert summary["coverage"] == {
        "expected_tool_names": 1,
        "must_succeed_first_pass": 1,
        "category": 2,
        "eval_mode": 2,
        "behavior_tags": 2,
        "side_effect_risk": 2,
        "source_trace_id": 1,
        "notes": 2,
        "reference_assistant_text": 1,
    }
    assert summary["category_counts"] == {
        "conversation": 1,
        "memory-and-skills": 1,
    }
    assert summary["missing_provenance"] == [
        {
            "example_id": "ex-2",
            "label": "thanks-no-tool",
            "missing_metadata_fields": ["source_trace_id", "reference_assistant_text"],
        }
    ]
