"""Tests for evals/compaction_regression.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "evals" / "compaction_regression.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compaction_regression", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataset_examples_cover_observed_trace_case():
    module = _load_module()

    trace_case = next(
        example
        for example in module.DATASET_EXAMPLES
        if example["inputs"]["case_id"] == "trace_wrapper_summary_must_be_rejected"
    )

    assert trace_case["metadata"]["source_trace_id"] == "019de3dd-d90c-74c3-bda2-270f258ab4ab"
    assert "DO NOT CHANGE ANY CODE" in trace_case["outputs"]["required_constraint_substrings"]


def test_run_case_reports_wrapper_poisoning_on_current_behavior():
    module = _load_module()

    case = next(
        example
        for example in module.DATASET_EXAMPLES
        if example["inputs"]["case_id"] == "fallback_must_ignore_latest_wrapper_human_message"
    )

    result = module.run_case(case["inputs"])

    assert module.COMPACTION_WRAPPER_TEXT in result["active_task"]
    assert result["used_fallback"] is True


def test_evaluator_constraints_fails_when_required_constraint_missing():
    module = _load_module()

    result = module.evaluator_constraints(
        {"outputs": {"normalized_summary": "## Active Task\nInspect repo\n"}},
        {"outputs": {"required_constraint_substrings": ["DO NOT CHANGE ANY CODE"]}},
    )

    assert result == {
        "key": "constraints_preserved",
        "score": 0,
        "comment": "missing=['DO NOT CHANGE ANY CODE']",
    }


def test_evaluator_forbidden_substrings_detects_wrapper_text():
    module = _load_module()

    result = module.evaluator_forbidden_substrings(
        {
            "outputs": {
                "normalized_summary": "## Active Task\nYou are in the middle of a conversation that has been summarized.\n"
            }
        },
        {
            "outputs": {
                "forbidden_substrings": [
                    "You are in the middle of a conversation that has been summarized"
                ]
            }
        },
    )

    assert result == {
        "key": "forbidden_substrings_absent",
        "score": 0,
        "comment": ("present=['You are in the middle of a conversation that has been summarized']"),
    }


def test_ensure_dataset_only_uploads_missing_case_ids():
    module = _load_module()

    class FakeDataset:
        id = "dataset-123"
        name = module.DEFAULT_DATASET

    class FakeClient:
        def __init__(self):
            self.created_examples = None

        def list_datasets(self, dataset_name=None, limit=None):
            assert dataset_name == module.DEFAULT_DATASET
            assert limit == 1
            yield FakeDataset()

        def list_examples(self, dataset_name=None, limit=None):
            assert dataset_name == module.DEFAULT_DATASET
            assert limit == 200
            yield {"inputs": {"case_id": "placeholder_summary_replaced"}}

        def create_examples(self, dataset_id=None, examples=None):
            self.created_examples = (dataset_id, examples)

    client = FakeClient()
    result = module.ensure_dataset(client, module.DEFAULT_DATASET)

    assert result["created_examples"] == len(module.DATASET_EXAMPLES) - 1
    assert result["total_examples"] == len(module.DATASET_EXAMPLES)
    assert client.created_examples is not None
    dataset_id, examples = client.created_examples
    assert dataset_id == "dataset-123"
    assert {example["inputs"]["case_id"] for example in examples} == {
        example["inputs"]["case_id"]
        for example in module.DATASET_EXAMPLES
        if example["inputs"]["case_id"] != "placeholder_summary_replaced"
    }


def test_run_eval_handles_object_like_evaluation_results(monkeypatch):
    module = _load_module()

    class FakeMetric:
        def __init__(self, key, score):
            self.key = key
            self.score = score

    class FakeEvaluationResults:
        def __init__(self, results):
            self.results = results

    class FakeResults:
        experiment_name = "compaction-exp"
        url = "https://example.com/compaction-exp"

        def __iter__(self):
            yield {
                "run": {
                    "outputs": {
                        "used_fallback": True,
                        "active_task": "Inspect repo",
                        "goal": "Report findings",
                    }
                },
                "example": {"inputs": {"case_id": "trace_wrapper_summary_must_be_rejected"}},
                "evaluation_results": FakeEvaluationResults(
                    [
                        FakeMetric("should_fallback", 1),
                        FakeMetric("active_task_preserved", 0),
                    ]
                ),
            }

    monkeypatch.setattr(module, "evaluate", lambda *args, **kwargs: FakeResults())

    result = module.run_eval(
        client=None, dataset_name=module.DEFAULT_DATASET, experiment_prefix="x"
    )

    assert result["experiment_name"] == "compaction-exp"
    assert result["summary"] == {
        "active_task_preserved": 0.0,
        "should_fallback": 1.0,
    }
    assert result["examples"] == [
        {
            "case_id": "trace_wrapper_summary_must_be_rejected",
            "used_fallback": True,
            "active_task": "Inspect repo",
            "goal": "Report findings",
            "metrics": {
                "should_fallback": 1,
                "active_task_preserved": 0,
            },
        }
    ]
