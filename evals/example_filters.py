#!/usr/bin/env python3
"""Helpers for filtering LangSmith dataset examples by eval metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _get_outputs(example: Any) -> dict[str, Any]:
    if hasattr(example, "outputs"):
        return example.outputs or {}
    if isinstance(example, dict):
        return example.get("outputs", {}) or {}
    return {}


def _normalize_list(values: Iterable[str] | None) -> list[str] | None:
    if not values:
        return None
    normalized = [value for value in values if isinstance(value, str) and value]
    return normalized or None


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def matches_filters(
    example: Any,
    *,
    categories: list[str] | None = None,
    eval_modes: list[str] | None = None,
    behavior_tags: list[str] | None = None,
    side_effect_risks: list[str] | None = None,
) -> bool:
    outputs = _get_outputs(example)

    category = outputs.get("category")
    if categories and category not in categories:
        return False

    eval_mode = outputs.get("eval_mode")
    if eval_modes and eval_mode not in eval_modes:
        return False

    if behavior_tags:
        observed_tags = set(_normalize_tags(outputs.get("behavior_tags")))
        if not observed_tags.intersection(behavior_tags):
            return False

    side_effect_risk = outputs.get("side_effect_risk")
    return (not side_effect_risks) or (side_effect_risk in side_effect_risks)


def filter_examples(
    examples: Iterable[Any],
    *,
    categories: list[str] | None = None,
    eval_modes: list[str] | None = None,
    behavior_tags: list[str] | None = None,
    side_effect_risks: list[str] | None = None,
    example_limit: int | None = None,
) -> list[Any]:
    categories = _normalize_list(categories)
    eval_modes = _normalize_list(eval_modes)
    behavior_tags = _normalize_list(behavior_tags)
    side_effect_risks = _normalize_list(side_effect_risks)

    filtered = [
        example
        for example in examples
        if matches_filters(
            example,
            categories=categories,
            eval_modes=eval_modes,
            behavior_tags=behavior_tags,
            side_effect_risks=side_effect_risks,
        )
    ]
    if example_limit is not None:
        return filtered[:example_limit]
    return filtered
