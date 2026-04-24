from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from deepclaw.runtime_controller import (
    RuntimeRoute,
    build_rebuild_prompt,
    choose_runtime_route,
    estimate_prompt_budget,
    truncate_reference_summary,
)


def _snapshot(*, messages=None, continuity_checkpoint=None, working_state=None):
    return SimpleNamespace(
        messages=list(messages or []),
        continuity_checkpoint=dict(continuity_checkpoint or {}),
        working_state=dict(working_state or {}),
    )


def test_estimate_prompt_budget_counts_live_history_and_summary():
    estimate = estimate_prompt_budget(
        snapshot=_snapshot(
            messages=[HumanMessage(content="hello"), HumanMessage(content="world")],
            continuity_checkpoint={"artifact_refs": [{"path": "/tmp/raw.md", "label": "raw"}]},
        ),
        user_text="continue",
        pending_summary="handoff",
        system_overhead_chars=50,
    )

    assert estimate.live_message_count == 2
    assert estimate.live_history_chars >= len("helloworld")
    assert estimate.pending_summary_chars == len("handoff")
    assert estimate.current_user_chars == len("continue")
    assert estimate.estimated_total_chars > estimate.live_history_chars


def test_choose_runtime_route_prefers_summary_truncation_when_summary_absorbs_overflow():
    estimate = estimate_prompt_budget(
        snapshot=_snapshot(messages=[HumanMessage(content="a" * 40)]),
        user_text="b" * 20,
        pending_summary="c" * 80,
        system_overhead_chars=20,
    )

    plan = choose_runtime_route(
        estimate, can_compact=True, fit_budget_chars=120, rebuild_budget_chars=400
    )

    assert plan.route is RuntimeRoute.TRUNCATE_ARTIFACTS_ONLY


def test_choose_runtime_route_prefers_rebuild_when_live_context_is_far_over_budget():
    estimate = estimate_prompt_budget(
        snapshot=_snapshot(messages=[HumanMessage(content="a" * 500) for _ in range(6)]),
        user_text="continue",
        pending_summary="handoff",
        system_overhead_chars=20,
    )

    plan = choose_runtime_route(
        estimate, can_compact=True, fit_budget_chars=200, rebuild_budget_chars=300
    )

    assert plan.route is RuntimeRoute.REBUILD_FROM_STATE


def test_choose_runtime_route_falls_back_to_hard_recovery_without_compaction():
    estimate = estimate_prompt_budget(
        snapshot=_snapshot(messages=[HumanMessage(content="a" * 300)]),
        user_text="continue",
        pending_summary="handoff",
        system_overhead_chars=20,
    )

    plan = choose_runtime_route(
        estimate, can_compact=False, fit_budget_chars=100, rebuild_budget_chars=120
    )

    assert plan.route is RuntimeRoute.HARD_OVERFLOW_RECOVERY


def test_truncate_reference_summary_preserves_artifact_references():
    summary = "\n".join(
        ["headline"]
        + [f"detail {idx}" for idx in range(100)]
        + ["- checkpoint_artifact: `/tmp/checkpoint.json`", "Use read_file on artifacts if needed."]
    )

    truncated = truncate_reference_summary(summary, max_chars=120)

    assert "Summary truncated for runtime budget" in truncated
    assert "/tmp/checkpoint.json" in truncated
    assert "read_file" in truncated


def test_build_rebuild_prompt_uses_state_and_artifacts():
    prompt = build_rebuild_prompt(
        user_text="Finish the patch",
        snapshot=_snapshot(
            messages=[HumanMessage(content="older message")],
            continuity_checkpoint={
                "current_goal": "Ship PR 5",
                "next_action": "Run the tests",
                "verification_status": "pending",
                "active_blockers": ["Need the failing stack trace"],
                "relevant_files": ["deepclaw/gateway.py"],
                "pending_verification": ["pytest"],
                "artifact_refs": [{"path": "/tmp/raw.md", "kind": "history", "label": "history"}],
                "reference_summary": {"items": ["Historical summary may be stale."]},
            },
        ),
        thread_state={
            "summary_artifact_path": "/tmp/summary.md",
            "checkpoint_artifact_path": "/tmp/checkpoint.json",
            "raw_history_artifact_paths": ["/tmp/raw.md"],
        },
        handoff_summary="condensed handoff",
        route=RuntimeRoute.REBUILD_FROM_STATE,
    )

    assert "Route: rebuild_from_state" in prompt
    assert "Ship PR 5" in prompt
    assert "Need the failing stack trace" in prompt
    assert "`/tmp/checkpoint.json`" in prompt
    assert "Finish the patch" in prompt
