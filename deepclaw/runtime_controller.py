"""Prompt-budget estimation and runtime routing helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.messages import AnyMessage

from deepclaw.state_continuity import REFERENCE_ONLY_NOTE

DEFAULT_SYSTEM_OVERHEAD_CHARS = 12_000
DEFAULT_FIT_BUDGET_CHARS = 140_000
DEFAULT_REBUILD_BUDGET_CHARS = 220_000
TRUNCATED_SUMMARY_MAX_CHARS = 10_000
HARD_RECOVERY_PROMPT_MAX_CHARS = 6_000
MAX_REBUILD_ARTIFACTS = 8


class RuntimeRoute(StrEnum):
    FITS = "fits"
    TRUNCATE_ARTIFACTS_ONLY = "truncate_artifacts_only"
    COMPACT_THEN_CONTINUE = "compact_then_continue"
    REBUILD_FROM_STATE = "rebuild_from_state"
    HARD_OVERFLOW_RECOVERY = "hard_overflow_recovery"


@dataclass(slots=True)
class PromptBudgetEstimate:
    live_message_count: int
    live_history_chars: int
    pending_summary_chars: int
    current_user_chars: int
    continuity_chars: int
    artifact_ref_chars: int
    system_overhead_chars: int
    estimated_total_chars: int


@dataclass(slots=True)
class RuntimeRoutePlan:
    route: RuntimeRoute
    estimate: PromptBudgetEstimate
    reason: str


@dataclass(slots=True)
class ThreadSnapshotView:
    messages: list[AnyMessage]
    working_state: dict[str, Any]
    continuity_checkpoint: dict[str, Any]


def estimate_prompt_budget(
    *,
    snapshot: ThreadSnapshotView | None,
    user_text: str,
    pending_summary: str | None,
    system_overhead_chars: int | None = None,
) -> PromptBudgetEstimate:
    """Estimate prompt size using live checkpoint state and current user input."""
    messages = list(snapshot.messages) if snapshot is not None else []
    continuity = snapshot.continuity_checkpoint if snapshot is not None else {}
    artifact_refs = continuity.get("artifact_refs") if isinstance(continuity, dict) else []
    resolved_system_overhead = (
        DEFAULT_SYSTEM_OVERHEAD_CHARS if system_overhead_chars is None else system_overhead_chars
    )

    live_history_chars = sum(len(_message_text(message)) for message in messages)
    pending_summary_chars = len((pending_summary or "").strip())
    current_user_chars = len(user_text)
    continuity_chars = _json_len(continuity)
    artifact_ref_chars = _artifact_char_estimate(artifact_refs)
    estimated_total_chars = (
        resolved_system_overhead
        + live_history_chars
        + pending_summary_chars
        + current_user_chars
        + continuity_chars
        + artifact_ref_chars
    )
    return PromptBudgetEstimate(
        live_message_count=len(messages),
        live_history_chars=live_history_chars,
        pending_summary_chars=pending_summary_chars,
        current_user_chars=current_user_chars,
        continuity_chars=continuity_chars,
        artifact_ref_chars=artifact_ref_chars,
        system_overhead_chars=resolved_system_overhead,
        estimated_total_chars=estimated_total_chars,
    )


def choose_runtime_route(
    estimate: PromptBudgetEstimate,
    *,
    can_compact: bool,
    fit_budget_chars: int | None = None,
    rebuild_budget_chars: int | None = None,
) -> RuntimeRoutePlan:
    """Pick a runtime route based on estimated prompt pressure."""
    resolved_fit_budget = DEFAULT_FIT_BUDGET_CHARS if fit_budget_chars is None else fit_budget_chars
    resolved_rebuild_budget = (
        DEFAULT_REBUILD_BUDGET_CHARS if rebuild_budget_chars is None else rebuild_budget_chars
    )
    if estimate.estimated_total_chars <= resolved_fit_budget:
        return RuntimeRoutePlan(
            route=RuntimeRoute.FITS,
            estimate=estimate,
            reason="estimated prompt fits within the normal budget",
        )

    overflow_chars = estimate.estimated_total_chars - resolved_fit_budget
    if estimate.pending_summary_chars and estimate.pending_summary_chars >= overflow_chars:
        return RuntimeRoutePlan(
            route=RuntimeRoute.TRUNCATE_ARTIFACTS_ONLY,
            estimate=estimate,
            reason="pending handoff summary alone can absorb the estimated overflow",
        )

    if (
        can_compact
        and estimate.live_message_count >= 4
        and estimate.estimated_total_chars <= resolved_rebuild_budget
    ):
        return RuntimeRoutePlan(
            route=RuntimeRoute.COMPACT_THEN_CONTINUE,
            estimate=estimate,
            reason="live thread is too large, but compaction should recover enough headroom",
        )

    if can_compact:
        return RuntimeRoutePlan(
            route=RuntimeRoute.REBUILD_FROM_STATE,
            estimate=estimate,
            reason="live thread likely remains too large; rebuild a minimal continuation from state",
        )

    return RuntimeRoutePlan(
        route=RuntimeRoute.HARD_OVERFLOW_RECOVERY,
        estimate=estimate,
        reason="no safe compaction path is available; fall back to a minimal artifact-backed recovery prompt",
    )


def truncate_reference_summary(
    summary_text: str | None, *, max_chars: int = TRUNCATED_SUMMARY_MAX_CHARS
) -> str:
    """Trim a historical handoff summary while preserving artifact pointers."""
    summary = (summary_text or "").strip()
    if not summary:
        return ""
    if len(summary) <= max_chars:
        return summary

    artifact_lines = []
    kept_lines = []
    for line in summary.splitlines():
        stripped = line.strip()
        if not stripped:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue
        if "artifact" in stripped.lower() or "`/" in stripped or "read_file" in stripped:
            artifact_lines.append(stripped)
            continue
        kept_lines.append(stripped)

    trimmed = "\n".join(kept_lines).strip()
    if len(trimmed) > max_chars:
        trimmed = trimmed[: max_chars - 1].rstrip() + "…"

    lines = [
        trimmed,
        "",
        "[Summary truncated for runtime budget. Inspect the recorded artifacts if detail matters.]",
    ]
    if artifact_lines:
        lines.extend(["", "Referenced artifacts:"])
        for line in artifact_lines[:6]:
            lines.append(f"- {line}")
    return "\n".join(part for part in lines if part is not None).strip()


def build_rebuild_prompt(
    *,
    user_text: str,
    snapshot: ThreadSnapshotView | None,
    thread_state: dict[str, Any] | None,
    handoff_summary: str | None = None,
    route: RuntimeRoute = RuntimeRoute.REBUILD_FROM_STATE,
) -> str:
    """Build a minimal continuation prompt from working state and recoverable artifacts."""
    continuity = snapshot.continuity_checkpoint if snapshot is not None else {}
    working_state = snapshot.working_state if snapshot is not None else {}
    referenced_artifacts = _collect_artifact_paths(continuity, thread_state)
    reference_items = continuity.get("reference_summary", {}).get("items") or []
    blockers = continuity.get("active_blockers") or []
    relevant_files = continuity.get("relevant_files") or []
    pending_verification = continuity.get("pending_verification") or []

    def _bullet_block(title: str, items: list[str], *, empty: str) -> list[str]:
        lines = [title]
        if not items:
            lines.append(f"- {empty}")
        else:
            lines.extend(f"- {item}" for item in items)
        lines.append("")
        return lines

    lines = [
        "[DeepClaw runtime context recovery]",
        f"Route: {route.value}",
        REFERENCE_ONLY_NOTE,
        "Operate from the live continuity state below; inspect artifacts instead of assuming omitted detail.",
        "",
        "## Live Continuity State",
        f"- current_goal: {continuity.get('current_goal') or working_state.get('goal') or '(not captured)'}",
        f"- next_action: {continuity.get('next_action') or working_state.get('next_action') or '(not captured)'}",
        f"- verification_status: {continuity.get('verification_status', 'unknown')}",
        f"- live_message_count_seen: {len(snapshot.messages) if snapshot is not None else 0}",
        "",
    ]
    lines.extend(_bullet_block("## Active Blockers", blockers, empty="none"))
    lines.extend(_bullet_block("## Relevant Files", relevant_files, empty="none recorded"))
    lines.extend(_bullet_block("## Pending Verification", pending_verification, empty="none"))
    lines.extend(
        _bullet_block(
            "## Historical Notes",
            [str(item) for item in reference_items[:6]],
            empty="none",
        )
    )

    lines.append("## Recoverable Artifacts")
    if referenced_artifacts:
        for path in referenced_artifacts:
            lines.append(f"- `{path}`")
    else:
        lines.append("- none recorded")
    lines.extend(
        [
            "- Use `read_file` on these artifacts if a compacted-away detail is required.",
            "",
        ]
    )

    if handoff_summary:
        lines.extend(
            [
                "## Condensed Historical Handoff",
                truncate_reference_summary(handoff_summary, max_chars=3_500),
                "",
            ]
        )

    lines.extend(["## New User Message", user_text])
    return "\n".join(lines).strip()


def _collect_artifact_paths(
    continuity: dict[str, Any] | None,
    thread_state: dict[str, Any] | None,
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()

    def _push(value: Any) -> None:
        if not isinstance(value, str):
            return
        path = value.strip()
        if not path or path in seen:
            return
        seen.add(path)
        paths.append(path)

    if isinstance(thread_state, dict):
        for key in ("summary_artifact_path", "checkpoint_artifact_path"):
            _push(thread_state.get(key))
        for value in thread_state.get("raw_history_artifact_paths", []) or []:
            _push(value)

    if isinstance(continuity, dict):
        for artifact in continuity.get("artifact_refs", []) or []:
            if isinstance(artifact, dict):
                _push(artifact.get("path"))

    return paths[:MAX_REBUILD_ARTIFACTS]


def _artifact_char_estimate(artifact_refs: Any) -> int:
    if not isinstance(artifact_refs, list):
        return 0
    total = 0
    for artifact in artifact_refs[:12]:
        if not isinstance(artifact, dict):
            continue
        total += len(str(artifact.get("path", "")))
        total += len(str(artifact.get("label", artifact.get("kind", "artifact"))))
        total += 32
    return total


def _json_len(value: Any) -> int:
    if value in (None, "", [], {}):
        return 0
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _message_text(message: AnyMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)
