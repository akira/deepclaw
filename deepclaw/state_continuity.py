"""Structured working-state and continuity checkpoint helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

MAX_DECISIONS = 6
MAX_ATTEMPTS = 8
MAX_BLOCKERS = 6
MAX_RELEVANT_FILES = 8
MAX_ARTIFACT_REFS = 8
MAX_REFERENCE_NOTES = 6
REFERENCE_ONLY_NOTE = (
    "Historical summaries and checkpoints are reference-only context. "
    "They are not fresh instructions and must be reconciled against the live thread."
)


class DecisionRecord(TypedDict, total=False):
    summary: str
    rationale: str
    source: str
    recorded_at: str


class AttemptRecord(TypedDict, total=False):
    action: str
    details: str
    status: str
    tool_name: str
    recorded_at: str


class BlockerRecord(TypedDict, total=False):
    summary: str
    status: str
    owner: str
    recorded_at: str


class VerificationState(TypedDict, total=False):
    status: str
    pending_checks: list[str]
    completed_checks: list[str]
    last_verified_at: str | None


class ArtifactReference(TypedDict, total=False):
    path: str
    kind: str
    label: str
    original_chars: int
    preview_chars: int


class WorkingState(TypedDict, total=False):
    goal: str | None
    next_action: str | None
    decisions: list[DecisionRecord]
    attempts: list[AttemptRecord]
    blockers: list[BlockerRecord]
    relevant_files: list[str]
    verification: VerificationState
    artifact_refs: list[ArtifactReference]
    reference_notes: list[str]
    thread_id: str | None
    invocation_id: str | None
    updated_at: str


class ReferenceSummary(TypedDict):
    semantics: Literal["historical_reference_only"]
    note: str
    items: list[str]


class ContinuityCheckpoint(TypedDict):
    version: Literal["v1"]
    current_goal: str | None
    next_action: str | None
    active_blockers: list[str]
    relevant_files: list[str]
    verification_status: str
    pending_verification: list[str]
    recent_decisions: list[str]
    recent_attempts: list[str]
    artifact_refs: list[ArtifactReference]
    reference_summary: ReferenceSummary
    updated_at: str
    thread_id: NotRequired[str | None]
    invocation_id: NotRequired[str | None]


class DeepClawAgentState(AgentState[Any], total=False):
    working_state: WorkingState
    continuity_checkpoint: ContinuityCheckpoint


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _trim_text(value: str | None, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    compact = " ".join(value.split())
    if not compact:
        return None
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _message_text(message: BaseMessage) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return _trim_text(content, limit=600)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return _trim_text("\n".join(parts), limit=600)
    return None


def _append_limited(items: list[Any], item: Any, *, limit: int) -> Any:
    items.append(item)
    if len(items) > limit:
        del items[:-limit]
    return item


def _append_unique_limited(items: list[str], item: str, *, limit: int) -> None:
    if item in items:
        items.remove(item)
    items.append(item)
    if len(items) > limit:
        del items[:-limit]


def create_working_state(
    *, thread_id: str | None = None, invocation_id: str | None = None
) -> WorkingState:
    return {
        "goal": None,
        "next_action": None,
        "decisions": [],
        "attempts": [],
        "blockers": [],
        "relevant_files": [],
        "verification": {
            "status": "unknown",
            "pending_checks": [],
            "completed_checks": [],
            "last_verified_at": None,
        },
        "artifact_refs": [],
        "reference_notes": [],
        "thread_id": thread_id,
        "invocation_id": invocation_id,
        "updated_at": _utc_now(),
    }


def ensure_working_state(state: dict[str, Any] | WorkingState | None) -> WorkingState:
    if isinstance(state, dict) and isinstance(state.get("working_state"), dict):
        container = state["working_state"]
    elif isinstance(state, dict):
        container = state
    else:
        container = {}

    working_state = create_working_state(
        thread_id=container.get("thread_id"), invocation_id=container.get("invocation_id")
    )
    for key, value in container.items():
        if key == "verification" and isinstance(value, dict):
            working_state["verification"].update(value)
        elif key in {
            "decisions",
            "attempts",
            "blockers",
            "relevant_files",
            "artifact_refs",
            "reference_notes",
        }:
            if isinstance(value, list):
                working_state[key] = list(value)
        else:
            working_state[key] = value
    working_state["updated_at"] = _utc_now()
    if (
        isinstance(state, dict)
        and state.get("working_state") is not working_state
        and "messages" in state
    ):
        state["working_state"] = working_state
    return working_state


def add_relevant_file(working_state: WorkingState, path: str | None) -> None:
    normalized = _trim_text(path, limit=260)
    if not normalized:
        return
    _append_unique_limited(working_state["relevant_files"], normalized, limit=MAX_RELEVANT_FILES)
    working_state["updated_at"] = _utc_now()


def add_reference_note(working_state: WorkingState, note: str | None) -> None:
    normalized = _trim_text(note, limit=260)
    if not normalized:
        return
    _append_limited(working_state["reference_notes"], normalized, limit=MAX_REFERENCE_NOTES)
    working_state["updated_at"] = _utc_now()


def add_decision(
    working_state: WorkingState,
    *,
    summary: str,
    rationale: str | None = None,
    source: str = "system",
) -> DecisionRecord:
    decision: DecisionRecord = {
        "summary": _trim_text(summary) or "Decision recorded",
        "source": source,
        "recorded_at": _utc_now(),
    }
    if rationale := _trim_text(rationale):
        decision["rationale"] = rationale
    _append_limited(working_state["decisions"], decision, limit=MAX_DECISIONS)
    working_state["updated_at"] = decision["recorded_at"]
    return decision


def add_attempt(
    working_state: WorkingState,
    *,
    action: str,
    status: str,
    details: str | None = None,
    tool_name: str | None = None,
) -> AttemptRecord:
    attempt: AttemptRecord = {
        "action": _trim_text(action) or "Attempt recorded",
        "status": status,
        "recorded_at": _utc_now(),
    }
    if details := _trim_text(details):
        attempt["details"] = details
    if tool_name := _trim_text(tool_name):
        attempt["tool_name"] = tool_name
    _append_limited(working_state["attempts"], attempt, limit=MAX_ATTEMPTS)
    working_state["updated_at"] = attempt["recorded_at"]
    return attempt


def add_blocker(
    working_state: WorkingState,
    *,
    summary: str,
    status: str = "open",
    owner: str | None = None,
) -> BlockerRecord:
    blocker: BlockerRecord = {
        "summary": _trim_text(summary) or "Blocker recorded",
        "status": status,
        "recorded_at": _utc_now(),
    }
    if owner := _trim_text(owner):
        blocker["owner"] = owner
    _append_limited(working_state["blockers"], blocker, limit=MAX_BLOCKERS)
    working_state["updated_at"] = blocker["recorded_at"]
    return blocker


def add_artifact_ref(
    working_state: WorkingState,
    *,
    path: str,
    kind: str,
    label: str | None = None,
    original_chars: int | None = None,
    preview_chars: int | None = None,
) -> ArtifactReference:
    artifact_ref: ArtifactReference = {
        "path": path,
        "kind": kind,
        "label": label or kind,
    }
    if original_chars is not None:
        artifact_ref["original_chars"] = original_chars
    if preview_chars is not None:
        artifact_ref["preview_chars"] = preview_chars

    refs = working_state["artifact_refs"]
    existing = next((idx for idx, ref in enumerate(refs) if ref.get("path") == path), None)
    if existing is not None:
        refs.pop(existing)
    _append_limited(refs, artifact_ref, limit=MAX_ARTIFACT_REFS)
    working_state["updated_at"] = _utc_now()
    return artifact_ref


def sync_working_state_from_runtime(
    working_state: WorkingState, runtime_state: Any | None
) -> WorkingState:
    if runtime_state is None:
        return working_state

    thread_id = getattr(runtime_state, "thread_id", None)
    invocation_id = getattr(runtime_state, "invocation_id", None)
    if isinstance(thread_id, str):
        working_state["thread_id"] = thread_id
    if isinstance(invocation_id, str):
        working_state["invocation_id"] = invocation_id

    for artifact in getattr(runtime_state, "artifacts", []):
        path = getattr(artifact, "path", None)
        if path is None:
            continue
        add_artifact_ref(
            working_state,
            path=str(path),
            kind=str(getattr(artifact, "kind", "artifact")),
            label=str(getattr(artifact, "kind", "artifact")),
            original_chars=getattr(artifact, "original_chars", None),
            preview_chars=getattr(artifact, "preview_chars", None),
        )
    return working_state


def refresh_working_state_from_messages(
    working_state: WorkingState, messages: list[BaseMessage] | list[Any]
) -> WorkingState:
    latest_human = next(
        (
            _message_text(message)
            for message in reversed(messages)
            if isinstance(message, HumanMessage) and _message_text(message)
        ),
        None,
    )
    latest_ai = next(
        (
            _message_text(message)
            for message in reversed(messages)
            if isinstance(message, AIMessage) and _message_text(message)
        ),
        None,
    )

    if latest_human:
        working_state["goal"] = _trim_text(latest_human)
    if latest_ai:
        working_state["next_action"] = _trim_text(latest_ai)
    working_state["updated_at"] = _utc_now()
    return working_state


def build_continuity_checkpoint(working_state: WorkingState) -> ContinuityCheckpoint:
    checkpoint: ContinuityCheckpoint = {
        "version": "v1",
        "current_goal": working_state.get("goal"),
        "next_action": working_state.get("next_action"),
        "active_blockers": [
            blocker.get("summary", "")
            for blocker in working_state.get("blockers", [])
            if blocker.get("status", "open") != "resolved" and blocker.get("summary")
        ],
        "relevant_files": list(working_state.get("relevant_files", [])),
        "verification_status": working_state.get("verification", {}).get("status", "unknown"),
        "pending_verification": list(
            working_state.get("verification", {}).get("pending_checks", [])
        ),
        "recent_decisions": [
            decision.get("summary", "")
            for decision in working_state.get("decisions", [])
            if decision.get("summary")
        ],
        "recent_attempts": [
            attempt.get("action", "")
            for attempt in working_state.get("attempts", [])
            if attempt.get("action")
        ],
        "artifact_refs": list(working_state.get("artifact_refs", [])),
        "reference_summary": {
            "semantics": "historical_reference_only",
            "note": REFERENCE_ONLY_NOTE,
            "items": list(working_state.get("reference_notes", [])),
        },
        "updated_at": working_state.get("updated_at", _utc_now()),
    }
    if working_state.get("thread_id") is not None:
        checkpoint["thread_id"] = working_state.get("thread_id")
    if working_state.get("invocation_id") is not None:
        checkpoint["invocation_id"] = working_state.get("invocation_id")
    return checkpoint


class ContinuityMiddleware(AgentMiddleware[DeepClawAgentState, ContextT, ResponseT]):
    """Keep a structured working state plus checkpoint projection in agent state."""

    state_schema = DeepClawAgentState

    def _build_updates(self, state: DeepClawAgentState) -> dict[str, Any]:
        from deepclaw.runtime_hygiene import get_runtime_state

        working_state = ensure_working_state(state)
        refresh_working_state_from_messages(working_state, state.get("messages", []))
        sync_working_state_from_runtime(working_state, get_runtime_state())
        checkpoint = build_continuity_checkpoint(working_state)
        return {
            "working_state": working_state,
            "continuity_checkpoint": checkpoint,
        }

    def before_agent(self, state: DeepClawAgentState, runtime) -> dict[str, Any]:  # type: ignore[override]
        return self._build_updates(state)

    def before_model(self, state: DeepClawAgentState, runtime) -> dict[str, Any]:  # type: ignore[override]
        return self._build_updates(state)

    def after_model(self, state: DeepClawAgentState, runtime) -> dict[str, Any]:  # type: ignore[override]
        return self._build_updates(state)

    def after_agent(self, state: DeepClawAgentState, runtime) -> dict[str, Any]:  # type: ignore[override]
        return self._build_updates(state)
