"""Safety middleware for DeepClaw.

Intercepts tool calls to enforce safety policies:
- Shell commands checked against dangerous patterns (critical = blocked, warning = interrupt)
- File writes checked against deny list of sensitive paths
- URL fetches checked against SSRF blocklist
- Tool output scanned and redacted for credential leaks
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

from deepclaw.approval_state import aget_thread_approved_keys
from deepclaw.config import load_config
from deepclaw.safety import (
    check_command,
    check_url_safety_sync,
    check_write_path,
    format_warning,
    redact_secrets,
)

logger = logging.getLogger(__name__)

_USER_INTENT_BASH_RE = re.compile(
    r"\b(run|execute|command|cmd)\b.*\bbash\b|\bbash\b.*\b(run|execute|command|cmd)\b",
    re.IGNORECASE,
)
_USER_INTENT_INLINE_PYTHON_RE = re.compile(r"\bpython(?:3)?\s+-c\b", re.IGNORECASE)

# Tool names from DeepAgents that we intercept
_EXECUTE_TOOL = "execute"
_WRITE_FILE_TOOL = "write_file"
_EDIT_FILE_TOOL = "edit_file"
_WEB_FETCH_TOOLS = frozenset({"web_fetch", "http_request", "fetch_url"})

# Tools whose output should be scanned for credential leaks
_REDACT_OUTPUT_TOOLS = frozenset(
    {
        _EXECUTE_TOOL,
        _WRITE_FILE_TOOL,
        "read_file",
        "grep",
        "glob",
        "ls",
        *_WEB_FETCH_TOOLS,
    }
)


def _blocked_tool_message(tool_call: dict, reason: str) -> ToolMessage:
    """Create an error ToolMessage for a blocked tool call."""
    return ToolMessage(
        content=f"BLOCKED: {reason}",
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
        status="error",
    )


def _normalize_tool_args(args: Any) -> str:
    """Return a stable JSON fingerprint fragment for tool arguments."""
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return repr(args)


def _tool_call_fingerprint(tool_call: Mapping[str, Any]) -> str:
    """Return a stable fingerprint for a tool call name + arguments."""
    return f"{tool_call.get('name', '')}:{_normalize_tool_args(tool_call.get('args', {}))}"


def _tool_result_error_text(result: Any) -> str | None:
    """Extract a structured error string from a tool result when present."""
    if not isinstance(result, ToolMessage):
        return None

    content = result.content
    if isinstance(content, list):
        content_text = " ".join(str(part) for part in content)
    else:
        content_text = str(content or "")
    content_text = content_text.strip()

    if result.status == "error":
        return content_text or "tool returned error status"

    if not content_text:
        return None

    try:
        parsed = json.loads(content_text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, Mapping):
        error = parsed.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()

    if '"error"' in content_text or content_text.startswith("{error"):
        return content_text
    if content_text.startswith("Error:") or content_text.startswith("BLOCKED:"):
        return content_text
    return None


def _extract_last_user_text(state: Mapping[str, Any] | None) -> str:
    """Best-effort extraction of the most recent user message text from request state."""
    if not state:
        return ""

    messages = state.get("messages")
    if not isinstance(messages, list | tuple):
        return ""

    for message in reversed(messages):
        if isinstance(message, Mapping):
            role = str(message.get("role", "")).lower()
            if role != "user":
                continue
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)

        msg_type = str(getattr(message, "type", "")).lower()
        if msg_type not in {"human", "user"}:
            continue

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content)

    return ""


def _user_intent_requires_review(state: Mapping[str, Any] | None) -> str | None:
    """Return a warning reason when the user's request explicitly asks for risky inline execution."""
    user_text = _extract_last_user_text(state)
    if not user_text:
        return None

    if _USER_INTENT_INLINE_PYTHON_RE.search(user_text):
        return "User explicitly requested inline Python execution (`python -c`)"

    if _USER_INTENT_BASH_RE.search(user_text):
        return "User explicitly requested bash execution"

    return None


def _passthrough_env_warning() -> str | None:
    """Return an approval warning when shell commands inherit allowlisted sensitive env vars."""
    try:
        passthrough = list(load_config().terminal.env_passthrough)
    except Exception:
        passthrough = []
    if not passthrough:
        return None
    vars_text = ", ".join(sorted(passthrough))
    return (
        "Shell commands in this session inherit allowlisted sensitive environment variables: "
        f"{vars_text}"
    )


def _warning_keys(
    matches: list[Any],
    intent_warning: str | None,
    passthrough_warning: str | None,
) -> list[str]:
    """Return stable approval keys for session-scoped warning persistence."""
    keys: list[str] = []
    for match in matches:
        category = getattr(match, "category", "unknown")
        keys.append(f"dangerous:{category}")

    if intent_warning == "User explicitly requested inline Python execution (`python -c`)":
        keys.append("intent:inline_python")
    elif intent_warning == "User explicitly requested bash execution":
        keys.append("intent:bash")

    if passthrough_warning:
        try:
            passthrough = sorted(load_config().terminal.env_passthrough)
        except Exception:
            passthrough = []
        if passthrough:
            keys.append(f"env_passthrough:{','.join(passthrough)}")

    return list(dict.fromkeys(keys))


def _is_thread_approved(approved_keys: set[str] | None, warning_keys: list[str]) -> bool:
    """Return True when all warning keys were previously approved for this thread."""
    if not warning_keys or not approved_keys:
        return False
    return all(key in approved_keys for key in warning_keys)


def _extract_thread_id(request: Any) -> str | None:
    """Best-effort extraction of the active graph thread ID from the request runtime."""
    runtime = getattr(request, "runtime", None)
    config = getattr(runtime, "config", None)
    if isinstance(config, Mapping):
        configurable = config.get("configurable")
        if isinstance(configurable, Mapping):
            thread_id = configurable.get("thread_id")
            if thread_id:
                return str(thread_id)
    return None


def _is_cron_allowlisted(thread_id: str | None, warning_keys: list[str]) -> bool:
    """Return True when a cron thread's warning keys are explicitly allowlisted."""
    if not thread_id or not thread_id.startswith("cron-") or not warning_keys:
        return False
    try:
        allowlist = set(load_config().terminal.cron_approval_allowlist)
    except Exception:
        allowlist = set()
    return bool(allowlist) and all(key in allowlist for key in warning_keys)


def _check_execute(
    tool_call: dict,
    state: Mapping[str, Any] | None = None,
    *,
    thread_id: str | None = None,
    approved_keys: set[str] | None = None,
) -> ToolMessage | None:
    """Check a shell command for dangerous patterns.

    Returns a ToolMessage if the command should be blocked, None otherwise.
    Critical severity commands are hard-blocked. Warning severity commands
    trigger a human-in-the-loop interrupt.
    """
    command = tool_call.get("args", {}).get("command", "")
    if not command:
        return None

    matches = check_command(command)
    intent_warning = _user_intent_requires_review(state)
    passthrough_warning = _passthrough_env_warning()
    warning_keys = _warning_keys(matches, intent_warning, passthrough_warning)
    warnings = [
        reason
        for reason in (
            format_warning(command, matches) if matches else None,
            intent_warning,
            passthrough_warning,
        )
        if reason
    ]
    if not warnings:
        return None

    warning_text = "\n\n".join(warnings)
    has_critical = any(m.severity == "critical" for m in matches)

    # Critical-severity hard-block runs before any approval bypass, so a prior
    # session approval of a warning-level command in the same category cannot
    # silently authorize a critical command (e.g. `killall` -> `kill -9 -1`
    # both share the `mass_process_kill` category).
    if has_critical:
        logger.warning("Blocked critical command: %s", command)
        return _blocked_tool_message(tool_call, warning_text)

    if _is_thread_approved(approved_keys, warning_keys):
        logger.info(
            "Skipping approval for thread %s; warnings already approved for this session: %s",
            thread_id,
            warning_keys,
        )
        return None

    if _is_cron_allowlisted(thread_id, warning_keys):
        logger.info(
            "Skipping approval for cron thread %s; warnings allowlisted: %s",
            thread_id,
            warning_keys,
        )
        return None

    # Warning-level: ask the human
    logger.info("Requesting approval for command: %s", command)
    decision = interrupt(
        {
            "type": "safety_review",
            "tool": tool_call["name"],
            "command": command,
            "scope": "once",
            "approval_keys": warning_keys,
            "warning": warning_text,
            "message": f"⚠️ Potentially dangerous command:\n\n```\n{command}\n```\n\n{warning_text}\n\nApprove or deny?",
        }
    )

    if isinstance(decision, dict) and decision.get("type") == "approve":
        return None  # Proceed with execution

    reason = "Command rejected by user"
    if isinstance(decision, dict) and decision.get("message"):
        reason = decision["message"]
    return _blocked_tool_message(tool_call, reason)


def _check_write_path(tool_call: dict) -> ToolMessage | None:
    """Check a file write/edit path against the deny list."""
    path = tool_call.get("args", {}).get("path", "")
    if not path:
        return None

    is_safe, reason = check_write_path(path)
    if is_safe:
        return None

    logger.warning("Blocked file write to %s: %s", path, reason)
    return _blocked_tool_message(tool_call, reason)


def _check_url(tool_call: dict) -> ToolMessage | None:
    """Check a URL against the SSRF blocklist."""
    url = tool_call.get("args", {}).get("url", "")
    if not url:
        return None

    is_safe, reason = check_url_safety_sync(url)
    if is_safe:
        return None

    logger.warning("Blocked URL fetch to %s: %s", url, reason)
    return _blocked_tool_message(tool_call, reason)


try:
    from langchain.agents.middleware.types import (  # type: ignore[import-untyped]
        AgentMiddleware,
        ContextT,
        ResponseT,
    )
except ImportError:
    AgentMiddleware = None  # type: ignore[assignment, misc]

try:
    from langgraph.prebuilt.tool_node import ToolCallRequest  # noqa: F401
except ImportError:
    ToolCallRequest = None  # type: ignore[assignment, misc]

if AgentMiddleware is not None and ToolCallRequest is not None:
    StateT = Any

    class SafetyMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
        """Middleware that enforces safety policies on tool calls.

        - Shell commands: critical patterns blocked, warning patterns interrupt for approval
        - File writes: denied paths blocked outright
        - URL fetches: SSRF-unsafe URLs blocked
        - All tool output: credential patterns redacted
        - Repeated identical failing tool calls: blocked after two consecutive failures
        """

        def __init__(self) -> None:
            super().__init__()
            self._recent_failures: dict[str, dict[str, Any]] = {}

        def _check_repeated_failure(
            self, thread_id: str | None, tool_call: Mapping[str, Any]
        ) -> ToolMessage | None:
            """Block a third consecutive identical failing tool call in the same thread."""
            if not thread_id:
                return None

            record = self._recent_failures.get(thread_id)
            fingerprint = _tool_call_fingerprint(tool_call)
            if not record or record.get("fingerprint") != fingerprint:
                return None
            if int(record.get("count", 0)) < 2:
                return None

            error_preview = str(record.get("error") or "previous tool failure").strip()
            tool_name = str(tool_call.get("name") or "tool")
            reason = (
                "Repeated identical failing tool call blocked after "
                f"{record['count']} consecutive failures. "
                f"`{tool_name}` with the same arguments already failed repeatedly. "
                f"Last error: {error_preview}"
            )
            logger.warning(
                "Blocked repeated identical failing tool call on thread %s: %s", thread_id, reason
            )
            return _blocked_tool_message(dict(tool_call), reason)

        def _record_tool_result(
            self,
            thread_id: str | None,
            tool_call: Mapping[str, Any],
            result: ToolMessage | Command[Any],
        ) -> None:
            """Track consecutive identical failures per thread and reset on success/change."""
            if not thread_id:
                return

            fingerprint = _tool_call_fingerprint(tool_call)
            error_text = _tool_result_error_text(result)
            if error_text is None:
                self._recent_failures.pop(thread_id, None)
                return

            record = self._recent_failures.get(thread_id)
            if (
                record
                and record.get("fingerprint") == fingerprint
                and record.get("error") == error_text
            ):
                record["count"] = int(record.get("count", 0)) + 1
                return

            self._recent_failures[thread_id] = {
                "fingerprint": fingerprint,
                "error": error_text,
                "count": 1,
            }

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        ) -> ToolMessage | Command[Any]:
            """Intercept tool calls for safety checks, then redact output."""
            tool_call = request.tool_call
            tool_name = tool_call["name"]
            request_state = getattr(request, "state", None)
            thread_id = _extract_thread_id(request)

            repeated_failure = self._check_repeated_failure(thread_id, tool_call)
            if repeated_failure is not None:
                return repeated_failure

            # --- Pre-execution safety gates ---

            if tool_name == _EXECUTE_TOOL:
                approved_keys = await aget_thread_approved_keys(thread_id) if thread_id else set()
                blocked = _check_execute(
                    tool_call,
                    request_state,
                    thread_id=thread_id,
                    approved_keys=approved_keys,
                )
                if blocked is not None:
                    return blocked

            elif tool_name in (_WRITE_FILE_TOOL, _EDIT_FILE_TOOL):
                blocked = _check_write_path(tool_call)
                if blocked is not None:
                    return blocked

            elif tool_name in _WEB_FETCH_TOOLS:
                blocked = _check_url(tool_call)
                if blocked is not None:
                    return blocked

            # --- Execute the tool ---
            result = await handler(request)

            # --- Post-execution: redact secrets from output ---
            if isinstance(result, ToolMessage) and tool_name in _REDACT_OUTPUT_TOOLS:
                content = result.content
                if isinstance(content, str):
                    redacted = redact_secrets(content)
                    if redacted != content:
                        logger.info("Redacted credentials from %s output", tool_name)
                        result = ToolMessage(
                            content=redacted,
                            name=result.name,
                            tool_call_id=result.tool_call_id,
                            status=result.status,
                        )

            self._record_tool_result(thread_id, tool_call, result)
            return result
else:
    SafetyMiddleware = None  # type: ignore[assignment, misc]
    logger.warning("SafetyMiddleware unavailable — langchain middleware types not installed")
