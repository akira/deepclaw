"""Safety middleware for DeepClaw.

Intercepts tool calls to enforce safety policies:
- Shell commands checked against dangerous patterns (critical = blocked, warning = interrupt)
- File writes checked against deny list of sensitive paths
- URL fetches checked against SSRF blocklist
- Tool output scanned and redacted for credential leaks
"""

import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

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


def _check_execute(tool_call: dict, state: Mapping[str, Any] | None = None) -> ToolMessage | None:
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

    if has_critical:
        logger.warning("Blocked critical command: %s", command)
        return _blocked_tool_message(tool_call, warning_text)

    # Warning-level: ask the human
    logger.info("Requesting approval for command: %s", command)
    decision = interrupt(
        {
            "type": "safety_review",
            "tool": tool_call["name"],
            "command": command,
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
        """

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        ) -> ToolMessage | Command[Any]:
            """Intercept tool calls for safety checks, then redact output."""
            tool_call = request.tool_call
            tool_name = tool_call["name"]
            request_state = getattr(request, "state", None)

            # --- Pre-execution safety gates ---

            if tool_name == _EXECUTE_TOOL:
                blocked = _check_execute(tool_call, request_state)
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

            return result
else:
    SafetyMiddleware = None  # type: ignore[assignment, misc]
    logger.warning("SafetyMiddleware unavailable — langchain middleware types not installed")
