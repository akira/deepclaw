"""Safety middleware for DeepClaw.

Intercepts tool calls to enforce safety policies:
- Shell commands checked against dangerous patterns (critical = blocked, warning = interrupt)
- File writes checked against deny list of sensitive paths
- URL fetches checked against SSRF blocklist
- Tool output scanned and redacted for credential leaks
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

try:
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command, interrupt
except ImportError as exc:  # pragma: no cover - optional dependency fallback
    ToolMessage = None
    Command = object
    interrupt = None
    _LANGCHAIN_CORE_IMPORT_ERROR = exc

from deepclaw.safety import (
    check_command,
    check_url_safety_sync,
    check_write_path,
    format_warning,
    redact_secrets,
)

logger = logging.getLogger(__name__)

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


def _blocked_tool_message(tool_call: dict, reason: str):
    """Create an error ToolMessage for a blocked tool call."""
    if ToolMessage is None:
        return {
            "content": f"BLOCKED: {reason}",
            "name": tool_call["name"],
            "tool_call_id": tool_call["id"],
            "status": "error",
        }
    return ToolMessage(
        content=f"BLOCKED: {reason}",
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
        status="error",
    )


def _check_execute(tool_call: dict):
    """Check a shell command for dangerous patterns.

    Returns a ToolMessage if the command should be blocked, None otherwise.
    Critical severity commands are hard-blocked. Warning severity commands
    trigger a human-in-the-loop interrupt.
    """
    command = tool_call.get("args", {}).get("command", "")
    if not command:
        return None

    matches = check_command(command)
    if not matches:
        return None

    warning_text = format_warning(command, matches)
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


def _check_write_path(tool_call: dict):
    """Check a file write/edit path against the deny list."""
    path = tool_call.get("args", {}).get("path", "")
    if not path:
        return None

    is_safe, reason = check_write_path(path)
    if is_safe:
        return None

    logger.warning("Blocked file write to %s: %s", path, reason)
    return _blocked_tool_message(tool_call, reason)


def _check_url(tool_call: dict):
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

if AgentMiddleware is not None and ToolCallRequest is not None and ToolMessage is not None and interrupt is not None:
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

            # --- Pre-execution safety gates ---

            if tool_name == _EXECUTE_TOOL:
                blocked = _check_execute(tool_call)
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
