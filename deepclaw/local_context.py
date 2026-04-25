"""Deepagents CLI-style local context middleware for DeepClaw."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, Protocol, cast, runtime_checkable

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        AgentState,
        ModelRequest,
        ModelResponse,
        PrivateStateAttr,
    )
except ImportError as exc:  # pragma: no cover - optional dependency fallback
    AgentMiddleware = object
    AgentState = dict
    ModelRequest = ModelResponse = PrivateStateAttr = object
    _LANGCHAIN_IMPORT_ERROR = exc

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import ExecuteResponse
    from deepagents.middleware.summarization import SummarizationEvent
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_DETECT_SCRIPT_TIMEOUT = 30


@runtime_checkable
class ExecutableBackend(Protocol):
    """Backend protocol for synchronous shell execution."""

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse: ...


@runtime_checkable
class AsyncExecutableBackend(Protocol):
    """Backend protocol for asynchronous shell execution."""

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse: ...


DETECT_CONTEXT_SCRIPT = r"""bash <<'__DEEPCLAW_CONTEXT_EOF__'
CWD="$(pwd)"
echo "## Local Context"
echo ""
echo "**Current Directory**: \`${CWD}\`"
echo ""

IN_GIT=false
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IN_GIT=true
fi

ROOT=""
$IN_GIT && ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"

if [ -f pyproject.toml ] || [ -f package.json ] || [ -f Cargo.toml ] || [ -f go.mod ] || [ -n "$ROOT" ]; then
  echo "**Project**:"
  [ -f pyproject.toml ] || [ -f setup.py ] && echo "- Language: python"
  [ -f package.json ] && echo "- Language: javascript/typescript"
  [ -f Cargo.toml ] && echo "- Language: rust"
  [ -f go.mod ] && echo "- Language: go"
  [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ] && echo "- Project root: \`${ROOT}\`"
  { [ -d .venv ] || [ -d venv ]; } && echo "- Environments: .venv"
  [ -d node_modules ] && echo "- Environments: node_modules"
  echo ""
fi

_DCT="$(mktemp -d)" || exit 1
trap 'rm -rf "$_DCT"' EXIT

(
  PKG=""
  if [ -f uv.lock ]; then PKG="Python: uv"
  elif [ -f poetry.lock ]; then PKG="Python: poetry"
  elif [ -f Pipfile.lock ] || [ -f Pipfile ]; then PKG="Python: pipenv"
  elif [ -f pyproject.toml ]; then PKG="Python: pip"
  elif [ -f requirements.txt ]; then PKG="Python: pip"
  fi

  NODE_PKG=""
  if [ -f bun.lockb ] || [ -f bun.lock ]; then NODE_PKG="Node: bun"
  elif [ -f pnpm-lock.yaml ]; then NODE_PKG="Node: pnpm"
  elif [ -f yarn.lock ]; then NODE_PKG="Node: yarn"
  elif [ -f package-lock.json ] || [ -f package.json ]; then NODE_PKG="Node: npm"
  fi

  [ -n "$NODE_PKG" ] && PKG="${PKG:+${PKG}, }${NODE_PKG}"
  [ -n "$PKG" ] && echo "**Package Manager**: ${PKG}" && echo ""
) > "$_DCT/pkg" 2>/dev/null &

(
  RT=""
  if command -v python3 >/dev/null 2>&1; then
    PV="$(python3 --version 2>/dev/null | awk '{print $2}')"
    [ -n "$PV" ] && RT="Python ${PV}"
  fi
  if command -v node >/dev/null 2>&1; then
    NV="$(node --version 2>/dev/null | sed 's/^v//')"
    [ -n "$NV" ] && RT="${RT:+${RT}, }Node ${NV}"
  fi
  [ -n "$RT" ] && echo "**Detected Runtimes**: ${RT}" && echo ""
) > "$_DCT/runtime" 2>/dev/null &

(
  if $IN_GIT; then
    BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    LINE="**Git**: Current branch \`${BRANCH}\`"
    DC="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    if [ "${DC:-0}" -gt 0 ]; then
      if [ "$DC" -eq 1 ]; then LINE="${LINE}, 1 uncommitted change"
      else LINE="${LINE}, ${DC} uncommitted changes"
      fi
    fi
    echo "$LINE"
    echo ""
  fi
) > "$_DCT/git" 2>/dev/null &

(
  TC=""
  if [ -f Makefile ] && grep -qE '^tests?:' Makefile 2>/dev/null; then TC="make test"
  elif [ -f pyproject.toml ] && { grep -q '\[tool\.pytest' pyproject.toml 2>/dev/null || [ -d tests ] || [ -d test ]; }; then TC="pytest"
  elif [ -f package.json ] && grep -q '"test"' package.json 2>/dev/null; then TC="npm test"
  fi
  [ -n "$TC" ] && echo "**Run Tests**: \`${TC}\`" && echo ""
) > "$_DCT/test" 2>/dev/null &

(
  EXCL='node_modules|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.tox|\.coverage|dist|build'
  FILES="$(ls -1 2>/dev/null | grep -vE "^(${EXCL})$" | sort -u)"
  if [ -n "$FILES" ]; then
    TOTAL="$(echo "$FILES" | wc -l | tr -d ' ')"
    SHOWN_FILES="$(echo "$FILES" | head -20)"
    SHOWN="$(echo "$SHOWN_FILES" | wc -l | tr -d ' ')"
    if [ "$SHOWN" -lt "$TOTAL" ]; then echo "**Files** (showing ${SHOWN} of ${TOTAL}):"
    else echo "**Files** (${TOTAL}):"
    fi
    echo "$SHOWN_FILES" | while IFS= read -r f; do
      [ -d "$f" ] && echo "- ${f}/" || echo "- ${f}"
    done
    echo ""
  fi
) > "$_DCT/files" 2>/dev/null &

wait
cat "$_DCT/pkg" "$_DCT/runtime" "$_DCT/git" "$_DCT/test" "$_DCT/files"
__DEEPCLAW_CONTEXT_EOF__
"""


class LocalContextState(AgentState):
    """State fields used by local context middleware."""

    local_context: NotRequired[str]
    _local_context_refreshed_at_cutoff: NotRequired[Annotated[int, PrivateStateAttr]]


class LocalContextMiddleware(AgentMiddleware):
    """Cache and inject compact local environment context into the system prompt."""

    state_schema = LocalContextState

    def __init__(self, backend: ExecutableBackend | AsyncExecutableBackend) -> None:
        self.backend = backend

    @staticmethod
    def _handle_detect_result(result: ExecuteResponse) -> str | None:
        output = result.output.strip() if result.output else ""
        if result.exit_code is None or result.exit_code != 0:
            logger.warning(
                "Local context detection failed with exit code %s; output: %.200s",
                result.exit_code,
                output or "(empty)",
            )
            return None
        return output or None

    def _run_detect_script(self) -> str | None:
        if not isinstance(self.backend, ExecutableBackend):
            return None
        try:
            result = self.backend.execute(DETECT_CONTEXT_SCRIPT, timeout=_DETECT_SCRIPT_TIMEOUT)
        except NotImplementedError:
            return None
        except Exception:
            logger.warning("Local context detection failed", exc_info=True)
            return None
        return self._handle_detect_result(result)

    async def _arun_detect_script(self) -> str | None:
        if isinstance(self.backend, AsyncExecutableBackend) and asyncio.iscoroutinefunction(
            self.backend.aexecute
        ):
            try:
                result = await self.backend.aexecute(
                    DETECT_CONTEXT_SCRIPT,
                    timeout=_DETECT_SCRIPT_TIMEOUT,
                )
            except Exception:
                logger.warning("Async local context detection failed", exc_info=True)
                return None
            return self._handle_detect_result(result)

        return await asyncio.to_thread(self._run_detect_script)

    @staticmethod
    def _refresh_cutoff(state: LocalContextState) -> int | None:
        raw_event = state.get("_summarization_event")
        if raw_event is None:
            return None
        event: SummarizationEvent = raw_event
        cutoff = event.get("cutoff_index")
        if cutoff == state.get("_local_context_refreshed_at_cutoff"):
            return None
        return cutoff

    def before_agent(
        self,
        state: LocalContextState,
        runtime: Runtime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        cutoff = self._refresh_cutoff(state)
        if cutoff is not None:
            output = self._run_detect_script()
            if output:
                return {
                    "local_context": output,
                    "_local_context_refreshed_at_cutoff": cutoff,
                }
            return {"_local_context_refreshed_at_cutoff": cutoff}

        if state.get("local_context"):
            return None

        output = self._run_detect_script()
        return {"local_context": output} if output else None

    async def abefore_agent(
        self,
        state: LocalContextState,
        runtime: Runtime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        cutoff = self._refresh_cutoff(state)
        if cutoff is not None:
            output = await self._arun_detect_script()
            if output:
                return {
                    "local_context": output,
                    "_local_context_refreshed_at_cutoff": cutoff,
                }
            return {"_local_context_refreshed_at_cutoff": cutoff}

        if state.get("local_context"):
            return None

        output = await self._arun_detect_script()
        return {"local_context": output} if output else None

    def _get_modified_request(self, request: ModelRequest) -> ModelRequest | None:
        state = cast("LocalContextState", request.state)
        local_context = state.get("local_context", "")
        if not local_context:
            return None

        system_prompt = request.system_prompt or ""
        return request.override(system_prompt=f"{system_prompt}\n\n{local_context}")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._get_modified_request(request) or request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._get_modified_request(request) or request)


__all__ = [
    "AsyncExecutableBackend",
    "DETECT_CONTEXT_SCRIPT",
    "ExecutableBackend",
    "LocalContextMiddleware",
    "LocalContextState",
]
