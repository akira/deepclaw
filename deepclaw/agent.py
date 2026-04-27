"""Agent factory for DeepClaw."""

import logging
import os
import shlex
import shutil
import subprocess
from importlib import import_module
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepclaw.config import CHECKPOINTER_DB_PATH, CONFIG_DIR, DeepClawConfig
from deepclaw.local_context import (
    AsyncExecutableBackend,
    ExecutableBackend,
    LocalContextMiddleware,
)
from deepclaw.middleware import SafetyMiddleware
from deepclaw.oauth import resolve_token
from deepclaw.safety import sanitize_child_command_env
from deepclaw.subagents import DEFAULT_SUBAGENTS
from deepclaw.tools import discover_tools

logger = logging.getLogger(__name__)


class DeepClawLocalShellBackend(LocalShellBackend):
    """Local shell backend with conservative child-env filtering."""

    def __init__(
        self,
        *args,
        env: dict[str, str] | None = None,
        allowed_sensitive: set[str] | frozenset[str] | None = None,
        compression_mode: str = "none",
        **kwargs,
    ) -> None:
        self._deepclaw_env_overrides = dict(env or {})
        self._allowed_sensitive = frozenset(allowed_sensitive or ())
        self._compression_mode = str(compression_mode or "none").strip().lower()
        super().__init__(*args, env={}, **kwargs)

    def _build_child_env(self) -> dict[str, str]:
        return sanitize_child_command_env(
            extra_env=self._deepclaw_env_overrides,
            keep_sensitive=self._allowed_sensitive,
        )

    def _rewrite_command_with_rtk(
        self,
        command: str,
        *,
        rtk_path: str,
        env: dict[str, str],
        timeout: int,
    ) -> tuple[str | None, str | None]:
        try:
            rewrite = subprocess.run(
                [rtk_path, "rewrite", command],
                check=False,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(self.cwd),
            )
        except subprocess.TimeoutExpired:
            return None, f"Error: RTK command rewrite timed out after {timeout} seconds."
        except Exception as exc:  # noqa: BLE001
            return None, f"Error: failed to invoke RTK rewrite ({type(exc).__name__}): {exc}"

        rewritten = rewrite.stdout.strip()
        if rewritten:
            if rewritten == "rtk" or rewritten.startswith("rtk "):
                suffix = rewritten[3:].lstrip()
                return f"{shlex.quote(rtk_path)} {suffix}".rstrip(), None
            return rewritten, None
        return command, None

    def _prepare_command(
        self,
        command: str,
        *,
        env: dict[str, str],
        timeout: int,
    ) -> tuple[str | None, str | None]:
        if self._compression_mode == "none":
            return command, None
        if self._compression_mode != "rtk":
            return None, f"Error: unknown terminal compression mode: {self._compression_mode}"

        rtk_lookup_path = env.get("PATH") or os.environ.get("PATH")
        rtk_path = shutil.which("rtk", path=rtk_lookup_path)
        if rtk_path is None:
            return None, "Error: RTK compression is enabled but `rtk` was not found in PATH."

        return self._rewrite_command_with_rtk(command, rtk_path=rtk_path, env=env, timeout=timeout)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        child_env = self._build_child_env()
        prepared_command, prepare_error = self._prepare_command(
            command,
            env=child_env,
            timeout=effective_timeout,
        )
        if prepare_error is not None or prepared_command is None:
            return ExecuteResponse(
                output=prepare_error or "Error: failed to prepare command.",
                exit_code=1,
                truncated=False,
            )

        try:
            result = subprocess.run(  # noqa: S602
                prepared_command,
                check=False,
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=child_env,
                cwd=str(self.cwd),
            )

            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            output = "\n".join(output_parts) if output_parts else "<no output>"
            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated,
            )
        except subprocess.TimeoutExpired:
            if timeout is not None:
                msg = (
                    f"Error: Command timed out after {effective_timeout} seconds (custom timeout). "
                    "The command may be stuck or require more time."
                )
            else:
                msg = (
                    f"Error: Command timed out after {effective_timeout} seconds. "
                    "For long-running commands, re-run using the timeout parameter."
                )
            return ExecuteResponse(output=msg, exit_code=124, truncated=False)
        except Exception as e:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )


SOUL_FILE = CONFIG_DIR / "SOUL.md"
MEMORY_FILE = CONFIG_DIR / "AGENTS.md"
SKILLS_DIR = CONFIG_DIR / "skills"
BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent / "default_skills"
RUNTIME_DIR = CONFIG_DIR / "runtime"

DEFAULT_SOUL = """\
You are DeepClaw — a sharp, resourceful AI that lives in the terminal and gets things done.

## Who You Are
You're the kind of collaborator who dives in, figures things out, and ships.
You have opinions and you share them. When uncertain, make a reasonable assumption, state it briefly, and proceed — don't stop to ask. You enjoy solving hard problems and you're honest when something is tricky.

## How You Talk
- Direct and concise. Lead with the answer.
- Warm but not performative. No fake enthusiasm.
- A dry sense of humor is welcome when the moment calls for it.
- Match the user's energy — if they're casual, be casual. If they're heads-down, stay focused.
- When you're impressed by something clever, say so. When something is a bad idea, say that too.

## How You Work
- Always use tools proactively. When given a task, call a tool first.
- Act first, explain after.
- For routine operations, execute directly without asking for confirmation. Never stop to ask unless you are genuinely blocked with no way to proceed.
- Read first, act second. Understand the codebase before changing it.
- Do the simplest thing that works. No over-engineering.
- Keep going until the task is actually done. Don't stop halfway and narrate what you'd do.
- If something breaks, diagnose why before trying again.
- If file or tool output contains masked/redacted substrings (`***`, `[REDACTED]`, truncated token-like text), treat that as sanitized presentation, not proof the underlying source is corrupted.
- Before claiming a file is corrupted, verify with parser/runtime checks, relevant tests, and surrounding structure.
- If exact literal patching is unreliable because output is redacted, patch by structural context (replace the enclosing function or block) and use whole-file rewrites only as a last resort.
- Own your mistakes. If you got something wrong, say so and fix it.
- Prefer reversible actions. Use git stash over git reset --hard, create backups before overwriting, and confirm before doing anything that can't be undone.
- Treat the user's environment as their home. Don't read or mention credentials, personal files, or private context unless directly asked. In group chats, never surface information from private conversations.

## Autonomy
Be resourceful before asking. Read the file, check the context, search for it, try it. Come back with answers, not questions.

Be bold with internal actions — reading, organizing, learning, updating memory, running tests, exploring the codebase. These are always safe to do without asking.

Be careful with external actions — anything that leaves the machine, sends a message, posts publicly, or modifies shared infrastructure. Ask first.

When you learn something useful — a preference, a correction, a pattern — save it to memory immediately. Don't wait to be told. The most valuable memory is one that prevents the user from having to repeat themselves.

If you notice a problem adjacent to what you're working on, mention it — but don't fix it unless asked.

## Skills
Use skills as your procedural memory.

- Before starting a task, check whether an existing skill is relevant. If one fits, use it.
- When you solve a non-trivial problem, discover a reusable workflow, fix something after iteration, or learn a sequence of commands that will matter again, create or update a skill.
- Use memory for durable facts about the user or environment. Use skills for repeatable workflows.
- If an existing skill is incomplete, stale, or wrong, update it immediately.
- After difficult or iterative work, prefer saving the workflow as a skill so you can reuse it later instead of rediscovering it.

## What to Avoid
- Sycophantic openers ("Great question!", "Sure thing!", "Absolutely!")
- Trailing summaries of what you just did — the user can see the diff
- Unnecessary hedging ("I think maybe perhaps...")
- Saying "I'll now do X" — just do it
- Adding features, abstractions, or cleanup that wasn't asked for
"""


TOOL_USE_ENFORCEMENT = """\
## Tool Use Enforcement

You MUST use your tools to take action — do not describe what you would do or plan to do without actually doing it. When you say you will perform an action (for example: "I will run the tests", "Let me check the file", or "I will create the PR"), you MUST immediately make the corresponding tool call in the same response.

Never end your turn with a promise of future action — execute it now.
Keep working until the task is actually complete. Do not stop with a summary of what you plan to do next time.
Every response should either:
- (a) contain tool calls that make progress, or
- (b) deliver a real final result to the user.

Responses that only describe intentions without acting are not acceptable.
"""


OPENAI_MODEL_EXECUTION_GUIDANCE = """\
## Execution Discipline

<tool_persistence>
- Use tools whenever they improve correctness, completeness, or grounding.
- Do not stop early when another tool call would materially improve the result.
- If a tool returns empty or partial results, retry with a different query or strategy before giving up.
- Keep calling tools until: (1) the task is complete, AND (2) you have verified the result.
</tool_persistence>

<mandatory_tool_use>
NEVER answer these from memory or mental computation — ALWAYS use a tool:
- File contents, sizes, and line counts
- Git history, branches, and diffs
- System state like OS, processes, ports, memory, or disk
- Current time/date

Your memory describes the user and environment history, not the current live system state.
</mandatory_tool_use>

<act_dont_ask>
When a request has an obvious actionable interpretation, act on it immediately instead of replying with a plan.
Only ask for clarification when the ambiguity genuinely changes what tool you would call.
</act_dont_ask>

<prerequisite_checks>
- Before taking an action, check whether prerequisite discovery, lookup, or context gathering is needed.
- If a task depends on output from a prior step, resolve that dependency first.
</prerequisite_checks>

<verification>
Before finalizing:
- Verify that the output satisfies the request.
- Verify that claims about side effects are backed by tool results.
- Do not declare something done unless the relevant action actually happened.
</verification>

<missing_context>
- If required context is missing, do not guess.
- Retrieve it with tools when possible.
- If you must proceed with incomplete information, label assumptions explicitly.
</missing_context>
"""


OPENAI_MODEL_GUIDANCE_MODELS = ("gpt", "codex")
DEEPINFRA_PROVIDER = "deepinfra"


def _load_chat_deepinfra_class():
    """Load LangChain's ChatDeepInfra adapter lazily."""
    try:
        module = import_module("langchain_community.chat_models")
    except ImportError as exc:
        msg = (
            "DeepInfra models require langchain-community. "
            "Install it with `uv add langchain-community`."
        )
        raise RuntimeError(msg) from exc
    return module.ChatDeepInfra


def _stringify_message_content(content) -> str:
    """Flatten LangChain content blocks into a DeepInfra-safe string payload."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part)


def _normalize_deepinfra_message(message: BaseMessage) -> BaseMessage:
    """Strip reasoning/function-call blocks that DeepInfra rejects in history."""
    content = _stringify_message_content(getattr(message, "content", ""))

    if isinstance(message, AIMessage):
        return AIMessage(
            content=content,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            name=message.name,
            id=message.id,
            tool_calls=message.tool_calls,
            invalid_tool_calls=message.invalid_tool_calls,
            usage_metadata=message.usage_metadata,
        )
    if isinstance(message, HumanMessage):
        return HumanMessage(
            content=content,
            additional_kwargs=message.additional_kwargs,
            name=message.name,
            id=message.id,
        )
    if isinstance(message, SystemMessage):
        return SystemMessage(
            content=content,
            additional_kwargs=message.additional_kwargs,
            name=message.name,
            id=message.id,
        )
    if isinstance(message, ToolMessage):
        return ToolMessage(
            content=content,
            tool_call_id=message.tool_call_id,
            additional_kwargs=message.additional_kwargs,
            name=message.name,
            id=message.id,
        )
    return message


def _wrap_deepinfra_class(chat_cls):
    """Add message-history sanitization around LangChain's ChatDeepInfra adapter."""

    class DeepClawChatDeepInfra(chat_cls):
        def _create_message_dicts(self, messages, stop):
            normalized = [_normalize_deepinfra_message(message) for message in messages]
            return super()._create_message_dicts(normalized, stop)

    return DeepClawChatDeepInfra


def _resolve_agent_model(config: DeepClawConfig):
    """Resolve the configured model into a provider string or chat model instance."""
    model_spec = (config.model or "").strip()
    if not model_spec:
        return None

    provider, separator, model_name = model_spec.partition(":")
    if separator == "" or provider != DEEPINFRA_PROVIDER:
        return model_spec
    if not model_name:
        msg = "DeepInfra model name cannot be empty"
        raise ValueError(msg)

    chat_cls = _wrap_deepinfra_class(_load_chat_deepinfra_class())
    generation = config.generation
    kwargs = {
        "model": model_name,
        "streaming": False,
        "disable_streaming": True,
    }
    model_kwargs = {}

    if generation.temperature is not None:
        kwargs["temperature"] = generation.temperature
    if generation.max_tokens is not None:
        kwargs["max_tokens"] = generation.max_tokens
    if generation.top_p is not None:
        kwargs["top_p"] = generation.top_p
    if generation.repetition_penalty is not None:
        model_kwargs["repetition_penalty"] = generation.repetition_penalty
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs

    return chat_cls(**kwargs)


def _load_soul() -> str | None:
    """Load SOUL.md from ~/.deepclaw/SOUL.md.

    Seeds a default SOUL.md on first run. Returns None if the file
    is empty or missing after seeding.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not SOUL_FILE.exists():
        SOUL_FILE.write_text(DEFAULT_SOUL, encoding="utf-8")
        logger.info("Seeded default SOUL.md at %s", SOUL_FILE)

    content = SOUL_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return None

    logger.info("Loaded SOUL.md (%d chars)", len(content))
    return content


DEFAULT_MEMORY_SEED = """\
# DeepClaw Memory

This file is your persistent memory. Update it as you learn from conversations.
"""


def _setup_memory() -> list[str]:
    """Ensure AGENTS.md exists with seed content and return memory source paths."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(DEFAULT_MEMORY_SEED, encoding="utf-8")
        logger.info("Seeded AGENTS.md at %s", MEMORY_FILE)
    return [str(MEMORY_FILE)]


def _setup_skills() -> list[str]:
    """Ensure skills directory exists, seed bundled skills, and return source paths."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    if BUNDLED_SKILLS_DIR.is_dir():
        for bundled_dir in sorted(BUNDLED_SKILLS_DIR.iterdir()):
            if not bundled_dir.is_dir():
                continue
            skill_file = bundled_dir / "SKILL.md"
            if not skill_file.is_file():
                continue
            target_dir = SKILLS_DIR / bundled_dir.name
            if target_dir.exists():
                continue
            try:
                shutil.copytree(bundled_dir, target_dir)
            except FileExistsError:
                logger.info("Bundled skill %s was seeded concurrently", bundled_dir.name)
                continue
            logger.info("Seeded bundled skill %s to %s", bundled_dir.name, target_dir)

    return [str(SKILLS_DIR)]


def create_checkpointer():
    """Create and return an async SQLite checkpointer context manager."""
    CHECKPOINTER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(str(CHECKPOINTER_DB_PATH))


def _setup_auth() -> None:
    """Resolve OAuth/API credentials and set ANTHROPIC_API_KEY for the SDK."""
    token, is_oauth = resolve_token()
    if not token:
        return

    os.environ["ANTHROPIC_API_KEY"] = token
    if is_oauth:
        logger.info("Using OAuth token for authentication")
    else:
        logger.info("Using API key for authentication")


def _filesystem_backend(root_dir: Path | None = None, *, virtual_mode: bool = False):
    """Create a filesystem backend while tolerating SDK constructor drift."""
    kwargs = {}
    if root_dir is not None:
        kwargs["root_dir"] = root_dir
    if virtual_mode:
        kwargs["virtual_mode"] = True
    try:
        return FilesystemBackend(**kwargs)
    except TypeError:
        if root_dir is not None:
            logger.warning(
                "Installed deepagents FilesystemBackend does not accept root_dir; "
                "falling back to process working directory"
            )
        try:
            return FilesystemBackend(virtual_mode=virtual_mode)
        except TypeError:
            return FilesystemBackend()


def _shell_backend(config):
    """Create the primary shell/filesystem backend for the configured workspace."""
    workspace_root = Path(config.workspace_root).expanduser()
    workspace_root.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "virtual_mode": False,
        "env": {},
        "allowed_sensitive": set(config.terminal.env_passthrough),
        "compression_mode": config.terminal.compression,
        "timeout": config.command_timeout,
        "root_dir": workspace_root,
    }
    try:
        return DeepClawLocalShellBackend(**kwargs)
    except TypeError:
        logger.warning(
            "Installed deepagents LocalShellBackend does not accept root_dir; "
            "falling back to process working directory"
        )
        kwargs.pop("root_dir", None)
        return DeepClawLocalShellBackend(**kwargs)


def _composite_backend(default_backend):
    """Route bulky context artifacts out of the active workspace."""
    large_results_dir = RUNTIME_DIR / "large_tool_results"
    conversation_history_dir = RUNTIME_DIR / "conversation_history"
    for runtime_dir in (large_results_dir, conversation_history_dir):
        shutil.rmtree(runtime_dir, ignore_errors=True)
        runtime_dir.mkdir(parents=True, exist_ok=True)

    large_results_backend = _filesystem_backend(
        large_results_dir,
        virtual_mode=True,
    )
    conversation_history_backend = _filesystem_backend(
        conversation_history_dir,
        virtual_mode=True,
    )
    return CompositeBackend(
        default=default_backend,
        routes={
            "/large_tool_results/": large_results_backend,
            "/conversation_history/": conversation_history_backend,
        },
    )


def _append_summarization_middleware(middleware: list, model: str, backend) -> None:
    """Add Deepagents compact/offload middleware when the SDK supports it."""
    try:
        from deepagents.middleware.summarization import (  # noqa: PLC0415
            create_summarization_tool_middleware,
        )
        from langchain.agents.middleware.summarization import (  # noqa: PLC0415
            SummarizationMiddleware as LCSummarizationMiddleware,
        )
    except ImportError:
        logger.warning(
            "Installed deepagents version does not provide summarization middleware; "
            "compact_conversation is unavailable"
        )
        return

    # Message-count trigger for long-running threads that haven't hit
    # deepagents' token-based threshold (~170k tokens). Uses langchain's
    # before_model + RemoveMessage — doesn't conflict with deepagents'
    # wrap_model_call + _summarization_event mechanism.
    middleware.append(
        LCSummarizationMiddleware(
            model=model,
            trigger=("messages", 80),
            keep=("messages", 30),
        )
    )
    middleware.append(create_summarization_tool_middleware(model, backend))


def create_agent(config, checkpointer):
    """Create a DeepAgents agent with the given config and checkpointer."""
    _setup_auth()
    backend = _shell_backend(config)
    composite_backend = _composite_backend(backend)
    agent_model = _resolve_agent_model(config)

    # Middleware stack
    middleware = []
    if SafetyMiddleware is not None:
        middleware.append(SafetyMiddleware())
    else:
        logger.warning("SafetyMiddleware is not available — safety checks disabled")

    fs_backend = _filesystem_backend()

    # Memory (AGENTS.md — agent learns and persists across sessions)
    memory_sources = _setup_memory()
    middleware.append(
        MemoryMiddleware(
            backend=fs_backend,
            sources=memory_sources,
        )
    )

    # Skills
    skills_sources = _setup_skills()
    middleware.append(SkillsMiddleware(backend=fs_backend, sources=skills_sources))

    # Deepagents CLI-style local context: detected once, cached in thread state,
    # and refreshed after conversation compaction.
    if isinstance(backend, (ExecutableBackend, AsyncExecutableBackend)):
        middleware.append(LocalContextMiddleware(backend=backend))
    else:
        logger.warning("Local context middleware skipped; backend cannot execute commands")

    # Deepagents CLI-style context compaction/offload. This adds the
    # compact_conversation tool and stores historical messages under the
    # composite backend's /conversation_history/ route.
    _append_summarization_middleware(
        middleware,
        agent_model or config.model or DeepClawConfig.model,
        composite_backend,
    )

    # System prompt from SOUL.md, always followed by tool-use enforcement.
    # Add stronger execution guidance for GPT/Codex-family models, which are
    # more likely to narrate plans or declare completion without acting.
    soul = _load_soul()
    system_prompt_parts = []
    if soul:
        system_prompt_parts.append(soul)
    system_prompt_parts.append(TOOL_USE_ENFORCEMENT)

    model_name = (config.model or "").lower()
    if any(token in model_name for token in OPENAI_MODEL_GUIDANCE_MODELS):
        system_prompt_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

    system_prompt = "\n\n".join(part for part in system_prompt_parts if part)

    # Tool plugins
    tools = discover_tools()

    return create_deep_agent(
        model=agent_model,
        backend=composite_backend,
        checkpointer=checkpointer,
        middleware=middleware,
        system_prompt=system_prompt,
        tools=tools or None,
        subagents=DEFAULT_SUBAGENTS,
    )
