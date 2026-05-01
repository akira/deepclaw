"""Tests for deepclaw.agent helpers."""

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from deepclaw import agent as agent_mod
from deepclaw.config import DeepClawConfig


class TestSetupSkills:
    def test_seeds_bundled_skills_into_config_dir(self, tmp_path, monkeypatch):
        bundled_dir = tmp_path / "bundled"
        installed_dir = tmp_path / "installed"
        bundled_skill = bundled_dir / "systematic-debugging"
        bundled_skill.mkdir(parents=True)
        (bundled_skill / "SKILL.md").write_text(
            "---\nname: systematic-debugging\ndescription: Debug carefully\n---\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(agent_mod, "BUNDLED_SKILLS_DIR", bundled_dir)
        monkeypatch.setattr(agent_mod, "SKILLS_DIR", installed_dir)

        result = agent_mod._setup_skills()

        installed_skill = installed_dir / "systematic-debugging" / "SKILL.md"
        assert result == [str(installed_dir)]
        assert installed_skill.is_file()
        assert "Debug carefully" in installed_skill.read_text(encoding="utf-8")

    def test_does_not_overwrite_existing_installed_skill(self, tmp_path, monkeypatch):
        bundled_dir = tmp_path / "bundled"
        installed_dir = tmp_path / "installed"
        bundled_skill = bundled_dir / "systematic-debugging"
        bundled_skill.mkdir(parents=True)
        (bundled_skill / "SKILL.md").write_text(
            "---\nname: systematic-debugging\ndescription: bundled copy\n---\n",
            encoding="utf-8",
        )

        installed_skill_dir = installed_dir / "systematic-debugging"
        installed_skill_dir.mkdir(parents=True)
        installed_skill = installed_skill_dir / "SKILL.md"
        installed_skill.write_text(
            "---\nname: systematic-debugging\ndescription: local copy\n---\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(agent_mod, "BUNDLED_SKILLS_DIR", bundled_dir)
        monkeypatch.setattr(agent_mod, "SKILLS_DIR", installed_dir)

        agent_mod._setup_skills()

        assert "local copy" in installed_skill.read_text(encoding="utf-8")

    def test_ignores_concurrent_seed_race(self, tmp_path, monkeypatch):
        bundled_dir = tmp_path / "bundled"
        installed_dir = tmp_path / "installed"
        bundled_skill = bundled_dir / "systematic-debugging"
        bundled_skill.mkdir(parents=True)
        (bundled_skill / "SKILL.md").write_text(
            "---\nname: systematic-debugging\ndescription: bundled copy\n---\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(agent_mod, "BUNDLED_SKILLS_DIR", bundled_dir)
        monkeypatch.setattr(agent_mod, "SKILLS_DIR", installed_dir)

        def _racing_copytree(src, dst):
            raise FileExistsError(dst)

        monkeypatch.setattr(agent_mod.shutil, "copytree", _racing_copytree)

        result = agent_mod._setup_skills()

        assert result == [str(installed_dir)]


class TestDeepClawLocalShellBackend:
    def test_prepare_command_uses_rtk_rewrite_when_enabled(self, monkeypatch):
        backend = agent_mod.DeepClawLocalShellBackend.__new__(agent_mod.DeepClawLocalShellBackend)
        backend._compression_mode = "rtk"
        monkeypatch.setattr(
            agent_mod.shutil,
            "which",
            lambda name, path=None: "/usr/bin/rtk" if name == "rtk" else None,
        )
        monkeypatch.setattr(
            backend,
            "_rewrite_command_with_rtk",
            lambda command, *, rtk_path, env, timeout: (f"{rtk_path} git status", None),
        )

        wrapped, error = backend._prepare_command(
            "git status && echo hi",
            env={"PATH": "/usr/bin"},
            timeout=30,
        )

        assert error is None
        assert wrapped == "/usr/bin/rtk git status"

    def test_prepare_command_errors_when_rtk_enabled_but_binary_missing(self, monkeypatch):
        backend = agent_mod.DeepClawLocalShellBackend.__new__(
            agent_mod.DeepClawLocalShellBackend,
        )
        backend._compression_mode = "rtk"
        monkeypatch.setattr(agent_mod.shutil, "which", lambda _name, path=None: None)

        wrapped, error = backend._prepare_command(
            "git status",
            env={"PATH": "/usr/bin"},
            timeout=30,
        )

        assert wrapped is None
        assert "RTK compression is enabled" in error

    def test_prepare_command_uses_process_path_when_child_env_omits_path(self, monkeypatch):
        backend = agent_mod.DeepClawLocalShellBackend.__new__(agent_mod.DeepClawLocalShellBackend)
        backend._compression_mode = "rtk"
        monkeypatch.setattr(agent_mod.os, "environ", {"PATH": "/process/bin"})
        monkeypatch.setattr(
            agent_mod.shutil,
            "which",
            lambda name, path=None: (
                "/process/bin/rtk" if name == "rtk" and path == "/process/bin" else None
            ),
        )
        monkeypatch.setattr(
            backend,
            "_rewrite_command_with_rtk",
            lambda command, *, rtk_path, env, timeout: (f"{rtk_path} git status", None),
        )

        wrapped, error = backend._prepare_command(
            "git status",
            env={},
            timeout=30,
        )

        assert error is None
        assert wrapped == "/process/bin/rtk git status"

    def test_prepare_command_errors_when_compression_mode_is_unknown(self):
        backend = agent_mod.DeepClawLocalShellBackend.__new__(
            agent_mod.DeepClawLocalShellBackend,
        )
        backend._compression_mode = "gzip"

        wrapped, error = backend._prepare_command(
            "git status",
            env={"PATH": "/usr/bin"},
            timeout=30,
        )

        assert wrapped is None
        assert "unknown terminal compression mode" in error

    def test_prepare_command_leaves_command_unchanged_when_disabled(self):
        backend = agent_mod.DeepClawLocalShellBackend.__new__(
            agent_mod.DeepClawLocalShellBackend,
        )
        backend._compression_mode = "none"

        wrapped, error = backend._prepare_command(
            "git status",
            env={"PATH": "/usr/bin"},
            timeout=30,
        )

        assert error is None
        assert wrapped == "git status"

    def test_execute_uses_original_command_when_compression_is_off(self, monkeypatch):
        backend = agent_mod.DeepClawLocalShellBackend(root_dir=".", virtual_mode=False, timeout=30)
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(command, 0, stdout="plain-output\n", stderr="")

        monkeypatch.setattr(agent_mod.subprocess, "run", fake_run)

        result = backend.execute("printf plain-output")

        assert result.exit_code == 0
        assert result.output == "plain-output\n"
        assert captured["command"] == "printf plain-output"

    def test_execute_uses_rtk_rewritten_command_when_compression_is_on(self, monkeypatch):
        backend = agent_mod.DeepClawLocalShellBackend(
            root_dir=".",
            virtual_mode=False,
            timeout=30,
            compression_mode="rtk",
        )
        captured = {"rewrite": None, "executed": None}
        monkeypatch.setattr(
            agent_mod.shutil,
            "which",
            lambda name, path=None: "/usr/bin/rtk" if name == "rtk" else None,
        )

        def fake_run(command, **kwargs):
            if isinstance(command, list):
                captured["rewrite"] = command
                return subprocess.CompletedProcess(command, 0, stdout="rtk git status\n", stderr="")
            captured["executed"] = command
            return subprocess.CompletedProcess(command, 0, stdout="rtk-output\n", stderr="")

        monkeypatch.setattr(agent_mod.subprocess, "run", fake_run)

        result = backend.execute("git status")

        assert result.exit_code == 0
        assert result.output == "rtk-output\n"
        assert captured["rewrite"] == ["/usr/bin/rtk", "rewrite", "git status"]
        assert captured["executed"] == "/usr/bin/rtk git status"

    def test_execute_falls_back_to_original_command_when_rtk_has_no_rewrite(self, monkeypatch):
        backend = agent_mod.DeepClawLocalShellBackend(
            root_dir=".",
            virtual_mode=False,
            timeout=30,
            compression_mode="rtk",
        )
        captured = {"rewrite": None, "executed": None}
        monkeypatch.setattr(
            agent_mod.shutil,
            "which",
            lambda name, path=None: "/usr/bin/rtk" if name == "rtk" else None,
        )

        def fake_run(command, **kwargs):
            if isinstance(command, list):
                captured["rewrite"] = command
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
            captured["executed"] = command
            return subprocess.CompletedProcess(command, 0, stdout="plain-output\n", stderr="")

        monkeypatch.setattr(agent_mod.subprocess, "run", fake_run)

        result = backend.execute("printf plain-output")

        assert result.exit_code == 0
        assert result.output == "plain-output\n"
        assert captured["rewrite"] == ["/usr/bin/rtk", "rewrite", "printf plain-output"]
        assert captured["executed"] == "printf plain-output"

    def test_prepare_command_uses_rewrite_stdout_even_when_rtk_returns_nonzero(self, monkeypatch):
        backend = agent_mod.DeepClawLocalShellBackend(
            root_dir=".",
            virtual_mode=False,
            timeout=30,
            compression_mode="rtk",
        )
        monkeypatch.setattr(
            agent_mod.shutil,
            "which",
            lambda name, path=None: "/usr/bin/rtk" if name == "rtk" else None,
        )

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 3, stdout="rtk git status\n", stderr="")

        monkeypatch.setattr(agent_mod.subprocess, "run", fake_run)

        wrapped, error = backend._prepare_command(
            "git status",
            env={"PATH": "/usr/bin"},
            timeout=30,
        )

        assert error is None
        assert wrapped == "/usr/bin/rtk git status"

    @pytest.mark.skipif(shutil.which("rtk") is None, reason="rtk not installed")
    def test_execute_with_real_rtk_matches_plain_output_for_simple_command(self, monkeypatch):
        rtk_dir = Path(shutil.which("rtk")).resolve().parent
        monkeypatch.setenv("PATH", f"{rtk_dir}:{os.environ.get('PATH', '')}")

        plain_backend = agent_mod.DeepClawLocalShellBackend(
            root_dir=".", virtual_mode=False, timeout=30
        )
        rtk_backend = agent_mod.DeepClawLocalShellBackend(
            root_dir=".",
            virtual_mode=False,
            timeout=30,
            compression_mode="rtk",
        )

        plain = plain_backend.execute("printf hello-from-backend")
        compressed = rtk_backend.execute("printf hello-from-backend")

        assert plain.exit_code == 0
        assert compressed.exit_code == 0
        assert plain.output == "hello-from-backend"
        assert compressed.output == "hello-from-backend"


class TestContextManagementSettings:
    def test_uses_fraction_based_compaction_and_filesystem_eviction_defaults(self):
        settings = agent_mod._context_management_settings()

        assert settings["summarization"]["trigger"] == ("fraction", 0.85)
        assert settings["summarization"]["keep"] == ("fraction", 0.10)
        assert settings["summarization"]["truncate_args_settings"] == {
            "trigger": ("fraction", 0.50),
            "keep": ("fraction", 0.10),
            "max_length": 2000,
            "truncation_text": "...(argument truncated)",
        }
        assert settings["filesystem"] == {
            "tool_token_limit_before_evict": 20000,
            "human_message_token_limit_before_evict": 50000,
        }

    def test_uses_structured_checkpoint_summary_prompt(self):
        prompt = agent_mod.CONTEXT_CHECKPOINT_SUMMARY_PROMPT

        assert "## Active Task" in prompt
        assert "## Goal" in prompt
        assert "## Constraints & Preferences" in prompt
        assert "## Completed Actions" in prompt
        assert "## Active State" in prompt
        assert "## Next Steps" in prompt


class TestCompactionFallbackSummary:
    def test_replaces_useless_placeholder_with_structured_fallback(self):
        messages = [
            HumanMessage(
                content="Please fix context bloat in DeepClaw and keep compact_conversation available."
            ),
            AIMessage(
                content="I switched to DeepAgents middleware and reviewed the trace evidence."
            ),
        ]

        result = agent_mod._normalize_compaction_summary(
            "Previous conversation was too long to summarize.",
            messages,
        )

        assert "## Active Task" in result
        assert "context bloat in DeepClaw" in result
        assert "## Goal" in result
        assert "Please fix context bloat in DeepClaw" in result
        assert "## Completed Actions" in result
        assert "reviewed the trace evidence" in result
        assert "## Next Steps" in result
        assert "Continue the concrete active task" in result
        assert "Continue the user's in-progress objective" not in result

    def test_replaces_generic_structured_summary_that_loses_task_identity(self):
        messages = [
            HumanMessage(
                content="Clone the OpenSWE repo and prepare a detailed adaptation report for DeepClaw."
            ),
            AIMessage(content="I cloned the repo and started reading the prompt and system files."),
        ]
        summary = """## Active Task
None

## Goal
Continue the user's in-progress objective without losing critical prior context.

## Constraints & Preferences
- Preserve prior decisions.

## Completed Actions
- Cloned the repo.

## Active State
- Repo available locally.

## Next Steps
- Continue from the latest user request.
"""

        result = agent_mod._normalize_compaction_summary(summary, messages)

        assert "Clone the OpenSWE repo" in result
        assert "Continue the user's in-progress objective" not in result
        assert "Continue from the latest user request" not in result
        assert "Build directly on this latest completed work" in result

    def test_keeps_useful_summary_that_mentions_none_as_normal_text(self):
        messages = [
            HumanMessage(
                content="Audit the branch and explain whether none of the commits regress compaction."
            ),
        ]
        summary = """## Active Task
Audit the branch and explain whether none of the commits regress compaction.

## Goal
Determine whether the compaction fix is safe to merge.

## Constraints & Preferences
- None beyond preserving the current branch state.

## Completed Actions
- Compared the current branch against origin/main.

## Active State
- Branch is fix/compaction-task-handoff.

## Next Steps
- Summarize findings and prepare the PR description.
"""

        assert agent_mod._normalize_compaction_summary(summary, messages) == summary.strip()

    def test_keeps_useful_summary_content(self):
        summary = "## Active Task\nShip the fix\n\n## Goal\nReduce context bloat"

        assert agent_mod._normalize_compaction_summary(summary, []) == summary


class TestDeepClawSummarizationFactory:
    def test_builds_custom_summarization_middleware_with_structured_prompt(self, monkeypatch):
        captured = {}

        class FakeSummarizationMiddleware:
            def __init__(self, **kwargs):
                captured["summarization_kwargs"] = kwargs

        fake_module = types.ModuleType("deepagents.middleware.summarization")
        fake_module.SummarizationMiddleware = FakeSummarizationMiddleware
        fake_module.SummarizationToolMiddleware = type("FakeSummarizationToolMiddleware", (), {})
        monkeypatch.setitem(sys.modules, "deepagents.middleware.summarization", fake_module)

        middleware = agent_mod._create_deepclaw_summarization_middleware("test:model", "backend")

        assert (
            captured["summarization_kwargs"]["summary_prompt"]
            == agent_mod.CONTEXT_CHECKPOINT_SUMMARY_PROMPT
        )
        assert middleware.name == "deepclaw_summarization"

    def test_falls_back_to_absolute_token_thresholds_without_model_profile(self, monkeypatch):
        captured = {"calls": []}

        class FakeSummarizationMiddleware:
            def __init__(self, **kwargs):
                captured["calls"].append(kwargs)
                if kwargs["trigger"] == ("fraction", 0.85):
                    raise ValueError(
                        "Model profile information is required to use fractional token limits"
                    )

        fake_module = types.ModuleType("deepagents.middleware.summarization")
        fake_module.SummarizationMiddleware = FakeSummarizationMiddleware
        fake_module.SummarizationToolMiddleware = type("FakeSummarizationToolMiddleware", (), {})
        monkeypatch.setitem(sys.modules, "deepagents.middleware.summarization", fake_module)

        agent_mod._create_deepclaw_summarization_middleware(
            "baseten:moonshotai/Kimi-K2.6", "backend"
        )

        assert len(captured["calls"]) == 2
        assert captured["calls"][1]["trigger"] == ("tokens", 80000)
        assert captured["calls"][1]["keep"] == ("messages", 8)
        assert captured["calls"][1]["truncate_args_settings"]["trigger"] == (
            "tokens",
            40000,
        )

    def test_builds_manual_compaction_tool_middleware(self, monkeypatch):
        class FakeSummarizationMiddleware:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeSummarizationToolMiddleware:
            def __init__(self, summarization):
                self.summarization = summarization

        fake_module = types.ModuleType("deepagents.middleware.summarization")
        fake_module.SummarizationMiddleware = FakeSummarizationMiddleware
        fake_module.SummarizationToolMiddleware = FakeSummarizationToolMiddleware
        monkeypatch.setitem(sys.modules, "deepagents.middleware.summarization", fake_module)

        middleware = agent_mod._create_deepclaw_summarization_tool_middleware(
            "test:model", "backend"
        )

        assert middleware.name == "deepclaw_summarization_tool"
        assert middleware.summarization.name == "deepclaw_summarization"

    def test_builds_production_subagents_with_manual_compaction(self, monkeypatch):
        monkeypatch.setattr(
            agent_mod,
            "_create_deepclaw_summarization_tool_middleware",
            lambda model, backend: f"compact:{model}:{backend}",
        )

        subagents = agent_mod._build_deepclaw_subagents("test:model", "backend")

        assert subagents[0]["name"] == "general-purpose"
        assert subagents[0]["middleware"] == ["compact:test:model:backend"]
        assert {spec["name"] for spec in subagents[1:]} == {"researcher", "coder", "sysadmin"}
        assert all(spec["middleware"] == ["compact:test:model:backend"] for spec in subagents)
        assert all(
            spec is not source
            for spec, source in zip(subagents[1:], agent_mod.DEFAULT_SUBAGENTS, strict=False)
        )

    def test_patches_deepagents_production_summarization_factory(self, monkeypatch):
        import deepagents.graph as deepagents_graph

        original_factory = deepagents_graph.create_summarization_middleware
        monkeypatch.setattr(
            agent_mod,
            "_create_deepclaw_summarization_middleware",
            lambda model, backend: ("custom", model, backend),
        )

        with agent_mod._patched_deepagents_summarization_factory():
            assert deepagents_graph.create_summarization_middleware("m", "b") == (
                "custom",
                "m",
                "b",
            )

        assert deepagents_graph.create_summarization_middleware is original_factory

    def test_serializes_factory_patch_with_lock(self, monkeypatch):
        events = []

        class FakeLock:
            def __enter__(self):
                events.append("enter")
                return self

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
                return False

        monkeypatch.setattr(agent_mod, "_DEEPAGENTS_SUMMARIZATION_FACTORY_LOCK", FakeLock())

        with agent_mod._patched_deepagents_summarization_factory():
            events.append("inside")

        assert events == ["enter", "inside", "exit"]


class FakeMiddlewareBackend:
    def __init__(self):
        self.writes = []
        self.download_requests = []

    def write(self, path, content):
        self.writes.append((path, content))
        return types.SimpleNamespace(error=None)

    async def awrite(self, path, content):
        self.writes.append((path, content))
        return types.SimpleNamespace(error=None)

    def download_files(self, paths):
        self.download_requests.append(tuple(paths))
        return []


def _build_test_summarization_middleware(monkeypatch, backend):
    import deepagents.middleware.summarization as deep_sum

    class FakeLCSummarizationMiddleware:
        def __init__(
            self,
            model,
            trigger,
            keep,
            token_counter,
            summary_prompt,
            trim_tokens_to_summarize,
        ):
            self.model = model
            self.trigger = trigger
            self.keep = keep
            self.token_counter = token_counter
            self.summary_prompt = summary_prompt
            self.trim_tokens_to_summarize = trim_tokens_to_summarize

        def _get_profile_limits(self):
            return 1000

        def _should_summarize(self, messages, total_tokens):
            trigger_type, trigger_value = self.trigger
            if trigger_type == "fraction":
                return total_tokens >= int(self._get_profile_limits() * trigger_value)
            return False

        def _determine_cutoff_index(self, messages):
            return max(1, len(messages) - 2)

        def _partition_messages(self, messages, cutoff_index):
            return messages[:cutoff_index], messages[cutoff_index:]

        def _create_summary(self, messages):
            return "## Active Task\nStub summary"

        async def _acreate_summary(self, messages):
            return self._create_summary(messages)

        def _partial_token_counter(self, messages):
            return 60

    monkeypatch.setattr(deep_sum, "LCSummarizationMiddleware", FakeLCSummarizationMiddleware)

    return agent_mod._create_deepclaw_summarization_middleware("test:model", backend)


class TestRuntimeContextManagementBehavior:
    def test_wrap_model_call_summarizes_and_offloads_history(self, monkeypatch):
        backend = FakeMiddlewareBackend()
        middleware = _build_test_summarization_middleware(monkeypatch, backend)

        middleware._lc_helper.token_counter = lambda messages, tools=None: 900
        monkeypatch.setattr(middleware, "_determine_cutoff_index", lambda messages: 2)
        monkeypatch.setattr(
            middleware, "_create_summary", lambda messages: "## Active Task\nSummarized work"
        )

        captured = {}

        def handler(request):
            captured["messages"] = request.messages
            return ModelResponse(result=[AIMessage(content="ok")])

        request = ModelRequest(
            model=Mock(),
            messages=[
                HumanMessage(content="old user request"),
                AIMessage(content="old assistant work"),
                HumanMessage(content="recent request"),
                AIMessage(content="recent assistant work"),
            ],
            tools=[],
            state={},
            runtime=Runtime(context=None),
        )

        response = middleware.wrap_model_call(request, handler)

        assert isinstance(response, ExtendedModelResponse)
        assert backend.writes
        offload_path, offload_content = backend.writes[0]
        assert offload_path.startswith("/conversation_history/")
        assert "old user request" in offload_content
        assert "old assistant work" in offload_content
        summary_message = captured["messages"][0]
        assert offload_path in summary_message.content
        assert "## Active Task\nSummarized work" in summary_message.content
        assert response.command.update["_summarization_event"]["file_path"] == offload_path

    def test_wrap_model_call_truncates_old_tool_args_before_handler(self, monkeypatch):
        backend = FakeMiddlewareBackend()
        middleware = _build_test_summarization_middleware(monkeypatch, backend)

        middleware._lc_helper.token_counter = lambda messages, tools=None: 600
        monkeypatch.setattr(middleware, "_determine_truncate_cutoff_index", lambda messages: 1)

        oversized_call = {
            "name": "write_file",
            "args": {"file_path": "/tmp/demo.txt", "content": "x" * 2500},
            "id": "call_1",
            "type": "tool_call",
        }
        captured = {}

        def handler(request):
            captured["messages"] = request.messages
            return ModelResponse(result=[AIMessage(content="ok")])

        request = ModelRequest(
            model=Mock(),
            messages=[
                AIMessage(content="tool call", tool_calls=[oversized_call]),
                HumanMessage(content="recent user message"),
            ],
            tools=[],
            state={},
            runtime=Runtime(context=None),
        )

        middleware.wrap_model_call(request, handler)

        truncated_content = captured["messages"][0].tool_calls[0]["args"]["content"]
        assert truncated_content.endswith("...(argument truncated)")
        assert len(truncated_content) < len(oversized_call["args"]["content"])

    def test_filesystem_middleware_offloads_large_tool_results(self):
        backend = FakeMiddlewareBackend()
        middleware = FilesystemMiddleware(
            backend=backend,
            tool_token_limit_before_evict=1,
            human_message_token_limit_before_evict=50000,
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={},
            stream_writer=lambda *_args, **_kwargs: None,
            tools=[],
            tool_call_id="call_1",
            store=None,
        )
        request = ToolCallRequest(
            tool_call={"name": "search_files", "args": {}, "id": "call_1", "type": "tool_call"},
            tool=None,
            state={},
            runtime=runtime,
        )

        result = middleware.wrap_tool_call(
            request,
            lambda _request: ToolMessage(
                content="X" * 40,
                name="search_files",
                tool_call_id="call_1",
            ),
        )

        assert backend.writes[0][0] == "/large_tool_results/call_1"
        assert "/large_tool_results/call_1" in result.content

    def test_filesystem_middleware_offloads_large_human_messages(self):
        backend = FakeMiddlewareBackend()
        middleware = FilesystemMiddleware(
            backend=backend,
            tool_token_limit_before_evict=20000,
            human_message_token_limit_before_evict=1,
        )
        captured = {}

        def handler(request):
            captured["messages"] = request.messages
            return ModelResponse(result=[AIMessage(content="ok")])

        request = ModelRequest(
            model=Mock(),
            messages=[
                HumanMessage(content="small"),
                HumanMessage(content="Y" * 40),
            ],
            tools=[],
            state={},
            runtime=Runtime(context=None),
        )

        response = middleware.wrap_model_call(request, handler)

        assert isinstance(response, ExtendedModelResponse)
        offload_path, offload_content = backend.writes[0]
        assert offload_path.startswith("/conversation_history/")
        assert offload_content == "Y" * 40
        assert offload_path in str(captured["messages"][-1].content)
        assert (
            response.command.update["messages"][0].additional_kwargs["lc_evicted_to"]
            == offload_path
        )


class TestCreateAgent:
    def test_wires_cli_style_context_backends_and_manual_compaction_tool(
        self, tmp_path, monkeypatch
    ):
        captured = {}

        class FakeShellBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def execute(self, command, *, timeout=None):
                return None

        class FakeFilesystemBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeCompositeBackend:
            def __init__(self, *, default, routes):
                self.default = default
                self.routes = routes

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return "agent"


        monkeypatch.setattr(agent_mod, "_load_soul", lambda: "soul")
        monkeypatch.setattr(agent_mod, "_setup_memory", lambda: ["/memory/AGENTS.md"])
        monkeypatch.setattr(agent_mod, "_setup_skills", lambda: ["/skills"])
        monkeypatch.setattr(agent_mod, "discover_tools", list)
        monkeypatch.setattr(agent_mod, "DeepClawLocalShellBackend", FakeShellBackend)
        monkeypatch.setattr(agent_mod, "FilesystemBackend", FakeFilesystemBackend)
        monkeypatch.setattr(agent_mod, "CompositeBackend", FakeCompositeBackend)
        monkeypatch.setattr(agent_mod, "RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr(agent_mod, "create_deep_agent", fake_create_deep_agent)
        monkeypatch.setattr(
            agent_mod,
            "_create_deepclaw_summarization_tool_middleware",
            lambda model, backend: ("compact-tool", model, backend),
        )
        monkeypatch.setattr(
            agent_mod,
            "_build_deepclaw_subagents",
            lambda model, backend: [("subagent-compact-tool", model, backend)],
        )
        monkeypatch.setattr(
            agent_mod,
            "_patched_deepagents_summarization_factory",
            lambda: __import__("contextlib").nullcontext(),
        )

        stale_file = tmp_path / "runtime" / "large_tool_results" / "stale.txt"
        stale_file.parent.mkdir(parents=True)
        stale_file.write_text("stale", encoding="utf-8")
        config = DeepClawConfig(model="test:model", workspace_root=str(tmp_path))

        result = agent_mod.create_agent(config, checkpointer="checkpointer")

        assert result == "agent"
        assert isinstance(captured["backend"], FakeCompositeBackend)
        assert "/large_tool_results/" in captured["backend"].routes
        assert "/conversation_history/" in captured["backend"].routes
        assert captured["backend"].default.kwargs["root_dir"] == tmp_path
        assert not stale_file.exists()
        assert (
            captured["backend"].routes["/large_tool_results/"].kwargs["root_dir"]
            == tmp_path / "runtime" / "large_tool_results"
        )
        assert (
            captured["backend"].routes["/conversation_history/"].kwargs["root_dir"]
            == tmp_path / "runtime" / "conversation_history"
        )
        assert not any(
            isinstance(middleware, FilesystemMiddleware) for middleware in captured["middleware"]
        )
        assert any(
            isinstance(middleware, agent_mod.LocalContextMiddleware)
            for middleware in captured["middleware"]
        )
        assert any(
            middleware[0] == "compact-tool" and middleware[1] == "test:model"
            for middleware in captured["middleware"]
            if isinstance(middleware, tuple)
        )
        assert captured["subagents"] == [
            ("subagent-compact-tool", "test:model", captured["backend"])
        ]

    def test_includes_openai_execution_guidance_for_gpt_models(self, tmp_path, monkeypatch):
        captured = {}

        class FakeShellBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def execute(self, command, *, timeout=None):
                return None

        class FakeFilesystemBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeCompositeBackend:
            def __init__(self, *, default, routes):
                self.default = default
                self.routes = routes

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return "agent"


        monkeypatch.setattr(agent_mod, "_load_soul", lambda: "soul")
        monkeypatch.setattr(agent_mod, "_setup_memory", lambda: ["/memory/AGENTS.md"])
        monkeypatch.setattr(agent_mod, "_setup_skills", lambda: ["/skills"])
        monkeypatch.setattr(agent_mod, "discover_tools", list)
        monkeypatch.setattr(agent_mod, "DeepClawLocalShellBackend", FakeShellBackend)
        monkeypatch.setattr(agent_mod, "FilesystemBackend", FakeFilesystemBackend)
        monkeypatch.setattr(agent_mod, "CompositeBackend", FakeCompositeBackend)
        monkeypatch.setattr(agent_mod, "RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr(agent_mod, "create_deep_agent", fake_create_deep_agent)
        monkeypatch.setattr(
            agent_mod,
            "_create_deepclaw_summarization_tool_middleware",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(agent_mod, "_build_deepclaw_subagents", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            agent_mod,
            "_patched_deepagents_summarization_factory",
            lambda: __import__("contextlib").nullcontext(),
        )

        config = DeepClawConfig(model="openai:gpt-5.3-codex", workspace_root=str(tmp_path))
        result = agent_mod.create_agent(config, checkpointer="checkpointer")

        assert result == "agent"
        assert agent_mod.TOOL_USE_ENFORCEMENT in captured["system_prompt"]
        assert agent_mod.OPENAI_MODEL_EXECUTION_GUIDANCE in captured["system_prompt"]

    def test_skips_openai_execution_guidance_for_non_gpt_models(self, tmp_path, monkeypatch):
        captured = {}

        class FakeShellBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def execute(self, command, *, timeout=None):
                return None

        class FakeFilesystemBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeCompositeBackend:
            def __init__(self, *, default, routes):
                self.default = default
                self.routes = routes

        def fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return "agent"


        monkeypatch.setattr(agent_mod, "_load_soul", lambda: "soul")
        monkeypatch.setattr(agent_mod, "_setup_memory", lambda: ["/memory/AGENTS.md"])
        monkeypatch.setattr(agent_mod, "_setup_skills", lambda: ["/skills"])
        monkeypatch.setattr(agent_mod, "discover_tools", list)
        monkeypatch.setattr(agent_mod, "DeepClawLocalShellBackend", FakeShellBackend)
        monkeypatch.setattr(agent_mod, "FilesystemBackend", FakeFilesystemBackend)
        monkeypatch.setattr(agent_mod, "CompositeBackend", FakeCompositeBackend)
        monkeypatch.setattr(agent_mod, "RUNTIME_DIR", tmp_path / "runtime")
        monkeypatch.setattr(agent_mod, "create_deep_agent", fake_create_deep_agent)
        monkeypatch.setattr(
            agent_mod,
            "_create_deepclaw_summarization_tool_middleware",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(agent_mod, "_build_deepclaw_subagents", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            agent_mod,
            "_patched_deepagents_summarization_factory",
            lambda: __import__("contextlib").nullcontext(),
        )

        config = DeepClawConfig(model="anthropic:claude-sonnet-4-6", workspace_root=str(tmp_path))
        result = agent_mod.create_agent(config, checkpointer="checkpointer")

        assert result == "agent"
        assert agent_mod.TOOL_USE_ENFORCEMENT in captured["system_prompt"]
        assert agent_mod.OPENAI_MODEL_EXECUTION_GUIDANCE not in captured["system_prompt"]
