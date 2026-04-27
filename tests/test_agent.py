"""Tests for deepclaw.agent helpers."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

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


class TestCreateAgent:
    def test_wires_cli_style_context_backends_and_summarization(self, tmp_path, monkeypatch):
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

        def fake_append_summarization(middleware, model, backend):
            middleware.append(("summarization", model, backend))

        monkeypatch.setattr(agent_mod, "_setup_auth", lambda: None)
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
            agent_mod, "_append_summarization_middleware", fake_append_summarization
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
        assert any(
            isinstance(middleware, agent_mod.LocalContextMiddleware)
            for middleware in captured["middleware"]
        )
        assert any(
            middleware[0] == "summarization" and middleware[1] == "test:model"
            for middleware in captured["middleware"]
            if isinstance(middleware, tuple)
        )

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

        monkeypatch.setattr(agent_mod, "_setup_auth", lambda: None)
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
            agent_mod, "_append_summarization_middleware", lambda *args, **kwargs: None
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

        monkeypatch.setattr(agent_mod, "_setup_auth", lambda: None)
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
            agent_mod, "_append_summarization_middleware", lambda *args, **kwargs: None
        )

        config = DeepClawConfig(model="anthropic:claude-sonnet-4-6", workspace_root=str(tmp_path))
        result = agent_mod.create_agent(config, checkpointer="checkpointer")

        assert result == "agent"
        assert agent_mod.TOOL_USE_ENFORCEMENT in captured["system_prompt"]
        assert agent_mod.OPENAI_MODEL_EXECUTION_GUIDANCE not in captured["system_prompt"]
