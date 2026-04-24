"""Tests for derived project context middleware."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

from deepclaw import agent as agent_mod
from deepclaw.project_context import (
    DerivedProjectContextMiddleware,
    build_derived_project_context_block,
)


def _git(path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)


class TestBuildDerivedProjectContextBlock:
    def test_python_repo_context_is_compact_and_useful(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

        _git(tmp_path, "init", "-b", "main")
        _git(tmp_path, "add", ".")
        _git(
            tmp_path,
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        )

        block = build_derived_project_context_block(tmp_path)

        assert block is not None
        assert "## Local Project Context" in block
        assert f"- cwd: `{tmp_path.resolve()}`" in block
        assert "git branch: `main` (clean)" in block
        assert "language/runtime: Python" in block
        assert "package manager: uv" in block
        assert "project shape: single-project repo" in block

    def test_monorepo_workspace_context_detects_shape_and_stack(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name":"repo","private":true}', encoding="utf-8")
        (tmp_path / "pnpm-workspace.yaml").write_text("packages:\n  - apps/*\n", encoding="utf-8")
        (tmp_path / "apps" / "web").mkdir(parents=True)
        (tmp_path / "apps" / "web" / "package.json").write_text('{"name":"web"}', encoding="utf-8")
        (tmp_path / "packages" / "shared").mkdir(parents=True)
        (tmp_path / "packages" / "shared" / "package.json").write_text(
            '{"name":"shared"}', encoding="utf-8"
        )

        block = build_derived_project_context_block(tmp_path)

        assert block is not None
        assert f"- cwd: `{tmp_path.resolve()}`" in block
        assert "language/runtime: Node.js" in block
        assert "package manager: pnpm" in block
        assert "project shape: pnpm workspace" in block

    def test_single_child_project_workspace_uses_child_as_project_root(self, tmp_path):
        (tmp_path / "api").mkdir()
        (tmp_path / "api" / "pyproject.toml").write_text(
            "[project]\nname='api'\n", encoding="utf-8"
        )

        block = build_derived_project_context_block(tmp_path)

        assert block is not None
        assert f"- cwd: `{tmp_path.resolve()}`" in block
        assert f"- project root: `{(tmp_path / 'api').resolve()}`" in block
        assert "language/runtime: Python" in block
        assert "project shape: single-project workspace" in block


class TestDerivedProjectContextMiddleware:
    def test_modify_request_appends_context_block(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        middleware = DerivedProjectContextMiddleware(workspace_root=tmp_path)
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage(content="Fix the failing test")],
            system_message=SystemMessage(content="Base instructions"),
            state={"messages": []},
            runtime=MagicMock(),
        )

        modified = middleware.modify_request(request)
        prompt_text = modified.system_prompt or ""

        assert "Base instructions" in prompt_text
        assert "## Local Project Context" in prompt_text
        assert f"- cwd: `{tmp_path.resolve()}`" in prompt_text


class TestCreateAgentProjectContextRegistration:
    def test_create_agent_registers_project_context_middleware(self, tmp_path, monkeypatch):
        captured = {}

        monkeypatch.setattr(agent_mod, "_setup_auth", lambda: None)
        monkeypatch.setattr(agent_mod, "_setup_memory", lambda: ["/tmp/AGENTS.md"])
        monkeypatch.setattr(agent_mod, "_setup_skills", lambda: ["/tmp/skills"])
        monkeypatch.setattr(agent_mod, "_load_soul", lambda: "SOUL")
        monkeypatch.setattr(agent_mod, "discover_tools", list)

        def _fake_create_deep_agent(**kwargs):
            captured.update(kwargs)
            return "agent"

        monkeypatch.setattr(agent_mod, "create_deep_agent", _fake_create_deep_agent)

        config = SimpleNamespace(
            model="anthropic:test",
            command_timeout=30,
            workspace_root=str(tmp_path),
        )

        result = agent_mod.create_agent(config, checkpointer=object())

        assert result == "agent"
        assert any(
            isinstance(middleware, DerivedProjectContextMiddleware)
            for middleware in captured["middleware"]
        )
