"""Tests for deepclaw.agent helpers."""

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
        monkeypatch.setattr(agent_mod, "LocalShellBackend", FakeShellBackend)
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
        monkeypatch.setattr(agent_mod, "LocalShellBackend", FakeShellBackend)
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
        monkeypatch.setattr(agent_mod, "LocalShellBackend", FakeShellBackend)
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
