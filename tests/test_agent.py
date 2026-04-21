"""Tests for deepclaw.agent helpers."""

from deepclaw import agent as agent_mod


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
