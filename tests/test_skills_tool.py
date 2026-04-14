"""Tests for the local skills management tool plugin."""

from deepclaw.tools import skills as skills_mod


class TestSkillsList:
    def test_lists_installed_skills(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)
        skill_dir = tmp_path / "web-research"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: web-research\ndescription: Research the web carefully\n---\n"
        )

        result = skills_mod.skills_list()

        assert result["count"] == 1
        assert result["skills"][0]["name"] == "web-research"
        assert result["skills"][0]["description"] == "Research the web carefully"


class TestSkillView:
    def test_returns_skill_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test Skill\n\nHello")

        result = skills_mod.skill_view("test-skill")

        assert result["name"] == "test-skill"
        assert "Hello" in result["content"]

    def test_missing_skill_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)

        result = skills_mod.skill_view("missing")

        assert "error" in result
        assert "Skill not found" in result["error"]


class TestSkillCreate:
    def test_creates_template_when_content_omitted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)

        result = skills_mod.skill_create("debug-helper", "Debug recurring issues")

        created = tmp_path / "debug-helper" / "SKILL.md"
        assert result["success"] is True
        assert created.is_file()
        content = created.read_text()
        assert "name: debug-helper" in content
        assert "description: Debug recurring issues" in content

    def test_rejects_invalid_skill_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)

        result = skills_mod.skill_create("../oops", "bad")

        assert "error" in result
        assert "Invalid skill name" in result["error"]

    def test_existing_skill_requires_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)
        skill_dir = tmp_path / "debug-helper"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("old")

        result = skills_mod.skill_create("debug-helper", "Debug recurring issues")

        assert "error" in result
        assert "already exists" in result["error"]


class TestSkillUpdate:
    def test_updates_existing_skill(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)
        skill_dir = tmp_path / "debug-helper"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("old")

        result = skills_mod.skill_update("debug-helper", "# New Content")

        assert result["success"] is True
        assert skill_file.read_text() == "# New Content"


class TestSkillInstall:
    def test_installs_from_skill_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path / "installed")
        src = tmp_path / "source-skill" / "SKILL.md"
        src.parent.mkdir()
        src.write_text("---\nname: imported\ndescription: Imported skill\n---\n")

        result = skills_mod.skill_install(str(src))

        installed = tmp_path / "installed" / "source-skill" / "SKILL.md"
        assert result["success"] is True
        assert result["name"] == "source-skill"
        assert installed.is_file()
        assert "Imported skill" in installed.read_text()

    def test_installs_from_directory_and_preserves_extra_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path / "installed")
        src_dir = tmp_path / "source-skill"
        src_dir.mkdir()
        (src_dir / "SKILL.md").write_text("# Imported")
        (src_dir / "notes.txt").write_text("extra")

        result = skills_mod.skill_install(str(src_dir), name="imported-dir")

        installed_dir = tmp_path / "installed" / "imported-dir"
        assert result["success"] is True
        assert (installed_dir / "SKILL.md").is_file()
        assert (installed_dir / "notes.txt").read_text() == "extra"

    def test_install_requires_skill_md_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path / "installed")
        src = tmp_path / "random.txt"
        src.write_text("hi")

        result = skills_mod.skill_install(str(src))

        assert "error" in result
        assert "SKILL.md" in result["error"]

    def test_install_existing_skill_requires_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path / "installed")
        existing = tmp_path / "installed" / "imported"
        existing.mkdir(parents=True)
        (existing / "SKILL.md").write_text("old")
        src = tmp_path / "SKILL.md"
        src.write_text("new")

        result = skills_mod.skill_install(str(src), name="imported")

        assert "error" in result
        assert "already exists" in result["error"]


class TestSkillDelete:
    def test_deletes_existing_skill(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)
        skill_dir = tmp_path / "delete-me"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Delete Me")

        result = skills_mod.skill_delete("delete-me")

        assert result["success"] is True
        assert result["action"] == "deleted"
        assert not skill_dir.exists()

    def test_delete_missing_skill_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path)

        result = skills_mod.skill_delete("missing")

        assert "error" in result
        assert "Skill not found" in result["error"]
