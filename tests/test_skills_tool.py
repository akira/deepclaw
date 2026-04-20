"""Tests for the local skills management tool plugin."""

import urllib.error

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


class TestDefaultInstallName:
    def test_uses_parent_directory_name_for_skill_file(self):
        src = skills_mod.Path("/tmp/source-skill/SKILL.md")
        assert skills_mod._default_install_name(src) == "source-skill"

    def test_falls_back_to_lowercase_stem_for_root_skill_file(self):
        src = skills_mod.Path("/SKILL.md")
        assert skills_mod._default_install_name(src) == "skill"


class TestSkillsShHelpers:
    def test_extracts_identifier_from_url(self):
        assert (
            skills_mod._skills_sh_identifier(
                "https://skills.sh/vercel-labs/agent-browser/agent-browser"
            )
            == "vercel-labs/agent-browser/agent-browser"
        )

    def test_resolves_repo_and_skill_path_from_detail_page(self, monkeypatch):
        html = (
            "<html><body>"
            "npx skills add https://github.com/vercel-labs/agent-browser --skill browser/agent-browser"
            "</body></html>"
        )
        monkeypatch.setattr(skills_mod, "_http_get", lambda url, accept=None: html.encode("utf-8"))

        result = skills_mod._resolve_skills_sh_source(
            "https://skills.sh/vercel-labs/agent-browser/agent-browser"
        )

        assert result == {
            "identifier": "vercel-labs/agent-browser/agent-browser",
            "repo": "vercel-labs/agent-browser",
            "skill_path": "browser/agent-browser",
            "detail_url": "https://skills.sh/vercel-labs/agent-browser/agent-browser",
        }


class TestSkillsSearchRemote:
    def test_returns_remote_results(self, monkeypatch):
        monkeypatch.setattr(
            skills_mod,
            "_http_get_json",
            lambda url, accept="application/json": {
                "skills": [
                    {
                        "id": "vercel-labs/agent-browser/agent-browser",
                        "skillId": "agent-browser",
                        "name": "agent-browser",
                        "installs": 195769,
                        "source": "vercel-labs/agent-browser",
                    }
                ]
            },
        )

        result = skills_mod.skills_search_remote("browser", limit=5)

        assert result["count"] == 1
        assert result["query"] == "browser"
        assert (
            result["skills"][0]["url"]
            == "https://skills.sh/vercel-labs/agent-browser/agent-browser"
        )
        assert result["skills"][0]["repo"] == "vercel-labs/agent-browser"

    def test_returns_error_on_network_failure(self, monkeypatch):
        def _fail(url, accept="application/json"):
            raise urllib.error.URLError("boom")

        monkeypatch.setattr(skills_mod, "_http_get_json", _fail)

        result = skills_mod.skills_search_remote("browser")

        assert "error" in result
        assert "skills.sh" in result["error"]


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

    def test_installs_from_skills_sh_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path / "installed")
        monkeypatch.setattr(
            skills_mod,
            "_resolve_skills_sh_source",
            lambda source: {
                "identifier": "vercel-labs/agent-browser/agent-browser",
                "repo": "vercel-labs/agent-browser",
                "skill_path": "browser/agent-browser",
                "detail_url": "https://skills.sh/vercel-labs/agent-browser/agent-browser",
            },
        )

        def _fake_download(repo, path):
            if path == "browser/agent-browser":
                return {}
            if path == "skills/agent-browser":
                return {
                    "SKILL.md": b"---\nname: agent-browser\ndescription: Browser automation\n---\n",
                    "notes.txt": b"extra",
                    "templates/config.json": b"{}",
                }
            return {}

        monkeypatch.setattr(skills_mod, "_download_github_directory", _fake_download)
        monkeypatch.setattr(
            skills_mod,
            "_find_github_skill_path",
            lambda repo, skill_name: "skills/agent-browser",
        )

        result = skills_mod.skill_install(
            "https://skills.sh/vercel-labs/agent-browser/agent-browser"
        )

        installed_dir = tmp_path / "installed" / "agent-browser"
        assert result["success"] is True
        assert result["name"] == "agent-browser"
        assert result["source_repo"] == "vercel-labs/agent-browser"
        assert result["source_skill_path"] == "browser/agent-browser"
        assert result["resolved_skill_path"] == "skills/agent-browser"
        assert (installed_dir / "SKILL.md").read_text() == (
            "---\nname: agent-browser\ndescription: Browser automation\n---\n"
        )
        assert (installed_dir / "notes.txt").read_text() == "extra"
        assert (installed_dir / "templates" / "config.json").read_text() == "{}"

    def test_skills_sh_install_without_skill_md_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_mod, "SKILLS_DIR", tmp_path / "installed")
        monkeypatch.setattr(
            skills_mod,
            "_resolve_skills_sh_source",
            lambda source: {
                "identifier": "vercel-labs/agent-browser/agent-browser",
                "repo": "vercel-labs/agent-browser",
                "skill_path": "browser/agent-browser",
                "detail_url": "https://skills.sh/vercel-labs/agent-browser/agent-browser",
            },
        )
        monkeypatch.setattr(
            skills_mod,
            "_download_github_directory",
            lambda repo, path: {"notes.txt": b"extra"},
        )

        result = skills_mod.skill_install(
            "https://skills.sh/vercel-labs/agent-browser/agent-browser"
        )

        assert "error" in result
        assert "SKILL.md" in result["error"]

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
