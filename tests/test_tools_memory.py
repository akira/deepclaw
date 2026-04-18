"""Tests for the explicit memory tools plugin."""

from pathlib import Path
from unittest.mock import patch

from deepclaw.tools import discover_tools


class TestMemoryToolsAvailable:
    def test_available_is_true(self):
        from deepclaw.tools import memory as memory_mod

        assert memory_mod.available() is True

    def test_get_tools_returns_four_tools(self):
        from deepclaw.tools import memory as memory_mod

        tools = memory_mod.get_tools()
        assert len(tools) == 4
        names = [tool.__name__ for tool in tools]
        assert names == [
            "memory_add",
            "memory_replace",
            "memory_remove",
            "memory_search",
        ]

    def test_discover_tools_includes_memory_tools(self):
        tools = discover_tools()
        names = [getattr(tool, "__name__", "") for tool in tools]

        assert "memory_add" in names
        assert "memory_replace" in names
        assert "memory_remove" in names
        assert "memory_search" in names


class TestMemoryAdd:
    def test_add_appends_bullet_to_named_section(self, tmp_path: Path):
        from deepclaw.tools import memory as memory_mod

        memory_file = tmp_path / "AGENTS.md"
        memory_file.write_text("# DeepClaw Memory\n\n## User\n- Existing note\n")

        with patch("deepclaw.tools.memory.agent_module.MEMORY_FILE", memory_file):
            result = memory_mod.memory_add("Prefers short updates", section="User")

        assert result["status"] == "added"
        assert result["section"] == "User"
        text = memory_file.read_text()
        assert "- Existing note" in text
        assert "- Prefers short updates" in text

    def test_add_creates_section_when_missing(self, tmp_path: Path):
        from deepclaw.tools import memory as memory_mod

        memory_file = tmp_path / "AGENTS.md"
        memory_file.write_text("# DeepClaw Memory\n")

        with patch("deepclaw.tools.memory.agent_module.MEMORY_FILE", memory_file):
            result = memory_mod.memory_add("Needs follow-up", section="Project")

        assert result["status"] == "added"
        text = memory_file.read_text()
        assert "## Project" in text
        assert "- Needs follow-up" in text


class TestMemoryReplace:
    def test_replace_updates_unique_text(self, tmp_path: Path):
        from deepclaw.tools import memory as memory_mod

        memory_file = tmp_path / "AGENTS.md"
        memory_file.write_text("# DeepClaw Memory\n\n## User\n- Prefers short updates\n")

        with patch("deepclaw.tools.memory.agent_module.MEMORY_FILE", memory_file):
            result = memory_mod.memory_replace("short updates", "concise updates")

        assert result["status"] == "replaced"
        text = memory_file.read_text()
        assert "concise updates" in text
        assert "short updates" not in text

    def test_replace_errors_when_text_not_found(self, tmp_path: Path):
        from deepclaw.tools import memory as memory_mod

        memory_file = tmp_path / "AGENTS.md"
        memory_file.write_text("# DeepClaw Memory\n")

        with patch("deepclaw.tools.memory.agent_module.MEMORY_FILE", memory_file):
            result = memory_mod.memory_replace("missing", "new")

        assert "error" in result


class TestMemoryRemove:
    def test_remove_deletes_unique_line(self, tmp_path: Path):
        from deepclaw.tools import memory as memory_mod

        memory_file = tmp_path / "AGENTS.md"
        memory_file.write_text("# DeepClaw Memory\n\n## User\n- Prefers short updates\n")

        with patch("deepclaw.tools.memory.agent_module.MEMORY_FILE", memory_file):
            result = memory_mod.memory_remove("Prefers short updates")

        assert result["status"] == "removed"
        assert "Prefers short updates" not in memory_file.read_text()


class TestMemorySearch:
    def test_search_returns_matching_lines_with_line_numbers(self, tmp_path: Path):
        from deepclaw.tools import memory as memory_mod

        memory_file = tmp_path / "AGENTS.md"
        memory_file.write_text(
            "# DeepClaw Memory\n\n## User\n- Alex prefers Telegram\n- Alex likes concise updates\n"
        )

        with patch("deepclaw.tools.memory.agent_module.MEMORY_FILE", memory_file):
            result = memory_mod.memory_search("telegram")

        assert result["query"] == "telegram"
        assert result["count"] == 1
        assert result["matches"][0]["line_number"] == 4
        assert "Telegram" in result["matches"][0]["line"]
