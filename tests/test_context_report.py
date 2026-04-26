from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from deepclaw.config import DeepClawConfig
from deepclaw.context_report import build_context_report


def test_build_context_report_includes_layering_and_tokens(monkeypatch, tmp_path: Path):
    memory_file = tmp_path / "AGENTS.md"
    memory_file.write_text("# Memory\nRemember the thing.\n", encoding="utf-8")

    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text("Be sharp and useful.\n", encoding="utf-8")

    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "dummy-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: dummy-skill\ndescription: Test skill for status output\n---\n\n# Dummy Skill\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("deepclaw.context_report.agent_module.MEMORY_FILE", memory_file)
    monkeypatch.setattr("deepclaw.context_report.agent_module.SOUL_FILE", soul_file)
    monkeypatch.setattr("deepclaw.context_report.agent_module.SKILLS_DIR", skills_dir)

    checkpoint = {
        "channel_values": {
            "messages": [HumanMessage(content="hello"), AIMessage(content="hi there")],
            "memory_contents": {str(memory_file): memory_file.read_text(encoding="utf-8")},
            "skills_metadata": [
                {
                    "name": "dummy-skill",
                    "description": "Test skill for status output",
                    "path": str(skill_dir / "SKILL.md"),
                    "allowed_tools": [],
                }
            ],
            "local_context": "## Local Context\n\n**Current Directory**: `/tmp/example`",
        }
    }
    monkeypatch.setattr(
        "deepclaw.context_report._load_latest_checkpoint", lambda _thread_id: checkpoint
    )

    def dummy_tool(question: str) -> str:
        """Dummy tool.

        Args:
            question: The user's question.

        Returns:
            A canned answer.
        """

        return question

    monkeypatch.setattr("deepclaw.context_report.discover_tools", lambda: [dummy_tool])

    config = DeepClawConfig(model="openai:gpt-5", workspace_root=str(tmp_path / "workspace"))
    report = build_context_report(config, "thread-123")

    assert "🧠 Context breakdown" in report
    assert "System prompt layers (in order):" in report
    assert "SOUL.md" in report
    assert "Tool Use Enforcement" in report
    assert "OpenAI/Codex execution guidance" in report
    assert "MemoryMiddleware" in report
    assert "SkillsMiddleware" in report
    assert "LocalContextMiddleware" in report
    assert "Tool schemas JSON" in report
    assert "Thread messages" in report
    assert "dummy-skill" in report
    assert "dummy_tool" in report
    assert "Estimated active context subtotal" in report
