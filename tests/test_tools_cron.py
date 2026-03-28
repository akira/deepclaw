"""Tests for the cron tool plugin."""

import pytest

from deepclaw.tools.cron import (
    available,
    get_tools,
    list_scheduled_tasks,
    remove_scheduled_task,
    schedule_task,
    set_chat_context,
    set_jobs_path,
)


@pytest.fixture(autouse=True)
def _use_tmp_jobs(tmp_path):
    """Point all cron tool operations at a temp directory and set a default chat context."""
    jobs_file = tmp_path / "jobs.json"
    set_jobs_path(jobs_file)
    set_chat_context("telegram", "99999")
    yield
    # Reset isn't strictly needed since each test gets its own tmp_path,
    # but set_jobs_path is module-global so be tidy
    from deepclaw.scheduler import DEFAULT_JOBS_PATH

    set_jobs_path(DEFAULT_JOBS_PATH)
    set_chat_context("telegram", "")


class TestAvailable:
    def test_always_available(self):
        assert available() is True


class TestGetTools:
    def test_returns_three_tools(self):
        tools = get_tools()
        assert len(tools) == 3
        names = {t.__name__ for t in tools}
        assert names == {"schedule_task", "list_scheduled_tasks", "remove_scheduled_task"}


class TestScheduleTask:
    def test_schedule_valid_job(self):
        result = schedule_task(
            prompt="Check deploy health",
            cron_expr="0 9 * * *",
            name="health check",
        )
        assert result["status"] == "scheduled"
        assert result["cron_expr"] == "0 9 * * *"
        assert result["prompt"] == "Check deploy health"
        assert result["delivery"] == {"channel": "telegram", "chat_id": "99999"}
        assert "job_id" in result

    def test_schedule_invalid_cron(self):
        result = schedule_task(prompt="test", cron_expr="not a cron")
        assert "error" in result

    def test_schedule_default_name_from_prompt(self):
        result = schedule_task(prompt="Do something important", cron_expr="*/5 * * * *")
        assert result["status"] == "scheduled"
        assert result["name"] == "Do something important"

    def test_schedule_without_chat_context_errors(self):
        set_chat_context("telegram", "")
        result = schedule_task(prompt="will fail", cron_expr="0 9 * * *")
        assert "error" in result
        assert "chat context" in result["error"].lower()


class TestListScheduledTasks:
    def test_empty(self):
        result = list_scheduled_tasks()
        assert result["jobs"] == []

    def test_with_jobs(self):
        schedule_task(prompt="task 1", cron_expr="0 * * * *")
        schedule_task(prompt="task 2", cron_expr="0 0 * * *")

        result = list_scheduled_tasks()
        assert result["count"] == 2
        assert len(result["jobs"]) == 2


class TestRemoveScheduledTask:
    def test_remove_existing(self):
        added = schedule_task(prompt="to remove", cron_expr="0 * * * *")
        job_id = added["job_id"]

        result = remove_scheduled_task(job_id)
        assert result["status"] == "removed"

        remaining = list_scheduled_tasks()
        assert remaining["jobs"] == []

    def test_remove_not_found(self):
        result = remove_scheduled_task("nonexistent")
        assert "error" in result

    def test_remove_by_prefix(self):
        added = schedule_task(prompt="prefix test", cron_expr="0 * * * *")
        job_id = added["job_id"]
        prefix = job_id[:8]

        result = remove_scheduled_task(prefix)
        assert result["status"] == "removed"
