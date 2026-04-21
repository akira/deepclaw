"""Tests for deepclaw.scheduler module."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepclaw.scheduler import (
    CronJob,
    Scheduler,
    add_job,
    list_jobs,
    load_jobs,
    parse_cron_add,
    remove_job,
    save_jobs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(**overrides) -> CronJob:
    """Create a CronJob with sensible defaults, applying overrides."""
    defaults = {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "name": "test job",
        "cron_expr": "0 9 * * *",
        "prompt": "Say hello",
        "enabled": True,
        "delivery": {"channel": "telegram", "chat_id": "12345"},
        "last_run": None,
    }
    defaults.update(overrides)
    return CronJob(**defaults)


def _write_jobs_file(path: Path, jobs_data: list[dict]) -> None:
    """Write raw JSON data to a jobs file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs_data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CronJob dataclass defaults
# ---------------------------------------------------------------------------


class TestCronJobDefaults:
    def test_default_id_is_uuid(self):
        job = CronJob()
        assert len(job.id) == 36  # UUID format
        assert "-" in job.id

    def test_default_enabled_is_true(self):
        job = CronJob()
        assert job.enabled is True

    def test_default_last_run_is_none(self):
        job = CronJob()
        assert job.last_run is None

    def test_default_delivery_is_empty_dict(self):
        job = CronJob()
        assert job.delivery == {}

    def test_default_cron_expr(self):
        job = CronJob()
        assert job.cron_expr == "* * * * *"


# ---------------------------------------------------------------------------
# load_jobs
# ---------------------------------------------------------------------------


class TestLoadJobs:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_jobs(tmp_path / "nonexistent.json")
        assert result == []

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "jobs.json"
        f.write_text("", encoding="utf-8")
        assert load_jobs(f) == []

    def test_valid_file_loads_jobs(self, tmp_path):
        f = tmp_path / "jobs.json"
        _write_jobs_file(
            f,
            [
                {
                    "id": "abc-123",
                    "name": "daily",
                    "cron_expr": "0 9 * * *",
                    "prompt": "hello",
                    "enabled": True,
                    "delivery": {},
                    "last_run": None,
                }
            ],
        )
        jobs = load_jobs(f)
        assert len(jobs) == 1
        assert jobs[0].id == "abc-123"
        assert jobs[0].name == "daily"

    def test_invalid_json_returns_empty(self, tmp_path):
        f = tmp_path / "jobs.json"
        f.write_text("not json!", encoding="utf-8")
        assert load_jobs(f) == []


# ---------------------------------------------------------------------------
# save_jobs / roundtrip
# ---------------------------------------------------------------------------


class TestSaveJobs:
    def test_save_creates_file(self, tmp_path):
        f = tmp_path / "subdir" / "jobs.json"
        job = _make_job()
        save_jobs([job], f)
        assert f.is_file()

    def test_roundtrip(self, tmp_path):
        f = tmp_path / "jobs.json"
        original = _make_job()
        save_jobs([original], f)
        loaded = load_jobs(f)
        assert len(loaded) == 1
        assert loaded[0].id == original.id
        assert loaded[0].name == original.name
        assert loaded[0].cron_expr == original.cron_expr
        assert loaded[0].prompt == original.prompt
        assert loaded[0].enabled == original.enabled
        assert loaded[0].delivery == original.delivery
        assert loaded[0].last_run == original.last_run


# ---------------------------------------------------------------------------
# add_job
# ---------------------------------------------------------------------------


class TestAddJob:
    def test_creates_valid_job_with_uuid(self, tmp_path):
        f = tmp_path / "jobs.json"
        job = add_job(
            name="morning summary",
            cron_expr="0 9 * * *",
            prompt="Summarize todos",
            delivery={"channel": "telegram", "chat_id": "99"},
            path=f,
        )
        assert len(job.id) == 36
        assert job.name == "morning summary"
        assert job.cron_expr == "0 9 * * *"
        assert job.prompt == "Summarize todos"
        assert job.enabled is True

    def test_persists_to_file(self, tmp_path):
        f = tmp_path / "jobs.json"
        add_job("a", "* * * * *", "do thing", {}, path=f)
        add_job("b", "0 0 * * *", "other", {}, path=f)
        jobs = load_jobs(f)
        assert len(jobs) == 2


# ---------------------------------------------------------------------------
# remove_job
# ---------------------------------------------------------------------------


class TestRemoveJob:
    def test_removes_correct_job(self, tmp_path):
        f = tmp_path / "jobs.json"
        j1 = add_job("a", "* * * * *", "p1", {}, path=f)
        j2 = add_job("b", "0 0 * * *", "p2", {}, path=f)
        removed = remove_job(j1.id, f)
        assert removed is True
        remaining = load_jobs(f)
        assert len(remaining) == 1
        assert remaining[0].id == j2.id

    def test_nonexistent_id_returns_false(self, tmp_path):
        f = tmp_path / "jobs.json"
        add_job("a", "* * * * *", "p1", {}, path=f)
        assert remove_job("no-such-id", f) is False


# ---------------------------------------------------------------------------
# parse_cron_add
# ---------------------------------------------------------------------------


class TestParseCronAdd:
    def test_valid_input(self):
        cron_expr, prompt = parse_cron_add("0 9 * * * | Summarize my todo list")
        assert cron_expr == "0 9 * * *"
        assert prompt == "Summarize my todo list"

    def test_every_minute(self):
        cron_expr, prompt = parse_cron_add("* * * * * | ping")
        assert cron_expr == "* * * * *"
        assert prompt == "ping"

    def test_missing_pipe_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            parse_cron_add("0 9 * * * no pipe here")

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="required"):
            parse_cron_add("0 9 * * * | ")

    def test_empty_cron_raises(self):
        with pytest.raises(ValueError, match="required"):
            parse_cron_add(" | do something")

    def test_invalid_cron_raises(self):
        with pytest.raises(ValueError, match="Invalid cron"):
            parse_cron_add("99 99 99 99 99 | bad cron")

    def test_prompt_with_pipes(self):
        cron_expr, prompt = parse_cron_add("0 9 * * * | echo a | grep b")
        assert cron_expr == "0 9 * * *"
        assert prompt == "echo a | grep b"


# ---------------------------------------------------------------------------
# Scheduler.tick — due job detection
# ---------------------------------------------------------------------------


class TestSchedulerTick:
    @pytest.mark.asyncio
    async def test_tick_runs_due_job(self, tmp_path):
        f = tmp_path / "jobs.json"
        # Create a job that last ran 2 minutes ago with every-minute schedule
        past = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
        job = _make_job(cron_expr="* * * * *", last_run=past)
        save_jobs([job], f)

        agent = AsyncMock()
        agent.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="done")]})
        channel = AsyncMock()
        channel.name = "telegram"

        scheduler = Scheduler(
            jobs_path=f, agent=agent, checkpointer=None, channels={"telegram": channel}
        )
        await scheduler.tick()

        agent.ainvoke.assert_called_once()
        channel.send.assert_called_once_with("12345", "done")

    @pytest.mark.asyncio
    async def test_tick_skips_disabled_job(self, tmp_path):
        f = tmp_path / "jobs.json"
        past = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
        job = _make_job(cron_expr="* * * * *", last_run=past, enabled=False)
        save_jobs([job], f)

        agent = AsyncMock()

        scheduler = Scheduler(jobs_path=f, agent=agent, checkpointer=None)
        await scheduler.tick()

        agent.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_skips_not_yet_due_job(self, tmp_path):
        f = tmp_path / "jobs.json"
        # Job last ran just now with a daily schedule — not due yet
        now_iso = datetime.now(UTC).isoformat()
        job = _make_job(cron_expr="0 0 1 1 *", last_run=now_iso)  # once a year
        save_jobs([job], f)

        agent = AsyncMock()

        scheduler = Scheduler(jobs_path=f, agent=agent, checkpointer=None)
        await scheduler.tick()

        agent.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_updates_last_run(self, tmp_path):
        f = tmp_path / "jobs.json"
        past = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
        job = _make_job(cron_expr="* * * * *", last_run=past)
        save_jobs([job], f)

        agent = AsyncMock()
        agent.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="ok")]})
        channel = AsyncMock()
        channel.name = "telegram"

        scheduler = Scheduler(
            jobs_path=f, agent=agent, checkpointer=None, channels={"telegram": channel}
        )
        await scheduler.tick()

        updated = load_jobs(f)
        assert updated[0].last_run is not None
        assert updated[0].last_run != past

    @pytest.mark.asyncio
    async def test_tick_redacts_secrets_before_delivery(self, tmp_path):
        f = tmp_path / "jobs.json"
        past = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
        job = _make_job(cron_expr="* * * * *", last_run=past)
        save_jobs([job], f)

        secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
        agent = AsyncMock()
        agent.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content=secret)]})
        channel = AsyncMock()
        channel.name = "telegram"

        scheduler = Scheduler(
            jobs_path=f, agent=agent, checkpointer=None, channels={"telegram": channel}
        )
        await scheduler.tick()

        channel.send.assert_called_once()
        delivered = channel.send.call_args.args[1]
        assert secret not in delivered
        assert "[REDACTED]" in delivered

    @pytest.mark.asyncio
    async def test_run_job_uses_fresh_thread_id_per_invocation(self, tmp_path):
        f = tmp_path / "jobs.json"
        job = _make_job()

        agent = AsyncMock()
        agent.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="ok")]})
        channel = AsyncMock()
        channel.name = "telegram"

        scheduler = Scheduler(
            jobs_path=f, agent=agent, checkpointer=None, channels={"telegram": channel}
        )

        await scheduler.run_job(job)
        await scheduler.run_job(job)

        first_config = agent.ainvoke.await_args_list[0].kwargs["config"]
        second_config = agent.ainvoke.await_args_list[1].kwargs["config"]
        first_thread_id = first_config["configurable"]["thread_id"]
        second_thread_id = second_config["configurable"]["thread_id"]

        assert first_thread_id.startswith(f"cron-{job.id}-")
        assert second_thread_id.startswith(f"cron-{job.id}-")
        assert first_thread_id != second_thread_id


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    def test_list_returns_all(self, tmp_path):
        f = tmp_path / "jobs.json"
        add_job("a", "* * * * *", "p1", {}, path=f)
        add_job("b", "0 0 * * *", "p2", {}, path=f)
        assert len(list_jobs(f)) == 2

    def test_list_empty(self, tmp_path):
        f = tmp_path / "nonexistent.json"
        assert list_jobs(f) == []
