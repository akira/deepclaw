"""Application-level cron scheduler for DeepClaw.

Runs tasks on a schedule and delivers results to Telegram.
Jobs are stored in ~/.deepclaw/cron/jobs.json.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from croniter import croniter

logger = logging.getLogger(__name__)

DEFAULT_JOBS_PATH = Path("~/.deepclaw/cron/jobs.json").expanduser()


@dataclass
class CronJob:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    cron_expr: str = "* * * * *"
    prompt: str = ""
    enabled: bool = True
    delivery: dict = field(default_factory=dict)
    last_run: str | None = None


def load_jobs(path: Path = DEFAULT_JOBS_PATH) -> list[CronJob]:
    """Read jobs from JSON file. Returns empty list if file missing or empty."""
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        raw = json.loads(text)
        if not isinstance(raw, list):
            return []
        return [CronJob(**entry) for entry in raw]
    except (json.JSONDecodeError, TypeError, OSError) as exc:
        logger.warning(f"Could not load jobs from {path}: {exc}")
        return []


def save_jobs(jobs: list[CronJob], path: Path = DEFAULT_JOBS_PATH) -> None:
    """Write jobs to JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(job) for job in jobs]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def add_job(
    name: str,
    cron_expr: str,
    prompt: str,
    delivery: dict,
    path: Path = DEFAULT_JOBS_PATH,
) -> CronJob:
    """Create a new CronJob, append it to the jobs file, and return it."""
    jobs = load_jobs(path)
    job = CronJob(
        name=name,
        cron_expr=cron_expr,
        prompt=prompt,
        delivery=delivery,
    )
    jobs.append(job)
    save_jobs(jobs, path)
    return job


def remove_job(job_id: str, path: Path = DEFAULT_JOBS_PATH) -> bool:
    """Remove a job by ID. Returns True if a job was removed."""
    jobs = load_jobs(path)
    original_len = len(jobs)
    jobs = [j for j in jobs if j.id != job_id]
    if len(jobs) == original_len:
        return False
    save_jobs(jobs, path)
    return True


def list_jobs(path: Path = DEFAULT_JOBS_PATH) -> list[CronJob]:
    """Return all jobs."""
    return load_jobs(path)


def parse_cron_add(text: str) -> tuple[str, str]:
    """Parse the /cron_add argument into (cron_expr, prompt).

    Expected format: ``<5-field cron> | <prompt>``
    Example: ``0 9 * * * | Summarize my todo list``
    """
    parts = text.split("|", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("Expected format: <cron_expr> | <prompt>")
    cron_expr = parts[0].strip()
    prompt = parts[1].strip()
    if not cron_expr or not prompt:
        raise ValueError("Both cron expression and prompt are required")
    # Validate cron expression
    if not croniter.is_valid(cron_expr):
        raise ValueError(f"Invalid cron expression: {cron_expr}")
    return cron_expr, prompt


class Scheduler:
    """In-process asyncio cron scheduler."""

    def __init__(self, jobs_path: Path, agent, checkpointer, bot) -> None:
        self._jobs_path = jobs_path
        self._agent = agent
        self._checkpointer = checkpointer
        self._bot = bot
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the tick loop as a background asyncio task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Cancel the tick loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Scheduler stopped")

    async def _loop(self) -> None:
        """Run tick() every 60 seconds."""
        try:
            while True:
                await self.tick()
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    async def tick(self) -> None:
        """Check each enabled job and run those that are due."""
        jobs = load_jobs(self._jobs_path)
        now = datetime.now(timezone.utc)
        modified = False

        for job in jobs:
            if not job.enabled:
                continue
            if self._is_due(job, now):
                logger.info(f"Running due cron job: {job.name} ({job.id})")
                await self.run_job(job)
                job.last_run = now.isoformat()
                modified = True

        if modified:
            save_jobs(jobs, self._jobs_path)

    def _is_due(self, job: CronJob, now: datetime) -> bool:
        """Check if a job is due to run based on its cron expression and last_run."""
        if job.last_run:
            last = datetime.fromisoformat(job.last_run)
        else:
            # Never run before: use 2 minutes ago as base so it fires on first tick
            last = now.replace(second=0, microsecond=0) - timedelta(minutes=2)

        cron = croniter(job.cron_expr, last)
        next_run = cron.get_next(datetime)
        # Make next_run timezone-aware if it isn't
        if next_run.tzinfo is None:
            next_run = next_run.replace(tzinfo=timezone.utc)
        return next_run <= now

    async def run_job(self, job: CronJob) -> None:
        """Invoke the agent with the job's prompt and deliver the result."""
        thread_id = f"cron-{job.id}"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = await self._agent.ainvoke(
                {"messages": [{"role": "user", "content": job.prompt}]},
                config=config,
            )
            messages = result.get("messages", [])
            content = messages[-1].content if messages else "(no response)"
            if isinstance(content, list):
                response = "\n".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            else:
                response = str(content)
        except Exception:
            logger.exception(f"Cron job {job.name} ({job.id}) agent invocation failed")
            response = f"Cron job '{job.name}' failed to execute."

        chat_id = job.delivery.get("chat_id")
        if chat_id and self._bot:
            try:
                await self._bot.send_message(chat_id=int(chat_id), text=str(response))
            except Exception:
                logger.exception(f"Failed to deliver cron result to chat {chat_id}")
