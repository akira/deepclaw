"""Application-level cron scheduler for DeepClaw.

Runs tasks on a schedule and delivers results to Telegram.
Jobs are stored in ~/.deepclaw/cron/jobs.json.

Concurrency safety: a module-level threading.Lock serializes all read-modify-write
operations on the jobs file, working across both sync (tool layer) and async
(scheduler tick, telegram handlers) call sites. Atomic writes (temp file +
os.replace) prevent corruption from partial writes. The tick() loop reloads
jobs after each job run to avoid stale overwrites when schedule_task runs
concurrently.
"""

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

from croniter import croniter

from deepclaw.safety import redact_secrets

logger = logging.getLogger(__name__)

DEFAULT_JOBS_PATH = Path("~/.deepclaw/cron/jobs.json").expanduser()

# Module-level threading lock serializing all read-modify-write operations on
# jobs.json. Works across both sync and async call sites (unlike asyncio.Lock
# which is event-loop-bound). File I/O is fast and already blocking, so holding
# this lock briefly is fine.
_jobs_lock = threading.Lock()
CRON_SILENT_SENTINEL = "[SILENT]"
CRON_SYSTEM_PROMPT = f"""You are running as an isolated scheduled cron job in DeepClaw.
Your final response will be delivered automatically to the configured destination.
Do not ask follow-up questions or mention internal scheduling mechanics unless required.
Do not call outbound messaging or notification tools to deliver the result yourself.
If there is nothing new or nothing worth sending, reply with exactly {CRON_SILENT_SENTINEL} and nothing else.
"""


class DeliveryTarget(TypedDict, total=False):
    """Where to deliver cron job results."""

    channel: str
    chat_id: str


@dataclass
class CronJob:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    cron_expr: str = "* * * * *"
    prompt: str = ""
    enabled: bool = True
    delivery: DeliveryTarget = field(default_factory=dict)
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
        logger.warning("Could not load jobs from %s: %s", path, exc)
        return []


def save_jobs(jobs: list[CronJob], path: Path = DEFAULT_JOBS_PATH) -> None:
    """Write jobs to JSON file atomically.

    Writes to a temp file in the same directory, then os.replace() swaps it
    into place. This prevents corruption if the process is killed mid-write
    and ensures readers never see a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(job) for job in jobs]
    content = json.dumps(data, indent=2) + "\n"
    # Write to a temp file in the same directory (so os.replace is atomic on POSIX)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".jobs_tmp_", suffix=".json")
    try:
        # mkstemp creates with mode 0600; normalize to 0o644 to match
        # the default umask behavior of path.write_text
        os.chmod(tmp_path, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up temp file on failure
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def add_job(
    name: str,
    cron_expr: str,
    prompt: str,
    delivery: DeliveryTarget,
    path: Path = DEFAULT_JOBS_PATH,
) -> CronJob:
    """Create a new CronJob, append it to the jobs file, and return it.

    Acquires the module-level threading.Lock to prevent concurrent
    read-modify-write races. Works from both sync and async call sites.
    """
    with _jobs_lock:
        jobs = load_jobs(path)
        job = CronJob(
            name=name,
            cron_expr=cron_expr,
            prompt=prompt,
            delivery=delivery,
        )
        jobs.append(job)
        save_jobs(jobs, path)
    logger.info("Cron job added: %s (%s) cron='%s'", job.name, job.id, cron_expr)
    return job


def remove_job(job_id: str, path: Path = DEFAULT_JOBS_PATH) -> bool:
    """Remove a job by ID. Returns True if a job was removed.

    Acquires the module-level threading.Lock to prevent concurrent
    read-modify-write races. Works from both sync and async call sites.
    """
    with _jobs_lock:
        jobs = load_jobs(path)
        original_len = len(jobs)
        removed_job = next((j for j in jobs if j.id == job_id), None)
        jobs = [j for j in jobs if j.id != job_id]
        if len(jobs) == original_len:
            return False
        save_jobs(jobs, path)
    name = removed_job.name if removed_job else "?"
    logger.info("Cron job removed: %s (%s)", name, job_id)
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
    """In-process asyncio cron scheduler.

    Delivers results through registered Channel instances rather than
    direct platform references. Channels are keyed by name (e.g., "telegram").
    """

    def __init__(
        self,
        jobs_path: Path,
        agent,
        checkpointer=None,
        channels: dict | None = None,
        *,
        max_turns: int = 200,
        run_timeout: float = 900,
    ) -> None:
        self._jobs_path = jobs_path
        self._agent = agent
        self._channels: dict = channels or {}  # name -> Channel instance
        self._task: asyncio.Task | None = None
        self._max_turns = max(0, int(max_turns))
        self._run_timeout = None if run_timeout <= 0 else float(run_timeout)

    def update_agent(self, agent) -> None:
        """Swap the agent used for cron job invocations (e.g. after a /model switch)."""
        self._agent = agent

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
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
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
        """Check each enabled job and run those that are due.

        Loads jobs under the lock, identifies due jobs, then releases the lock
        while running each job (agent invocation can take minutes). Before each
        run, re-checks under the lock that the job still exists and is enabled
        (it may have been removed/disabled during this tick). After each job
        completes, re-acquires the lock, reloads the current jobs, updates
        last_run for that specific job, and saves. This prevents the stale-write
        race where tick() overwrites jobs added during long-running agent calls.
        """
        with _jobs_lock:
            jobs = load_jobs(self._jobs_path)
        now = datetime.now(UTC)

        for job in jobs:
            if not job.enabled:
                continue
            if not self._is_due(job, now):
                continue

            # Re-check under lock: job may have been removed or disabled
            # during this tick (which can span minutes for prior jobs)
            with _jobs_lock:
                current = load_jobs(self._jobs_path)
                fresh = next((j for j in current if j.id == job.id), None)
                if fresh is None or not fresh.enabled:
                    logger.info(
                        "Cron job %s (%s) removed/disabled during tick; skipping",
                        job.name,
                        job.id,
                    )
                    continue

            logger.info("Running due cron job: %s (%s)", job.name, job.id)
            await self.run_job(job)

            # Update last_run under lock, reloading to avoid stale overwrites
            with _jobs_lock:
                current_jobs = load_jobs(self._jobs_path)
                target = next((j for j in current_jobs if j.id == job.id), None)
                if target is None:
                    # Job was removed during run_job — nothing to update
                    logger.info(
                        "Cron job %s (%s) removed during run; skipping last_run update",
                        job.name,
                        job.id,
                    )
                    continue
                target.last_run = now.isoformat()
                try:
                    save_jobs(current_jobs, self._jobs_path)
                except OSError:
                    logger.exception(
                        "Failed to persist last_run for job %s (%s); continuing",
                        job.name,
                        job.id,
                    )

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
            next_run = next_run.replace(tzinfo=UTC)
        return next_run <= now

    async def run_job(self, job: CronJob) -> None:
        """Invoke the agent with the job's prompt and deliver the result."""
        thread_id = f"cron-{job.id}-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}
        if self._max_turns > 0:
            config["recursion_limit"] = self._max_turns

        try:
            invoke_coro = self._agent.ainvoke(
                {
                    "messages": [
                        {"role": "system", "content": CRON_SYSTEM_PROMPT},
                        {"role": "user", "content": job.prompt},
                    ]
                },
                config=config,
            )
            if self._run_timeout is None:
                result = await invoke_coro
            else:
                result = await asyncio.wait_for(invoke_coro, timeout=self._run_timeout)
            messages = result.get("messages", [])
            content = messages[-1].content if messages else "(no response)"
            if isinstance(content, list):
                response = "\n".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            else:
                response = str(content)
        except TimeoutError:
            logger.exception("Cron job %s (%s) timed out", job.name, job.id)
            response = f"Cron job '{job.name}' timed out after {int(self._run_timeout)} seconds."
        except Exception:
            logger.exception("Cron job %s (%s) agent invocation failed", job.name, job.id)
            response = f"Cron job '{job.name}' failed to execute."

        response = redact_secrets(str(response)).strip()
        if response == CRON_SILENT_SENTINEL:
            logger.info("Cron job '%s' returned silent sentinel; skipping delivery", job.name)
            return

        channel_name = job.delivery.get("channel", "")
        chat_id = job.delivery.get("chat_id", "")
        channel = self._channels.get(channel_name)

        if channel and chat_id:
            try:
                await channel.send(chat_id, str(response))
            except Exception:
                logger.exception(
                    "Failed to deliver cron result via %s to %s", channel_name, chat_id
                )
        elif chat_id:
            logger.warning("No channel '%s' registered for delivery", channel_name)
        else:
            logger.info("Cron job '%s' completed (no delivery target)", job.name)
