"""Cron scheduling tools for the agent.

Allows the agent to schedule, list, and remove recurring tasks.
Always available (no external dependencies).

The jobs_path can be overridden via set_jobs_path() for testing.
Chat context uses contextvars so concurrent async handlers don't collide.
"""

import contextvars
from pathlib import Path
from typing import Any

from deepclaw.scheduler import (
    DEFAULT_JOBS_PATH,
    add_job,
    list_jobs,
    remove_job,
)

_jobs_path: Path = DEFAULT_JOBS_PATH
_current_chat_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_current_chat_id", default=""
)
_current_channel: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_current_channel", default="telegram"
)


def set_jobs_path(path: Path) -> None:
    """Override the jobs file path (useful for testing and configuration)."""
    global _jobs_path  # noqa: PLW0603
    _jobs_path = path


def set_chat_context(channel: str, chat_id: str) -> None:
    """Set the current chat context so scheduled tasks auto-deliver to the right place.

    Called by the gateway before each agent invocation. Uses contextvars
    so each concurrent async task gets its own isolated context.
    """
    _current_channel.set(channel)
    _current_chat_id.set(chat_id)


def available() -> bool:
    """Always available — cron uses only local JSON storage."""
    return True


def schedule_task(
    prompt: str,
    cron_expr: str,
    name: str = "",
) -> dict[str, Any]:
    """Schedule a recurring task that runs on a cron schedule.

    The agent will be invoked with the given prompt at each scheduled time,
    and the result delivered back to this conversation automatically.

    Args:
        prompt: The instruction to execute on each run (e.g., "Check if the deploy is healthy").
        cron_expr: Standard 5-field cron expression (e.g., "0 9 * * *" for daily at 9am,
                   "*/30 * * * *" for every 30 minutes).
        name: Optional human-readable name for the job.

    Returns:
        Confirmation with job ID, schedule, and delivery target.
    """
    try:
        from croniter import croniter

        if not croniter.is_valid(cron_expr):
            return {"error": f"Invalid cron expression: {cron_expr}"}
    except Exception as exc:
        return {"error": f"Failed to validate cron expression: {exc}"}

    chat_id = _current_chat_id.get()
    channel = _current_channel.get()

    if not chat_id:
        return {"error": "No chat context available — cannot determine delivery target."}

    delivery = {"channel": channel, "chat_id": chat_id}

    job = add_job(
        name=name or prompt[:50],
        cron_expr=cron_expr,
        prompt=prompt,
        delivery=delivery,
        path=_jobs_path,
    )
    return {
        "status": "scheduled",
        "job_id": job.id,
        "name": job.name,
        "cron_expr": job.cron_expr,
        "prompt": job.prompt,
        "delivery": job.delivery,
    }


def list_scheduled_tasks() -> dict[str, Any]:
    """List all scheduled cron tasks.

    Returns:
        Dictionary with a list of all scheduled jobs and their details.
    """
    jobs = list_jobs(_jobs_path)
    if not jobs:
        return {"jobs": [], "message": "No scheduled tasks."}
    return {
        "jobs": [
            {
                "job_id": j.id,
                "name": j.name,
                "cron_expr": j.cron_expr,
                "prompt": j.prompt,
                "enabled": j.enabled,
                "last_run": j.last_run,
                "delivery": j.delivery,
            }
            for j in jobs
        ],
        "count": len(jobs),
    }


def remove_scheduled_task(job_id: str) -> dict[str, Any]:
    """Remove a scheduled cron task by its job ID.

    Args:
        job_id: The job ID to remove (full ID or prefix match).

    Returns:
        Confirmation of removal or error if not found.
    """
    jobs = list_jobs(_jobs_path)
    matches = [j for j in jobs if j.id.startswith(job_id)]

    if not matches:
        return {"error": f"No job found matching: {job_id}"}
    if len(matches) > 1:
        return {
            "error": f"Ambiguous ID prefix '{job_id}' matches {len(matches)} jobs. Be more specific.",
            "matches": [{"job_id": j.id, "name": j.name} for j in matches],
        }

    removed = remove_job(matches[0].id, _jobs_path)
    if removed:
        return {"status": "removed", "job_id": matches[0].id, "name": matches[0].name}
    return {"error": f"Failed to remove job: {matches[0].id}"}


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [schedule_task, list_scheduled_tasks, remove_scheduled_task]
