"""Heartbeat — periodic proactive monitoring for DeepClaw.

Reads a user-maintained HEARTBEAT.md checklist on a schedule, sends it
to the agent, and only notifies the user when something needs attention.
Silent when everything is OK.
"""

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from deepclaw.config import HeartbeatConfig

logger = logging.getLogger(__name__)

DEEPCLAW_DIR = Path("~/.deepclaw").expanduser()
HEARTBEAT_FILE = DEEPCLAW_DIR / "HEARTBEAT.md"

HEARTBEAT_OK = "HEARTBEAT_OK"

HEARTBEAT_PROMPT = """\
You are running a scheduled heartbeat check. Below is your checklist.

Go through each item. Check what you can using the tools available to you.
Be thorough but efficient — don't skip items, but don't over-investigate either.

If every item looks fine and nothing needs the user's attention, respond with
exactly: HEARTBEAT_OK

If something needs attention, give a brief, actionable summary — what's wrong,
how urgent it is, and what the user should do. No preamble, no filler.

Do NOT repeat items that are fine. Only report findings.

## Checklist

{checklist}
"""

DEFAULT_HEARTBEAT_SEED = """\
# Heartbeat Checklist

<!-- Add items for the agent to periodically check. Examples below. -->
<!-- Uncomment or replace with your own checks. -->

<!-- - Check if any git repos in ~/projects have uncommitted changes -->
<!-- - Check disk usage and warn if any partition is above 90% -->
<!-- - Check if any background services I care about are running -->
<!-- - Review and tidy up MEMORY.md if it's getting long -->
"""

# Patterns considered "empty" — template comments, blank checkboxes, headers only
_STRIP_PATTERNS = [
    re.compile(r"<!--.*?-->", re.DOTALL),  # HTML comments
    re.compile(r"^#+\s+.*$", re.MULTILINE),  # Markdown headers
    re.compile(r"^[-*]\s*\[[ x]?\]\s*$", re.MULTILINE),  # Empty checkboxes
    re.compile(r"^[-*]\s*$", re.MULTILINE),  # Bare list markers
]


def is_checklist_empty(content: str) -> bool:
    """Return True if the checklist has no actionable items.

    Strips HTML comments, headers, empty checkboxes, and whitespace
    to determine if there's anything for the agent to actually check.
    """
    stripped = content
    for pattern in _STRIP_PATTERNS:
        stripped = pattern.sub("", stripped)
    return not stripped.strip()


def is_heartbeat_ok(response: str) -> bool:
    """Return True if the agent response indicates nothing needs attention."""
    return HEARTBEAT_OK in response


def is_quiet_hours(config: HeartbeatConfig) -> bool:
    """Return True if current time falls within configured quiet hours."""
    if config.quiet_hours_start is None or config.quiet_hours_end is None:
        return False

    try:
        tz = ZoneInfo(config.timezone)
    except (KeyError, ValueError):
        logger.warning("Invalid timezone %s, defaulting to UTC", config.timezone)
        tz = ZoneInfo("UTC")

    now = datetime.now(tz)
    hour = now.hour
    start = config.quiet_hours_start
    end = config.quiet_hours_end

    if start <= end:
        # Simple range, e.g. 1-6
        return start <= hour < end
    else:
        # Midnight-wrapping range, e.g. 23-8
        return hour >= start or hour < end


def _seed_heartbeat() -> None:
    """Create default HEARTBEAT.md if it doesn't exist."""
    DEEPCLAW_DIR.mkdir(parents=True, exist_ok=True)
    if not HEARTBEAT_FILE.exists():
        HEARTBEAT_FILE.write_text(DEFAULT_HEARTBEAT_SEED, encoding="utf-8")
        logger.info("Seeded default HEARTBEAT.md at %s", HEARTBEAT_FILE)


def _load_checklist() -> str:
    """Read HEARTBEAT.md content. Returns empty string if missing."""
    if not HEARTBEAT_FILE.is_file():
        return ""
    return HEARTBEAT_FILE.read_text(encoding="utf-8")


class HeartbeatRunner:
    """Background runner that periodically checks HEARTBEAT.md."""

    def __init__(self, config: HeartbeatConfig, agent, channels: dict):
        self._config = config
        self._agent = agent
        self._channels = channels  # name -> Channel instance
        self._task: asyncio.Task | None = None
        self._consecutive_failures = 0

    async def start(self) -> None:
        """Start the heartbeat loop as a background asyncio task."""
        if not self._config.enabled:
            logger.info("Heartbeat disabled")
            return
        _seed_heartbeat()
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Heartbeat started (every %d min, notify=%s)",
            self._config.interval_minutes,
            self._config.notify_chat_id or "none",
        )

    async def stop(self) -> None:
        """Cancel the heartbeat loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Heartbeat stopped")

    async def _loop(self) -> None:
        """Run a heartbeat check on the configured interval."""
        interval = self._config.interval_minutes * 60
        try:
            while True:
                await asyncio.sleep(interval)
                await self._tick()
        except asyncio.CancelledError:
            raise

    async def _tick(self) -> None:
        """Run a single heartbeat check."""
        if is_quiet_hours(self._config):
            logger.debug("Heartbeat skipped — quiet hours")
            return

        checklist = _load_checklist()
        if is_checklist_empty(checklist):
            logger.debug("Heartbeat skipped — checklist empty")
            return

        prompt = HEARTBEAT_PROMPT.format(checklist=checklist)
        thread_id = "heartbeat"

        try:
            result = await self._agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            messages = result.get("messages", [])
            content = messages[-1].content if messages else ""
            if isinstance(content, list):
                response = "\n".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            else:
                response = str(content)

            self._consecutive_failures = 0
        except Exception:
            self._consecutive_failures += 1
            logger.exception("Heartbeat agent invocation failed (%d/%d)",
                             self._consecutive_failures, self._config.max_failures)
            if self._consecutive_failures >= self._config.max_failures:
                logger.error("Heartbeat disabled after %d consecutive failures", self._config.max_failures)
                await self.stop()
            return

        if is_heartbeat_ok(response):
            logger.info("Heartbeat OK — nothing needs attention")
            return

        # Something needs attention — notify
        logger.info("Heartbeat findings — notifying user")
        await self._notify(response)

    async def _notify(self, message: str) -> None:
        """Send heartbeat findings to the configured chat."""
        chat_id = self._config.notify_chat_id
        if not chat_id:
            logger.warning("Heartbeat has findings but no notify_chat_id configured")
            return

        channel = self._channels.get("telegram")
        if channel is None:
            logger.warning("No telegram channel available for heartbeat notification")
            return

        formatted = f"\U0001f514 *Heartbeat*\n\n{message}"
        try:
            await channel.send(chat_id, formatted)
        except Exception:
            logger.exception("Failed to deliver heartbeat notification")
