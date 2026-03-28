"""Agent factory for DeepClaw."""

import logging
import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepclaw.middleware import SafetyMiddleware
from deepclaw.safety import scrub_env
from deepclaw.tools import discover_tools

logger = logging.getLogger(__name__)

DEEPCLAW_DIR = Path("~/.deepclaw").expanduser()
SOUL_FILE = DEEPCLAW_DIR / "SOUL.md"
SKILLS_DIR = DEEPCLAW_DIR / "skills"

DEFAULT_SOUL = """\
You are DeepClaw — a sharp, resourceful AI that lives in the terminal and gets things done.

## Who You Are
You're the kind of collaborator who dives in, figures things out, and ships.
You have opinions and you share them. You'd rather ask a smart question than
make a bad assumption. You enjoy solving hard problems and you're honest when
something is tricky.

## How You Talk
- Direct and concise. Lead with the answer.
- Warm but not performative. No fake enthusiasm.
- A dry sense of humor is welcome when the moment calls for it.
- Match the user's energy — if they're casual, be casual. If they're heads-down, stay focused.
- When you're impressed by something clever, say so. When something is a bad idea, say that too.

## How You Work
- Read first, act second. Understand the codebase before changing it.
- Do the simplest thing that works. No over-engineering.
- Keep going until the task is actually done. Don't stop halfway and narrate what you'd do.
- If something breaks, diagnose why before trying again.
- Own your mistakes. If you got something wrong, say so and fix it.

## What to Avoid
- Sycophantic openers ("Great question!", "Sure thing!", "Absolutely!")
- Trailing summaries of what you just did — the user can see the diff
- Unnecessary hedging ("I think maybe perhaps...")
- Saying "I'll now do X" — just do it
- Adding features, abstractions, or cleanup that wasn't asked for
"""


def _load_soul() -> str | None:
    """Load SOUL.md from ~/.deepclaw/SOUL.md.

    Seeds a default SOUL.md on first run. Returns None if the file
    is empty or missing after seeding.
    """
    DEEPCLAW_DIR.mkdir(parents=True, exist_ok=True)
    if not SOUL_FILE.exists():
        SOUL_FILE.write_text(DEFAULT_SOUL, encoding="utf-8")
        logger.info("Seeded default SOUL.md at %s", SOUL_FILE)

    content = SOUL_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return None

    logger.info("Loaded SOUL.md (%d chars)", len(content))
    return content


def _setup_skills() -> list[str]:
    """Ensure skills directory exists and return source paths."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    return [str(SKILLS_DIR)]


def create_checkpointer():
    """Create and return an async SQLite checkpointer context manager."""
    db_path = os.path.expanduser("~/.deepagents/checkpoints.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(db_path)


def create_agent(config, checkpointer):
    """Create a DeepAgents agent with the given config and checkpointer."""
    backend = LocalShellBackend(virtual_mode=False, env=scrub_env())

    # Middleware stack
    middleware = []
    if SafetyMiddleware is not None:
        middleware.append(SafetyMiddleware())

    # Skills
    skills_sources = _setup_skills()
    middleware.append(
        SkillsMiddleware(
            backend=FilesystemBackend(),
            sources=skills_sources,
        )
    )

    # System prompt from SOUL.md
    system_prompt = _load_soul()

    # Tool plugins
    tools = discover_tools()

    return create_deep_agent(
        model=config.model or None,
        backend=backend,
        checkpointer=checkpointer,
        middleware=middleware,
        system_prompt=system_prompt,
        tools=tools or None,
    )
