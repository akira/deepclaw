# DeepClaw

A multi-platform AI agent built on [LangChain DeepAgents](https://github.com/langchain-ai/deepagents). DeepClaw extends DeepAgents with a messaging gateway, starting with Telegram and the terminal, so you can interact with your agent from anywhere.

## Quick Start

```bash
# Required
export TELEGRAM_BOT_TOKEN=<your-telegram-bot-token>
export ANTHROPIC_API_KEY=<your-anthropic-key>

# Install and run the Telegram bot
uv sync
uv run deepclaw

# Optional: install web search tools
uv sync --extra web
```

To use the interactive TUI instead (or alongside):

```bash
uv run deepclaw-tui
```

Both share the same `~/.deepagents/` workspace, so memory and skills persist across interfaces.

## Configuration

DeepClaw loads configuration from three layers (highest precedence first):

1. Shell environment variables
2. `~/.deepclaw/.env` file
3. `~/.deepclaw/config.yaml` file

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram Bot API token from [@BotFather](https://t.me/BotFather) |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key (default provider) |
| `OPENAI_API_KEY` | No | OpenAI API key (if using OpenAI models) |
| `DEEPCLAW_MODEL` | No | Model override, e.g. `openai:gpt-4o` (defaults to `claude-sonnet-4-6`) |
| `DEEPCLAW_ALLOWED_USERS` | No | Comma-separated list of Telegram user IDs or usernames for access control |
| `TAVILY_API_KEY` | No | Tavily API key for web search and extract tools |

*Or whichever provider API key matches your chosen model.

### Config File

```yaml
# ~/.deepclaw/config.yaml
model: "claude-sonnet-4-6"
telegram:
  bot_token: "..."
  allowed_users:
    - "123456789"
    - "myusername"
  streaming:
    enabled: true
    edit_interval: 1.0
    buffer_threshold: 100
workspace:
  root: "~/.deepclaw/workspace"
```

## How It Works

DeepClaw wires a DeepAgents agent to Telegram via long-polling. Each Telegram chat gets its own persistent conversation thread (using `chat_id` as the LangGraph `thread_id`), backed by SQLite checkpointing at `~/.deepagents/checkpoints.db`.

The agent has full DeepAgents capabilities out of the box:
- File operations (read, write, edit, glob, grep)
- Shell command execution
- Task planning (write_todos)
- Sub-agent spawning
- Context compression when conversations get long
- Skills and memory (AGENTS.md / SKILL.md)

## SOUL.md — Agent Identity

DeepClaw loads a `SOUL.md` file from `~/.deepclaw/SOUL.md` to define the agent's personality, communication style, and working behavior. This is prepended to the system prompt on every conversation.

A default SOUL.md is seeded on first run. Edit it to customize your agent's voice:

```bash
$EDITOR ~/.deepclaw/SOUL.md
```

## Memory

DeepClaw has persistent memory via `~/.deepclaw/AGENTS.md`. The agent learns from your interactions and writes observations back to this file — preferences, corrections, project context, workflow patterns.

Memory is loaded into the system prompt at session start and persists across conversations. The agent decides what's worth remembering based on built-in guidelines (e.g., user preferences yes, transient small talk no, credentials never).

You can view or edit it directly:

```bash
cat ~/.deepclaw/AGENTS.md
$EDITOR ~/.deepclaw/AGENTS.md
```

## Skills

DeepClaw supports skills via SKILL.md files in `~/.deepclaw/skills/`. Skills are specialized workflows the agent discovers and uses when relevant.

To add a skill, create a subdirectory with a SKILL.md:

```
~/.deepclaw/skills/
└── web-research/
    └── SKILL.md
```

SKILL.md format:
```markdown
---
name: web-research
description: Structured approach to conducting thorough web research
---

# Web Research Skill

## When to Use
- User asks you to research a topic
...
```

The agent sees skill names and descriptions in its system prompt, and reads the full SKILL.md content on demand (progressive disclosure).

## Heartbeat

DeepClaw can run periodic proactive checks via a `HEARTBEAT.md` checklist. The agent reviews the checklist on a schedule and **only notifies you when something needs attention** — silent when everything is OK.

A default (commented-out) `HEARTBEAT.md` is seeded at `~/.deepclaw/HEARTBEAT.md`. Uncomment or add your own checks:

```markdown
# Heartbeat Checklist
- Check if any git repos in ~/projects have uncommitted changes
- Check disk usage and warn if any partition is above 90%
- Review and tidy up MEMORY.md if it's getting long
```

Enable and configure in `~/.deepclaw/config.yaml`:

```yaml
heartbeat:
  enabled: true
  interval_minutes: 30
  notify_chat_id: "123456789"   # Telegram chat ID for notifications
  quiet_hours_start: 23          # Don't check 11 PM - 8 AM
  quiet_hours_end: 8
  timezone: "America/Los_Angeles"
  max_failures: 3                # Auto-disable after 3 consecutive errors
```

## Tool Plugins

DeepClaw uses a plugin system for optional tools. Each plugin in `deepclaw/tools/` is auto-discovered at startup. A plugin loads only if its dependencies are installed and required env vars are set.

| Plugin | Install | Env Var | Tools |
|---|---|---|---|
| `web_search` | `uv sync --extra web` | `TAVILY_API_KEY` | `web_search`, `web_extract` |

To add a new tool plugin, create a module in `deepclaw/tools/` that exports:
- `available() -> bool` — checks if deps and env vars are present
- `get_tools() -> list` — returns tool callables

## Telegram Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/new` | Start a fresh conversation thread |
| `/status` | Show current thread ID, model, and allowlist status |
| `/help` | List available commands |
| `/cron` | List all scheduled cron jobs |
| `/cron_add <expr> \| <prompt>` | Add a scheduled job, e.g. `/cron_add 0 9 * * * \| Summarize my todo list` |
| `/cron_rm <id_prefix>` | Remove a scheduled job by ID prefix |
| `/safety_test <cmd>` | Check a shell command for dangerous patterns |

## Safety

DeepClaw includes a layered safety system that actively gates tool execution via `SafetyMiddleware`:

- **Shell command gating** — every `execute` tool call is checked against 34+ dangerous patterns across 14 categories. Critical severity commands (`rm -rf`, `mkfs`, `dd`, `DROP TABLE`, fork bombs, piped remote execution) are **hard-blocked**. Warning severity commands (`git push --force`, `chmod -R`, `bash -c`) trigger a **human-in-the-loop interrupt** for approval via Telegram.
- **Write path deny list** — `write_file` and `edit_file` calls are blocked for sensitive paths: `~/.ssh/`, `~/.aws/`, `~/.gnupg/`, `~/.kube/`, shell configs (`~/.bashrc`, `~/.zshrc`), and system files (`/etc/passwd`, `/etc/shadow`, `/etc/sudoers`).
- **SSRF protection** — URL fetches are validated against private/internal network ranges (RFC 1918, loopback, link-local, CGNAT, cloud metadata endpoints). DNS failures are fail-closed.
- **Credential redaction** — tool output is scanned for 11 secret patterns (GitHub PATs, AWS keys, Slack tokens, OpenAI/Anthropic keys, Bearer tokens, generic `api_key=` assignments) and redacted with `[REDACTED]` before being sent to the LLM or streamed to Telegram.
- **Environment variable scrubbing** — child processes spawned by the agent receive a filtered environment. Only safe variables (`PATH`, `HOME`, `LANG`, toolchain paths, etc.) pass through. Any variable whose name contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `AUTH`, or `PRIVATE` is stripped, preventing credential leakage via shell commands.
- **Access control** — optional allowlist restricts bot usage to specific Telegram user IDs or usernames.

## Cron Scheduler

DeepClaw includes an in-process cron scheduler that runs agent prompts on a schedule and delivers results to Telegram. Jobs are stored in `~/.deepclaw/cron/jobs.json` and can be managed via the `/cron`, `/cron_add`, and `/cron_rm` Telegram commands.

## Daemon Deployment

Run DeepClaw as a background service on macOS (launchd) or Linux (systemd):

```bash
# Install the service file
deepclaw service install

# Check status
deepclaw service status

# Uninstall
deepclaw service uninstall
```

Logs are written to `~/.deepclaw/logs/`.

## Project Structure

```
deepclaw/
  pyproject.toml        # Dependencies and entry points
  deepclaw/
    __init__.py
    bot.py              # Telegram bot with streaming responses
    config.py           # Layered config loader (env > .env > yaml)
    safety.py           # Dangerous command detection, path deny list, credential redaction, SSRF protection
    middleware.py       # SafetyMiddleware — gates tool calls through safety checks
    tools/
      __init__.py       # Plugin discovery (discover_tools)
      web_search.py     # Tavily web search + extract plugin
    scheduler.py        # Application-level cron scheduler
    service.py          # systemd/launchd service management
  tests/
    test_bot.py
    test_config.py
    test_safety.py
    test_middleware.py
    test_scheduler.py
    test_service.py
  docs/
    DESIGN_DEEPCLAW.md  # Full design document
```

## Roadmap

See [docs/DESIGN_DEEPCLAW.md](docs/DESIGN_DEEPCLAW.md) for the full design. Next up:

- **Post-MVP** — Discord, Slack, Signal, webhooks, smart routing, skills guard, RL training, and more
