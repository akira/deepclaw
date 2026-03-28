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

DeepClaw includes a safety module that detects dangerous commands before they execute:

- **Dangerous command detection** — regex-based pattern matching for destructive shell commands (`rm -rf`, `mkfs`, `dd`), SQL statements (`DROP TABLE`, `TRUNCATE`), system config writes, fork bombs, piped remote execution, and more. Each pattern is categorized and assigned a severity level (critical/warning).
- **SSRF protection** — URL validation that blocks requests to private/internal networks (RFC 1918, loopback, link-local, cloud metadata endpoints).
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
    safety.py           # Dangerous command detection and SSRF protection
    scheduler.py        # Application-level cron scheduler
    service.py          # systemd/launchd service management
  tests/
    test_bot.py
    test_config.py
    test_safety.py
    test_scheduler.py
    test_service.py
  docs/
    DESIGN_DEEPCLAW.md  # Full design document
```

## Roadmap

See [docs/DESIGN_DEEPCLAW.md](docs/DESIGN_DEEPCLAW.md) for the full design. Next up:

- **Post-MVP** — Discord, Slack, Signal, webhooks, smart routing, skills guard, RL training, and more
