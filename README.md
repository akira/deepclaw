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

| Environment Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram Bot API token from [@BotFather](https://t.me/BotFather) |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key (default provider) |
| `OPENAI_API_KEY` | No | OpenAI API key (if using OpenAI models) |
| `DEEPCLAW_MODEL` | No | Model override, e.g. `openai:gpt-4o` (defaults to `claude-sonnet-4-6`) |

*Or whichever provider API key matches your chosen model.

## How It Works

DeepClaw wires a DeepAgents agent to Telegram via long-polling. Each Telegram chat gets its own persistent conversation thread (using `chat_id` as the LangGraph `thread_id`), backed by SQLite checkpointing at `~/.deepagents/checkpoints.db`.

The agent has full DeepAgents capabilities out of the box:
- File operations (read, write, edit, glob, grep)
- Shell command execution
- Task planning (write_todos)
- Sub-agent spawning
- Context compression when conversations get long
- Skills and memory (AGENTS.md / SKILL.md)

## Project Structure

```
deepclaw/
  pyproject.toml       # Dependencies and entry points
  deepclaw/
    __init__.py
    bot.py             # Telegram bot (~130 lines)
  docs/
    DESIGN_DEEPCLAW.md # Full design document
```

## Roadmap

See [docs/DESIGN_DEEPCLAW.md](docs/DESIGN_DEEPCLAW.md) for the full design. Beyond the prototype:

- **MVP 1** — Multi-channel gateway with streaming, media handling, and auto-reply pipeline
- **MVP 2** — Security (dangerous command approval, SSRF protection, access control)
- **MVP 3** — Daemon deployment (systemd/launchd) and configuration system
- **MVP 4** — Cron scheduler with delivery routing to channels
- **Post-MVP** — Discord, Slack, Signal, webhooks, smart routing, skills guard, RL training, and more
