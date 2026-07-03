# Live Telegram e2e smoke tests

These tests run the DeepClaw bot **in-process against real Telegram** and drive
it with a [Telethon](https://docs.telethon.dev) userbot that pairs, chats,
awaits streamed replies, and clicks the inline approval buttons. The agent
brain is scripted (no LLM calls), so the only thing under test is the Telegram
integration itself: server-side MarkdownV2 acceptance, entity parsing, button
clicks, and streaming edits.

They are **skipped by default** (in CI and locally) and only run when
`DEEPCLAW_E2E_TELEGRAM=1`.

## One-time setup

1. **Create a dedicated test bot.** Message [@BotFather](https://t.me/BotFather),
   create a new bot, and copy the token. Do **not** reuse your production
   DeepClaw bot token — the tests reset pairing state on every run.

2. **Get Telegram API credentials** for the userbot at
   [my.telegram.org](https://my.telegram.org) → API development tools. This
   yields an `api_id` and `api_hash`.

3. **Generate a Telethon StringSession** for the account that will play the
   human (a real account you control; consider a secondary account). This is
   interactive (phone number + login code) but only needed once:

   ```bash
   uv run --with telethon python -c "
   from telethon.sync import TelegramClient
   from telethon.sessions import StringSession
   import os
   with TelegramClient(StringSession(), int(os.environ['TELEGRAM_API_ID']), os.environ['TELEGRAM_API_HASH']) as c:
       print(c.session.save())
   "
   ```

   Treat the printed session string like a password — it grants full access to
   the account.

4. **Open the bot chat once** from the userbot account (send `/start` in any
   Telegram client), so the account can message the bot.

## Running

```bash
uv sync --extra dev --extra e2e

export DEEPCLAW_E2E_TELEGRAM=1
export DEEPCLAW_E2E_BOT_TOKEN=<test bot token from BotFather>
export TELEGRAM_API_ID=<api_id>
export TELEGRAM_API_HASH=<api_hash>
export TELETHON_SESSION=<string session from step 3>

uv run python -m pytest tests/e2e/ -v
```

The tests send a handful of messages per run and pace streaming edits at
~1/second, well under Telegram's flood limits. A full run takes on the order
of a minute.

## Running in CI

The `Live Telegram E2E` workflow (`.github/workflows/e2e-telegram-live.yml`)
runs these tests on demand via **Actions → Live Telegram E2E → Run workflow**.
It is never triggered by pushes or PRs. To enable it, add these repo secrets
(Settings → Secrets and variables → Actions):

- `DEEPCLAW_E2E_BOT_TOKEN`
- `TELEGRAM_API_ID`
- `TELEGRAM_API_HASH`
- `TELETHON_SESSION`

A concurrency group prevents two runs from polling the bot token at the same
time. The `TELETHON_SESSION` secret grants full access to the userbot account,
so use a dedicated secondary account and revoke the session from Telegram's
"Active Sessions" screen if it ever leaks.

## Notes

- The userbot account must be able to receive messages from the bot; if a test
  times out waiting for a reply, check that the bot isn't blocked and that no
  other process is polling the same bot token (Telegram allows only one
  poller — a `Conflict` error means your production bot is running with the
  same token).
- Sessions are revocable from Telegram's "Active Sessions" settings screen if
  the string leaks.
