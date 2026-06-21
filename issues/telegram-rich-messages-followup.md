# Telegram rich messages follow-up

## title
Add Telegram rich-message delivery for constructs that MarkdownV2 degrades

## summary
DeepClaw PR #108 improved Telegram readability on the existing MarkdownV2 path, but tables and other richer markdown constructs still degrade to plain text or bullet-like formatting. Hermes already supports Telegram Bot API 10.1 rich messages for these cases. DeepClaw should add a minimal, guarded rich-message path for final Telegram delivery.

## evidence
- DeepClaw currently sends final Telegram replies with `ParseMode.MARKDOWN_V2` and `sendMessage` / `editMessageText`.
- Markdown pipe tables in DeepClaw still appear as raw text in Telegram.
- Hermes’ Telegram adapter explicitly uses `sendRichMessage` for constructs the legacy path degrades, including tables, task lists, details blocks, and math.
- PTB 22.6 in this DeepClaw environment exposes async `Bot.do_api_request`, so raw Bot API calls are viable here.

## expected behavior
- Final Telegram replies that contain rich-only constructs should render natively when Telegram supports them.
- Ordinary replies should remain on the existing MarkdownV2 path.
- Unsupported rich endpoints or bad rich payloads should fall back safely to the current MarkdownV2 behavior.

## actual behavior
- DeepClaw formats final Telegram replies more cleanly after PR #108, but still cannot render native tables or other rich constructs.
- The current final-delivery flow chunks to Telegram’s 4096-char MarkdownV2 limit and does not attempt Bot API rich-message delivery.

## proposed fixes
1. Add a `telegram.rich_messages` config flag, defaulting to `true`.
2. Add Telegram rich-delivery helpers in `deepclaw/channels/telegram.py`:
   - bot capability detection via `do_api_request`
   - content eligibility detection (initially tables/task lists/details/math)
   - rich payload builder using raw markdown
   - conservative fallback classification
3. Keep streaming edits on the existing MarkdownV2 path.
4. On final delivery, attempt rich send/edit only when eligible; otherwise preserve current behavior.
5. Add tests for:
   - rich eligible content using raw markdown payloads
   - fallback to MarkdownV2 when rich is unsupported or rejected
   - no accidental use of normalized/escaped MarkdownV2 text as the rich payload
