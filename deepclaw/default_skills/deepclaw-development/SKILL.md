---
name: deepclaw-development
description: Develop, configure, and extend DeepClaw — adding tool plugins, tuning SOUL.md, managing the venv, and restarting the service.
version: 1.0.1
platforms: [linux]
tags: [deepclaw, tool-plugins, configuration]
required_environment_variables:
  - OPENAI_API_KEY
  - TAVILY_API_KEY
---

# DeepClaw Development

DeepClaw is an AI agent gateway built on DeepAgents, running as a systemd user service. Source at `/home/ubuntu/deepclaw/`, config at `~/.deepclaw/`.

## When to Use

- Developing or debugging DeepClaw itself
- Working on tool plugins, scheduler behavior, SOUL.md, service deployment, or bundled skills
- Verifying local DeepClaw runtime behavior against the source tree

## Key Paths

| Path | Purpose |
|------|---------|
| `/home/ubuntu/deepclaw/` | Source repo |
| `/home/ubuntu/deepclaw/.venv/` | Python venv (Python 3.14) |
| `~/.deepclaw/SOUL.md` | System prompt / personality (generated at first run) |
| `~/.deepclaw/memory file` | Cross-session context injected each run (see agent.py for filename) |
| `~/.deepclaw/config.yaml` | Model, telegram, heartbeat settings |
| `~/.deepclaw/skills/` | Skill markdown files |
| `/home/ubuntu/deepclaw/deepclaw/tools/` | Tool plugin directory |

## Venv Management

No pip in the venv — use uv instead:

```bash
cd /home/ubuntu/deepclaw
uv add <package>
uv remove <package>
/home/ubuntu/deepclaw/.venv/bin/python /tmp/test_script.py
```

## Running Tests

The project venv uses Python 3.14. Always invoke pytest through the project venv's python binary directly — using a different outer venv or global `uv run` can pick up the wrong interpreter and fail with `ModuleNotFoundError: No module named 'langchain_core'`.

## Deterministic First

- Prefer deterministic checks over free-form reasoning for state that can be measured directly.
- Use logs, git history, tests, service status, filesystem inspection, and direct tool output before inferring behavior in-model.
- For DeepClaw operations, especially avoid mental approximations for scheduler state, thread state, tool registration, or deployment status when the repo or service can answer directly.

```bash
/home/ubuntu/deepclaw/.venv/bin/python -m pytest tests/ -v --tb=short
# scoped:
/home/ubuntu/deepclaw/.venv/bin/python -m pytest tests/test_bot.py -v
```

If pytest is not yet installed in the project venv, add it via `uv pip install` targeting `.venv/bin/python`, along with `pytest-asyncio` and `anyio`.

## Service Management

```bash
systemctl --user restart deepclaw.service
systemctl --user status deepclaw.service --no-pager
journalctl --user -u deepclaw.service -f
```

## Verification

- Re-run the relevant DeepClaw tests from the project venv after changes.
- Confirm the expected tools or behaviors are present in logs or direct commands.
- If the service was changed, restart it and inspect `systemctl` or `journalctl` output before declaring success.

## Writing a Tool Plugin

Plugins live in `/home/ubuntu/deepclaw/deepclaw/tools/<name>.py`. Auto-discovered — no registration needed.

### Browser / Playwright thread-affinity pitfall

If a DeepClaw tool uses Playwright's **sync** API, do not store browser/page objects in `threading.local()` and assume later tool calls will land on the same thread. DeepClaw tool calls may execute on different worker threads, which causes failures like:
- `No active browser session. Call browser_navigate first.`
- `greenlet.error: cannot switch to a different thread`

Correct pattern:
- keep browser session state process-global
- run **all** Playwright operations on a single dedicated worker thread
- protect worker startup with a lock so concurrent first-use cannot create multiple workers

In practice this means a small queue + worker-thread dispatcher (or a BrowserManager actor) is safer than per-thread session state.

Every plugin must export:
- `available() -> bool` — return True if deps and vars are present
- `get_tools() -> list` — return list of callable functions

The function docstring (with Args/Returns sections) becomes the tool description shown to the model. Follow the same pattern as `web_search.py` or `browser.py`.

DeepClaw loads `~/.deepclaw/.env` into the process env at startup, so plugins can read vars via standard env lookups.

### Testing a plugin before restarting

Write a test script to `/tmp/`, set any required env vars, import and call `available()` and `get_tools()`, run it with the venv python. Verify `available()` returns True before restarting.

```bash
/home/ubuntu/deepclaw/.venv/bin/python /tmp/test_plugin.py
```

Confirm tools appear in service logs after restart:
```bash
journalctl --user -u deepclaw.service -n 30 --no-pager | grep "Loaded.*tools"
```

### CRITICAL: available() must not gate on env vars

If `available()` checks for API keys and they are absent from the service environment, the entire plugin is silently skipped — tools never register, no error is shown.

### Vision plugin safety pitfall

If a DeepClaw vision/image tool accepts remote URLs (for example `image_path` that may be `http://` or `https://`), do **not** assume existing web-fetch SSRF middleware will protect it. The middleware currently inspects tools with a `url` arg on the web fetch path, so a vision tool taking `image_path` can bypass that check unless the tool validates URLs itself.

Correct pattern:
- call `check_url_safety_sync()` (from `deepclaw.safety`) inside the vision tool before accepting a remote URL
- reject anything that is not a public, SSRF-safe URL with a clear user-visible error
- keep the tool docs honest: if you say "public image URL", enforce that in code

Rule: `available()` checks packages only. Credential errors surface at call time via the tool's return value.

```python
def available() -> bool:
    try:
        import some_package  # noqa: F401
        return True
    except ImportError:
        return False
```

### Adding credentials to DeepClaw

Add as `Environment=KEY=value` lines in the deepclaw systemd service file, or add to `~/.deepclaw/.env`. After editing the service file run `systemctl --user daemon-reload` then restart.

## Tuning SOUL.md for High Agency

The deployed `~/.deepclaw/SOUL.md` overrides DEFAULT_SOUL in `agent.py`. They can drift.

**Diagnostic:** Compare `~/.deepclaw/SOUL.md` against DEFAULT_SOUL in `/home/ubuntu/deepclaw/deepclaw/agent.py`. The source version may have a better Autonomy section that didn't make it into the deployed file.

**Low-agency symptoms:** Agent asks clarifying questions instead of acting, stops mid-task to confirm, hedges unnecessarily.

**Fixes:**
1. Replace "ask a smart question rather than make a bad assumption" with "make a reasonable assumption, state it briefly, and proceed"
2. Strengthen no-confirmation: "Never stop to ask unless you are genuinely blocked with no way to proceed"
3. Add/restore the Autonomy section: bold with internal actions (reading, testing, exploring); careful with external actions (sending messages, modifying shared infra); come back with answers not questions
4. If you want the agent to proactively save reusable workflows, add a **Skills** section telling it to use skills as procedural memory, check for relevant skills before starting, and create/update skills after non-trivial or iterative work.

**Important:** changing only `~/.deepclaw/SOUL.md` updates the live bot on that machine after restart, but it does **not** change what ships in the repo. If the behavior should persist for fresh installs / newly seeded configs, also patch `DEFAULT_SOUL` in `/home/ubuntu/deepclaw/deepclaw/agent.py` in the same PR.

## Memory / Context File

DeepClaw injects a context file at the start of each session. Keep it useful:
- User name, preferences, timezone
- Environment paths (venv, config, source)
- Available tools and how to use them
- Common patterns for the codebase

More context = fewer questions from the agent. Filename is set in `agent.py` (MEMORY_FILE constant).

### Important: do not force memory/skills to reload every turn unless you truly need to

DeepAgents caches middleware state like `memory_contents` and `skills_metadata` in the thread checkpoint. That means:
- adding memory with `memory_add` / `memory_replace` / `memory_remove`
- installing or changing skills with `skill_create` / `skill_update` / `skill_install` / `skill_delete`

does **not** automatically make the new data visible inside the same thread on the next turn if you rely only on `MemoryMiddleware` / `SkillsMiddleware`.

A tempting fix is custom middleware that pops those keys and reloads them on every turn. That works, but it is the blunt instrument and adds avoidable complexity / overhead.

Preferred fix: targeted checkpoint invalidation after mutations.

Implementation pattern that worked:
1. Keep normal `MemoryMiddleware` and `SkillsMiddleware` in `create_agent()`.
2. Add a helper (for example `state_cache.py`) that:
   - uses `checkpointer.aget_tuple({"configurable": {"thread_id": ..., "checkpoint_ns": ""}})`
   - copies the checkpoint with `copy_checkpoint(...)`
   - removes `memory_contents` and `skills_metadata` from `checkpoint["channel_values"]`
   - writes a new checkpoint with `create_checkpoint(...)` + `checkpointer.aput(...)`
3. Call that invalidation helper after successful mutating tool calls.
   - Good place: `SafetyMiddleware.awrap_tool_call()` after `handler(request)` returns for mutating tools
   - Tool names to watch: `memory_add`, `memory_replace`, `memory_remove`, `skill_create`, `skill_update`, `skill_install`, `skill_delete`
4. Also call the same invalidation helper from Telegram slash-command handlers after local mutations, because those bypass the normal agent tool-call path.

Why this is better:
- preserves the thread's message history
- avoids re-reading AGENTS.md and skills on every turn
- makes newly added memory / installed skills visible on the next turn without requiring `/clear`

Verification pattern:
- inspect checkpoints with `AsyncSqliteSaver.alist(...)` and confirm stale threads contain cached `memory_contents` / `skills_metadata`
- after invalidation, confirm those keys are gone but `messages` remain
- test both the helper itself and Telegram slash commands that should trigger invalidation

### Important middleware caching pitfall

DeepAgents `MemoryMiddleware` and `SkillsMiddleware` cache loaded state in the conversation thread state (`memory_contents`, `skills_metadata`). That means changes on disk can be invisible for the rest of the thread:
- editing `~/.deepclaw/AGENTS.md` via memory tools may not affect the current conversation until `/clear` or a fresh thread
- installing a skill under `~/.deepclaw/skills/` may not make the skill visible in the current conversation until `/clear` or a fresh thread

If you need memory/skills to refresh immediately, subclass the middleware and drop the cached keys before delegating:
```python
class ReloadingMemoryMiddleware(MemoryMiddleware):
    def before_agent(self, state, runtime, config=None):
        fresh_state = dict(state)
        fresh_state.pop("memory_contents", None)
        return super().before_agent(fresh_state, runtime, config)

    async def abefore_agent(self, state, runtime, config=None):
        fresh_state = dict(state)
        fresh_state.pop("memory_contents", None)
        return await super().abefore_agent(fresh_state, runtime, config)
```

Do the analogous thing for `SkillsMiddleware` with `skills_metadata`.

### Critical hook-signature pitfall when subclassing middleware

When overriding `before_agent` / `abefore_agent` on DeepAgents middleware, keep the method signature compatible with the framework runner. In practice:
- accept `config=None` (or `RunnableConfig | None = None`)
- do **not** make `config` required in your override

A required `config` parameter caused live Telegram requests to fail with:
- `TypeError: ReloadingMemoryMiddleware.abefore_agent() missing 1 required positional argument: 'config'`

This failure surfaces to the user as the generic Telegram error:
- `Sorry, something went wrong processing your message.`

If you see that symptom right after a middleware change, inspect `journalctl --user -u deepclaw.service -n 120 --no-pager` first.

## Adding Telegram Slash Commands

## Adding Telegram Slash Commands

Commands are registered in two places:

1. **Handler** — `application.add_handler(CommandHandler("name", cmd_fn))` in `run_telegram()`
2. **Dropdown** — `await application.bot.set_my_commands([BotCommand(...), ...])` in `post_init()`

Without `set_my_commands`, Telegram never shows the slash dropdown menu to users — handlers work but autocomplete doesn't appear. Leave `/pair`, `/start`, and debug-only commands out of the menu to reduce clutter.

**Standard command pattern** — all handlers must check auth before acting:
```python
async def cmd_foo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorize_chat(update):
        return
    if not is_user_allowed(update, context.bot_data.get(ALLOWED_USERS_KEY, set())):
        await update.message.reply_text(REJECTION_MESSAGE)
        return
    # handler logic
```

### Pattern for one-command subcommand UIs (`/skills`, `/cron`, etc.)

If you want a compact Telegram UX, prefer a single top-level command with subcommands parsed from `update.message.text`, rather than many separate Telegram commands.

A good pattern is:
- add one handler: `CommandHandler("skills", cmd_skills)`
- add one menu entry: `BotCommand("skills", "Browse or manage local skills")`
- parse subcommands with a helper like `_parse_skills_command()`
- support a default action when no subcommand is provided (for `/skills`, default to browse/list)
- keep usage strings explicit, e.g.:
  - `/skills view <name>`
  - `/skills create <name> | <description>`
  - `/skills install <path> [| <name>]`
  - `/skills delete <name>`
- when returning full skill content, chunk with `chunk_message(..., TELEGRAM_MESSAGE_LIMIT)` before `reply_text`

This worked well for local skill management because it exposed browse/view/create/install/delete behavior without cluttering Telegram's slash menu with many near-duplicate commands.

## Keeping PRs Clean When Local `main` Contains Deploy-Only Merges

DeepClaw local `main` may include private/local deploy commits like `deploy: merge ... locally` that should not appear in a GitHub PR. If you branch from that local `main`, your PR can accidentally include already-merged upstream work plus local deploy history, making the PR look much larger than it really is.

Safer pattern:
```bash
git fetch origin
git checkout -b feat/my-change origin/main
# ... commit work ...
```

If a PR branch is already polluted after upstream merged an earlier PR:
1. reset the feature branch to `origin/main`
2. cherry-pick only the real feature commits you want to keep
3. force-push the cleaned branch

Example cleanup flow that worked:
```bash
git checkout feat/my-branch
git reset --hard origin/main
git cherry-pick <feature-commit-1> <feature-commit-2>
git push --force-with-lease fork feat/my-branch
```

This is especially important when stacking PRs after a previous DeepClaw PR has already been merged upstream.

### `/skills` slash command pattern

A good Telegram wrapper for local skill management is a single `/skills` command with subcommands instead of many top-level slash commands. This keeps the Telegram menu small while still exposing the functionality.

Observed working shape:
- `/skills` or `/skills browse` — list installed skills
- `/skills view <name>` — show one `SKILL.md`
- `/skills create <name> | <description>` — create a templated skill
- `/skills install <path>` or `/skills install <path> | <name>` — import a skill
- `/skills delete <name>` — remove a skill
- aliases: treat `remove` and `rm` the same as `delete`

Implementation notes that worked well:
- add a small parser helper like `_parse_skills_command(raw_text) -> (subcommand, args)` so tests don't depend on ad-hoc string splitting in the handler
- use a formatter helper like `_format_skills_list()` so `/skills` output can be unit tested separately
- for `view`, chunk the combined header + file contents with `chunk_message(..., TELEGRAM_MESSAGE_LIMIT)` before replying
- for `create` and `install`, using `|` as the delimiter makes multi-word descriptions and paths manageable in Telegram text commands
- keep the help text and `set_my_commands()` menu in sync with the handler behavior — forgetting one of these is easy
- add focused tests for parse helpers plus happy-path `/skills` subcommands; monkeypatch the underlying `skill_*` functions instead of touching the real filesystem

### Good pattern for subcommand-style Telegram UX

For a feature with multiple operations, a single slash command with subcommands works well. Example that shipped successfully for local skill management:
- `/skills` or `/skills browse`
- `/skills view <name>`
- `/skills create <name> | <description>`
- `/skills install <path>`
- `/skills install <path> | <name>`

Implementation notes that worked well:
- add a tiny parser helper like `_parse_skills_command(raw_text) -> (subcommand, args)`
- make the bare command default to a sensible action (`browse` for `/skills`)
- keep write operations narrow and explicit; prefer simple `|`-separated arguments over trying to parse free-form natural language inside the command handler
- reuse the existing tool/plugin functions inside the slash handler (`skills_list`, `skill_view`, `skill_create`, `skill_install`) instead of duplicating business logic
- for long content like `SKILL.md`, chunk replies with `chunk_message(...)`
- update `/help` text and the Telegram `set_my_commands(...)` menu in the same PR so the command is both implemented and discoverable

### Testing slash commands

A lightweight test pattern that worked well in `tests/test_bot.py`:
- add `_make_slash_update()` and `_make_slash_context()` helpers
- monkeypatch the underlying tool functions (`skill_view`, `skill_create`, `skill_install`, etc.) rather than touching the real filesystem
- assert on `update.message.reply_text.call_args[0][0]` for the user-visible reply text
- test at least:
  - default/bare command behavior
  - one read path (`view`)
  - one write path (`create` or `install`)
  - usage text for invalid or unknown subcommands

## Ruff CI — Format Check vs Lint

The CI runs **two separate ruff steps**:
- `ruff check` — lint rules (SIM, E, F, I, etc.)
- `ruff format --check` — formatting (like black)

Always run both locally before pushing. A PR can pass lint but fail format — they are independent checks.

## Ruff Lint — SIM117

Ruff rule SIM117 rejects nested `with` statements. Always combine them:
```python
# wrong — SIM117 violation
with patch("module.A", 0):
    with patch("module.B") as mock:
        ...

# correct
with (
    patch("module.A", 0),
    patch("module.B") as mock,
):
    ...
```

## Hot-Reloading the Agent (model switch)

The agent is created once at startup via `create_agent(config, checkpointer)`. To switch models at runtime, rebuild both the agent and the gateway and update `bot_data`:

```python
from dataclasses import replace
new_config = replace(config, model=new_model)
context.bot_data[CONFIG_KEY] = new_config
checkpointer = context.bot_data["checkpointer_resolved"]
new_agent = create_agent(new_config, checkpointer)
context.bot_data["agent"] = new_agent
context.bot_data[GATEWAY_KEY] = Gateway(agent=new_agent, streaming_config=new_config.telegram.streaming)
```

**Pitfall:** `DeepClawConfig` is a `@dataclass` — `dataclasses.replace()` only works on real dataclass instances. Test mocks that use `SimpleNamespace(model=...)` will raise `TypeError: replace() should be called on dataclass instances`. Always use `DeepClawConfig(model=...)` in tests that exercise code paths touching the config.

**Pitfall: Agent self-reports wrong model after a runtime switch.** The LLM answers "what model are you" from conversation history or injected context (AGENTS.md), not the actual runtime config. If AGENTS.md has the old model name hardcoded, the bot keeps reporting it even after a `/model` switch. Two fixes needed: (1) inject `config.model` into the system prompt inside `create_agent()` as an explicit "Active Model" section, so the agent always knows what it's running as at construction time; (2) rewrite the `- Model:` line in AGENTS.md inside the `/model` command handler using `re.sub()` so memory stays accurate across restarts.

## Contributing via PR

DeepClaw source is at `akira/deepclaw` (origin) and the fork is at `BlueMeadow19/deepclaw` (remote name: `fork`).

```bash
# Branch from latest main
git checkout main && git pull origin main
git checkout -b feat/my-feature

# ... make changes ...

# Run tests through the project venv (not global uv run)
/home/ubuntu/deepclaw/.venv/bin/python -m pytest tests/ -v --tb=short

# Commit — stage only the files you want, exclude .beads/
git add deepclaw/... tests/...
git commit -m "feat: description"

# Push to fork and open PR
git push fork feat/my-feature
gh pr create --repo akira/deepclaw --head BlueMeadow19:feat/my-feature --title "..." --body "..."
```

**If .beads/ gets accidentally committed:** `git rm --cached .beads/issues.jsonl`, add `.beads/` to `.gitignore`, then `git commit --amend --no-edit` + `git push fork <branch> --force`.

## Deploy Changes Locally

After a branch is ready, merge to local main and restart:

```bash
git checkout main
git merge feat/my-feature --no-ff -m "deploy: merge <feature>"
systemctl --user restart deepclaw.service
systemctl --user status deepclaw.service --no-pager
```

## Beads Task Tracking

Beads is initialized inside the deepclaw repo (not the openclaw workspace):

```bash
cd /home/ubuntu/deepclaw
bd init          # first time only
bd list
bd create "Task title" -p 1   # p0=critical, p1=high, p2=medium, p3=low
bd dep add <child-id> <parent-id>   # child depends on (is blocked by) parent
```

Keep `.beads/` in `.gitignore` — it's local-only task state, not for the PR.

## Telegram Media Upload Handling

To let DeepClaw process Telegram image uploads, register a media handler alongside the text handler:

```python
application.add_handler(
    MessageHandler((filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND, handle_message)
)
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
```

Recommended pattern in `handle_message()`:
1. Accept messages with `text`, `caption`, `photo`, or `document` instead of early-returning on `message.text`
2. Download supported media to `~/.deepclaw/uploads/telegram/<chat_id>/`
3. Append the saved local path into the user prompt, e.g. `Attached image saved at local path: ...`
4. Explicitly tell the agent to call `vision_analyze` on that path if the image matters
5. For unsupported docs (PDF, arbitrary binaries), reply with a user-visible explanation instead of silently ignoring them

Useful helper split:
- `_looks_like_supported_image(document)` — accept `image/*` MIME types or image filename suffixes
- `_download_media_file(update)` — save the uploaded photo/document and return `(path, error)`
- `_build_incoming_text(update)` — combine caption/text with the saved file path note

Supported image suffixes used successfully: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`

## Vision Tool Deployment Notes

The `vision.py` plugin can be always-loaded if it uses only stdlib imports and returns credential/backend errors at call time. That way the tool is visible even when the backend key is missing.

Observed good behavior:
- `available()` returns `True`
- if `OPENAI_API_KEY` is missing, `vision_analyze()` returns a clear error like `Vision backend unavailable: OPENAI_API_KEY is not set...`
- browser screenshot -> local file path -> `vision_analyze(path, question)` is the intended loop

After deployment, verify with:
```bash
journalctl --user -u deepclaw.service -n 30 --no-pager
```
Look for:
- `Loaded 1 tools from plugin vision`

## Local Skill Management Workflow

DeepClaw already loads `~/.deepclaw/skills/` through `SkillsMiddleware`, but first-class management can live in a repo plugin, e.g. `deepclaw/tools/skills.py`.

A solid minimal toolset is:
- `skills_list()` — enumerate installed skills from `~/.deepclaw/skills`
- `skill_view(name)` — read one skill's `SKILL.md`
- `skill_create(name, description, content=None, overwrite=False)` — create a new skill or templated `SKILL.md`
- `skill_update(name, content)` — replace an existing skill's `SKILL.md`
- `skill_install(source_path, name=None, overwrite=False)` — import from a local `SKILL.md` file or directory containing one

Implementation notes that worked well:
- Import `CONFIG_DIR` from `deepclaw.config` and derive `SKILLS_DIR = CONFIG_DIR / "skills"` inside the plugin. Do **not** import `SKILLS_DIR` from `deepclaw.agent` in the tool plugin, because that drags in the full agent/deepagents dependency chain just to use the filesystem path.
- Validate skill names with a strict regex (lowercase letters, digits, `_`, `-`) to prevent path traversal and weird directory names.
- Constrain all writes to `~/.deepclaw/skills` and require explicit `overwrite=True` before replacing an existing installed skill.
- For installs, accept only:
  - a standalone `SKILL.md`
  - or a directory containing `SKILL.md`
- For standalone `SKILL.md` installs, derive the default installed skill name from the parent directory name. If the file is at filesystem root (for example `/SKILL.md`), fall back to `src.stem.lower()` so the default name becomes `skill` instead of an empty/invalid name.
- Preserve extra files when importing a directory via `shutil.copytree()`.
- Derive a useful description from YAML frontmatter `description:` when present; otherwise fall back to the first markdown heading or first non-empty line.
- Important compatibility pitfall: DeepClaw's local `/skills` tooling can read plain markdown `SKILL.md` files, but DeepAgents `SkillsMiddleware` is stricter and skips skills without valid YAML frontmatter. Symptom in logs: `Skipping .../SKILL.md: no valid YAML frontmatter found`, while `/skills` still lists the skill.
- Preferred fix: normalize skill content on `skill_create()`, `skill_update()`, and `skill_install()` so every stored `SKILL.md` has frontmatter like:
  - `---`
  - `name: my-skill`
  - `description: ...`
  - `---`
- Also add a startup migration in `_setup_skills()` that scans `~/.deepclaw/skills/*/SKILL.md` and upgrades legacy plain-markdown skills in place. This keeps old installed skills usable by `SkillsMiddleware` without manual reinstallation.
- When synthesizing frontmatter for a legacy skill, use the directory name as `name` and the first heading / extracted description as `description`, then prepend the original markdown body.
- Test by monkeypatching the plugin's `SKILLS_DIR` constant to a temp directory rather than touching the real `~/.deepclaw/skills`.
- Add a regression test that `_setup_skills()` upgrades a legacy skill file with no frontmatter.
- Update `tests/test_tools.py` so plugin discovery asserts the new skill-management tools are present.

## Available Tool Plugins (as of 2026-04)

| Plugin | Required vars | Notes |
|--------|--------------|-------|
| `web_search.py` | TAVILY_API_KEY | web_search + web_extract |
| `browser.py` | — | Local Playwright/Chromium by default; Browserbase env vars are optional fallback for cloud sessions |
| `cron.py` | — | schedule/list/remove tasks |
| `memory.py` | — | explicit AGENTS.md memory tools: add / replace / remove / search |
| `skills.py` | — | local skill management: list/view/create/update/install under `~/.deepclaw/skills` |

## Browser Tool Pitfall — Playwright Sync API and Threads

DeepClaw tool calls may run on different worker threads. The Playwright **sync** API binds browser/page objects to the thread that created them, so storing session state in `threading.local()` breaks multi-call workflows:
- `browser_navigate()` can succeed
- then `browser_snapshot()` / `browser_click()` on a later tool call can fail with either:
  - `No active browser session. Call browser_navigate first.`
  - or greenlet / "Cannot switch to a different thread" errors

**Fix pattern:** keep all browser operations on one dedicated browser thread and marshal every browser tool function onto that thread. Process-global session storage alone is not enough, because Playwright objects themselves are thread-affine.

Recommended implementation shape in `deepclaw/tools/browser.py`:
1. Keep `_SESSION` in process-global state, protected by a lock
2. Start one daemon browser worker thread lazily
3. Send each browser operation (`navigate`, `snapshot`, `click`, `type`, `press`, `scroll`, `screenshot`, `close`) through a queue/future helper like `_run_in_browser_thread()`
4. Add a regression test that reproduces cross-thread usage:
   - call `browser_navigate()` on one thread
   - call `browser_snapshot()` on another thread
   - assert it succeeds

This bug is easy to misdiagnose as "session storage lost" when the deeper cause is Playwright sync thread affinity.



### Vision tool implementation notes

For lightweight multimodal support, a DeepClaw tool plugin can call OpenAI's vision-capable chat completions API using only Python stdlib (`urllib`, `json`, `base64`, `mimetypes`) instead of adding the `openai` package as a runtime dependency.

Recommended pattern:
- `available()` should return `True` if the module has no non-stdlib deps
- check `OPENAI_API_KEY` inside the tool function and return a useful error if missing
- accept either a local file path or `http/https` image URL
- for local files, infer MIME type and send a `data:<mime>;base64,...` image URL payload
- test both string `message.content` responses and list-of-content-blocks responses from the API
- include one regression test that uses a tiny PNG fixture written from raw bytes and patches `urllib.request.urlopen`
