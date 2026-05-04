---
name: browser-harness
description: Use browser-use/browser-harness for direct CDP browser control, local Chrome attachment, or remote Browser Use cloud browsers. Read the local checkout docs first.
version: 1.0.0
author: browser-use
license: MIT
metadata:
  hermes:
    tags: [browser, cdp, automation, chrome, browser-use]
    homepage: https://github.com/browser-use/browser-harness
---

# browser-harness

Local checkout:
- `/home/ubuntu/Developer/browser-harness`

Read these first:
1. `/home/ubuntu/Developer/browser-harness/SKILL.md`
2. `/home/ubuntu/Developer/browser-harness/helpers.py`
3. `/home/ubuntu/Developer/browser-harness/install.md` for first-time setup or reconnects

## Invocation

Use the global command directly:

```bash
browser-harness <<'PY'
new_tab("https://browser-use.com")
wait_for_load()
print(page_info())
PY
```

## Important defaults

- Prefer `new_tab(url)` for first navigation instead of `goto(url)` so you do not clobber the active user tab.
- After every meaningful action, verify with `screenshot()` or `page_info()`.
- Search `/home/ubuntu/Developer/browser-harness/domain-skills/` before inventing a site-specific approach.
- Check `/home/ubuntu/Developer/browser-harness/interaction-skills/` if you hit tabs, uploads, dialogs, iframes, dropdowns, screenshots, or scrolling issues.

## Remote browser pattern

If `BROWSER_USE_API_KEY` is set, prefer remote browsers when no local Chrome is available or when parallel isolated browser sessions are needed.

Important bootstrap gotcha: on a fresh headless machine, do **not** make `start_remote_daemon()` the first stdin lines of `browser-harness`, because the wrapper calls `ensure_daemon()` before it executes stdin. Bootstrap the remote daemon from Python first, then use `browser-harness` with the matching `BU_NAME`.

```bash
cd /home/ubuntu/Developer/browser-harness
uv run python - <<'PY'
from admin import start_remote_daemon
start_remote_daemon("work")
PY

BU_NAME=work browser-harness <<'PY'
new_tab("https://example.com")
wait_for_load()
print(page_info())
PY
```

When finished, stop the remote daemon so the cloud browser is shut down cleanly:

```bash
cd /home/ubuntu/Developer/browser-harness
uv run python - <<'PY'
from admin import stop_remote_daemon
stop_remote_daemon("work")
PY
```

## Local browser bootstrap

For attaching to a real local Chrome profile, follow the repo's `install.md` exactly. Key rule: try attaching first; only escalate to `chrome://inspect/#remote-debugging` if `DevToolsActivePort` is genuinely missing.
