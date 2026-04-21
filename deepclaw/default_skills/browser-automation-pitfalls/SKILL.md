---
name: browser-automation-pitfalls
description: Known pitfalls and failure patterns when using browser automation (Browserbase/Playwright) for web tasks — anti-bot detection, dynamic form field ref collisions, iframe issues, and workarounds.
version: 1.0.0
author: DeepClaw
license: MIT
platforms: [linux, macos]
---

# Browser Automation Pitfalls

Known failure patterns encountered with the browser tool (Browserbase + Playwright).

## Ref Collisions on Dynamically Appearing Fields

**Problem:** When a form reveals a new field after you fill the first one (e.g., "Confirm password" appears after typing a password), the new field may share the same `ref` ID (e.g., `@e6`) as an existing element. `browser_type` will fail with: `Selector "@eN" matched 2 elements`.

**Workaround attempts (ranked by success):**
1. **Re-navigate and use `browser_type` immediately on fresh refs** — only works if the confirm field is visible on load, not dynamically revealed
2. **Tab navigation** — click the first field, press Tab twice (once past reveal button, once to confirm field), then type characters via `browser_press` — unreliable, characters may not land
3. **Vision + annotate** — use `browser_vision(annotate=True)` to identify the visual position, but it won't give a usable ref if the DOM ref is still ambiguous
4. **None reliably worked** — if the field is dynamically injected and shares a ref, you may be stuck

**Root cause:** The browser snapshot assigns refs based on DOM order at snapshot time. If a new element is injected mid-session with the same ordinal index, refs collide.

---

## Sites with Strong Anti-Bot Signup Defenses

These sites actively block automated account creation:

### Proton Mail (account.proton.me/mail/signup)
- **Confirm password field** appears dynamically after typing, causes ref collision (see above)
- **CAPTCHA** — iframe-based human verification appears after form submission
- **Fingerprinting** — detects headless/automated browser sessions
- **Result:** Full automated registration is not feasible without residential proxies + CAPTCHA solving

### General patterns to watch for:
- `iframe`-embedded forms (payment, CAPTCHA) — can't interact with them via snapshot refs
- `shadow-root` elements — don't appear in accessibility tree
- Bot detection that triggers on headless user-agent or missing mouse movement entropy

---

## iframe Form Elements

**Problem:** Elements inside `<iframe>` (e.g., Stripe payment forms, hCaptcha, reCAPTCHA) do not appear in `browser_snapshot`. You cannot click or type into them.

**Workaround:** None from within the current tool. Require frame-switching support which Browserbase's MCP bridge doesn't expose.

---

## Proton Mail Signup — What Was Tried

Attempted flow for `account.proton.me/mail/signup?plan=free`:
1. Navigated to free signup URL ✅
2. Typed password in first field (`@e4`) ✅
3. Clicked "Start using Proton Mail now" → triggered "Confirm password" field ✅
4. Tried `browser_type(@e6)` on confirm field → `matched 2 elements` ❌
5. Tried Tab navigation from password field → field not focused ❌
6. Tried `browser_vision(annotate=True)` → confirmed field at visual position but ref was still @e6 (collision) ❌
7. Conclusion: Cannot proceed past confirm password step with current tooling

---

## Single-Page Apps That Ignore URL Search Params

**Problem:** Some SPAs (e.g. rec.us) render their full content regardless of `?search=` query params in the URL — the params are client-side routing hints that only work when typed into the app's own search box, not when passed in the URL directly.

**Example:** `https://www.rec.us/sfrecpark?search=upper+noe` loads the full facility list unfiltered. The search param is silently ignored.

**Workaround:** Find the direct canonical URL for the specific resource (e.g. `https://www.rec.us/uppernoe`) by inspecting the page source or the reservable courts list on the parent site. This is more reliable than trying to drive search UI via URL params.

---

## Password Fields Inside Modal Dialogs (rec.us pattern)

**Problem:** A modal dialog has a scrollable inner container. The password field is below the fold. `browser_snapshot` sees it (e.g. `@e17`) but `browser_type` fails with `matched 6 elements` (or `matched 2 elements`) because the same ordinal ref exists multiple times across the page's shadow/nested DOM.

**Workaround that works:**
1. Use `browser_type` on the email field — works after dialog is fully loaded (may time out on first attempt; retry once)
2. Press `Tab` once to move focus to the password field
3. Verify focus: snapshot will show "Required" validation text under the field when focused
4. Type each character individually with `browser_press` — reliable for passwords
5. For long passwords, consider `browser_press("Control+a")` first to clear any partial content
6. Click submit button by ref

**Key insight:** `browser_press` always goes to the currently focused element, bypassing ref resolution entirely. Use it when `browser_type` fails due to ref collisions.

**Limitation:** Very slow for long passwords (one key press per character). No clipboard paste available in headless mode (`xdotool` requires a DISPLAY).

**Delegation workaround for long passwords:** If the parent agent is low on iteration budget, delegate to a subagent with explicit per-character instructions. Pass the password spelled out as individual characters in the goal string (e.g. "type: w a n ! M K J 8 ..."). Subagents have a fresh budget and can handle all the keypresses. This was verified working for rec.us login (19-char password, confirmed successful login with green avatar badge appearing).

**Practical solution for long passwords — delegate to a subagent:**
When the password is long (10+ chars), character-by-character typing will exhaust the main agent's iteration budget. Delegate to a subagent with its own budget. Pass the password characters explicitly spelled out in the goal.

Example delegation goal:
```
Log into rec.us with email X / password Y.
1. Navigate to URL, click Log In
2. browser_type email field
3. Tab to password field (verify focus by seeing "Required" text in snapshot)
4. browser_press each character one at a time: c h a r 1 ...
5. Click submit, screenshot and report result (success = initials avatar in top-right)
```

**rec.us specific:** Login success confirmed by green initials avatar (e.g. "AK") appearing in top-right corner and "Reservations"/"Fast Track" nav links becoming visible.

---

## X/Twitter — Blocks Automated Login (Camoufox and Browser Use / browser-harness)

**Problem:** X/Twitter aggressively blocks automated logins across multiple browser stacks. The email field can often be filled successfully, but clicking **Next** or submitting the form resets the flow back to the outer landing/login wall instead of advancing to password entry.

Observed across:
- **Camoufox** (fingerprint-spoofing Firefox)
- **Browser Use cloud + browser-harness** (remote CDP browser)

**Camoufox findings:**
- The form accepted input, but clicking Next ultimately led to the hidden body-text error:
  - `"Could not log you in now. Please try again later."`
- The page often stayed at:
  - `https://x.com/i/flow/login`
- Password field never appeared.

**Browser Use / browser-harness findings:**
- The page body could show the login modal text:
  - `Sign in to X`
  - `Phone, email, or username`
  - `Next`
- The email field was found and filled successfully, including with **character-by-character keypresses**.
- But the **Next** step failed consistently:
  1. coordinate click on the visible Next button
  2. JS click on the underlying button
  3. Enter key submission
- All three caused the page to bounce back to the outer landing state (`Happening now / Join today / Sign in`) instead of advancing.
- In browser-harness screenshots, the session could also appear to be behind a centered loading spinner/modal even while the live Browser Use session looked interactive to the human viewer.

**What was tried (all failed at submission stage):**
1. `fill()` / JS value setter + `input`/`change` events
2. character-by-character typing / sequential keypresses
3. coordinate click on **Next**
4. JS `.click()` on **Next**
5. pressing `Enter`
6. reloading the login flow and retrying from fresh state

**Root cause:** X appears to detect automated sessions via behavioral/fingerprinting signals and silently rejects or resets the login flow at submission time. The failure is **not** necessarily bad credentials if the same email/password work manually.

**Important diagnosis:**
- If typing works but **Next** resets the page back to the outer landing state, treat that as an X anti-bot/login-defense symptom.
- The blocker is the **submission step**, not necessarily input entry.

**Workaround:**
- Prefer a **cookie-backed session** or an existing logged-in browser/profile.
- For Browser Use, use a **cloud profile** or sync a real local Chrome profile and reuse it.
- Alternatively use the X API / OAuth tokens for programmatic access.

**Notes:**
- `document.body.innerText` is useful for detecting whether the inner login modal is present even when DOM selectors are flaky.
- browser-harness/browser-use may show a loading spinner/modal in screenshots while the human watching the live session sees the login form; trust the live session plus body text more than a single screenshot in that case.

---

## X/Twitter — Browser Use / browser-harness Fresh Login Also Fails Unreliably

**Context:** Tried with `browser-use/browser-harness` using Browser Use cloud browsers (remote daemon + `browser-harness`) after installing the upstream skill/docs and retrying multiple input/click strategies.

**What reliably worked:**
- Browser Use remote browser provisioning itself worked.
- `x.com/i/flow/login` loaded.
- The page sometimes exposed login-step text in `document.body.innerText`, including:
  - `Sign in to X`
  - `Phone, email, or username`
  - `Next`
- The email field could sometimes be found and filled, including a successful character-by-character entry of the email.

**What failed repeatedly:**
- The login modal state was unstable:
  - screenshots could show a centered white modal with spinner while body text suggested the login step existed
  - after waiting, the page often fell back to the outer landing page (`Happening now`, `Join today`, `Sign in`)
- The email field was not consistently exposed as a stable DOM target even when the login text was present.
- After successful email entry, every attempt to activate `Next` failed to advance:
  1. direct coordinate click on the known `Next` button center
  2. JS click on the `Next` control
  3. pressing `Enter`
  4. `Tab` to `Next` then `Enter`
- Post-submit behavior typically reset the page back to the outer landing state instead of proceeding to password entry.

**Interpretation:**
- This looks like X login gating / anti-bot behavior, not a Browser Use provisioning problem.
- The blocking point is specifically the transition after the email step (`Next`), not initial page load or typing itself.

**Practical guidance:**
- Do **not** spend many more iterations on fresh-login click variants once you confirm:
  - email entry works
  - `Next` resets or bounces back to landing
- Prefer a cookie-backed session instead:
  1. Browser Use cloud profile with existing X cookies
  2. synced local Chrome profile uploaded to Browser Use
  3. manual one-time login in the Browser Use live session, then reuse the saved profile later
- If you must experiment, the most informative signals are:
  - compare `document.body.innerText` against screenshot state
  - verify whether the email field value actually changed before pressing `Next`
  - treat spinner/landing-page oscillation as a sign to stop brute-forcing submission methods

## Yelp — Completely Blocks Automated Access

**Problem:** Yelp aggressively blocks all headless browser sessions with a "You have been blocked" page (cites bot detection from IP). Individual business pages are also blocked. No workaround via browser automation.

**What was tried:** Direct search URL, individual business pages, different URL variations — all blocked.

**Workaround:** Use Eater SF, The Infatuation, or Google Maps instead for restaurant data. These are editorially richer anyway.

**Note:** TripAdvisor also blocks similarly. Google/Bing show CAPTCHA challenges.

---

## DeepClaw Browser Tool Architecture

DeepClaw's browser tooling should be treated as a Playwright-based local browser plugin first, with Browserbase only as an optional remote backend. The safest default is local Chromium/Playwright with runtime credential checks instead of gating tool registration on API keys.

For DeepClaw browser plugins:
- use Playwright directly with local Chromium (`pw.chromium.launch(headless=True)`) when possible
- make `available()` check only for installed packages, not API keys
- validate optional remote credentials at call time, not load time, so the plugin still registers and can return a clear runtime error if needed

---

## Building a Playwright Browser Plugin for Third-Party Agents (DeepClaw Pattern)

**Context:** When adding browser tools to an agent framework that uses a plugin discovery system (e.g. DeepClaw's `discover_tools()`), there are several pitfalls.

### `page.accessibility` doesn't exist in Playwright >= 1.46
**Problem:** `page.accessibility.snapshot()` raises `AttributeError: 'Page' object has no attribute 'accessibility'` in newer Playwright versions.
**Fix:** Use `page.locator("body").aria_snapshot()` as fallback, or better — use `page.evaluate()` with custom JS to extract the accessibility tree yourself.

### JS-based ref system is more reliable than Playwright's built-in accessibility API
The most robust approach: inject refs directly into DOM elements via `page.evaluate()` using a JS snapshot function. Then find elements by ref using a second JS function that walks the DOM looking for `el._deepclaw_ref === refNum` and returns a CSS selector. This bypasses all Playwright version differences entirely.

```python
# Snapshot JS tags each element with _deepclaw_ref = counter
# Locator JS walks DOM to find element and returns CSS selector
selector = page.evaluate(FIND_REF_JS, ref_num)
locator = page.locator(selector)
```

### `available()` must NOT check for API keys — only packages
**Problem:** If `available()` returns False (e.g. because API keys aren't set), `discover_tools()` silently skips the entire plugin. The agent never sees the tools at all.
**Fix:** Only check for installed packages in `available()`. Do API key validation at call time inside the tool function, returning a clear error message.

```python
def available() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except ImportError:
        return False
```

### Snapshot pagination prevents truncation
For large pages, cap elements per call and return a hint:
```python
def browser_snapshot(offset: int = 0, limit: int = 100) -> dict:
    ...
    if offset + limit < total:
        result["hint"] = f"Call browser_snapshot(offset={offset + limit}) for more."
```

### Ruff lint rules to watch for in browser plugins
- **PLW2901**: Don't reuse loop variable name (e.g. `for line in ...: line = line.strip()` → rename to `raw_line`)
- **B904**: In `except` blocks, always `raise X from err` not bare `raise X`

---

## Snapshot Truncation — Use Vision as Fallback

**Problem:** `browser_snapshot(full=True)` truncates at ~8000 chars. Pages with many repeated elements (e.g. GitHub trending, search results, long lists) will be cut off mid-content.

**Workaround:** Use `browser_vision(question="...")` to take a screenshot and extract content visually. Ask a specific extraction question (e.g. "List all repos with their names, star counts, and stars today"). Vision can see the full rendered page regardless of snapshot limits.

**When to use vision over snapshot:**
- Listing pages with 10+ repeated items (trending, search results, feeds)
- When snapshot output ends with `[... content truncated ...]`
- When you need tabular data that renders cleanly visually

**Tip:** Scroll down first if content is below the fold, then call `browser_vision`.

---

## General Advice

- **Re-snapshot after every click** — refs change whenever DOM updates
- **For dynamically revealed fields**: navigate fresh, get snapshot, act — don't let stale refs linger
- **For sites requiring CAPTCHA**: inform the user upfront that full automation isn't feasible
- **Residential proxies**: Browserbase stealth warning says "Running WITHOUT residential proxies" — upgrading the Browserbase plan may help with detection but won't solve iframe or CAPTCHA issues
- **Use `browser_vision(annotate=True)` for spatial debugging** when snapshot refs are ambiguous
