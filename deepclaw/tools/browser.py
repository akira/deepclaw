"""Browser automation via local Playwright (Chromium).

Available when playwright is installed.

Uses Browserbase cloud if BROWSERBASE_API_KEY + BROWSERBASE_PROJECT_ID are set,
otherwise falls back to local headless Chromium (default, no keys needed).

Provides:
  - browser_navigate: load a URL
  - browser_snapshot: get accessibility tree with ref IDs for interaction
  - browser_click: click an element by ref
  - browser_type: type text into a field by ref
  - browser_press: press a keyboard key
  - browser_scroll: scroll the page
  - browser_screenshot: save a screenshot locally
  - browser_close: close the current session
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_API_KEY = "BROWSERBASE_API_KEY"
_ENV_PROJECT_ID = "BROWSERBASE_PROJECT_ID"
_SCREENSHOTS_DIR = Path("~/.deepclaw/browser_screenshots").expanduser()

_local = threading.local()

# JS to extract interactive + content elements with stable ref IDs
_SNAPSHOT_JS = """
() => {
    const INTERACTIVE = ['a', 'button', 'input', 'select', 'textarea', 'label'];
    const CONTENT = ['h1','h2','h3','h4','h5','h6','p','li','td','th','span','div'];
    let counter = 0;
    let nodes = [];

    function getRole(el) {
        const tag = el.tagName.toLowerCase();
        const type = el.getAttribute('type') || '';
        if (tag === 'a') return 'link';
        if (tag === 'button') return 'button';
        if (tag === 'input') {
            if (type === 'checkbox') return 'checkbox';
            if (type === 'radio') return 'radio';
            if (type === 'submit' || type === 'button') return 'button';
            return 'textbox';
        }
        if (tag === 'select') return 'combobox';
        if (tag === 'textarea') return 'textbox';
        if (tag === 'label') return 'label';
        return tag;
    }

    function getText(el) {
        // Get visible text, truncated
        const text = (el.innerText || el.textContent || el.value || el.getAttribute('placeholder') || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim().replace(/\\s+/g, ' ');
        return text.slice(0, 80);
    }

    function isVisible(el) {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
        return true;
    }

    function getDepth(el) {
        let depth = 0;
        let p = el.parentElement;
        while (p && p !== document.body) { depth++; p = p.parentElement; }
        return Math.min(depth, 8);
    }

    // Collect interactive elements
    document.querySelectorAll(INTERACTIVE.join(',')).forEach(el => {
        if (!isVisible(el)) return;
        counter++;
        el._deepclaw_ref = counter;
        const role = getRole(el);
        const text = getText(el);
        const href = el.getAttribute('href') || '';
        nodes.push({
            ref: counter,
            role,
            text,
            href: href.slice(0, 100),
            depth: getDepth(el),
            interactive: true,
            checked: el.checked !== undefined ? el.checked : null,
        });
    });

    // Collect headings for context
    document.querySelectorAll('h1,h2,h3,h4').forEach(el => {
        if (!isVisible(el)) return;
        if (el._deepclaw_ref) return; // already captured
        counter++;
        el._deepclaw_ref = counter;
        nodes.push({
            ref: counter,
            role: el.tagName.toLowerCase(),
            text: getText(el),
            href: '',
            depth: getDepth(el),
            interactive: false,
        });
    });

    return {nodes, title: document.title, url: window.location.href};
}
"""

# JS to find element by ref and return a CSS selector
_FIND_REF_JS = """
(refNum) => {
    function find(el) {
        if (el._deepclaw_ref === refNum) return el;
        for (let c of el.children) {
            const r = find(c);
            if (r) return r;
        }
        return null;
    }
    const el = find(document.body);
    if (!el) return null;
    // Build a unique selector
    if (el.id) return '#' + CSS.escape(el.id);
    // Use nth-of-type chain
    let parts = [];
    let cur = el;
    while (cur && cur !== document.body) {
        let tag = cur.tagName.toLowerCase();
        let idx = 1;
        let sib = cur.previousElementSibling;
        while (sib) { if (sib.tagName === cur.tagName) idx++; sib = sib.previousElementSibling; }
        parts.unshift(tag + ':nth-of-type(' + idx + ')');
        cur = cur.parentElement;
    }
    return 'body > ' + parts.join(' > ');
}
"""


def _get_env(key: str) -> str:
    val = os.environ.get(key, "")
    if val:
        return val
    env_file = Path("~/.deepclaw/.env").expanduser()
    if env_file.is_file():
        for raw_line in env_file.read_text().splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            k, _, v = stripped.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    return ""


def available() -> bool:
    """True if playwright is installed."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except ImportError:
        return False


def _get_session() -> dict:
    return getattr(_local, "session", {})


def _set_session(data: dict) -> None:
    _local.session = data


def _close_session() -> None:
    session = _get_session()
    try:
        if session.get("page"):
            session["page"].close()
        if session.get("browser"):
            session["browser"].close()
        if session.get("playwright"):
            session["playwright"].stop()
    except Exception:
        pass
    try:
        if session.get("bb") and session.get("session_id"):
            session["bb"].sessions.delete(session["session_id"])
    except Exception:
        pass
    _set_session({})


def _ensure_session(url: str | None = None) -> dict | None:
    session = _get_session()
    if session.get("page"):
        return session
    if not url:
        return None

    try:
        from playwright.sync_api import sync_playwright

        api_key = _get_env(_ENV_API_KEY)
        project_id = _get_env(_ENV_PROJECT_ID)
        pw = sync_playwright().start()

        if api_key and project_id:
            from browserbase import Browserbase
            bb = Browserbase(api_key=api_key)
            bb_session = bb.sessions.create(project_id=project_id)
            browser = pw.chromium.connect_over_cdp(bb_session.connect_url)
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            data_extra = {"bb": bb, "session_id": bb_session.id}
        else:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            data_extra = {"session_id": "local"}

        page = context.pages[0] if context.pages else context.new_page()
        data = {"playwright": pw, "browser": browser, "context": context, "page": page, **data_extra}
        _set_session(data)
        return data
    except Exception as e:
        logger.error("Failed to create browser session: %s", e)
        return None


def _build_snapshot(page) -> str:
    """Run JS snapshot and format as readable text with ref IDs."""
    try:
        result = page.evaluate(_SNAPSHOT_JS)
        nodes = result.get("nodes", [])
        lines = []
        for n in nodes:
            indent = "  " * n["depth"]
            ref = f"[e{n['ref']}]"
            role = n["role"]
            text = n["text"]
            extra = ""
            if n.get("href"):
                extra = f" -> {n['href']}"
            if n.get("checked") is True:
                extra += " [checked]"
            elif n.get("checked") is False:
                extra += " [unchecked]"
            interactive_mark = "" if n["interactive"] else " (info)"
            lines.append(f"{indent}{ref} {role}{interactive_mark}: {text}{extra}")
        return "\n".join(lines) if lines else "(no elements found)"
    except Exception as e:
        return f"(snapshot error: {e})"


def _locate_ref(page, ref: str):
    """Find an element by ref string and return a Playwright Locator."""
    ref_clean = ref.lstrip("@e").lstrip("e")
    try:
        ref_num = int(ref_clean)
    except ValueError as err:
        raise ValueError(f"Invalid ref: {ref!r} — use format like 'e5' or '@e5'") from err

    selector = page.evaluate(_FIND_REF_JS, ref_num)
    if not selector:
        raise ValueError(f"Ref {ref} not found. Call browser_snapshot() first to assign refs.")
    return page.locator(selector)


# ── Tool functions ──────────────────────────────────────────────────────────


def browser_navigate(url: str) -> dict[str, Any]:
    """Navigate to a URL in a browser session.

    Starts a new session automatically if one isn't active.
    Must be called before any other browser tool.

    Args:
        url: The URL to navigate to (e.g., "https://example.com").

    Returns:
        Dict with url, title, and status.
    """
    session = _get_session()
    if not session.get("page"):
        session = _ensure_session(url)
        if not session:
            return {"error": "Failed to start browser. Make sure playwright is installed: uv add playwright && playwright install chromium"}

    page = session["page"]
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(1)
        return {"success": True, "url": page.url, "title": page.title(), "status": response.status if response else None}
    except Exception as e:
        return {"error": str(e), "url": url}


def browser_snapshot(offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """Get the current page as a list of elements with ref IDs.

    Refs like [e5] can be passed to browser_click and browser_type.
    Use offset/limit to paginate through large pages.

    Args:
        offset: Skip first N elements (default: 0).
        limit: Max elements to return (default: 100, max: 200).

    Returns:
        Dict with snapshot text, total element count, url, and title.
    """
    session = _get_session()
    if not session.get("page"):
        return {"error": "No active browser session. Call browser_navigate first."}

    page = session["page"]
    try:
        result = page.evaluate(_SNAPSHOT_JS)
        nodes = result.get("nodes", [])
        total = len(nodes)
        limit = min(limit, 200)
        page_nodes = nodes[offset:offset + limit]

        lines = []
        for n in page_nodes:
            indent = "  " * n["depth"]
            ref = f"[e{n['ref']}]"
            role = n["role"]
            text = n["text"]
            extra = ""
            if n.get("href"):
                extra = f" -> {n['href']}"
            if n.get("checked") is True:
                extra += " [checked]"
            elif n.get("checked") is False:
                extra += " [unchecked]"
            info = "" if n["interactive"] else " (info)"
            lines.append(f"{indent}{ref} {role}{info}: {text}{extra}")

        snapshot_text = "\n".join(lines) if lines else "(no elements found)"
        result_dict = {
            "url": page.url,
            "title": page.title(),
            "snapshot": snapshot_text,
            "total_elements": total,
            "showing": f"{offset}–{offset + len(page_nodes)} of {total}",
        }
        if offset + limit < total:
            result_dict["hint"] = f"More elements available. Call browser_snapshot(offset={offset + limit}) to see next page."
        return result_dict
    except Exception as e:
        return {"error": str(e)}


def browser_click(ref: str) -> dict[str, Any]:
    """Click an element by its ref ID from browser_snapshot.

    Args:
        ref: Element reference like "e5" or "@e5".

    Returns:
        Dict with success status and updated url/title.
    """
    session = _get_session()
    if not session.get("page"):
        return {"error": "No active browser session. Call browser_navigate first."}

    page = session["page"]
    try:
        locator = _locate_ref(page, ref)
        locator.first.click(timeout=10000)
        time.sleep(0.8)
        return {"success": True, "url": page.url, "title": page.title()}
    except Exception as e:
        return {"error": str(e), "ref": ref}


def browser_type(ref: str, text: str) -> dict[str, Any]:
    """Type text into an input field by its ref ID. Clears the field first.

    Args:
        ref: Element reference from browser_snapshot (e.g., "e3").
        text: The text to type.

    Returns:
        Dict with success status.
    """
    session = _get_session()
    if not session.get("page"):
        return {"error": "No active browser session. Call browser_navigate first."}

    page = session["page"]
    try:
        locator = _locate_ref(page, ref)
        locator.first.fill(text, timeout=10000)
        return {"success": True, "ref": ref, "text": text}
    except Exception as e:
        return {"error": str(e), "ref": ref}


def browser_press(key: str) -> dict[str, Any]:
    """Press a keyboard key on the currently focused element.

    Args:
        key: Key name (e.g., "Enter", "Tab", "Escape", "ArrowDown").

    Returns:
        Dict with success status.
    """
    session = _get_session()
    if not session.get("page"):
        return {"error": "No active browser session. Call browser_navigate first."}

    page = session["page"]
    try:
        page.keyboard.press(key)
        time.sleep(0.3)
        return {"success": True, "key": key}
    except Exception as e:
        return {"error": str(e), "key": key}


def browser_scroll(direction: str = "down") -> dict[str, Any]:
    """Scroll the page up or down.

    Args:
        direction: "up" or "down" (default: "down").

    Returns:
        Dict with success status.
    """
    session = _get_session()
    if not session.get("page"):
        return {"error": "No active browser session. Call browser_navigate first."}

    page = session["page"]
    try:
        delta = 600 if direction == "down" else -600
        page.mouse.wheel(0, delta)
        time.sleep(0.4)
        return {"success": True, "direction": direction}
    except Exception as e:
        return {"error": str(e)}


def browser_screenshot() -> dict[str, Any]:
    """Take a screenshot of the current page and save it locally.

    Returns:
        Dict with the local file path to the screenshot.
    """
    session = _get_session()
    if not session.get("page"):
        return {"error": "No active browser session. Call browser_navigate first."}

    page = session["page"]
    try:
        _SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        path = _SCREENSHOTS_DIR / f"screenshot_{ts}.png"
        page.screenshot(path=str(path), full_page=False)
        return {"success": True, "path": str(path), "url": page.url}
    except Exception as e:
        return {"error": str(e)}


def browser_close() -> dict[str, Any]:
    """Close the current browser session and release resources.

    Call this when done with browser tasks.

    Returns:
        Dict confirming the session was closed.
    """
    session = _get_session()
    if not session.get("page"):
        return {"status": "no active session"}
    session_id = session.get("session_id", "unknown")
    _close_session()
    return {"status": "closed", "session_id": session_id}


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [
        browser_navigate,
        browser_snapshot,
        browser_click,
        browser_type,
        browser_press,
        browser_scroll,
        browser_screenshot,
        browser_close,
    ]
