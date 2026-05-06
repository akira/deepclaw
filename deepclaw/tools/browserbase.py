"""Browserbase provider tools for DeepClaw.

Implements the LangChain Browserbase integration pattern:
- browserbase_search: cheap discovery-first search
- browserbase_fetch: fast stateless fetch for simple pages
- browserbase_rendered_extract: full rendered read-only extraction via Stagehand
- browserbase_interactive_task: multi-step interactive browser task via Stagehand

Credential errors are returned at call time so the plugin remains visible.
"""

from __future__ import annotations

import asyncio
import os
from importlib import import_module
from pathlib import Path
from typing import Any

from deepclaw.safety import check_url_safety_sync

_ENV_API_KEY = "BROWSERBASE_API_KEY"
_ENV_PROJECT_ID = "BROWSERBASE_PROJECT_ID"
_ENV_RENDER_MODEL = "STAGEHAND_MODEL"
_ENV_AGENT_MODEL = "STAGEHAND_AGENT_MODEL"
_DEFAULT_RENDER_MODEL = "google/gemini-3-flash-preview"
_DEFAULT_AGENT_MODEL = "anthropic/claude-sonnet-4-6"
_MAX_SEARCH_RESULTS = 10
_DEFAULT_MAX_FETCH_CHARS = 12000
_DEFAULT_INTERACTIVE_MAX_STEPS = 20


def _get_env(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if value:
        return value
    env_file = Path("~/.deepclaw/.env").expanduser()
    if env_file.is_file():
        for raw_line in env_file.read_text().splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            env_key, _, env_value = stripped.partition("=")
            if env_key.strip() == key:
                return env_value.strip().strip('"').strip("'")
    return ""


def available() -> bool:
    """Return True when the Browserbase SDK is installed.

    Credentials are checked at call time so the tools still register and can
    provide clear errors when not configured.
    """

    try:
        import_module("browserbase")
        import_module("bs4")
    except ImportError:
        return False
    return True


def _error(message: str) -> dict[str, Any]:
    return {"error": message}


def _require_api_key() -> str | None:
    api_key = _get_env(_ENV_API_KEY)
    if api_key:
        return api_key
    return None


def _require_project_id() -> str | None:
    project_id = _get_env(_ENV_PROJECT_ID)
    if project_id:
        return project_id
    return None


def _check_public_url(url: str) -> str | None:
    is_safe, reason = check_url_safety_sync(url)
    if is_safe:
        return None
    return reason


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict | list | str | int | float | bool) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def _is_oversized_fetch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    oversized_markers = (
        "response body exceeded",
        "maximum allowed size",
        "exceeded the maximum",
        "exceeds the maximum",
        "1mb limit",
        "too large",
    )
    return any(marker in message for marker in oversized_markers)


def browserbase_search(query: str, num_results: int = 5) -> dict[str, Any]:
    """Search the web with Browserbase.

    Args:
        query: Search query to run.
        num_results: Number of results to return, clamped to 1..10.

    Returns:
        Dict containing the query and compact search results, or an error.
    """

    api_key = _require_api_key()
    if not api_key:
        return _error(
            "Browserbase search unavailable: BROWSERBASE_API_KEY is not set. "
            "Add it to ~/.deepclaw/.env or the environment."
        )

    Browserbase = import_module("browserbase").Browserbase
    client = Browserbase(api_key=api_key)
    response = client.search.web(
        query=query, num_results=max(1, min(num_results, _MAX_SEARCH_RESULTS))
    )
    results = [
        {
            "title": getattr(result, "title", None),
            "url": getattr(result, "url", None),
            "snippet": getattr(result, "snippet", None),
        }
        for result in getattr(response, "results", [])
    ]
    return {"query": query, "results": results}


def browserbase_fetch(
    url: str, use_proxy: bool = False, max_chars: int = _DEFAULT_MAX_FETCH_CHARS
) -> dict[str, Any]:
    """Fetch page content without a full browser session.

    Best for static pages and quick reads when no JS-heavy interaction is needed.
    max_chars only truncates text after a successful Browserbase raw fetch; it
    does not avoid Browserbase's raw 1MB response limit. If Browserbase reports
    an oversized raw response, use browserbase_rendered_extract or browser
    navigation instead of retrying browserbase_fetch with a different max_chars.

    Args:
        url: Public URL to fetch.
        use_proxy: Whether Browserbase proxies should be enabled.
        max_chars: Maximum extracted text length after a successful raw fetch.

    Returns:
        Dict with URL, status_code, and cleaned text, or an error.
    """

    if reason := _check_public_url(url):
        return _error(f"Browserbase fetch blocked: {reason}")

    api_key = _require_api_key()
    if not api_key:
        return _error(
            "Browserbase fetch unavailable: BROWSERBASE_API_KEY is not set. "
            "Add it to ~/.deepclaw/.env or the environment."
        )

    Browserbase = import_module("browserbase").Browserbase
    BeautifulSoup = import_module("bs4").BeautifulSoup

    client = Browserbase(api_key=api_key)
    try:
        response = client.fetch_api.create(url=url, proxies=use_proxy)
    except Exception as exc:  # noqa: BLE001 - Browserbase SDK raises provider-specific exceptions.
        if _is_oversized_fetch_error(exc):
            return {
                "error": "Browserbase fetch failed: oversized_response",
                "error_type": "oversized_response",
                "retryable_with_same_tool": False,
                "recommended_tool": "browserbase_rendered_extract",
                "url": url,
            }
        raise
    html = str(getattr(response, "content", "") or "")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = (soup.body or soup).get_text("\n", strip=True)
    return {
        "url": url,
        "status_code": getattr(response, "status_code", None),
        "text": text[: max(1, max_chars)],
    }


def browserbase_rendered_extract(start_url: str, instruction: str) -> dict[str, Any]:
    """Extract rendered content from a JS-heavy page via Browserbase + Stagehand.

    Args:
        start_url: Public page URL to open.
        instruction: Read-only extraction instruction.

    Returns:
        Dict containing the extracted result, session metadata, or an error.
    """

    if reason := _check_public_url(start_url):
        return _error(f"Browserbase rendered extract blocked: {reason}")

    try:
        return asyncio.run(_browserbase_rendered_extract(start_url, instruction))
    except RuntimeError as exc:
        return _error(f"Browserbase rendered extract failed: {exc}")


async def _browserbase_rendered_extract(start_url: str, instruction: str) -> dict[str, Any]:
    stagehand = await _create_stagehand(
        model_name=_get_env(_ENV_RENDER_MODEL) or _DEFAULT_RENDER_MODEL
    )
    if isinstance(stagehand, dict):
        return stagehand

    try:
        assert stagehand.page is not None
        await stagehand.page.goto(start_url)
        result = await stagehand.page.extract(instruction)
        return {
            "url": start_url,
            "mode": "rendered_extract",
            "result": _serialize(result),
        }
    finally:
        await stagehand.close()


def browserbase_interactive_task(
    start_url: str, task: str, max_steps: int = _DEFAULT_INTERACTIVE_MAX_STEPS
) -> dict[str, Any]:
    """Execute a multi-step interactive browser task via Browserbase + Stagehand.

    Args:
        start_url: Public page URL to open.
        task: Interactive browser task to perform.
        max_steps: Maximum Stagehand agent steps.

    Returns:
        Dict containing the interactive result, session metadata, or an error.
    """

    if reason := _check_public_url(start_url):
        return _error(f"Browserbase interactive task blocked: {reason}")

    try:
        return asyncio.run(_browserbase_interactive_task(start_url, task, max_steps=max_steps))
    except RuntimeError as exc:
        return _error(f"Browserbase interactive task failed: {exc}")


async def _browserbase_interactive_task(
    start_url: str, task: str, *, max_steps: int
) -> dict[str, Any]:
    stagehand = await _create_stagehand(
        model_name=_get_env(_ENV_AGENT_MODEL) or _DEFAULT_AGENT_MODEL
    )
    if isinstance(stagehand, dict):
        return stagehand

    try:
        assert stagehand.page is not None
        await stagehand.page.goto(start_url)
        agent = stagehand.agent(
            model=_get_env(_ENV_AGENT_MODEL) or _DEFAULT_AGENT_MODEL,
            instructions="Execute the browser task precisely and stop when done.",
        )
        result = await agent.execute(task, max_steps=max(1, max_steps))
        return {
            "url": start_url,
            "mode": "interactive_task",
            "result": _serialize(result),
        }
    finally:
        await stagehand.close()


async def _create_stagehand(*, model_name: str):
    api_key = _require_api_key()
    if not api_key:
        return _error(
            "Browserbase rendered tools unavailable: BROWSERBASE_API_KEY is not set. "
            "Add it to ~/.deepclaw/.env or the environment."
        )

    project_id = _require_project_id()
    if not project_id:
        return _error(
            "Browserbase rendered tools unavailable: BROWSERBASE_PROJECT_ID is not set. "
            "Add it to ~/.deepclaw/.env or the environment."
        )

    try:
        Stagehand = import_module("stagehand").Stagehand
    except ImportError:
        return _error(
            "Browserbase rendered tools require the `stagehand` package. Install it with `uv add stagehand`."
        )

    stagehand = Stagehand(
        env="BROWSERBASE",
        api_key=api_key,
        project_id=project_id,
        model_name=model_name,
    )
    await stagehand.init()
    return stagehand


def get_tools() -> list:
    """Return Browserbase integration tools."""

    return [
        browserbase_search,
        browserbase_fetch,
        browserbase_rendered_extract,
        browserbase_interactive_task,
    ]
