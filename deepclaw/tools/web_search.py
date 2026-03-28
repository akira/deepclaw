"""Web search and URL extraction via Tavily.

Available when TAVILY_API_KEY is set. Provides:
  - web_search: search the web for current information
  - web_extract: fetch and extract clean content from URLs
"""

import os
from collections.abc import Callable
from typing import Any, Literal

_TAVILY_API_KEY_VAR = "TAVILY_API_KEY"
_MAX_SEARCH_RESULTS = 20
_MAX_EXTRACT_URLS = 10

_client = None


def available() -> bool:
    """True if tavily-python is installed and TAVILY_API_KEY is set."""
    if not os.environ.get(_TAVILY_API_KEY_VAR):
        return False
    try:
        import tavily  # noqa: F401

        return True
    except ImportError:
        return False


def _get_client():
    global _client  # noqa: PLW0603
    if _client is None:
        from tavily import TavilyClient

        _client = TavilyClient(api_key=os.environ[_TAVILY_API_KEY_VAR])
    return _client


def _tavily_call(fn: Callable, error_prefix: str, **context) -> dict[str, Any]:
    """Execute a Tavily API call with standard error handling."""
    try:
        from tavily import (
            BadRequestError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            UsageLimitExceededError,
        )
        from tavily.errors import ForbiddenError
        from tavily.errors import TimeoutError as TavilyTimeoutError
    except ImportError as exc:
        return {"error": f"Required package not installed: {exc.name}", **context}

    try:
        return fn()
    except (
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        UsageLimitExceededError,
    ) as e:
        return {"error": f"{error_prefix}: {e!s}", **context}
    except Exception as e:
        return {"error": f"{error_prefix}: {e!s}", **context}


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict[str, Any]:
    """Search the web for current information.

    Args:
        query: The search query (be specific and detailed).
        max_results: Number of results to return (default: 5, max: 20).
        topic: Search topic — "general", "news", or "finance".
        include_raw_content: Include full page content (uses more tokens).

    Returns:
        Dictionary with results list, each containing title, url, content, and score.
    """
    max_results = min(max_results, _MAX_SEARCH_RESULTS)

    return _tavily_call(
        lambda: _get_client().search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        ),
        error_prefix="Web search error",
        query=query,
    )


def web_extract(
    urls: list[str],
) -> dict[str, Any]:
    """Extract clean content from one or more URLs.

    Args:
        urls: List of URLs to extract content from (max: 10).

    Returns:
        Dictionary with extracted results, each containing url and raw_content.
    """
    urls = urls[:_MAX_EXTRACT_URLS]

    return _tavily_call(
        lambda: _get_client().extract(urls=urls),
        error_prefix="Web extract error",
        urls=urls,
    )


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [web_search, web_extract]
