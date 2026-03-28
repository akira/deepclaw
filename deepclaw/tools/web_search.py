"""Web search and URL extraction via Tavily.

Available when TAVILY_API_KEY is set. Provides:
  - web_search: search the web for current information
  - web_extract: fetch and extract clean content from URLs
"""

import os
from typing import Any, Literal

_TAVILY_API_KEY_VAR = "TAVILY_API_KEY"

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


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict[str, Any]:
    """Search the web for current information.

    Args:
        query: The search query (be specific and detailed).
        max_results: Number of results to return (default: 5).
        topic: Search topic — "general", "news", or "finance".
        include_raw_content: Include full page content (uses more tokens).

    Returns:
        Dictionary with results list, each containing title, url, content, and score.
    """
    try:
        from tavily import (
            BadRequestError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            UsageLimitExceededError,
        )
        from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError
    except ImportError as exc:
        return {"error": f"Required package not installed: {exc.name}", "query": query}

    try:
        return _get_client().search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except (
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        UsageLimitExceededError,
    ) as e:
        return {"error": f"Web search error: {e!s}", "query": query}
    except Exception as e:
        return {"error": f"Web search error: {e!s}", "query": query}


def web_extract(
    urls: list[str],
) -> dict[str, Any]:
    """Extract clean content from one or more URLs.

    Args:
        urls: List of URLs to extract content from.

    Returns:
        Dictionary with extracted results, each containing url and raw_content.
    """
    try:
        from tavily import (
            BadRequestError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            UsageLimitExceededError,
        )
        from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError
    except ImportError as exc:
        return {"error": f"Required package not installed: {exc.name}", "urls": urls}

    try:
        return _get_client().extract(urls=urls)
    except (
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        UsageLimitExceededError,
    ) as e:
        return {"error": f"Web extract error: {e!s}", "urls": urls}
    except Exception as e:
        return {"error": f"Web extract error: {e!s}", "urls": urls}


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [web_search, web_extract]
