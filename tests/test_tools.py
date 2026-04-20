"""Tests for the tool plugin system plus web_search and vision plugins."""

import json
import os
import threading
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from deepclaw.tools import discover_tools

# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


class TestDiscoverTools:
    def test_returns_list(self):
        result = discover_tools()
        assert isinstance(result, list)

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_loads_tavily_when_available(self):
        try:
            import tavily  # noqa: F401
        except ImportError:
            pytest.skip("tavily-python not installed")

        tools = discover_tools()
        tool_names = [getattr(t, "__name__", "") for t in tools]
        assert "web_search" in tool_names
        assert "web_extract" in tool_names
        assert "vision_analyze" in tool_names
        assert "skills_list" in tool_names
        assert "skills_search_remote" in tool_names
        assert "skill_view" in tool_names
        assert "skill_create" in tool_names
        assert "skill_update" in tool_names
        assert "skill_install" in tool_names

    @patch.dict(os.environ, {}, clear=False)
    def test_skips_tavily_without_api_key(self):
        env = os.environ.copy()
        env.pop("TAVILY_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            tools = discover_tools()
            tool_names = [getattr(t, "__name__", "") for t in tools]
            assert "web_search" not in tool_names

    def test_handles_broken_plugin_gracefully(self):
        # discover_tools should not raise even if a plugin is broken
        tools = discover_tools()
        assert isinstance(tools, list)


# ---------------------------------------------------------------------------
# Web search plugin — available()
# ---------------------------------------------------------------------------


class TestWebSearchAvailable:
    def test_available_without_key(self):
        from deepclaw.tools import web_search as ws_mod

        with patch.dict(os.environ, {}, clear=True):
            assert ws_mod.available() is False

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_available_with_key_and_package(self):
        try:
            import tavily  # noqa: F401
        except ImportError:
            pytest.skip("tavily-python not installed")

        from deepclaw.tools import web_search as ws_mod

        assert ws_mod.available() is True

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    @patch.dict("sys.modules", {"tavily": None})
    def test_available_with_key_but_no_package(self):
        from deepclaw.tools import web_search as ws_mod

        assert ws_mod.available() is False


# ---------------------------------------------------------------------------
# Web search plugin — get_tools()
# ---------------------------------------------------------------------------


class TestWebSearchGetTools:
    def test_returns_two_tools(self):
        from deepclaw.tools import web_search as ws_mod

        tools = ws_mod.get_tools()
        assert len(tools) == 2
        names = [t.__name__ for t in tools]
        assert "web_search" in names
        assert "web_extract" in names


# ---------------------------------------------------------------------------
# Web search plugin — web_search() error handling
# ---------------------------------------------------------------------------


class TestWebSearchFunction:
    def test_returns_error_without_tavily_installed(self):
        from deepclaw.tools.web_search import web_search

        with (
            patch.dict("sys.modules", {"tavily": None}),
            patch("builtins.__import__", side_effect=ImportError("tavily")),
        ):
            result = web_search("test query")
            assert "error" in result

    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_returns_results_on_success(self):
        try:
            import tavily  # noqa: F401
        except ImportError:
            pytest.skip("tavily-python not installed")

        from deepclaw.tools import web_search as ws_mod

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"title": "Test", "url": "https://example.com", "content": "test", "score": 0.9}
            ],
            "query": "test query",
        }

        ws_mod._client = mock_client
        result = ws_mod.web_search("test query")
        assert "results" in result
        mock_client.search.assert_called_once()

        ws_mod._client = None


# ---------------------------------------------------------------------------
# Web search plugin — web_extract() error handling
# ---------------------------------------------------------------------------


class TestWebExtractFunction:
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"})
    def test_returns_results_on_success(self):
        try:
            import tavily  # noqa: F401
        except ImportError:
            pytest.skip("tavily-python not installed")

        from deepclaw.tools import web_search as ws_mod

        mock_client = MagicMock()
        mock_client.extract.return_value = {
            "results": [{"url": "https://example.com", "raw_content": "page content"}],
        }

        ws_mod._client = mock_client
        result = ws_mod.web_extract(["https://example.com"])
        assert "results" in result
        mock_client.extract.assert_called_once()

        ws_mod._client = None


# ---------------------------------------------------------------------------
# Browser plugin session persistence
# ---------------------------------------------------------------------------


class TestBrowserPluginSessionPersistence:
    def test_session_survives_cross_thread_tool_calls(self):
        from deepclaw.tools import browser as browser_mod

        browser_mod._set_session({"page": "sentinel", "session_id": "local"})
        result: dict[str, object] = {}

        def _read_session() -> None:
            result.update(browser_mod._get_session())

        thread = threading.Thread(target=_read_session)
        thread.start()
        thread.join()

        assert result["page"] == "sentinel"
        assert result["session_id"] == "local"

        browser_mod._set_session({})

    def test_browser_thread_startup_is_synchronized(self):
        from deepclaw.tools import browser as browser_mod

        starts: list[str] = []

        class FakeThread:
            def __init__(self, target=None, name=None, daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon
                self._alive = False
                starts.append(name or "thread")

            def is_alive(self):
                return self._alive

            def start(self):
                self._alive = True

            def join(self):
                return None

        real_thread = threading.Thread
        browser_mod._BROWSER_THREAD = None
        with patch("deepclaw.tools.browser.threading.Thread", FakeThread):
            threads = [real_thread(target=browser_mod._ensure_browser_thread) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert len(starts) == 1


# ---------------------------------------------------------------------------
# Vision plugin
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestVisionPlugin:
    def test_available(self):
        from deepclaw.tools import vision as vision_mod

        assert vision_mod.available() is True

    def test_get_tools(self):
        from deepclaw.tools import vision as vision_mod

        tools = vision_mod.get_tools()
        assert len(tools) == 1
        assert tools[0].__name__ == "vision_analyze"

    def test_returns_error_without_openai_key(self):
        from deepclaw.tools.vision import vision_analyze

        with patch.dict(os.environ, {}, clear=True):
            result = vision_analyze("/tmp/example.png", "What is in this image?")

        assert "error" in result
        assert "OPENAI_API_KEY" in result["error"]

    def test_returns_error_for_missing_file(self):
        from deepclaw.tools.vision import vision_analyze

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            result = vision_analyze("/tmp/does-not-exist.png", "What is in this image?")

        assert "error" in result
        assert "Image not found" in result["error"]

    def test_local_file_success(self, tmp_path):
        from deepclaw.tools.vision import vision_analyze

        image_path = tmp_path / "sample.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        def fake_urlopen(req, timeout=60):
            assert timeout == 60
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["model"] == "gpt-4.1-mini"
            assert payload["messages"][0]["content"][0]["text"] == "Describe it"
            image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
            assert image_url.startswith("data:image/png;base64,")
            return _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "A tiny test image.",
                            }
                        }
                    ]
                }
            )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("deepclaw.tools.vision.request.urlopen", side_effect=fake_urlopen),
        ):
            result = vision_analyze(str(image_path), "Describe it")

        assert result["success"] is True
        assert result["answer"] == "A tiny test image."
        assert result["source"] == "file"
        assert result["image_path"] == str(image_path)
        assert result["mime_type"] == "image/png"
        assert result["size_bytes"] == 8

    def test_remote_url_success(self):
        from deepclaw.tools.vision import vision_analyze

        def fake_urlopen(req, timeout=60):
            assert timeout == 60
            payload = json.loads(req.data.decode("utf-8"))
            assert (
                payload["messages"][0]["content"][1]["image_url"]["url"]
                == "https://example.com/cat.png"
            )
            return _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "A cat looking at the camera."}
                                ],
                            }
                        }
                    ]
                }
            )

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("deepclaw.tools.vision.check_url_safety_sync", return_value=(True, "")),
            patch("deepclaw.tools.vision.request.urlopen", side_effect=fake_urlopen),
        ):
            result = vision_analyze("https://example.com/cat.png", "What is here?")

        assert result["success"] is True
        assert result["answer"] == "A cat looking at the camera."
        assert result["source"] == "url"

    def test_remote_url_rejected_when_not_ssrf_safe(self):
        from deepclaw.tools.vision import vision_analyze

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch(
                "deepclaw.tools.vision.check_url_safety_sync",
                return_value=(False, "IP 127.0.0.1 is in a blocked network range"),
            ),
        ):
            result = vision_analyze("http://127.0.0.1/cat.png", "What is here?")

        assert "error" in result
        assert "Remote image URL is not allowed" in result["error"]
        assert "SSRF-safe" in result["error"]

    def test_http_error_surfaces_details(self, tmp_path):
        from deepclaw.tools.vision import vision_analyze

        image_path = tmp_path / "sample.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        http_error = HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )
        http_error.read = lambda: b'{"error":{"message":"bad key"}}'

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch("deepclaw.tools.vision.request.urlopen", side_effect=http_error),
        ):
            result = vision_analyze(str(image_path), "Describe it")

        assert "error" in result
        assert "HTTP 401" in result["error"]
        assert "bad key" in result["error"]
