"""Tests for the TTS tool plugin."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepclaw.tools.tts import available, get_tools, text_to_speech


class TestTTSAvailability:
    def test_available_returns_true(self):
        assert available() is True

    def test_get_tools_returns_text_to_speech(self):
        tools = get_tools()
        assert len(tools) == 1
        assert tools[0].__name__ == "text_to_speech"


class TestTTSValidation:
    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            result = text_to_speech("Hello world")
        assert "error" in result
        assert "OPENAI_API_KEY" in result["error"]

    def test_empty_text(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            result = text_to_speech("")
        assert "error" in result
        assert "No text" in result["error"]

    def test_long_text(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            result = text_to_speech("x" * 5000)
        assert "error" in result
        assert "too long" in result["error"]

    def test_invalid_voice(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            result = text_to_speech("Hello", voice="not-a-voice")
        assert "error" in result
        assert "Unsupported voice" in result["error"]

    def test_voice_case_insensitive(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            result = text_to_speech("Hello", voice="ALLOY")
        # Should pass validation, fail on HTTP
        assert "error" in result
        assert "HTTP" in result["error"] or "request failed" in result["error"]


class TestTTSSuccessPath:
    def test_successful_tts_write(self, tmp_path):
        fake_audio = b"fake mp3 bytes"
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("deepclaw.tools.tts.request.urlopen") as mock_urlopen,
        ):
            mock_resp = MagicMock()
            mock_resp.read.return_value = fake_audio
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            result = text_to_speech("Hello world")

        assert result["success"] is True
        assert result["text_length"] == 11
        assert result["audio_bytes"] == len(fake_audio)
        assert result["voice"] == "alloy"
        assert result["model"] == "gpt-4o-mini-tts"
        assert Path(result["audio_path"]).exists()
        assert Path(result["audio_path"]).read_bytes() == fake_audio

    def test_custom_output_path(self, tmp_path):
        custom_path = tmp_path / "custom.mp3"
        fake_audio = b"fake mp3 bytes"
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("deepclaw.tools.tts.request.urlopen") as mock_urlopen,
        ):
            mock_resp = MagicMock()
            mock_resp.read.return_value = fake_audio
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            result = text_to_speech("Hello", output_path=str(custom_path))

        assert result["success"] is True
        assert result["audio_path"] == str(custom_path)
        assert custom_path.exists()

    def test_custom_voice_and_model(self, tmp_path):
        fake_audio = b"fake mp3 bytes"
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("deepclaw.tools.tts.request.urlopen") as mock_urlopen,
        ):
            mock_resp = MagicMock()
            mock_resp.read.return_value = fake_audio
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            result = text_to_speech("Hello", voice="nova", model="tts-1")

        assert result["success"] is True
        assert result["voice"] == "nova"
        assert result["model"] == "tts-1"


class TestTTSHTTPError:
    def test_http_401(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("deepclaw.tools.tts.request.urlopen") as mock_urlopen,
        ):
            from urllib import error

            mock_urlopen.side_effect = error.HTTPError(
                url="https://api.openai.com/v1/audio/speech",
                code=401,
                msg="Unauthorized",
                hdrs={},
                fp=None,
            )
            result = text_to_speech("Hello")

        assert "error" in result
        assert "HTTP 401" in result["error"]

    def test_url_error(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            patch("deepclaw.tools.tts.request.urlopen") as mock_urlopen,
        ):
            from urllib import error

            mock_urlopen.side_effect = error.URLError("Connection refused")
            result = text_to_speech("Hello")

        assert "error" in result
        assert "request failed" in result["error"]
