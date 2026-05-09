"""Text-to-speech (TTS) tool for DeepClaw.

Provides:
  - text_to_speech: convert text to an audio file using OpenAI's TTS API

Uses OpenAI's text-to-speech API via stdlib HTTP calls, so the plugin loads
without extra Python dependencies. Requires OPENAI_API_KEY at call time.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request

_OPENAI_API_KEY_VAR = "OPENAI_API_KEY"
_OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
_DEFAULT_MODEL = "gpt-4o-mini-tts"
_DEFAULT_VOICE = "alloy"
_SUPPORTED_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
}


def available() -> bool:
    """This plugin uses only the Python standard library, so it is always available."""
    return True


def _output_dir() -> Path:
    return Path.home() / ".deepclaw" / "tts"


def text_to_speech(
    text: str,
    voice: str = _DEFAULT_VOICE,
    model: str = _DEFAULT_MODEL,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Convert text to speech and save as an audio file.

    Uses OpenAI's text-to-speech API. The resulting audio file is saved locally
    and its path is returned.

    When the user wants the audio played or sent (e.g. via Telegram), include
    the line `MEDIA:<audio_path>` in your response so the gateway can deliver
    the audio file.

    For Telegram voice bubbles (round playable messages), also include
    `[[audio_as_voice]]` on a separate line before the MEDIA: line.

    Args:
        text: The text to convert to speech. Maximum length is 4096 characters.
        voice: The voice to use. Options: alloy, ash, ballad, coral, echo,
            fable, onyx, nova, sage, shimmer (default: alloy).
        model: The TTS model to use (default: gpt-4o-mini-tts).
        output_path: Optional full path for the output audio file. If not
            provided, a file is created under ~/.deepclaw/tts/.

    Returns:
        Dictionary containing the path to the saved audio file, voice used,
        model used, and text length.
    """
    api_key = os.environ.get(_OPENAI_API_KEY_VAR, "").strip()
    if not api_key:
        return {
            "error": (
                "TTS backend unavailable: OPENAI_API_KEY is not set. "
                "Set it in the environment or ~/.deepclaw/.env, then try again."
            ),
            "text": text,
        }

    if not text:
        return {"error": "No text provided for TTS conversion.", "text": text}

    if len(text) > 4096:
        return {
            "error": (
                f"Text too long for TTS: {len(text)} characters (max 4096). "
                "Please shorten the text."
            ),
            "text": text,
        }

    voice = voice.lower().strip()
    if voice not in _SUPPORTED_VOICES:
        supported = ", ".join(sorted(_SUPPORTED_VOICES))
        return {
            "error": (f"Unsupported voice '{voice}'. Supported voices: {supported}"),
            "voice": voice,
            "text": text,
        }

    if output_path:
        out_path = Path(output_path).expanduser()
    else:
        out_dir = _output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tts_{uuid.uuid4().hex[:12]}.mp3"
        out_path = out_dir / filename

    payload = {
        "model": model,
        "voice": voice,
        "input": text,
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        _OPENAI_TTS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            audio_bytes = resp.read()
        out_path.write_bytes(audio_bytes)
        return {
            "success": True,
            "audio_path": str(out_path),
            "voice": voice,
            "model": model,
            "text_length": len(text),
            "audio_bytes": len(audio_bytes),
        }
    except error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        return {
            "error": f"TTS backend HTTP {e.code}: {details}",
            "text": text,
            "voice": voice,
            "model": model,
        }
    except error.URLError as e:
        return {
            "error": f"TTS backend request failed: {e.reason}",
            "text": text,
            "voice": voice,
            "model": model,
        }
    except OSError as e:
        return {
            "error": f"Failed to write audio file to {out_path}: {e}",
            "text": text,
            "voice": voice,
            "model": model,
        }


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [text_to_speech]
