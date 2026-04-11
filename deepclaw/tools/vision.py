"""Vision analysis for local screenshots and remote image URLs.

Provides:
  - vision_analyze: answer a natural-language question about an image

Works especially well with browser_screenshot() output from browser.py.
Uses OpenAI's vision-capable chat completions API via stdlib HTTP calls, so the
plugin loads without extra Python dependencies. Requires OPENAI_API_KEY at call
time.
"""

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from deepclaw.safety import check_url_safety_sync

_OPENAI_API_KEY_VAR = "OPENAI_API_KEY"
_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MODEL = "gpt-4.1-mini"
_SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}


def available() -> bool:
    """This plugin uses only the Python standard library, so it is always available."""
    return True


def _resolve_image_reference(image_path: str) -> tuple[str, dict[str, Any]]:
    if image_path.startswith(("http://", "https://")):
        is_safe, reason = check_url_safety_sync(image_path)
        if not is_safe:
            raise ValueError(
                f"Remote image URL is not allowed: {reason}. Only public, SSRF-safe image URLs are supported."
            )
        return image_path, {"source": "url", "image_path": image_path}

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _encoding = mimetypes.guess_type(path.name)
    if mime_type not in _SUPPORTED_MIME_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_MIME_TYPES))
        raise ValueError(
            f"Unsupported image type for {path.name!r}: {mime_type or 'unknown'}. Supported: {supported}"
        )

    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    metadata = {
        "source": "file",
        "image_path": str(path),
        "mime_type": mime_type,
        "size_bytes": len(image_bytes),
    }
    return data_url, metadata


def _extract_answer(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("Vision backend returned no choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"} and item.get("text"):
                texts.append(str(item["text"]).strip())
        if texts:
            return "\n\n".join(t for t in texts if t)

    raise ValueError("Vision backend returned an unexpected response shape")


def vision_analyze(image_path: str, question: str, model: str = _DEFAULT_MODEL) -> dict[str, Any]:
    """Analyze an image and answer a specific question about it.

    Accepts either a local file path (for example, a browser_screenshot() result)
    or a public http/https image URL.

    Args:
        image_path: Local path to an image file, or a public image URL.
        question: What to look for in the image.
        model: Vision-capable OpenAI model to use (default: gpt-4.1-mini).

    Returns:
        Dictionary containing the answer, plus metadata about the analyzed image.
    """
    api_key = os.environ.get(_OPENAI_API_KEY_VAR, "").strip()
    if not api_key:
        return {
            "error": (
                "Vision backend unavailable: OPENAI_API_KEY is not set. "
                "Set it in the environment or ~/.deepclaw/.env, then try again."
            ),
            "image_path": image_path,
            "question": question,
        }

    try:
        image_ref, metadata = _resolve_image_reference(image_path)
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e), "image_path": image_path, "question": question}
    except OSError as e:
        return {
            "error": f"Could not read image {image_path!r}: {e}",
            "image_path": image_path,
            "question": question,
        }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_ref}},
                ],
            }
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        _OPENAI_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            response_payload = json.loads(resp.read().decode("utf-8"))
        answer = _extract_answer(response_payload)
        return {
            "success": True,
            "answer": answer,
            "model": model,
            **metadata,
        }
    except error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        return {
            "error": f"Vision backend HTTP {e.code}: {details}",
            "image_path": image_path,
            "question": question,
            "model": model,
        }
    except error.URLError as e:
        return {
            "error": f"Vision backend request failed: {e.reason}",
            "image_path": image_path,
            "question": question,
            "model": model,
        }
    except (ValueError, json.JSONDecodeError) as e:
        return {
            "error": f"Vision backend returned an invalid response: {e}",
            "image_path": image_path,
            "question": question,
            "model": model,
        }


def get_tools() -> list:
    """Return the tool callables for this plugin."""
    return [vision_analyze]
