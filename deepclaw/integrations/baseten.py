"""Baseten-specific model helpers for DeepClaw."""

from importlib import import_module

from langchain_core.messages import ToolMessage

from deepclaw.config import DeepClawConfig

BASETEN_PROVIDER = "baseten"


def load_chat_baseten_class():
    """Load LangChain's ChatBaseten adapter lazily."""
    try:
        module = import_module("langchain_baseten")
    except ImportError as exc:
        msg = (
            "Baseten models require langchain-baseten. Install it with `uv add langchain-baseten`."
        )
        raise RuntimeError(msg) from exc
    return module.ChatBaseten


def _sanitize_tool_message_content(content):
    """Convert OpenAI-incompatible structured tool output into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
                continue
            if block.get("type") == "image":
                mime_type = block.get("mime_type") or "image"
                text_parts.append(f"[image omitted: {mime_type}]")
        return "\n".join(part for part in text_parts if part)
    return str(content)


def _sanitize_tool_messages(messages):
    """Normalize ToolMessage content to plain strings for OpenAI-compatible providers."""
    sanitized_messages = []
    for message in messages:
        if isinstance(message, ToolMessage) and not isinstance(message.content, str):
            sanitized_messages.append(
                message.model_copy(
                    update={"content": _sanitize_tool_message_content(message.content)}
                )
            )
        else:
            sanitized_messages.append(message)
    return sanitized_messages


def resolve_baseten_model(config: DeepClawConfig):
    """Resolve a baseten:* model spec into a wrapped ChatBaseten model."""
    model_spec = (config.model or "").strip()
    provider, separator, model_name = model_spec.partition(":")
    if separator == "" or provider != BASETEN_PROVIDER:
        return model_spec
    if not model_name:
        msg = "Baseten model name cannot be empty"
        raise ValueError(msg)

    chat_cls = load_chat_baseten_class()

    class WrappedChatBaseten(chat_cls):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            messages = self._convert_input(input_).to_messages()
            sanitized_messages = _sanitize_tool_messages(messages)
            return super()._get_request_payload(sanitized_messages, stop=stop, **kwargs)

    generation = config.generation
    kwargs = {
        "streaming": True,
        "disable_streaming": False,
        "stream_usage": True,
    }
    if model_name.startswith(("https://", "http://")):
        kwargs["model_url"] = model_name
    else:
        kwargs["model"] = model_name

    model_kwargs = {}
    if generation.temperature is not None:
        kwargs["temperature"] = generation.temperature
    if generation.max_tokens is not None:
        kwargs["max_tokens"] = generation.max_tokens
    if generation.top_p is not None:
        kwargs["top_p"] = generation.top_p
    if generation.repetition_penalty is not None:
        model_kwargs["repetition_penalty"] = generation.repetition_penalty
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs

    return WrappedChatBaseten(**kwargs)
