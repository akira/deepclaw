"""Baseten-specific model helpers for DeepClaw."""

from importlib import import_module

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
    generation = config.generation
    kwargs = {
        "streaming": False,
        "disable_streaming": True,
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

    return chat_cls(**kwargs)
