"""Fireworks AI-specific model helpers for DeepClaw."""

from importlib import import_module

from deepclaw.config import DeepClawConfig

FIREWORKS_PROVIDER = "fireworks"


def load_chat_fireworks_class():
    """Load LangChain's ChatFireworks adapter lazily."""
    try:
        module = import_module("langchain_community.chat_models")
    except ImportError as exc:
        msg = (
            "Fireworks models require langchain-community. "
            "Install it with `uv add langchain-community`."
        )
        raise RuntimeError(msg) from exc
    return module.ChatFireworks


def resolve_fireworks_model(config: DeepClawConfig):
    """Resolve a fireworks:* model spec into a wrapped ChatFireworks model."""
    model_spec = (config.model or "").strip()
    provider, separator, model_name = model_spec.partition(":")
    provider = provider.strip()
    model_name = model_name.strip()
    if separator == "" or provider != FIREWORKS_PROVIDER:
        return model_spec
    if not model_name:
        msg = "Fireworks model name cannot be empty"
        raise ValueError(msg)

    chat_cls = load_chat_fireworks_class()
    generation = config.generation
    kwargs = {
        "model": model_name,
        "streaming": True,
    }
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
