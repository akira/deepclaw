"""Headroom model wrapping helpers for DeepClaw."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from deepclaw.config import HeadroomConfig

logger = logging.getLogger(__name__)


def wrap_model_with_headroom(
    model: str | BaseChatModel,
    headroom_config: HeadroomConfig,
) -> str | BaseChatModel:
    """Wrap a chat model with Headroom's LangChain model wrapper when enabled."""
    if not headroom_config.enabled:
        return model

    try:
        from deepagents._models import resolve_model
        from headroom.integrations import HeadroomChatModel
    except ImportError as exc:  # pragma: no cover - exercised via runtime only
        msg = (
            "Headroom compression is enabled but the Headroom LangChain integration "
            "is not installed. Add `headroom-ai[langchain]` to the environment."
        )
        raise RuntimeError(msg) from exc

    resolved_model = resolve_model(model)
    wrapped_model = HeadroomChatModel(resolved_model)
    logger.info("Headroom prompt compression enabled for %s", resolved_model.__class__.__name__)
    return wrapped_model
