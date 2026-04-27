"""Headroom model wrapping helpers for DeepClaw."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.base import RunnableBinding

from deepclaw.config import HeadroomConfig

logger = logging.getLogger(__name__)


def _invoke_kwargs(stop: list[str] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    invoke_kwargs = dict(kwargs)
    if stop is not None:
        invoke_kwargs.setdefault("stop", stop)
    invoke_kwargs.pop("run_manager", None)
    return invoke_kwargs


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

    class DeepClawHeadroomChatModel(HeadroomChatModel):
        """Headroom model wrapper that emits DeepClaw-specific savings logs."""

        def _optimize_messages(self, messages):  # type: ignore[override]
            optimized_messages, metrics = super()._optimize_messages(messages)
            logger.info(
                "Headroom savings: %s -> %s tokens (%s saved, %.1f%%) transforms=%s",
                metrics.tokens_before,
                metrics.tokens_after,
                metrics.tokens_saved,
                metrics.savings_percent,
                metrics.transforms_applied,
            )
            return optimized_messages, metrics

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            optimized_messages, _metrics = self._optimize_messages(messages)
            if isinstance(self.wrapped_model, RunnableBinding):
                response = self.wrapped_model.invoke(
                    optimized_messages,
                    **_invoke_kwargs(stop, kwargs),
                )
                if isinstance(response, ChatResult):
                    return response
                return ChatResult(generations=[ChatGeneration(message=response)])
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            if isinstance(self.wrapped_model, RunnableBinding):
                optimized_messages, _metrics = self._optimize_messages(messages)
                response = await self.wrapped_model.ainvoke(
                    optimized_messages,
                    **_invoke_kwargs(stop, kwargs),
                )
                if isinstance(response, ChatResult):
                    return response
                return ChatResult(generations=[ChatGeneration(message=response)])
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        def _stream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Iterator[Any]:
            if isinstance(self.wrapped_model, RunnableBinding):
                optimized_messages, _metrics = self._optimize_messages(messages)
                yield from self.wrapped_model.stream(
                    optimized_messages,
                    **_invoke_kwargs(stop, kwargs),
                )
                return
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

        async def _astream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> AsyncIterator[Any]:
            if isinstance(self.wrapped_model, RunnableBinding):
                optimized_messages, _metrics = self._optimize_messages(messages)
                async for chunk in self.wrapped_model.astream(
                    optimized_messages,
                    **_invoke_kwargs(stop, kwargs),
                ):
                    yield chunk
                return
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk

        def bind_tools(self, tools, **kwargs):  # type: ignore[override]
            new_wrapped = self.wrapped_model.bind_tools(tools, **kwargs)
            rebound = DeepClawHeadroomChatModel(new_wrapped)
            rebound.headroom_config = self.headroom_config
            rebound.mode = self.mode
            rebound.auto_detect_provider = self.auto_detect_provider
            return rebound

    resolved_model = resolve_model(model)
    wrapped_model = DeepClawHeadroomChatModel(resolved_model)
    logger.info("Headroom prompt compression enabled for %s", resolved_model.__class__.__name__)
    return wrapped_model
