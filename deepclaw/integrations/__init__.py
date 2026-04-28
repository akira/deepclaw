"""Provider integrations for DeepClaw model backends."""

from deepclaw.config import DeepClawConfig

from .baseten import BASETEN_PROVIDER, resolve_baseten_model
from .deepinfra import DEEPINFRA_PROVIDER, resolve_deepinfra_model


def resolve_provider_model(config: DeepClawConfig):
    """Resolve provider-specific model adapters, or return the raw model string."""
    for resolver in (resolve_baseten_model, resolve_deepinfra_model):
        resolved = resolver(config)
        if resolved != (config.model or "").strip():
            return resolved
    return (config.model or "").strip()


__all__ = [
    "BASETEN_PROVIDER",
    "DEEPINFRA_PROVIDER",
    "resolve_baseten_model",
    "resolve_deepinfra_model",
    "resolve_provider_model",
]
