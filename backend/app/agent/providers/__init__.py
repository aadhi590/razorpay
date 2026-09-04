"""LLM provider abstraction for the recovery agent.

The agent loop depends only on :class:`LLMProvider`. Concrete implementations:
:class:`GeminiProvider` (Google Gemini) and :class:`GroqProvider` (Groq,
OpenAI-compatible). :func:`make_provider` picks one from ``AgentConfig.provider``
(env ``LLM_PROVIDER``). Tests inject scripted / reactive fakes instead.
"""
from __future__ import annotations

from app.agent.config import AgentConfig
from app.agent.providers.base import (
    AuthError,
    LLMProvider,
    MalformedResponseError,
    ProviderError,
    ProviderUnavailable,
    RateLimitedError,
    ToolSpec,
    TransientError,
)
from app.agent.providers.gemini import GeminiProvider
from app.agent.providers.groq import GroqProvider


def make_provider(config: AgentConfig | None = None) -> LLMProvider:
    """Build the configured provider. Raises :class:`ProviderUnavailable` when the
    selected backend has no API key (the runner catches this and fails safe)."""
    config = config or AgentConfig.from_settings()
    if config.provider == "groq":
        return GroqProvider(config)
    return GeminiProvider(config)


__all__ = [
    "LLMProvider",
    "GeminiProvider",
    "GroqProvider",
    "make_provider",
    "ToolSpec",
    "ProviderError",
    "AuthError",
    "RateLimitedError",
    "TransientError",
    "MalformedResponseError",
    "ProviderUnavailable",
]
