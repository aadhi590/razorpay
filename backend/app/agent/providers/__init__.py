"""LLM provider abstraction for the recovery agent.

The agent loop depends only on :class:`LLMProvider`. The initial concrete
implementation is :class:`GeminiProvider` (Google Gemini, free tier). Tests
inject scripted / reactive fakes instead of calling a live API.
"""
from __future__ import annotations

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

__all__ = [
    "LLMProvider",
    "GeminiProvider",
    "ToolSpec",
    "ProviderError",
    "AuthError",
    "RateLimitedError",
    "TransientError",
    "MalformedResponseError",
    "ProviderUnavailable",
]
