"""Provider protocol + typed error hierarchy.

A provider's one job: given the system prompt, the running conversation, and the
tool specs, return **one** :class:`~app.agent.schemas.ProviderTurn` that tells
the loop, unambiguously, either "the model wants to call tool X with args A" or
"the model produced text instead of a tool call" (a protocol violation the loop
handles). It must never itself decide what the agent does next.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.agent.schemas import ProviderTurn


@dataclass(frozen=True)
class ToolSpec:
    """Provider-agnostic description of a callable tool."""

    name: str
    description: str
    # JSON-schema-ish dict: {"type": "object", "properties": {...}, "required": [...]}
    parameters: dict[str, Any]


class Message(Protocol):
    role: str
    content: Any


class LLMProvider(Protocol):
    model: str

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> ProviderTurn:
        """Run exactly one model turn. Synchronous, single request, no batching."""
        ...


# --- errors ------------------------------------------------------------

class ProviderError(Exception):
    """Base class for all provider failures."""


class AuthError(ProviderError):
    """Missing / invalid / unauthorised API key. Never retried."""


class RateLimitedError(ProviderError):
    """HTTP 429 / quota exhausted. Retried at most once, with backoff."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class TransientError(ProviderError):
    """5xx / connection reset / timeout. Retried a small bounded number of times."""


class MalformedResponseError(ProviderError):
    """Response was not valid JSON, or had no usable candidate/parts."""


class ProviderUnavailable(ProviderError):
    """The provider cannot be used at all (e.g. no API key configured)."""
