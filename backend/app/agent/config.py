"""Agent configuration.

Environment-driven, with conservative defaults chosen for the Gemini **free
tier**:

    GEMINI_API_KEY          -- read from environment / .env only (never logged)
    GEMINI_MODEL            -- default "gemini-flash-latest" (see app/config.py:
                               the requested "gemini-2.5-flash" is rejected by
                               this key's generateContent endpoint)
    GEMINI_MAX_TURNS        -- default 4  (hard ceiling on agent reasoning turns)
    GEMINI_TIMEOUT_SECONDS  -- default 20 (per Gemini HTTP request)

The remaining knobs are agent-internal (not worth an app-wide setting) and are
also deliberately small: at most a couple of transient retries, no aggressive
429 retrying, a low temperature, and a tight output-token cap.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class AgentConfig:
    api_key: str | None
    model: str
    max_turns: int
    timeout_seconds: float

    # agent-internal, free-tier conservative
    max_invalid_requests: int = 3          # bad/again -> terminate safely
    max_transient_retries: int = 2         # 5xx / network only
    max_rate_limit_retries: int = 1        # 429: at most ONE bounded retry
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 8.0
    temperature: float = 0.2
    max_output_tokens: int = 512

    @classmethod
    def from_settings(cls) -> "AgentConfig":
        return cls(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            max_turns=max(1, int(settings.GEMINI_MAX_TURNS)),
            timeout_seconds=float(settings.GEMINI_TIMEOUT_SECONDS),
        )

    @property
    def has_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())
