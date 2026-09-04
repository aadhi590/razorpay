"""Agent configuration.

Environment-driven, with conservative defaults chosen for a free tier:

    LLM_PROVIDER             -- "gemini" (default) or "groq"
    GEMINI_API_KEY          -- read from environment / .env only (never logged)
    GEMINI_MODEL            -- default "gemini-flash-latest"
    GROQ_API_KEY            -- read from environment / .env only (never logged)
    GROQ_MODEL              -- default "openai/gpt-oss-120b"
    GEMINI_MAX_TURNS        -- default 4  (hard ceiling on agent reasoning turns)
    GEMINI_TIMEOUT_SECONDS  -- default 20 (per model HTTP request)

``api_key`` / ``model`` on this object are already resolved to the *active*
provider, so both concrete providers just read ``config.api_key`` and
``config.model``.

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
    provider: str = "gemini"              # "gemini" | "groq"
    thinking_budget: int = -1             # -1 -> field omitted; 0 -> thinking off (Gemini only)
    base_url: str | None = None           # provider REST base (Groq); None -> provider default

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
        provider = (getattr(settings, "LLM_PROVIDER", "gemini") or "gemini").strip().lower()
        if provider == "groq":
            api_key = settings.GROQ_API_KEY
            model = settings.GROQ_MODEL
            base_url = getattr(settings, "GROQ_BASE_URL", None)
        else:
            provider = "gemini"
            api_key = settings.GEMINI_API_KEY
            model = settings.GEMINI_MODEL
            base_url = None
        max_turns = int(getattr(settings, "AGENT_MAX_TURNS", 0)) or int(
            settings.GEMINI_MAX_TURNS
        )
        timeout = float(getattr(settings, "AGENT_TIMEOUT_SECONDS", 0.0)) or float(
            settings.GEMINI_TIMEOUT_SECONDS
        )
        return cls(
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url,
            max_turns=max(1, max_turns),
            timeout_seconds=timeout,
            thinking_budget=int(getattr(settings, "GEMINI_THINKING_BUDGET", -1)),
        )

    @property
    def has_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())
