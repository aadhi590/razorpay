"""Explicit, incrementally-accumulated agent state for one recovery event.

Nothing here is pre-populated with database content that Gemini has not asked
for. The ``*_context`` / ``eligible_actions`` / ``quantitative_scores`` slots
stay ``None`` until the corresponding tool actually runs. ``actions_attempted``
is seeded from the database because the **guardrails** need it regardless of
what Gemini has observed -- but it is never surfaced to Gemini except through a
tool result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.schemas import ToolTraceEntry


@dataclass
class RecoveryAgentState:
    recovery_event_id: int
    dry_run: bool

    # ---- populated only when a tool is actually called -----------------
    event_context: dict[str, Any] | None = None
    customer_context: dict[str, Any] | None = None
    payment_context: dict[str, Any] | None = None
    subscription_context: dict[str, Any] | None = None
    recovery_history: dict[str, Any] | None = None
    eligible_actions: list[dict[str, Any]] | None = None
    quantitative_scores: list[dict[str, Any]] | None = None
    # observed historical incrementality the agent looked up, keyed by action_type
    action_incrementality: dict[str, Any] | None = None

    # ---- application-tracked facts (not fed to Gemini directly) -------
    actions_attempted: list[str] = field(default_factory=list)   # from DB + this run
    current_attempt: int = 1
    executed_actions: list[dict[str, Any]] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)

    # ---- the turn-by-turn record (this is what proves the loop is real)
    trace: list[ToolTraceEntry] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    # ---- counters / accounting --------------------------------------
    turns_used: int = 0
    latency_ms: int = 0
    invalid_requests: int = 0
    transient_errors: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    # ---- termination ------------------------------------------------
    final_status: str | None = None        # RunStatus
    stop_reason: str | None = None
    chosen_action: str | None = None
    customer_message: str | None = None
    reasoning_summary: str = ""
    escalation_required: bool = False
    escalation_type: str | None = None

    # ---- Hinglish voice (TTS audio FILE only -- not a call, not delivered) --
    voice_generated: bool = False
    voice_reason: str | None = None        # tts_disabled | tts_generation_failed | empty_message
    audio_url: str | None = None
    audio_path: str | None = None
    voice_engine: str | None = None

    # cache of read-tool results, to avoid a duplicate DB hit if Gemini
    # asks for the exact same read again (still a genuine turn).
    _read_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_usage(self, prompt: int | None, output: int | None, total: int | None) -> None:
        self.prompt_tokens += prompt or 0
        self.output_tokens += output or 0
        self.total_tokens += total or ((prompt or 0) + (output or 0))

    def token_usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
