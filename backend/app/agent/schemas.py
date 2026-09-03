"""Strict schemas for agent turns, tool calls, and the final run result.

Every model turn is a *structured* decision (a validated tool call), never
free-form text. The final result is the single object the API and the audit
trail consume.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- terminal reasons (Section 12 / 13) -----------------------------------

STOP_REASONS = (
    "payment_recovered",
    "no_eligible_actions",
    "max_attempts_reached",
    "expected_value_below_threshold",
    "customer_already_recovered",
    "control_event",
    "action_executed_awaiting_outcome",
    "guardrail_violation",
    "quota_or_api_failure",
    "max_turns_reached",
    "repeated_invalid_output",
    "other",
)
StopReason = Literal[STOP_REASONS]  # type: ignore[valid-type]

ESCALATION_TYPES = ("manual_review", "merchant_support", "receivables_team")
EscalationType = Literal[ESCALATION_TYPES]  # type: ignore[valid-type]

RunStatus = Literal[
    "completed",       # Gemini stopped or finalised cleanly
    "escalated",       # Gemini escalated
    "failed_safe",     # a guardrail / quota / provider failure forced termination
]


# --- provider-level turn -------------------------------------------------

class ToolCall(BaseModel):
    """A single tool the model asked the application to run this turn."""

    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Gemini 3.x returns a per-call ``thoughtSignature`` that MUST be echoed back
    # with the functionCall in conversation history, or the next request 400s.
    thought_signature: str | None = None


class ProviderTurn(BaseModel):
    """What the provider returned for one generate() call."""

    model_config = ConfigDict(extra="forbid")
    tool_call: ToolCall | None = None
    raw_text: str | None = None          # only set when the model failed to call a tool
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


# --- persisted per-turn trace record -----------------------------------

class ToolTraceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    terminal: bool
    result_summary: str
    guardrail_code: str | None = None
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None


# --- final result -----------------------------------------------------

class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recovery_event_id: int
    agent: Literal["gemini"] = "gemini"
    model: str
    dry_run: bool

    status: RunStatus
    stop_reason: str
    decision: str                          # e.g. "execute:whatsapp_nudge" / "stop:payment_recovered"
    chosen_action: str | None = None
    customer_message: str | None = None    # Hinglish, personalised
    reasoning_summary: str = ""            # concise rationale, NOT hidden chain-of-thought
    escalation_required: bool = False
    escalation_type: str | None = None

    # Hinglish TTS: a real audio FILE synthesized from customer_message and
    # retrievable at audio_url. NOT a phone call; NOT delivered to the customer
    # through any channel in this stage. voice_reason is set (tts_disabled /
    # tts_generation_failed / empty_message) whenever voice_generated is False.
    voice_generated: bool = False
    voice_reason: str | None = None
    audio_url: str | None = None
    voice_engine: str | None = None

    turns_used: int
    actions_attempted: list[str] = Field(default_factory=list)
    actions_executed: list[str] = Field(default_factory=list)
    outcome: dict[str, Any] | None = None

    latency_ms: int
    token_usage: dict[str, int] = Field(default_factory=dict)

    quantitative_scores: list[dict[str, Any]] | None = None
    # observed historical incrementality the agent consulted this run, if any,
    # keyed by action_type (from get_historical_incrementality_for_action)
    action_incrementality: dict[str, Any] | None = None
    # observed recent-vs-baseline lift trend the agent consulted this run, if any,
    # keyed by action_type (from get_action_lift_trend)
    action_lift_trend: dict[str, Any] | None = None
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
