"""The bounded, tool-driven agent loop.

Design invariant (Section 7): the ONLY thing that decides what happens on each
iteration is ``turn.tool_call.name`` -- the tool Gemini itself named this turn.
There is no branch anywhere in this file that selects a tool, or decides to
stop / escalate / finalise, based on the turn index, the recovery event's
fields, or the accumulated history. The loop only:

  1. asks the provider for the next tool call, given the conversation so far;
  2. validates it (argument schema + guardrails for mutating tools);
  3. executes it via the application;
  4. feeds the structured result back;
  5. repeats -- until Gemini calls a terminal tool, or a hard limit
     (MAX_TURNS / repeated-invalid / provider-unavailable) forces a safe stop.

If a different recovery event reached any given iteration, Gemini could
legitimately choose a different next tool, and this code would execute it
unchanged.
"""
from __future__ import annotations

import time
from typing import Any

from sqlalchemy.orm import Session

from app.agent.config import AgentConfig
from app.agent.guardrails import check_execute_action, load_event
from app.agent.prompts import (
    initial_user_message,
    retry_user_message,
    system_prompt,
    tool_call_entry,
    tool_result_entry,
)
from app.agent.providers.base import (
    AuthError,
    LLMProvider,
    MalformedResponseError,
    ProviderUnavailable,
    RateLimitedError,
    TransientError,
)
from app.agent.schemas import ProviderTurn, ToolTraceEntry
from app.agent.state import RecoveryAgentState
from app.agent.tools import TOOLS, ALL_TOOLS
from app.agent.tools.base import ToolContext

_VALIDATION_FAIL = "argument_validation_failed"


class RecoveryAgent:
    def __init__(
        self,
        db: Session,
        provider: LLMProvider,
        config: AgentConfig | None = None,
        *,
        razorpay_client: Any = None,
        voice_service: Any = None,
    ) -> None:
        self.db = db
        self.provider = provider
        self.config = config or AgentConfig.from_settings()
        self.razorpay_client = razorpay_client
        if voice_service is None:
            from app.services.voice import VoiceService

            voice_service = VoiceService.from_settings()
        self.voice_service = voice_service
        self._tool_specs = [t.spec() for t in ALL_TOOLS]

    # ------------------------------------------------------------------
    def run(self, recovery_event_id: int, *, dry_run: bool = True) -> RecoveryAgentState:
        state = RecoveryAgentState(
            recovery_event_id=recovery_event_id, dry_run=dry_run
        )
        # Seed guardrail-relevant facts from the DB. NOT sent to Gemini -- it
        # only learns these by calling a tool.
        event = load_event(self.db, recovery_event_id)
        if event is not None:
            state.actions_attempted = [i.action_type for i in event.interventions]
            state.current_attempt = len(event.interventions) + 1

        conversation: list[dict[str, Any]] = [
            initial_user_message(recovery_event_id, dry_run)
        ]
        run_started = time.monotonic()

        while state.turns_used < self.config.max_turns:
            # --- OBSERVE + REASON: one genuine model turn -----------------
            try:
                turn = self.provider.generate(
                    system_prompt=system_prompt(self.config),
                    conversation=conversation,
                    tools=self._tool_specs,
                )
            except AuthError as exc:
                self._fail_safe(state, "quota_or_api_failure",
                                f"Gemini auth error: {exc}")
                break
            except RateLimitedError as exc:
                self._fail_safe(state, "quota_or_api_failure",
                                f"Gemini rate limit / quota: {exc}")
                break
            except (TransientError, MalformedResponseError, ProviderUnavailable) as exc:
                state.transient_errors += 1
                state.errors.append(f"{type(exc).__name__}: {exc}")
                if state.transient_errors > self.config.max_transient_retries:
                    self._fail_safe(state, "quota_or_api_failure",
                                    f"Gemini unavailable: {exc}")
                    break
                conversation.append(retry_user_message(
                    "The previous attempt failed transiently. Retry the next "
                    "step by calling a tool."
                ))
                continue

            state.turns_used += 1
            state.record_usage(
                turn.prompt_tokens, turn.output_tokens, turn.total_tokens
            )

            # --- CHOOSE: Gemini named a tool (or failed to) -------------
            if turn.tool_call is None:
                if self._handle_no_tool_call(state, conversation, turn):
                    break
                continue

            call = turn.tool_call
            conversation.append(
                tool_call_entry(call.name, call.arguments, call.thought_signature)
            )

            # --- APPLICATION validates + executes ----------------------
            result, terminal = self._dispatch(state, call, turn)
            # --- OBSERVE RESULT: fed back for the next genuine turn ----
            conversation.append(tool_result_entry(call.name, result))

            if terminal:
                break
        else:
            # loop exited because turns_used == max_turns
            self._force_stop(state, "max_turns_reached", "reached the turn limit")

        state.latency_ms = int((time.monotonic() - run_started) * 1000)
        if state.final_status is None:
            self._force_stop(state, "other", "run ended without an explicit decision")
        return state

    # ------------------------------------------------------------------
    def _handle_no_tool_call(
        self,
        state: RecoveryAgentState,
        conversation: list[dict[str, Any]],
        turn: ProviderTurn,
    ) -> bool:
        """Return True if the run must terminate."""
        state.invalid_requests += 1
        state.errors.append("model produced text instead of a tool call")
        if state.invalid_requests >= self.config.max_invalid_requests:
            self._force_stop(
                state, "repeated_invalid_output",
                "model repeatedly failed to produce a valid tool call",
            )
            return True
        conversation.append(
            {"type": "model_text", "text": turn.raw_text[:500]}
        )
        conversation.append(
            retry_user_message(
                "That was not a tool call. You must call exactly one tool. "
                "If you are finished, call stop_recovery or escalate_recovery."
            )
        )
        return False

    # ------------------------------------------------------------------
    def _dispatch(
        self, state: RecoveryAgentState, call, turn: ProviderTurn
    ) -> tuple[dict[str, Any], bool]:
        tool = TOOLS.get(call.name)
        if tool is None:
            state.invalid_requests += 1
            payload = {
                "error": "unknown_tool",
                "message": f"{call.name!r} is not a tool",
                "available_tools": sorted(TOOLS),
            }
            self._trace(state, call, payload, ok=False, terminal=False, turn=turn)
            return self._maybe_bail(state, payload)

        # 1. argument schema validation
        try:
            args = self._validate_args(tool, call.arguments)
        except ValueError as exc:
            state.invalid_requests += 1
            payload = {"error": _VALIDATION_FAIL, "message": str(exc)}
            self._trace(state, call, payload, ok=False, terminal=False,
                        turn=turn, guardrail_code=_VALIDATION_FAIL)
            return self._maybe_bail(state, payload)

        # 2. guardrails for mutating tools -- unconditional application code
        if tool.mutating:
            guard = check_execute_action(
                load_event(self.db, state.recovery_event_id),
                args.get("action_type", ""),
            )
            if not guard.ok:
                state.invalid_requests += 1
                payload = {
                    "error": "guardrail_violation",
                    "code": guard.code,
                    "message": guard.message,
                }
                self._trace(state, call, payload, ok=False, terminal=False,
                            turn=turn, guardrail_code=guard.code)
                return self._maybe_bail(state, payload)

        # 3. execute
        ctx = ToolContext(
            db=self.db, state=state, dry_run=state.dry_run,
            razorpay_client=self.razorpay_client,
            voice_service=self.voice_service,
        )
        try:
            payload = tool.run(ctx, args)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not crash the run
            state.errors.append(f"tool {tool.name} raised {type(exc).__name__}: {exc}")
            payload = {"error": "tool_execution_error", "message": str(exc)[:200]}
            self._trace(state, call, payload, ok=False, terminal=False, turn=turn)
            return payload, False

        state.decisions.append(f"{tool.name}({args.get('action_type') or ''})".strip())
        self._trace(state, call, payload, ok="error" not in payload,
                    terminal=tool.terminal, turn=turn)
        return payload, tool.terminal

    # ------------------------------------------------------------------
    def _maybe_bail(
        self, state: RecoveryAgentState, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        if state.invalid_requests >= self.config.max_invalid_requests:
            self._force_stop(
                state, "guardrail_violation",
                "too many invalid / disallowed tool requests",
            )
            return payload, True
        return payload, False

    @staticmethod
    def _validate_args(tool, raw: dict[str, Any]) -> dict[str, Any]:
        schema = tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])
        raw = raw or {}
        for key in required:
            if key not in raw or raw[key] in (None, ""):
                raise ValueError(f"missing required argument {key!r}")
        cleaned: dict[str, Any] = {}
        for key, val in raw.items():
            if key not in props:
                continue  # ignore unknown keys rather than fail hard
            spec = props[key]
            enum = spec.get("enum")
            if enum is not None and val not in enum:
                raise ValueError(
                    f"argument {key!r}={val!r} is not one of {enum}"
                )
            cleaned[key] = val
        return cleaned

    # ------------------------------------------------------------------
    def _trace(
        self,
        state: RecoveryAgentState,
        call,
        payload: dict[str, Any],
        *,
        ok: bool,
        terminal: bool,
        turn: ProviderTurn,
        guardrail_code: str | None = None,
    ) -> None:
        state.trace.append(
            ToolTraceEntry(
                turn=state.turns_used,
                tool=call.name,
                arguments=dict(call.arguments or {}),
                ok=ok,
                terminal=terminal,
                result_summary=_summarise(payload),
                guardrail_code=guardrail_code,
                latency_ms=turn.latency_ms or None,
                prompt_tokens=turn.prompt_tokens,
                output_tokens=turn.output_tokens,
            )
        )

    def _fail_safe(self, state: RecoveryAgentState, reason: str, message: str) -> None:
        state.final_status = "failed_safe"
        state.stop_reason = reason
        state.errors.append(message)
        state.reasoning_summary = state.reasoning_summary or message

    def _force_stop(self, state: RecoveryAgentState, reason: str, message: str) -> None:
        if state.final_status is None:
            status = "failed_safe" if reason in (
                "repeated_invalid_output", "guardrail_violation",
                "quota_or_api_failure",
            ) else "completed"
            state.final_status = status
            state.stop_reason = reason
            state.reasoning_summary = state.reasoning_summary or message


def _summarise(payload: dict[str, Any]) -> str:
    if "error" in payload:
        return f"error: {payload.get('code') or payload['error']}"
    keys = [k for k in ("executed", "simulated", "stopped", "stop_reason",
                        "escalated", "escalation_type", "observed",
                        "payment_recovered", "source", "recommended_by_expected_value",
                        "observed_incremental_lift", "model_predicted_uplift_for_context",
                        "computable")
            if k in payload]
    if keys:
        return ", ".join(f"{k}={payload[k]}" for k in keys)
    return ", ".join(sorted(payload)[:6])
