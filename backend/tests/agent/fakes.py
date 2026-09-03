"""Fake LLM providers for the agent test-suite.

The test-suite NEVER calls a live Gemini API. Two fakes matter:

* :class:`ScriptedProvider` -- returns a fixed queue of tool calls / errors.
  Used to prove the loop executes whatever the model asks, in order.
* :class:`ReactiveProvider` -- picks the next tool by *inspecting the last tool
  result in the conversation*, the way a real model reacts to what it observed.
  Used to prove different event states drive different tool sequences through
  the same loop code.
"""
from __future__ import annotations

from typing import Any

from app.agent.schemas import ProviderTurn, ToolCall


def _turn(name: str, args: dict[str, Any]) -> ProviderTurn:
    return ProviderTurn(
        tool_call=ToolCall(name=name, arguments=args),
        model="fake",
        latency_ms=1,
        prompt_tokens=12,
        output_tokens=6,
        total_tokens=18,
    )


class ScriptedProvider:
    model = "fake-scripted"

    def __init__(self, script: list[Any]) -> None:
        # each item: (tool_name, args) | Exception | ProviderTurn
        self._queue = list(script)
        self.conversations: list[list[dict[str, Any]]] = []
        self.tools_offered: list[str] = []

    def generate(self, *, system_prompt, conversation, tools) -> ProviderTurn:
        self.conversations.append([dict(e) for e in conversation])
        self.tools_offered = [t.name for t in tools]
        if not self._queue:
            return _turn("stop_recovery",
                         {"stop_reason": "other",
                          "reasoning_summary": "script exhausted"})
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, ProviderTurn):
            return item
        name, args = item
        return _turn(name, args)


class ReactiveProvider:
    """Genuinely branches on observed tool results."""

    model = "fake-reactive"

    def __init__(self) -> None:
        self.decisions: list[str] = []
        self.saw_tool_result_before_second_call = False

    def generate(self, *, system_prompt, conversation, tools) -> ProviderTurn:
        calls = [e for e in conversation if e["type"] == "tool_call"]
        results = [e for e in conversation if e["type"] == "tool_result"]
        if len(calls) == 1 and results:
            self.saw_tool_result_before_second_call = True

        last = results[-1] if results else None
        name, args = self._decide([c["name"] for c in calls], last)
        self.decisions.append(name)
        return _turn(name, args)

    @staticmethod
    def _decide(called: list[str], last: dict[str, Any] | None):
        if last is None:
            return "get_recovery_event_context", {}

        p = last["payload"]
        src = last["name"]

        if src == "get_recovery_event_context":
            if p.get("payment_recovered"):
                return "stop_recovery", {
                    "stop_reason": "payment_recovered",
                    "reasoning_summary": "payment already recovered",
                }
            if p.get("attempt_number", 1) >= p.get("max_attempts", 3):
                return "escalate_recovery", {
                    "escalation_type": "manual_review",
                    "reasoning_summary": "at the attempt cap; needs a human",
                }
            if p.get("actions_already_attempted"):
                return "get_available_recovery_actions", {}
            return "get_action_scores", {}

        if src == "get_action_scores":
            scores = p.get("scores") or []
            if not scores:
                return "stop_recovery", {
                    "stop_reason": "no_eligible_actions",
                    "reasoning_summary": "no eligible action to score",
                }
            best = p.get("recommended_by_expected_value") or scores[0]["action"]
            return "execute_recovery_action", {
                "action_type": best,
                "customer_message": (
                    "Hi, aapka recent payment complete nahi ho paaya. "
                    "Hum ek secure link bhej sakte hain jisse aap dobara try karein."
                ),
                "reason": "highest model expected value among eligible actions",
            }

        if src == "get_available_recovery_actions":
            elig = p.get("eligible_actions") or []
            if not elig:
                return "stop_recovery", {
                    "stop_reason": "no_eligible_actions",
                    "reasoning_summary": "all actions already attempted",
                }
            return "execute_recovery_action", {
                "action_type": elig[0]["action_type"],
                "customer_message": "Hi, ek aur koshish karte hain — thoda time lagega.",
                "reason": "only remaining eligible action",
            }

        if src == "execute_recovery_action":
            if p.get("error"):
                return "stop_recovery", {
                    "stop_reason": "guardrail_violation",
                    "reasoning_summary": f"blocked: {p.get('code')}",
                }
            return "observe_recovery_outcome", {}

        if src == "observe_recovery_outcome":
            if p.get("payment_recovered"):
                return "stop_recovery", {
                    "stop_reason": "payment_recovered",
                    "reasoning_summary": "payment recovered after the action",
                }
            return "stop_recovery", {
                "stop_reason": "action_executed_awaiting_outcome",
                "reasoning_summary": "action executed; outcome not yet known",
            }

        return "stop_recovery", {
            "stop_reason": "other", "reasoning_summary": "nothing left to do",
        }


class RaisingProvider:
    model = "fake-raising"

    def __init__(self, exc: BaseException, *, times: int | None = None) -> None:
        self._exc = exc
        self._times = times
        self.calls = 0

    def generate(self, *, system_prompt, conversation, tools) -> ProviderTurn:
        self.calls += 1
        if self._times is None or self.calls <= self._times:
            raise self._exc
        return _turn("stop_recovery",
                     {"stop_reason": "other", "reasoning_summary": "recovered"})
