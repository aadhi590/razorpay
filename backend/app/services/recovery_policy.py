"""Recovery decision policy.

The policy answers a single question: *given the current state of a recovery
event, which intervention actions are worth attempting, and in what order?*

It is deliberately isolated from :class:`RecoveryOrchestratorService` so a
future ML/uplift model or LLM-backed agent can be dropped in by implementing
the :class:`RecoveryPolicy` protocol, without touching orchestration,
persistence, or the API layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.services.recovery_config import (
    ACTION_TYPES,
    DEFAULT_REASON_MULTIPLIER,
    DEFAULT_RELIABILITY,
    DIMINISHING_RETURNS_DECAY,
    FAILURE_REASONS,
    MAX_RECOVERY_PROB,
    MIN_RECOVERY_PROB,
    PREMIUM_ACTION,
    PREMIUM_MIN_AMOUNT_PAISE,
    PREMIUM_MIN_PRIORITY,
)


@dataclass(frozen=True)
class PolicyContext:
    """Everything the policy is allowed to look at when making a decision."""

    failure_reason: str | None
    amount_paise: int
    priority: int
    is_control: bool
    attempt_number: int  # 1-based: this decision would produce the Nth attempt
    prior_action_types: list[str]
    customer_successful_payments: int = 0
    customer_failed_payments: int = 0
    # Identifies the row a DB-backed policy (e.g. MLRecoveryPolicy) needs for
    # point-in-time feature extraction. Optional and ignored by the rules
    # policy, so it stays backward compatible.
    recovery_event_id: int | None = None

    def reliability_proxy(self) -> float:
        """Observable stand-in for the generator's hidden reliability score:
        the customer's historical payment success ratio."""
        total = self.customer_successful_payments + self.customer_failed_payments
        if total <= 0:
            return DEFAULT_RELIABILITY
        return self.customer_successful_payments / total


@dataclass(frozen=True)
class CandidateAction:
    action_type: str
    cost_paise: int
    estimated_recovery_probability: float
    expected_value_paise: float
    score: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class PolicyDecision:
    should_intervene: bool
    candidates: list[CandidateAction]  # ranked best-first
    rationale: str

    @property
    def selected(self) -> CandidateAction | None:
        return self.candidates[0] if self.candidates else None


class RecoveryPolicy(Protocol):
    """Replaceable decision strategy."""

    name: str

    def decide(self, context: PolicyContext) -> PolicyDecision: ...


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class RulesBasedRecoveryPolicy:
    """Deterministic policy derived from the generator's intervention model.

    The estimated recovery probability for an action is::

        p = effectiveness(action)
            * reason_multiplier(failure_reason)
            * decay ** (attempt_number - 1)      # diminishing returns
            * (0.5 + reliability_proxy)           # customer-history adjustment

    clamped to the generator's [0.02, 0.9] bounds. Candidates are ranked by
    expected recovered value ``p * amount_paise - cost_paise``.

    Action types already attempted on the event are never re-proposed, so the
    orchestrator escalates to a different action on each attempt instead of
    repeating an ineffective one. ``priority`` / ``amount`` gate the expensive
    ``method_switch_prompt`` away from low-value recoveries.
    """

    name = "rules_v1"

    def decide(self, context: PolicyContext) -> PolicyDecision:
        if context.is_control:
            return PolicyDecision(
                should_intervene=False,
                candidates=[],
                rationale="control event: policy does not intervene",
            )

        untried = [
            action
            for action in ACTION_TYPES
            if action not in context.prior_action_types
        ]
        if not untried:
            return PolicyDecision(
                should_intervene=False,
                candidates=[],
                rationale="all candidate action types have already been attempted",
            )

        reason_multiplier = self._reason_multiplier(context.failure_reason)
        decay = DIMINISHING_RETURNS_DECAY ** (context.attempt_number - 1)
        reliability = context.reliability_proxy()
        allow_premium = (
            context.priority >= PREMIUM_MIN_PRIORITY
            or context.amount_paise >= PREMIUM_MIN_AMOUNT_PAISE
        )

        candidates: list[CandidateAction] = []
        for action in untried:
            if (
                action == PREMIUM_ACTION
                and not allow_premium
                and len(untried) > 1
            ):
                # Low-value recovery: skip the expensive method-switch prompt
                # unless it is the only remaining option.
                continue

            cfg = ACTION_TYPES[action]
            effectiveness = float(cfg["effectiveness"])
            cost = int(cfg["cost_paise"])

            raw_p = effectiveness * reason_multiplier * decay
            p = _clamp(
                raw_p * (0.5 + reliability),
                MIN_RECOVERY_PROB,
                MAX_RECOVERY_PROB,
            )
            expected_value = p * context.amount_paise - cost

            candidates.append(
                CandidateAction(
                    action_type=action,
                    cost_paise=cost,
                    estimated_recovery_probability=round(p, 4),
                    expected_value_paise=round(expected_value, 2),
                    score=round(expected_value, 2),
                    confidence=round(p, 4),
                    reason=(
                        f"p(recover)~{p:.2f} (effectiveness={effectiveness}, "
                        f"reason_multiplier={reason_multiplier}, decay={decay:.2f}, "
                        f"reliability={reliability:.2f}); "
                        f"EV~{expected_value:.0f} paise on attempt "
                        f"{context.attempt_number}"
                    ),
                )
            )

        # Rank by expected recovered value, then prefer the cheaper action.
        candidates.sort(key=lambda c: (-c.score, c.cost_paise))

        return PolicyDecision(
            should_intervene=bool(candidates),
            candidates=candidates,
            rationale=(
                f"ranked {len(candidates)} untried action(s) by expected recovered "
                f"value; failure_reason={context.failure_reason} "
                f"(multiplier={reason_multiplier}), priority={context.priority}, "
                f"attempt={context.attempt_number} (decay={decay:.2f}), "
                f"reliability={reliability:.2f}"
            ),
        )

    @staticmethod
    def _reason_multiplier(failure_reason: str | None) -> float:
        if failure_reason is None:
            return DEFAULT_REASON_MULTIPLIER
        return FAILURE_REASONS.get(failure_reason, DEFAULT_REASON_MULTIPLIER)
