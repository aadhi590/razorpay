"""ML-backed recovery policy.

``MLRecoveryPolicy`` is a drop-in :class:`~app.services.recovery_policy.RecoveryPolicy`
that ranks candidate actions by the model's predicted expected recovered value:

    expected_value = P(recovery | context, action) * payment_amount - action_cost

The model is a *decision tool*, not the agent. It scores; this policy ranks;
the orchestrator persists; a future agent will sit on top.

Robustness: if the artifact is missing / invalid, or the recovery event cannot
be scored (no ``recovery_event_id`` in the context, DB error, malformed
features), the policy transparently falls back to
:class:`RulesBasedRecoveryPolicy` and says so in the decision rationale. It
never raises into the recovery workflow.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.ml.config import ACTION_COST_PAISE, ACTIONS
from app.ml.inference.predictor import RecoveryModel
from app.ml.models.artifact import ModelUnavailable
from app.services.recovery_policy import (
    CandidateAction,
    PolicyContext,
    PolicyDecision,
    RecoveryPolicy,
    RulesBasedRecoveryPolicy,
)


class MLRecoveryPolicy:
    """RecoveryPolicy backed by the recovery-response model, with a rules fallback."""

    def __init__(
        self,
        db: Session,
        *,
        model: RecoveryModel | None = None,
        fallback: RecoveryPolicy | None = None,
        log_predictions: bool = False,
    ) -> None:
        self.db = db
        self._fallback: RecoveryPolicy = fallback or RulesBasedRecoveryPolicy()
        self._log_predictions = log_predictions
        try:
            self._model = model or RecoveryModel.load()
        except ModelUnavailable:
            self._model = None
        self.name = (
            f"ml:{self._model.version}"
            if self._model is not None
            else f"{self._fallback.name}+ml_unavailable"
        )

    @property
    def model_version(self) -> str | None:
        """The live model artifact version, or ``None`` when falling back to
        rules. Read by the orchestrator's assignment audit trail."""
        return self._model.version if self._model is not None else None

    # -- RecoveryPolicy ------------------------------------------
    def decide(self, context: PolicyContext) -> PolicyDecision:
        if context.is_control:
            return PolicyDecision(
                should_intervene=False,
                candidates=[],
                rationale="control event: policy does not intervene",
            )

        reason = self._fallback_reason(context)
        if reason is not None:
            return self._fallback_decision(context, reason)

        untried = [a for a in ACTIONS if a not in context.prior_action_types]
        if not untried:
            return PolicyDecision(
                should_intervene=False,
                candidates=[],
                rationale="all candidate action types have already been attempted",
            )

        try:
            scores = self._model.predict_for_event(
                self.db.connection(),
                int(context.recovery_event_id),
                actions=untried,
                as_of=datetime.now(timezone.utc),
            )
        except Exception as exc:  # noqa: BLE001 - never break the workflow
            return self._fallback_decision(
                context, f"ML scoring failed ({type(exc).__name__}: {exc})"
            )

        candidates: list[CandidateAction] = []
        for action in untried:
            s = scores.get(action)
            if s is None:
                continue
            p = float(s.probability)
            cost = ACTION_COST_PAISE.get(action, 0)
            ev = p * context.amount_paise - cost
            candidates.append(
                CandidateAction(
                    action_type=action,
                    cost_paise=cost,
                    estimated_recovery_probability=round(p, 4),
                    expected_value_paise=round(ev, 2),
                    score=round(ev, 2),
                    confidence=round(p, 4),
                    reason=(
                        f"{self.name}: P(recover|{action})={p:.3f}, "
                        f"EV~{ev:.0f} paise "
                        f"(cost {cost}, amount {context.amount_paise})"
                    ),
                )
            )

        if not candidates:
            return self._fallback_decision(
                context, "model returned no usable action scores"
            )

        candidates.sort(key=lambda c: (-c.score, c.cost_paise))

        if self._log_predictions:
            self._log(context, candidates)

        return PolicyDecision(
            should_intervene=True,
            candidates=candidates,
            rationale=(
                f"{self.name} ranked {len(candidates)} untried action(s) by "
                f"predicted expected recovered value "
                f"(top: {candidates[0].action_type} "
                f"p={candidates[0].estimated_recovery_probability}, "
                f"EV~{candidates[0].expected_value_paise:.0f} paise)"
            ),
        )

    # -- fallback plumbing ------------------------------------
    def _fallback_reason(self, context: PolicyContext) -> str | None:
        if self._model is None:
            return "model artifact unavailable"
        if context.recovery_event_id is None:
            return "no recovery_event_id in policy context"
        return None

    def _fallback_decision(
        self, context: PolicyContext, why: str
    ) -> PolicyDecision:
        decision = self._fallback.decide(context)
        return PolicyDecision(
            should_intervene=decision.should_intervene,
            candidates=decision.candidates,
            rationale=(
                f"[ML fallback -> {self._fallback.name}: {why}] "
                f"{decision.rationale}"
            ),
        )

    def _log(self, context: PolicyContext, candidates: list[CandidateAction]) -> None:
        from app.ml.monitoring.drift import log_prediction

        log_prediction(
            {
                "model_version": self.name,
                "recovery_event_id": context.recovery_event_id,
                "attempt_number": context.attempt_number,
                "scores": {
                    c.action_type: c.estimated_recovery_probability
                    for c in candidates
                },
            }
        )
