"""Uplift-aware recovery policy.

``UpliftRecoveryPolicy`` is a drop-in
:class:`~app.services.recovery_policy.RecoveryPolicy` that ranks candidate
actions by their **incremental** economic value rather than by raw predicted
recovery probability::

    baseline          = mu_0(X)                    # P(recover | no intervention)
    treated           = mu_a(X)                    # P(recover | action a)
    uplift_a          = treated - baseline
    incremental_value = uplift_a * payment_amount_paise - action_cost_paise

Actions are ranked by ``incremental_value``. A customer who would very likely
recover anyway (high baseline, low uplift) is correctly deprioritised even if
their raw recovery probability under the action is high.

This policy does **not** replace :class:`MLRecoveryPolicy`. It is selected
explicitly (``?policy=uplift``). If the uplift artifact is missing / invalid, or
the event cannot be scored, it falls back -- first to :class:`MLRecoveryPolicy`
(which itself falls back to the rules policy), so the recovery workflow can
never break.

Safety: control events get no intervention; already-tried actions are excluded;
the orchestrator still enforces ``MAX_INTERVENTION_ATTEMPTS`` and the terminal
status checks. When no action clears ``min_net_incremental_value_paise`` the policy
declines to intervene and says why -- the economically-honest outcome for a
causal policy, and the reason it is opt-in.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.ml.uplift.config import INTERVENTION_COST_PAISE, TREATMENT_ACTIONS
from app.ml.uplift.inference.predictor import UpliftModel
from app.ml.uplift.models.artifact import UpliftModelUnavailable
from app.services.recovery_policy import (
    CandidateAction,
    PolicyContext,
    PolicyDecision,
    RecoveryPolicy,
    RulesBasedRecoveryPolicy,
)

_ACTIONS = list(TREATMENT_ACTIONS)


class UpliftRecoveryPolicy:
    """RecoveryPolicy backed by the uplift artifact, with an ML -> rules fallback."""

    def __init__(
        self,
        db: Session,
        *,
        model: UpliftModel | None = None,
        fallback: RecoveryPolicy | None = None,
        min_net_incremental_value_paise: float | None = 0.0,
    ) -> None:
        self.db = db
        self._min_value = min_net_incremental_value_paise
        self._fallback: RecoveryPolicy = fallback or self._default_fallback(db)
        try:
            self._model = model or UpliftModel.load()
        except UpliftModelUnavailable:
            self._model = None
        self.name = (
            f"uplift:{self._model.version}"
            if self._model is not None
            else f"{self._fallback.name}+uplift_unavailable"
        )

    @staticmethod
    def _default_fallback(db: Session) -> RecoveryPolicy:
        try:
            from app.services.ml_recovery_policy import MLRecoveryPolicy

            return MLRecoveryPolicy(db)
        except Exception:  # noqa: BLE001 - never let fallback construction break serving
            return RulesBasedRecoveryPolicy()

    @property
    def model_version(self) -> str | None:
        """Live uplift artifact version, or ``None`` when falling back. Read by
        the orchestrator's assignment audit trail."""
        return self._model.version if self._model is not None else None

    # -- RecoveryPolicy ----------------------------------------
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

        untried = [a for a in _ACTIONS if a not in context.prior_action_types]
        if not untried:
            return PolicyDecision(
                should_intervene=False,
                candidates=[],
                rationale="all candidate action types have already been attempted",
            )

        try:
            event_uplift = self._model.predict_for_event(
                self.db.connection(),
                int(context.recovery_event_id),
                actions=untried,
                as_of=datetime.now(timezone.utc),
            )
        except Exception as exc:  # noqa: BLE001 - never break the workflow
            return self._fallback_decision(
                context, f"uplift scoring failed ({type(exc).__name__}: {exc})"
            )

        baseline = event_uplift.baseline_probability
        candidates: list[CandidateAction] = []
        for a in event_uplift.actions:
            candidates.append(
                CandidateAction(
                    action_type=a.action,
                    cost_paise=a.cost_paise,
                    estimated_recovery_probability=round(a.treatment_probability, 4),
                    expected_value_paise=round(a.net_incremental_value_paise, 2),
                    score=round(a.net_incremental_value_paise, 2),
                    confidence=round(max(0.0, min(1.0, a.treatment_probability)), 4),
                    reason=(
                        f"{self.name}: baseline P(recover|no action)={baseline:.3f}, "
                        f"P(recover|{a.action})={a.treatment_probability:.3f}, "
                        f"uplift={a.uplift:+.3f}; incremental revenue "
                        f"~{a.incremental_expected_revenue_paise:.0f} paise "
                        f"- cost {a.cost_paise} = net {a.net_incremental_value_paise:.0f} paise"
                    ),
                )
            )
        candidates.sort(key=lambda c: (-c.score, c.cost_paise))

        if self._min_value is not None and (
            not candidates or candidates[0].score <= self._min_value
        ):
            top = candidates[0] if candidates else None
            return PolicyDecision(
                should_intervene=False,
                candidates=[],
                rationale=(
                    f"{self.name}: no action has net incremental value "
                    f"> {self._min_value:.0f} paise "
                    f"(best: {top.action_type if top else 'n/a'} "
                    f"@ {top.score if top else 'n/a'} paise, "
                    f"baseline P(recover)={baseline:.3f}); "
                    "intervening would not create incremental recovered revenue"
                ),
            )

        return PolicyDecision(
            should_intervene=True,
            candidates=candidates,
            rationale=(
                f"{self.name} ranked {len(candidates)} untried action(s) by net "
                f"incremental value (baseline P(recover)={baseline:.3f}; "
                f"top: {candidates[0].action_type} uplift-driven net "
                f"{candidates[0].score:.0f} paise)"
            ),
        )

    # -- fallback plumbing ------------------------------------
    def _fallback_reason(self, context: PolicyContext) -> str | None:
        if self._model is None:
            return "uplift artifact unavailable"
        if context.recovery_event_id is None:
            return "no recovery_event_id in policy context"
        return None

    def _fallback_decision(self, context: PolicyContext, why: str) -> PolicyDecision:
        decision = self._fallback.decide(context)
        return PolicyDecision(
            should_intervene=decision.should_intervene,
            candidates=decision.candidates,
            rationale=f"[uplift fallback -> {self._fallback.name}: {why}] {decision.rationale}",
        )
