"""Recovery orchestration.

:class:`RecoveryOrchestratorService` turns a policy decision into persisted
state. For one recovery event it: loads the event and its payment/customer
context, asks the (replaceable) :class:`RecoveryPolicy` what to do, and then
creates the ``Intervention`` / ``AgentEvent`` / ``AuditLog`` rows and updates
``RecoveryEvent`` (and ``Payment``) accordingly -- all inside a single
transaction per recovery event.

It does NOT simulate or record an ``Outcome``: an outcome reflects something
observed after an intervention actually runs, which is the responsibility of a
separate (future) component. That is why interventions are created here with
status ``"pending"`` and ``executed_at = NULL``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog
from app.models.interventions import Intervention
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription
from app.services.experimentation import (
    DEFAULT_EXPERIMENT_CONFIG,
    ActionAssigner,
)
from app.services.recovery_config import (
    MAX_INTERVENTION_ATTEMPTS,
    STATUS_ABANDONED,
    STATUS_CLOSED,
    STATUS_OPEN,
)
from app.services.recovery_policy import (
    CandidateAction,
    PolicyContext,
    PolicyDecision,
    RecoveryPolicy,
    RulesBasedRecoveryPolicy,
)

ORCHESTRATOR_ACTOR = "recovery_orchestrator"

# OrchestrationOutcome.disposition values
DISP_INTERVENTION_CREATED = "intervention_created"
DISP_SKIPPED_CONTROL = "skipped_control"
DISP_ALREADY_RECOVERED = "already_recovered"
DISP_MAX_ATTEMPTS = "max_attempts_reached"
DISP_NOT_OPEN = "not_open"
DISP_NO_CANDIDATE = "no_candidate"
DISP_ERROR = "error"


class RecoveryEventNotFound(Exception):
    def __init__(self, recovery_event_id: int) -> None:
        super().__init__(f"Recovery event {recovery_event_id} not found")
        self.recovery_event_id = recovery_event_id


@dataclass
class OrchestrationOutcome:
    recovery_event: RecoveryEvent
    payment: Payment
    previous_status: str
    disposition: str
    action_taken: bool
    decision_reason: str
    attempt_number: int | None = None
    selected_action: str | None = None
    confidence: float | None = None
    candidates: list[CandidateAction] = field(default_factory=list)
    intervention: Intervention | None = None
    agent_event: AgentEvent | None = None
    error: str | None = None
    # Action-assignment / experimentation metadata. ``propensity`` is the exact
    # probability with which ``selected_action`` was assigned (1.0 by default,
    # i.e. pure exploitation of the top-ranked candidate).
    propensity: float | None = None
    exploration: bool = False
    assignment: dict | None = None

    def as_response_dict(self) -> dict:
        return {
            "recovery_event_id": self.recovery_event.id,
            "payment_id": self.payment.id,
            "is_control": self.recovery_event.is_control,
            "variant": self.recovery_event.variant,
            "priority": self.recovery_event.priority,
            "previous_status": self.previous_status,
            "recovery_event_status": self.recovery_event.status,
            "disposition": self.disposition,
            "action_taken": self.action_taken,
            "attempt_number": self.attempt_number,
            "selected_action": self.selected_action,
            "decision_reason": self.decision_reason,
            "confidence": self.confidence,
            "intervention_id": (
                self.intervention.id if self.intervention is not None else None
            ),
            "agent_event_id": (
                self.agent_event.id if self.agent_event is not None else None
            ),
            "candidates": [asdict(c) for c in self.candidates],
            "propensity": self.propensity,
            "exploration": self.exploration,
            "assignment": self.assignment,
            "error": self.error,
        }


@dataclass
class BatchOrchestrationResult:
    considered: int
    interventions_created: int = 0
    closed: int = 0
    abandoned: int = 0
    skipped: int = 0
    errors: int = 0
    outcomes: list[OrchestrationOutcome] = field(default_factory=list)


class RecoveryOrchestratorService:
    def __init__(
        self,
        db: Session,
        policy: RecoveryPolicy | None = None,
        assigner: ActionAssigner | None = None,
    ) -> None:
        self.db = db
        self.policy: RecoveryPolicy = policy or RulesBasedRecoveryPolicy()
        # Default assigner is disabled (exploit only, propensity 1.0) so the
        # orchestrator's behaviour is unchanged unless an experiment config is
        # explicitly injected.
        self.assigner: ActionAssigner = assigner or ActionAssigner(
            DEFAULT_EXPERIMENT_CONFIG
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def orchestrate_event(self, recovery_event_id: int) -> OrchestrationOutcome:
        """Run the orchestrator for a single recovery event, committing the
        resulting state change atomically."""
        recovery_event = self._load_event(recovery_event_id)
        try:
            outcome = self._orchestrate(recovery_event)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return outcome

    def run_batch(self, limit: int | None = None) -> BatchOrchestrationResult:
        """Process eligible events: status ``open`` and not control. Each event
        is orchestrated in its own transaction so one failure does not discard
        the rest of the batch."""
        events = self._eligible_open_treatment_events(limit)
        result = BatchOrchestrationResult(considered=len(events))

        for recovery_event in events:
            try:
                outcome = self._orchestrate(recovery_event)
                self.db.commit()
            except Exception as exc:  # noqa: BLE001 - recorded per event
                self.db.rollback()
                result.errors += 1
                result.outcomes.append(
                    OrchestrationOutcome(
                        recovery_event=recovery_event,
                        payment=recovery_event.payment,
                        previous_status=recovery_event.status,
                        disposition=DISP_ERROR,
                        action_taken=False,
                        decision_reason=f"orchestration failed: {exc}",
                        error=str(exc),
                    )
                )
                continue

            result.outcomes.append(outcome)
            if outcome.disposition == DISP_INTERVENTION_CREATED:
                result.interventions_created += 1
            elif recovery_event.status == STATUS_ABANDONED:
                result.abandoned += 1
            elif recovery_event.status == STATUS_CLOSED:
                result.closed += 1
            else:
                result.skipped += 1

        return result

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    _LOAD_OPTIONS = (
        selectinload(RecoveryEvent.interventions).selectinload(
            Intervention.outcome
        ),
        selectinload(RecoveryEvent.payment)
        .selectinload(Payment.subscription)
        .selectinload(Subscription.customer),
    )

    def _load_event(self, recovery_event_id: int) -> RecoveryEvent:
        stmt = (
            select(RecoveryEvent)
            .where(RecoveryEvent.id == recovery_event_id)
            .options(*self._LOAD_OPTIONS)
        )
        event = self.db.scalars(stmt).one_or_none()
        if event is None:
            raise RecoveryEventNotFound(recovery_event_id)
        return event

    def _eligible_open_treatment_events(
        self, limit: int | None
    ) -> list[RecoveryEvent]:
        stmt = (
            select(RecoveryEvent)
            .where(
                RecoveryEvent.status == STATUS_OPEN,
                RecoveryEvent.is_control.is_(False),
            )
            .order_by(RecoveryEvent.priority.desc(), RecoveryEvent.id.asc())
            .options(*self._LOAD_OPTIONS)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.db.scalars(stmt).all())

    # ------------------------------------------------------------------
    # Core decision + persistence (no commit -- caller owns the transaction)
    # ------------------------------------------------------------------

    def _orchestrate(self, recovery_event: RecoveryEvent) -> OrchestrationOutcome:
        payment = recovery_event.payment
        previous_status = recovery_event.status
        prior = sorted(recovery_event.interventions, key=lambda i: i.id)
        prior_action_types = [i.action_type for i in prior]
        attempt_number = len(prior) + 1

        # 1. Non-open events are terminal; leave them untouched.
        if recovery_event.status != STATUS_OPEN:
            return self._outcome(
                recovery_event,
                payment,
                previous_status,
                disposition=DISP_NOT_OPEN,
                decision_reason=(
                    f"recovery event is not open (status={recovery_event.status})"
                ),
            )

        # 2. Control events never receive interventions.
        if recovery_event.is_control:
            return self._outcome(
                recovery_event,
                payment,
                previous_status,
                disposition=DISP_SKIPPED_CONTROL,
                decision_reason="control event: policy does not intervene",
            )

        # 3. Already recovered -> close the event.
        if self._is_already_recovered(payment, prior):
            recovery_event.status = STATUS_CLOSED
            if recovery_event.closed_at is None:
                recovery_event.closed_at = payment.recovered_at or _utcnow()
            self._audit(
                recovery_event,
                action="recovery_event_closed",
                reason="payment already recovered; no further intervention needed",
                metadata={"attempt_number": attempt_number},
            )
            return self._outcome(
                recovery_event,
                payment,
                previous_status,
                disposition=DISP_ALREADY_RECOVERED,
                decision_reason="payment already recovered",
            )

        # 4. Stopping rule: the generator's attempt cap.
        if len(prior) >= MAX_INTERVENTION_ATTEMPTS:
            reason = (
                f"max intervention attempts reached "
                f"({len(prior)}/{MAX_INTERVENTION_ATTEMPTS})"
            )
            recovery_event.status = STATUS_ABANDONED
            self._audit(
                recovery_event,
                action="recovery_event_abandoned",
                reason=reason,
                metadata={"prior_action_types": prior_action_types},
            )
            return self._outcome(
                recovery_event,
                payment,
                previous_status,
                disposition=DISP_MAX_ATTEMPTS,
                decision_reason=reason,
            )

        # 5. Ask the policy.
        customer = payment.subscription.customer
        context = PolicyContext(
            failure_reason=payment.failure_reason,
            amount_paise=payment.amount,
            priority=recovery_event.priority,
            is_control=recovery_event.is_control,
            attempt_number=attempt_number,
            prior_action_types=prior_action_types,
            customer_successful_payments=customer.total_successful_payments,
            customer_failed_payments=customer.total_failed_payments,
            recovery_event_id=recovery_event.id,
        )
        decision = self.policy.decide(context)

        # 6. Nothing worth trying -> abandon.
        if decision.selected is None:
            recovery_event.status = STATUS_ABANDONED
            self._audit(
                recovery_event,
                action="recovery_event_abandoned",
                reason=decision.rationale,
                metadata={"prior_action_types": prior_action_types},
            )
            return self._outcome(
                recovery_event,
                payment,
                previous_status,
                disposition=DISP_NO_CANDIDATE,
                decision_reason=decision.rationale,
            )

        # 6b. The policy RANKS candidates; the experimentation layer ASSIGNS the
        #     action that is actually executed and records its exact propensity.
        #     With the default (disabled) config this returns the top-ranked
        #     candidate with propensity 1.0 and exploration=False.
        assignment = self.assigner.assign(
            decision=decision,
            recovery_event_id=recovery_event.id,
            policy_name=self.policy.name,
            model_version=getattr(self.policy, "model_version", None),
        )
        selected = self._candidate_for_action(decision, assignment.chosen_action)

        # 7. Create the intervention + decision + audit records.
        intervention = Intervention(
            recovery_event_id=recovery_event.id,
            action_type=selected.action_type,
            status="pending",
            cost_paise=selected.cost_paise,
            agent_reason=selected.reason[:500],
            executed_at=None,
        )
        self.db.add(intervention)
        self.db.flush()

        agent_event = AgentEvent(
            recovery_event_id=recovery_event.id,
            event_type="intervention_decision",
            input_context={
                "failure_reason": payment.failure_reason,
                "amount_paise": payment.amount,
                "priority": recovery_event.priority,
                "variant": recovery_event.variant,
                "attempt_number": attempt_number,
                "prior_action_types": prior_action_types,
                "reliability_proxy": round(context.reliability_proxy(), 4),
                "policy": self.policy.name,
                "candidates": [
                    {
                        "action_type": c.action_type,
                        "score": c.score,
                        "estimated_recovery_probability": (
                            c.estimated_recovery_probability
                        ),
                        "cost_paise": c.cost_paise,
                    }
                    for c in decision.candidates
                ],
                "selected_action": selected.action_type,
                "assignment": assignment.as_context_dict(),
            },
            decision=f"execute_{selected.action_type}"[:500],
            confidence=selected.confidence,
        )
        self.db.add(agent_event)

        self._audit(
            recovery_event,
            action=f"intervention_{selected.action_type}_selected",
            reason=selected.reason,
            metadata={
                "attempt_number": attempt_number,
                "intervention_cost_paise": selected.cost_paise,
                "ranked_candidates": [
                    c.action_type for c in decision.candidates
                ],
                "policy": self.policy.name,
                "assigned_action": assignment.chosen_action,
                "assignment_propensity": assignment.propensity,
                "assignment_exploration": assignment.exploration,
                "assignment_mechanism": assignment.assignment_mechanism,
                "experiment_id": assignment.experiment_id,
            },
        )

        # Mirror the generator: each attempt bumps the payment retry counter.
        payment.retry_count += 1
        self.db.flush()

        return self._outcome(
            recovery_event,
            payment,
            previous_status,
            disposition=DISP_INTERVENTION_CREATED,
            action_taken=True,
            attempt_number=attempt_number,
            selected_action=selected.action_type,
            decision_reason=selected.reason,
            confidence=selected.confidence,
            candidates=decision.candidates,
            intervention=intervention,
            agent_event=agent_event,
            propensity=assignment.propensity,
            exploration=assignment.exploration,
            assignment=assignment.as_context_dict(),
        )

    @staticmethod
    def _candidate_for_action(
        decision: PolicyDecision, action_type: str
    ) -> CandidateAction:
        for candidate in decision.candidates:
            if candidate.action_type == action_type:
                return candidate
        # Unreachable: the assigner only ever returns an action drawn from
        # decision.candidates.
        raise ValueError(
            f"assigned action {action_type!r} is not among the policy candidates"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _outcome(
        recovery_event: RecoveryEvent,
        payment: Payment,
        previous_status: str,
        *,
        disposition: str,
        decision_reason: str,
        action_taken: bool = False,
        attempt_number: int | None = None,
        selected_action: str | None = None,
        confidence: float | None = None,
        candidates: list[CandidateAction] | None = None,
        intervention: Intervention | None = None,
        agent_event: AgentEvent | None = None,
        propensity: float | None = None,
        exploration: bool = False,
        assignment: dict | None = None,
    ) -> OrchestrationOutcome:
        return OrchestrationOutcome(
            recovery_event=recovery_event,
            payment=payment,
            previous_status=previous_status,
            disposition=disposition,
            action_taken=action_taken,
            decision_reason=decision_reason,
            attempt_number=attempt_number,
            selected_action=selected_action,
            confidence=confidence,
            candidates=candidates or [],
            intervention=intervention,
            agent_event=agent_event,
            propensity=propensity,
            exploration=exploration,
            assignment=assignment,
        )

    @staticmethod
    def _is_already_recovered(
        payment: Payment, prior: list[Intervention]
    ) -> bool:
        if payment.status == "success" or payment.recovered_at is not None:
            return True
        return any(
            i.outcome is not None and i.outcome.payment_recovered
            for i in prior
        )

    def _audit(
        self,
        recovery_event: RecoveryEvent,
        *,
        action: str,
        reason: str,
        metadata: dict | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            recovery_event_id=recovery_event.id,
            actor=ORCHESTRATOR_ACTOR,
            action=action,
            reason=reason[:500],
            event_metadata=metadata,
        )
        self.db.add(entry)
        return entry


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
