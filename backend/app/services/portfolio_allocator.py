"""Portfolio scarcity allocator.

The recovery agent and the orchestrator reason about **one** recovery event at a
time. Neither of them reasons about the fact that, at any given moment, there is
a whole batch of simultaneously-open, actionable recovery events competing for
the same limited resource -- contact-frequency headroom, compliance budget, ops
capacity -- and that spending that resource on one event means not spending it on
another.

:class:`PortfolioAllocator` is a **deterministic, read-only application service**
(the same authority level as :mod:`app.agent.guardrails`, *not* a new agent or
decision-maker). Given the batch of currently-open eligible recovery events and
an explicit capacity constraint, it:

1. reuses the existing policy scoring (:func:`app.services.policy_factory.resolve_policy`
   -> :meth:`RecoveryPolicy.decide`, the *identical* call the orchestrator makes)
   to get each event's best available action and its expected value **net of
   action cost**;
2. ranks the eligible events by that expected value (highest first);
3. applies the capacity constraint: the top ``capacity`` positive-value events
   are ``"act"``, the remainder are ``"skip"``;
4. states, for every ``"skip"``, the *specific* reason -- which rank cutoff it
   fell below, or that its best action has non-positive expected value -- never a
   generic "not enough budget";
5. quantifies what the constraint costs: expected value captured by the ``"act"``
   set vs. the expected value that would have been captured with unlimited
   capacity.

It never executes an action, never creates an ``Intervention``, never touches the
agent loop. Acting on an ``"act"``-ranked event still goes through the existing
agent-run endpoint and the existing guardrails, unchanged.

Eligibility is **not** re-invented here: the batch filter is the orchestrator's
(:meth:`RecoveryOrchestratorService._eligible_open_treatment_events` -- status
``open`` and not control) and each event is then passed through the exact same
:func:`app.agent.guardrails.check_event_actionable` check the agent run performs,
so this service can never select an event the guardrails would reject.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.guardrails import (
    _LOAD_OPTIONS,
    build_policy_context,
    check_event_actionable,
)
from app.models.recovery_events import RecoveryEvent
from app.services.policy_factory import DEFAULT_POLICY, resolve_policy
from app.services.recovery_config import STATUS_OPEN

# Sensible default when the caller does not pin a capacity. Kept small on purpose:
# the whole point of the feature is to make the scarcity tradeoff visible, and a
# default at or above the current open-batch size would hide it.
DEFAULT_BATCH_CAPACITY = 3

RANKING_BASIS_RAW_EV = "raw_expected_value"

_RANKING_NOTE = (
    "Events are ranked by raw expected value (net of action cost) from the "
    "{policy} policy -- the same scoring the orchestrator uses. The scoring layer "
    "exposes no variance or confidence interval on a single event's expected "
    "value, so this ranking is NOT uncertainty-adjusted; near-ties are broken by "
    "the action's recovery probability, then by lower action cost, then by event "
    "id. Acting on an 'act' event still runs through the normal agent endpoint "
    "and guardrails -- this allocation is advisory."
)


@dataclass(frozen=True)
class AllocatedEvent:
    """One recovery event's place in the allocation.

    ``expected_value_paise`` is the best available action's expected value net of
    its cost (``P(recover) * amount_paise - action_cost_paise``), taken straight
    from the policy's ranked top candidate. ``rank`` is 1-based across all events
    that had at least one candidate action; it is ``None`` only when the policy
    proposed no action at all for the event.
    """

    recovery_event_id: int
    payment_id: int
    priority: int
    amount_paise: int
    failure_reason: str | None
    best_action: str | None
    expected_value_paise: float | None
    recovery_probability: float | None
    action_cost_paise: int | None
    rank: int | None
    decision: str  # "act" | "skip"
    reason: str


@dataclass(frozen=True)
class PortfolioAllocationResult:
    computable: bool
    reason: str | None
    policy: str
    capacity: int
    ranking_basis: str
    generated_at: datetime

    total_open_eligible_events: int      # passed the guardrail actionability check
    events_ranked: int                   # of those, had >=1 candidate action
    events_without_actionable_option: int
    capacity_used: int

    act: list[AllocatedEvent] = field(default_factory=list)
    skip: list[AllocatedEvent] = field(default_factory=list)

    expected_value_captured_paise: float = 0.0
    expected_value_if_unlimited_paise: float = 0.0
    expected_value_forgone_to_capacity_paise: float = 0.0

    note: str = ""


class PortfolioAllocator:
    def __init__(self, db: Session, *, policy_name: str | None = None) -> None:
        self.db = db
        self.policy_name = (policy_name or DEFAULT_POLICY).lower()
        # resolve_policy raises ValueError for an unknown name -- surfaced to the
        # route as a 400, same as the orchestrator route.
        self._policy = resolve_policy(self.policy_name, db)

    # ------------------------------------------------------------------
    def allocate(self, capacity: int | None = None) -> PortfolioAllocationResult:
        cap = DEFAULT_BATCH_CAPACITY if capacity is None else int(capacity)
        if cap < 1:
            cap = 1
        now = datetime.now(timezone.utc)
        note = _RANKING_NOTE.format(policy=self._policy.name)

        events = self._open_eligible_events()
        actionable = [e for e in events if check_event_actionable(e).ok]

        if not actionable:
            return PortfolioAllocationResult(
                computable=False,
                reason=(
                    "no open, non-control recovery event currently passes the "
                    "guardrail actionability check; there is nothing to allocate"
                ),
                policy=self._policy.name,
                capacity=cap,
                ranking_basis=RANKING_BASIS_RAW_EV,
                generated_at=now,
                total_open_eligible_events=0,
                events_ranked=0,
                events_without_actionable_option=0,
                capacity_used=0,
                note=note,
            )

        # 1. score every actionable event with the SAME policy call the
        #    orchestrator makes -- no parallel scoring mechanism.
        scored: list[tuple[RecoveryEvent, object | None]] = []
        for event in actionable:
            decision = self._policy.decide(build_policy_context(event))
            scored.append((event, decision.selected))

        rankable = [(e, top) for (e, top) in scored if top is not None]
        no_option = [e for (e, top) in scored if top is None]

        # 2. rank by raw expected value (net of cost); deterministic tie-breaks.
        rankable.sort(
            key=lambda pair: (
                -float(pair[1].score),
                -float(pair[1].confidence),
                int(pair[1].cost_paise),
                int(pair[0].id),
            )
        )

        positive = [(e, top) for (e, top) in rankable if float(top.score) > 0.0]
        act_count = min(cap, len(positive))
        cutoff_ev = (
            float(positive[act_count - 1][1].score) if act_count > 0 else None
        )

        act: list[AllocatedEvent] = []
        skip: list[AllocatedEvent] = []

        for idx, (event, top) in enumerate(rankable):
            rank = idx + 1
            ev = float(top.score)
            common = dict(
                recovery_event_id=event.id,
                payment_id=event.payment_id,
                priority=event.priority,
                amount_paise=event.payment.amount,
                failure_reason=event.payment.failure_reason,
                best_action=top.action_type,
                expected_value_paise=round(ev, 2),
                recovery_probability=round(
                    float(top.estimated_recovery_probability), 4
                ),
                action_cost_paise=int(top.cost_paise),
                rank=rank,
            )
            if ev <= 0.0:
                skip.append(
                    AllocatedEvent(
                        **common,
                        decision="skip",
                        reason=(
                            f"best available action '{top.action_type}' has a "
                            f"non-positive expected value ({ev:.0f} paise); acting "
                            "would not create expected recovered value, so it is "
                            "not allocated capacity regardless of headroom"
                        ),
                    )
                )
            elif rank <= act_count:
                act.append(
                    AllocatedEvent(
                        **common,
                        decision="act",
                        reason=(
                            f"ranked #{rank} of {len(rankable)} by expected value; "
                            f"within the capacity limit of {cap}"
                        ),
                    )
                )
            else:
                places_below = rank - act_count
                skip.append(
                    AllocatedEvent(
                        **common,
                        decision="skip",
                        reason=(
                            f"ranked #{rank} of {len(rankable)} by expected value, "
                            f"which is {places_below} place(s) below the capacity "
                            f"cutoff at rank #{act_count} "
                            f"(cutoff expected value {cutoff_ev:.0f} paise). The "
                            f"capacity limit of {cap} is fully consumed by the "
                            f"{act_count} higher-value event(s) above it"
                        ),
                    )
                )

        # events the policy could not propose any action for -- reported, not
        # silently dropped, and never ranked/acted.
        for event in no_option:
            skip.append(
                AllocatedEvent(
                    recovery_event_id=event.id,
                    payment_id=event.payment_id,
                    priority=event.priority,
                    amount_paise=event.payment.amount,
                    failure_reason=event.payment.failure_reason,
                    best_action=None,
                    expected_value_paise=None,
                    recovery_probability=None,
                    action_cost_paise=None,
                    rank=None,
                    decision="skip",
                    reason=(
                        f"the {self._policy.name} policy proposes no eligible "
                        "action for this event (e.g. every action already tried, "
                        "or no action clears the policy's minimum value bar); it "
                        "is not rankable and receives no capacity"
                    ),
                )
            )

        captured = sum(a.expected_value_paise or 0.0 for a in act)
        if_unlimited = sum(float(top.score) for (_e, top) in positive)
        forgone = if_unlimited - captured

        return PortfolioAllocationResult(
            computable=True,
            reason=None,
            policy=self._policy.name,
            capacity=cap,
            ranking_basis=RANKING_BASIS_RAW_EV,
            generated_at=now,
            total_open_eligible_events=len(actionable),
            events_ranked=len(rankable),
            events_without_actionable_option=len(no_option),
            capacity_used=len(act),
            act=act,
            skip=skip,
            expected_value_captured_paise=round(captured, 2),
            expected_value_if_unlimited_paise=round(if_unlimited, 2),
            expected_value_forgone_to_capacity_paise=round(forgone, 2),
            note=note,
        )

    # ------------------------------------------------------------------
    def _open_eligible_events(self) -> list[RecoveryEvent]:
        """The orchestrator's batch population: status ``open`` and not control,
        highest priority first. Identical filter to
        :meth:`RecoveryOrchestratorService._eligible_open_treatment_events`."""
        stmt = (
            select(RecoveryEvent)
            .where(
                RecoveryEvent.status == STATUS_OPEN,
                RecoveryEvent.is_control.is_(False),
            )
            .order_by(RecoveryEvent.priority.desc(), RecoveryEvent.id.asc())
            .options(*_LOAD_OPTIONS)
        )
        return list(self.db.scalars(stmt).all())
