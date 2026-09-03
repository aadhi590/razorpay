"""Read-only recovery analytics.

Nothing in this module writes to the database. Every method issues
SELECT-only statements and aggregates **in SQL** (``GROUP BY`` + aggregate
functions / ``FILTER`` / ``CASE``); whole tables are never pulled into Python.

Definition of "recovered"
-------------------------
A ``RecoveryEvent`` is counted as *recovered* when **either**:

* one of its interventions has an ``Outcome`` with ``payment_recovered = TRUE``
  -- the treatment path: an observed, successful intervention; **or**
* its ``Payment.recovered_at IS NOT NULL``
  -- the control / natural-recovery path, where no ``Outcome`` row exists
  (the outcomes API only ever writes ``Outcome`` rows, and control events have
  no interventions at all).

``recovered_value`` for an event is the full ``Payment.amount`` of a recovered
event (there is no partial-recovery modelling in the prototype). The per-action
endpoint instead attributes ``Outcome.recovered_amount_paise`` to the action
whose intervention produced the successful outcome.

Monetary values are returned as ``Decimal`` rupees (paise / 100, 2 dp). Every
ratio guards against division by zero.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import Select, and_, case, distinct, func, or_, select
from sqlalchemy.orm import Session

from app.models.agent_events import AgentEvent
from app.models.experiment import Experiment
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription
from app.schemas.analytics import (
    ActionIncrementalityResponse,
    ActionsResponse,
    ActionStats,
    AssignmentActionStats,
    AssignmentCoverageResponse,
    ControlVsTreatmentResponse,
    ExperimentResult,
    ExperimentsResponse,
    ExperimentVariantStats,
    GroupStats,
    RecoveryImpactResponse,
    SummaryResponse,
)
from app.services.experimentation import (
    MIN_DISTINCT_EVENTS_PER_ACTION,
    MIN_PROPENSITY_FOR_OVERLAP,
)

_CENTS = Decimal("0.01")

# Recent window for `action_lift_trend`. The synthetic dataset spans ~9 months
# with ~1.2-1.5k recovery events per month fairly evenly distributed, so a 90-day
# calendar window holds >1000 events per action -- enough for a stable Wilson
# interval -- while still being genuinely "recent". The prior (baseline) window
# is everything before it, and the two are disjoint.
RECENT_WINDOW_DAYS = 90


def _rupees(paise) -> Decimal:
    """paise (int / Decimal / None) -> Decimal rupees, 2 dp."""
    if paise is None:
        paise = 0
    return (Decimal(paise) / Decimal(100)).quantize(_CENTS, rounding=ROUND_HALF_UP)


def _ratio(numerator, denominator) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


class AnalyticsService:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Shared building block: one row per RecoveryEvent with a `recovered`
    # boolean and the payment amount, computed entirely in SQL.
    # ------------------------------------------------------------------
    def _recovery_event_facts(self) -> Select:
        recovered = or_(
            func.coalesce(func.bool_or(Outcome.payment_recovered), False),
            Payment.recovered_at.isnot(None),
        ).label("recovered")

        return (
            select(
                RecoveryEvent.id.label("re_id"),
                RecoveryEvent.status.label("status"),
                RecoveryEvent.is_control.label("is_control"),
                RecoveryEvent.experiment_id.label("experiment_id"),
                RecoveryEvent.variant.label("variant"),
                Payment.amount.label("amount"),
                recovered,
            )
            .select_from(RecoveryEvent)
            .join(Payment, Payment.id == RecoveryEvent.payment_id)
            .outerjoin(
                Intervention,
                Intervention.recovery_event_id == RecoveryEvent.id,
            )
            .outerjoin(Outcome, Outcome.intervention_id == Intervention.id)
            .group_by(RecoveryEvent.id, Payment.id)
            .subquery("re_facts")
        )

    # ------------------------------------------------------------------
    # GET /summary
    # ------------------------------------------------------------------
    def summary(self) -> SummaryResponse:
        f = self._recovery_event_facts()

        ev = self.db.execute(
            select(
                func.count().label("total"),
                func.count().filter(f.c.status == "open").label("open"),
                func.count().filter(f.c.status == "closed").label("closed"),
                func.count().filter(f.c.status == "abandoned").label("abandoned"),
                func.count().filter(f.c.recovered).label("recovered"),
                func.coalesce(func.sum(f.c.amount), 0).label("value"),
                func.coalesce(
                    func.sum(case((f.c.recovered, f.c.amount), else_=0)), 0
                ).label("recovered_value"),
            ).select_from(f)
        ).one()

        failed = self.db.execute(
            select(
                func.count().label("count"),
                func.coalesce(func.sum(Payment.amount), 0).label("value"),
            ).where(Payment.failure_reason.isnot(None))
        ).one()

        avg_amount = self.db.execute(
            select(func.coalesce(func.avg(Payment.amount), 0))
        ).scalar_one()

        total_interventions = self.db.execute(
            select(func.count()).select_from(Intervention)
        ).scalar_one()
        total_outcomes = self.db.execute(
            select(func.count()).select_from(Outcome)
        ).scalar_one()

        return SummaryResponse(
            total_failed_payments=failed.count,
            total_recovery_events=ev.total,
            open_events=ev.open,
            closed_events=ev.closed,
            abandoned_events=ev.abandoned,
            recovered_events=ev.recovered,
            total_recovered_value=_rupees(ev.recovered_value),
            total_failed_payment_value=_rupees(failed.value),
            overall_recovery_rate=_ratio(ev.recovered, ev.total),
            average_payment_amount=_rupees(avg_amount),
            total_interventions=total_interventions,
            total_outcomes=total_outcomes,
        )

    # ------------------------------------------------------------------
    # GET /control-vs-treatment
    # ------------------------------------------------------------------
    def control_vs_treatment(self) -> ControlVsTreatmentResponse:
        f = self._recovery_event_facts()
        rows = self.db.execute(
            select(
                f.c.is_control,
                func.count().label("events"),
                func.count().filter(f.c.recovered).label("recovered"),
                func.coalesce(func.sum(f.c.amount), 0).label("value"),
                func.coalesce(
                    func.sum(case((f.c.recovered, f.c.amount), else_=0)), 0
                ).label("recovered_value"),
            )
            .select_from(f)
            .group_by(f.c.is_control)
        ).all()

        by_flag = {bool(r.is_control): r for r in rows}

        def group(is_control: bool, name: str) -> GroupStats:
            r = by_flag.get(is_control)
            events = r.events if r else 0
            recovered = r.recovered if r else 0
            return GroupStats(
                group=name,
                recovery_events=events,
                recovered_events=recovered,
                recovery_rate=_ratio(recovered, events),
                total_payment_value=_rupees(r.value if r else 0),
                recovered_value=_rupees(r.recovered_value if r else 0),
            )

        control = group(True, "control")
        treatment = group(False, "treatment")

        absolute_lift = round(
            treatment.recovery_rate - control.recovery_rate, 6
        )
        relative_lift = (
            round(absolute_lift / control.recovery_rate, 6)
            if control.recovery_rate
            else None
        )

        return ControlVsTreatmentResponse(
            control=control,
            treatment=treatment,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
        )

    # ------------------------------------------------------------------
    # GET /actions
    # ------------------------------------------------------------------
    def actions(self) -> ActionsResponse:
        recovered_re = case(
            (Outcome.payment_recovered.is_(True), Intervention.recovery_event_id)
        )
        recovered_amt = case(
            (Outcome.payment_recovered.is_(True), Outcome.recovered_amount_paise),
            else_=0,
        )

        rows = self.db.execute(
            select(
                Intervention.action_type,
                func.count(Intervention.id).label("interventions"),
                func.count(func.distinct(Intervention.recovery_event_id)).label(
                    "distinct_events"
                ),
                func.count(func.distinct(recovered_re)).label("recovered_events"),
                func.coalesce(func.sum(recovered_amt), 0).label("recovered_value"),
                func.coalesce(func.sum(Intervention.cost_paise), 0).label("cost"),
            )
            .select_from(Intervention)
            .outerjoin(Outcome, Outcome.intervention_id == Intervention.id)
            .group_by(Intervention.action_type)
            .order_by(Intervention.action_type)
        ).all()

        actions = [
            ActionStats(
                action_type=r.action_type,
                interventions=r.interventions,
                distinct_recovery_events=r.distinct_events,
                recovered_events=r.recovered_events,
                recovery_rate=_ratio(r.recovered_events, r.distinct_events),
                recovered_value=_rupees(r.recovered_value),
                intervention_cost=_rupees(r.cost),
                cost_per_recovery=(
                    _rupees(Decimal(r.cost) / Decimal(r.recovered_events))
                    if r.recovered_events
                    else None
                ),
            )
            for r in rows
        ]
        return ActionsResponse(actions=actions)

    # ------------------------------------------------------------------
    # GET /experiments
    # ------------------------------------------------------------------
    def experiments(self) -> ExperimentsResponse:
        f = self._recovery_event_facts()
        rows = self.db.execute(
            select(
                f.c.experiment_id,
                Experiment.name.label("experiment_name"),
                f.c.variant,
                func.count().label("events"),
                func.count().filter(f.c.recovered).label("recovered"),
                func.coalesce(func.sum(f.c.amount), 0).label("value"),
                func.coalesce(
                    func.sum(case((f.c.recovered, f.c.amount), else_=0)), 0
                ).label("recovered_value"),
            )
            .select_from(f)
            .outerjoin(Experiment, Experiment.id == f.c.experiment_id)
            .group_by(f.c.experiment_id, Experiment.name, f.c.variant)
            .order_by(f.c.experiment_id, f.c.variant)
        ).all()

        grouped: dict[int | None, list] = defaultdict(list)
        for r in rows:
            grouped[r.experiment_id].append(r)

        results: list[ExperimentResult] = []
        for exp_id, variant_rows in grouped.items():
            variants = [
                ExperimentVariantStats(
                    experiment_id=r.experiment_id,
                    experiment_name=r.experiment_name,
                    variant=r.variant,
                    recovery_events=r.events,
                    recovered_events=r.recovered,
                    recovery_rate=_ratio(r.recovered, r.events),
                    payment_value=_rupees(r.value),
                    recovered_value=_rupees(r.recovered_value),
                )
                for r in variant_rows
            ]
            rate_by_variant = {v.variant: v.recovery_rate for v in variants}
            c_rate = rate_by_variant.get("control")
            t_rate = rate_by_variant.get("treatment")
            if c_rate is not None and t_rate is not None:
                absolute_lift = round(t_rate - c_rate, 6)
                relative_lift = (
                    round(absolute_lift / c_rate, 6) if c_rate else None
                )
            else:
                absolute_lift = relative_lift = None

            results.append(
                ExperimentResult(
                    experiment_id=exp_id,
                    experiment_name=variant_rows[0].experiment_name,
                    variants=variants,
                    absolute_lift=absolute_lift,
                    relative_lift=relative_lift,
                )
            )

        results.sort(key=lambda e: (e.experiment_id is None, e.experiment_id or 0))
        return ExperimentsResponse(experiments=results)

    # ------------------------------------------------------------------
    # GET /assignment-coverage
    # ------------------------------------------------------------------
    def assignment_coverage(self) -> AssignmentCoverageResponse:
        """Inspect whether the experiment has collected statistically useful
        randomized-assignment data: per-action volume, observed recovery rate,
        and the distribution of assignment propensities (with overlap /
        positivity warnings).

        The propensity is read from
        ``AgentEvent.input_context -> 'assignment' -> 'propensity'`` (written by
        the orchestrator's action-assignment layer and by the generator's
        ``--randomized-assignment`` mode). Interventions with no logged
        propensity are still counted but flagged -- inverse-propensity
        estimators cannot use them.
        """
        assign = AgentEvent.input_context["assignment"]
        chosen = func.coalesce(
            assign["chosen_action"].as_string(),
            AgentEvent.input_context["selected_action"].as_string(),
        )

        facts = (
            select(
                Intervention.id.label("iv_id"),
                Intervention.action_type.label("action"),
                Intervention.recovery_event_id.label("re_id"),
                Subscription.customer_id.label("customer_id"),
                func.coalesce(
                    func.bool_or(Outcome.payment_recovered), False
                ).label("recovered"),
                func.max(assign["propensity"].as_float()).label("propensity"),
                func.bool_or(assign["exploration"].as_boolean()).label(
                    "exploration"
                ),
            )
            .select_from(Intervention)
            .join(
                RecoveryEvent,
                RecoveryEvent.id == Intervention.recovery_event_id,
            )
            .join(Payment, Payment.id == RecoveryEvent.payment_id)
            .join(Subscription, Subscription.id == Payment.subscription_id)
            .outerjoin(Outcome, Outcome.intervention_id == Intervention.id)
            .outerjoin(
                AgentEvent,
                and_(
                    AgentEvent.recovery_event_id
                    == Intervention.recovery_event_id,
                    AgentEvent.event_type == "intervention_decision",
                    chosen == Intervention.action_type,
                ),
            )
            .where(RecoveryEvent.is_control.is_(False))
            .group_by(Intervention.id, Subscription.customer_id)
            .subquery("assign_facts")
        )

        rows = self.db.execute(
            select(
                facts.c.action,
                func.count().label("interventions"),
                func.count(distinct(facts.c.re_id)).label("distinct_events"),
                func.count(distinct(facts.c.customer_id)).label(
                    "distinct_customers"
                ),
                func.count().filter(facts.c.recovered).label("recovered"),
                func.count()
                .filter(facts.c.propensity.isnot(None))
                .label("propensity_logged"),
                func.avg(facts.c.propensity).label("avg_propensity"),
                func.min(facts.c.propensity).label("min_propensity"),
                func.max(facts.c.propensity).label("max_propensity"),
                func.count()
                .filter(facts.c.exploration.is_(True))
                .label("exploration_count"),
            )
            .select_from(facts)
            .group_by(facts.c.action)
            .order_by(facts.c.action)
        ).all()

        event_counts = self.db.execute(
            select(
                func.count().label("total"),
                func.count().filter(RecoveryEvent.is_control.is_(True)).label(
                    "control"
                ),
            ).select_from(RecoveryEvent)
        ).one()

        total_interventions = sum(r.interventions for r in rows)
        total_logged = sum(r.propensity_logged for r in rows)

        actions = [
            AssignmentActionStats(
                action_type=r.action,
                interventions=r.interventions,
                proportion=_ratio(r.interventions, total_interventions),
                distinct_recovery_events=r.distinct_events,
                distinct_customers=r.distinct_customers,
                recovered=r.recovered,
                observed_recovery_rate=_ratio(r.recovered, r.interventions),
                propensity_logged=r.propensity_logged,
                avg_propensity=(
                    round(float(r.avg_propensity), 6)
                    if r.avg_propensity is not None
                    else None
                ),
                min_propensity=(
                    round(float(r.min_propensity), 6)
                    if r.min_propensity is not None
                    else None
                ),
                max_propensity=(
                    round(float(r.max_propensity), 6)
                    if r.max_propensity is not None
                    else None
                ),
                exploration_count=r.exploration_count,
            )
            for r in rows
        ]

        warnings = self._assignment_warnings(
            actions, total_interventions, total_logged
        )

        return AssignmentCoverageResponse(
            total_recovery_events=event_counts.total,
            control_events=event_counts.control,
            treatment_events=event_counts.total - event_counts.control,
            treatment_interventions_total=total_interventions,
            treatment_interventions_with_logged_propensity=total_logged,
            propensity_coverage=_ratio(total_logged, total_interventions),
            actions=actions,
            warnings=warnings,
            note=(
                "SYNTHETIC BENCHMARK. Propensity is P(assigned action | context) "
                "after eligibility filtering. epsilon-greedy / randomized "
                "assignment does NOT by itself make this data causal -- see "
                "app/services/experimentation/README.md for the assumptions "
                "still required."
            ),
        )

    # ------------------------------------------------------------------
    # GET /recovery-impact
    # ------------------------------------------------------------------
    def recovery_impact(
        self,
        *,
        since: "datetime | None" = None,
        experiment_id: int | None = None,
    ) -> "RecoveryImpactResponse":
        """Measured incremental money recovered by the recovery system, using
        the generator's randomized control/treatment split.

        Control vs treatment is read from ``RecoveryEvent.is_control`` (the
        explicit stored column). In the current data this is exactly equivalent
        to "has zero Intervention rows" -- control events get no interventions
        -- but the column is the authoritative source and is what the rest of
        the analytics service uses.

        "Recovered" for an event (same definition as the other analytics
        endpoints): an ``Outcome.payment_recovered`` is true for one of its
        interventions, OR ``Payment.recovered_at IS NOT NULL`` (the control /
        natural-recovery path, which has no Outcome row).

        Recovered revenue for an event is the full ``Payment.amount`` (the
        prototype has no partial-recovery modelling), summed per group.

        Incremental revenue formula (explicit and auditable)::

            control_recovery_rate = recovered_control_events / control_group_size
            incremental_revenue_recovered_paise =
                treated_recovered_revenue_paise
                - round(control_recovery_rate * treated_at_risk_amount_paise)

        i.e. the treated group's actually-recovered revenue, minus the revenue
        the *control* recovery rate would have produced on the treated group's
        total at-risk amount. It is left signed: a negative value means
        treatment underperformed the control baseline on this batch.

        Confidence: a Newcombe/Wilson 95% interval for
        ``treated_recovery_rate - control_recovery_rate`` (see
        :mod:`app.services.proportion_stats`). ``confidence_method`` states
        exactly which method produced the note --
        ``"newcombe_wilson_95_difference"`` when computed,
        ``"not_computed"`` when the control or treated group is empty.
        """
        from app.services.proportion_stats import newcombe_difference_interval

        recovered = or_(
            func.coalesce(func.bool_or(Outcome.payment_recovered), False),
            Payment.recovered_at.isnot(None),
        ).label("recovered")

        facts = (
            select(
                RecoveryEvent.is_control.label("is_control"),
                Payment.amount.label("amount"),
                recovered,
            )
            .select_from(RecoveryEvent)
            .join(Payment, Payment.id == RecoveryEvent.payment_id)
            .outerjoin(
                Intervention, Intervention.recovery_event_id == RecoveryEvent.id
            )
            .outerjoin(Outcome, Outcome.intervention_id == Intervention.id)
            .group_by(RecoveryEvent.id, Payment.id)
        )
        if since is not None:
            facts = facts.where(RecoveryEvent.created_at >= since)
        if experiment_id is not None:
            facts = facts.where(RecoveryEvent.experiment_id == experiment_id)
        facts = facts.subquery("impact_facts")

        rows = self.db.execute(
            select(
                facts.c.is_control,
                func.count().label("n"),
                func.count().filter(facts.c.recovered).label("rec"),
                func.coalesce(func.sum(facts.c.amount), 0).label("at_risk"),
                func.coalesce(
                    func.sum(case((facts.c.recovered, facts.c.amount), else_=0)), 0
                ).label("rec_amount"),
            )
            .select_from(facts)
            .group_by(facts.c.is_control)
        ).all()

        by = {bool(r.is_control): r for r in rows}
        c = by.get(True)
        t = by.get(False)

        control_n = int(c.n) if c else 0
        treated_n = int(t.n) if t else 0
        rec_control = int(c.rec) if c else 0
        rec_treated = int(t.rec) if t else 0
        control_at_risk = int(c.at_risk) if c else 0
        treated_at_risk = int(t.at_risk) if t else 0
        control_rec_rev = int(c.rec_amount) if c else 0
        treated_rec_rev = int(t.rec_amount) if t else 0

        filters = {
            "since": since.isoformat() if since is not None else None,
            "experiment_id": experiment_id,
        }
        base = dict(
            filters=filters,
            control_group_size=control_n,
            treated_group_size=treated_n,
            recovered_control_events=rec_control,
            recovered_treated_events=rec_treated,
            control_at_risk_amount_paise=control_at_risk,
            treated_at_risk_amount_paise=treated_at_risk,
            control_recovered_revenue_paise=control_rec_rev,
            treated_recovered_revenue_paise=treated_rec_rev,
            total_recovered_revenue_paise=treated_rec_rev,
        )

        if control_n == 0 or treated_n == 0:
            missing = "control" if control_n == 0 else "treated"
            return RecoveryImpactResponse(
                computable=False,
                reason=f"no {missing} events match the filter; "
                "incremental impact cannot be computed",
                control_recovery_rate=None,
                treated_recovery_rate=None,
                incremental_recovery_rate=None,
                incremental_recovery_rate_ci_95=None,
                incremental_revenue_recovered_paise=None,
                confidence_note=(
                    f"not computable: {missing} group is empty "
                    f"(control_n={control_n}, treated_n={treated_n})"
                ),
                confidence_method="not_computed",
                **base,
            )

        control_rate = rec_control / control_n
        treated_rate = rec_treated / treated_n
        incremental_rate = treated_rate - control_rate
        incremental_revenue = treated_rec_rev - round(control_rate * treated_at_risk)

        ci = newcombe_difference_interval(
            rec_treated, treated_n, rec_control, control_n
        )

        return RecoveryImpactResponse(
            computable=True,
            reason=None,
            control_recovery_rate=round(control_rate, 6),
            treated_recovery_rate=round(treated_rate, 6),
            incremental_recovery_rate=round(incremental_rate, 6),
            incremental_recovery_rate_ci_95=ci.as_list(),
            incremental_revenue_recovered_paise=int(incremental_revenue),
            confidence_note=self._impact_confidence_note(
                incremental_rate, ci, control_n, treated_n
            ),
            confidence_method="newcombe_wilson_95_difference",
            **base,
        )

    @staticmethod
    def _impact_confidence_note(
        incremental_rate: float,
        ci,
        control_n: int,
        treated_n: int,
    ) -> str:
        lo, hi = ci.as_list()
        parts = [
            f"95% CI for the incremental recovery rate: [{lo}, {hi}] "
            f"(Newcombe/Wilson, not a p-value)."
        ]
        if ci.excludes_zero:
            parts.append(
                "The interval excludes zero, so the lift is unlikely to be "
                "sampling noise at this sample size."
            )
        else:
            parts.append(
                "The interval includes zero, so this is a directional estimate "
                "only -- not distinguishable from noise at this sample size."
            )
        # One soft threshold: n<30 is the conventional small-sample rule of
        # thumb. The Wilson interval stays valid below it, but the point
        # estimate is fragile, so we say so.
        if control_n < 30:
            parts.append(
                f"Control group is very small (n={control_n} < 30); treat even "
                "the interval cautiously."
            )
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Per-action observed incrementality (consumed by the agent tool
    # `get_historical_incrementality_for_action`, not exposed as an HTTP route).
    # ------------------------------------------------------------------
    def action_incrementality(
        self,
        action_type: str,
        *,
        experiment_id: int | None = None,
        since: "datetime | None" = None,
        until: "datetime | None" = None,
    ) -> "ActionIncrementalityResponse":
        """Observed, measured incremental recovery lift for one ``action_type``.

        Deliberately reuses the batch endpoint's machinery, not a second copy:

        * the per-event ``recovered`` / ``is_control`` facts come from the shared
          :meth:`_recovery_event_facts` subquery (identical definitions to every
          other analytics method);
        * the **control group is the same randomized control arm** the batch
          ``recovery_impact`` uses -- ``RecoveryEvent.is_control`` -- not a new
          "zero interventions" definition;
        * ``observed_incremental_lift`` is ``action_rate - control_rate``, the
          exact same subtraction ``recovery_impact.incremental_recovery_rate``
          performs, grouped by action instead of by the whole treated arm;
        * the 95% CI is the same :func:`newcombe_difference_interval` call
          (treated = a, control = b);
        * the small-sample discipline is the same: a rate is never reported for
          a control-empty scope or for an action with fewer than
          ``MIN_DISTINCT_EVENTS_PER_ACTION`` historical uses.

        ``experiment_id`` scopes both groups to one experiment (used only by the
        tests, mirroring :meth:`recovery_impact`); the agent tool always calls
        this globally.

        ``since`` / ``until`` restrict **both** the action arm and the control
        arm to ``RecoveryEvent.created_at`` in ``[since, until)`` -- the same
        window applied identically to both, so the incremental-lift subtraction
        stays valid. Used by :meth:`action_lift_trend` to compute the lift over a
        recent window vs. a disjoint prior window; ``None`` (the default) is the
        all-time scope the agent tool uses directly.
        """
        from app.services.proportion_stats import newcombe_difference_interval

        f = self._recovery_event_facts()

        treated_event_ids = select(Intervention.recovery_event_id).where(
            Intervention.action_type == action_type
        )
        treated_where = f.c.re_id.in_(treated_event_ids)
        control_where = f.c.is_control.is_(True)
        if experiment_id is not None:
            treated_where = and_(treated_where, f.c.experiment_id == experiment_id)
            control_where = and_(control_where, f.c.experiment_id == experiment_id)
        if since is not None or until is not None:
            window = select(RecoveryEvent.id)
            if since is not None:
                window = window.where(RecoveryEvent.created_at >= since)
            if until is not None:
                window = window.where(RecoveryEvent.created_at < until)
            treated_where = and_(treated_where, f.c.re_id.in_(window))
            control_where = and_(control_where, f.c.re_id.in_(window))

        def _counts(where_clause) -> tuple[int, int]:
            row = self.db.execute(
                select(
                    func.count().label("n"),
                    func.count().filter(f.c.recovered).label("rec"),
                )
                .select_from(f)
                .where(where_clause)
            ).one()
            return int(row.n), int(row.rec)

        treated_n, rec_treated = _counts(treated_where)
        control_n, rec_control = _counts(control_where)

        base = dict(
            action_type=action_type,
            treated_group_size=treated_n,
            control_group_size=control_n,
            recovered_treated_events=rec_treated,
            recovered_control_events=rec_control,
        )

        if control_n == 0:
            return ActionIncrementalityResponse(
                computable=False,
                reason="no_control_baseline",
                observed_recovery_rate_for_action=None,
                baseline_control_recovery_rate=None,
                observed_incremental_lift=None,
                observed_incremental_lift_ci_95=None,
                sample_size_note=(
                    "No randomized control events match this scope, so there is "
                    "no baseline to measure incremental lift against."
                ),
                confidence_method="not_computed",
                **base,
            )
        if treated_n < MIN_DISTINCT_EVENTS_PER_ACTION:
            reason = (
                "no_historical_data" if treated_n == 0 else "insufficient_historical_data"
            )
            return ActionIncrementalityResponse(
                computable=False,
                reason=reason,
                observed_recovery_rate_for_action=None,
                baseline_control_recovery_rate=None,
                observed_incremental_lift=None,
                observed_incremental_lift_ci_95=None,
                sample_size_note=(
                    f"Only {treated_n} historical use(s) of {action_type!r} "
                    f"(< {MIN_DISTINCT_EVENTS_PER_ACTION}, the minimum for a stable "
                    "per-action estimate -- the same threshold the assignment-"
                    "coverage checks use). No lift is reported, to avoid a "
                    "fabricated rate from a near-empty sample."
                ),
                confidence_method="not_computed",
                **base,
            )

        action_rate = rec_treated / treated_n
        control_rate = rec_control / control_n
        incremental = action_rate - control_rate
        ci = newcombe_difference_interval(
            rec_treated, treated_n, rec_control, control_n
        )

        return ActionIncrementalityResponse(
            computable=True,
            reason=None,
            observed_recovery_rate_for_action=round(action_rate, 6),
            baseline_control_recovery_rate=round(control_rate, 6),
            observed_incremental_lift=round(incremental, 6),
            observed_incremental_lift_ci_95=ci.as_list(),
            sample_size_note=self._action_incrementality_note(
                action_type, ci, treated_n
            ),
            confidence_method="newcombe_wilson_95_difference",
            **base,
        )

    @staticmethod
    def _action_incrementality_note(action_type: str, ci, treated_n: int) -> str:
        lo, hi = ci.as_list()
        parts = [
            f"Based on {treated_n} historical uses of {action_type!r} vs the "
            "randomized control baseline.",
            f"95% CI for the incremental lift: [{lo}, {hi}] (Newcombe/Wilson, "
            "not a p-value) -- the same confidence standard as the batch-level "
            "recovery-impact metric.",
        ]
        if ci.excludes_zero:
            parts.append(
                "The interval excludes zero, so the lift is unlikely to be "
                "sampling noise at this sample size."
            )
        else:
            parts.append(
                "The interval includes zero, so this is a directional estimate "
                "only -- not distinguishable from noise at this sample size."
            )
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Recent-vs-baseline lift trend (consumed by the agent tool
    # `get_action_lift_trend`, not exposed as an HTTP route).
    # ------------------------------------------------------------------
    def action_lift_trend(
        self,
        action_type: str,
        *,
        experiment_id: int | None = None,
        now: "datetime | None" = None,
    ) -> "ActionLiftTrendResponse":
        """Is one ``action_type``'s *observed* recovery performance trending over
        time -- eroding, improving, or flat?

        This is the time-window extension of :meth:`action_incrementality`. It
        does **not** introduce a second statistical method: it calls
        ``action_incrementality`` three times (all-time; the recent
        ``RECENT_WINDOW_DAYS``-day window; the disjoint prior window) and then
        applies the *same* :func:`newcombe_difference_interval` used everywhere
        else -- here to the action's recovery rate in the recent window vs. the
        prior window.

        Honesty rules, matching the rest of this module:

        * the recent and prior windows are **disjoint** (``created_at >= cutoff``
          vs. ``created_at < cutoff``), so the two proportions are independent and
          Newcombe's assumption holds -- comparing "recent" against "all-time"
          would compare a sample against a superset of itself, which is invalid;
        * if either window has fewer than ``MIN_DISTINCT_EVENTS_PER_ACTION`` uses
          of the action (the same floor the all-time tool enforces), or no
          control events, the result is ``computable=False`` with an explicit
          reason -- no trend is fabricated from a thin window;
        * ``trend_direction`` is ``"stable_or_insufficient_data"`` unless the
          interval lies **entirely** on one side of zero -- a wide interval that
          straddles zero is reported as no detectable trend, not spun as one;
        * the control-arm recovery rate for both windows is reported too, so a
          system-wide shift (which would move every action) is visible rather
          than being misattributed to this action.

        ``experiment_id`` isolates a constructed dataset for the tests, mirroring
        :meth:`action_incrementality` / :meth:`recovery_impact`; the agent tool
        always calls this globally. ``now`` pins the window boundary for
        deterministic tests; it defaults to the current time.
        """
        from datetime import timedelta

        from app.schemas.analytics import ActionLiftTrendResponse
        from app.services.proportion_stats import newcombe_difference_interval

        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=RECENT_WINDOW_DAYS)
        window_desc = (
            f"recovery events created in the {RECENT_WINDOW_DAYS} days before "
            f"{now.date().isoformat()}"
        )

        all_time = self.action_incrementality(
            action_type, experiment_id=experiment_id
        )
        recent = self.action_incrementality(
            action_type, experiment_id=experiment_id, since=cutoff
        )
        prior = self.action_incrementality(
            action_type, experiment_id=experiment_id, until=cutoff
        )

        base = dict(
            action_type=action_type,
            recent_window_description=window_desc,
            recent_window_days=RECENT_WINDOW_DAYS,
            all_time_lift=all_time.observed_incremental_lift,
            all_time_window_size=all_time.treated_group_size,
            recent_window_size=recent.treated_group_size,
            baseline_window_size=prior.treated_group_size,
        )
        _blank = dict(
            recent_window_lift=None,
            baseline_window_lift=None,
            recent_window_action_recovery_rate=None,
            baseline_window_action_recovery_rate=None,
            recent_window_control_recovery_rate=None,
            baseline_window_control_recovery_rate=None,
            trend_confidence_interval=None,
            confidence_method="not_computed",
        )

        if not recent.computable:
            return ActionLiftTrendResponse(
                computable=False,
                reason="insufficient_recent_data",
                trend_direction="stable_or_insufficient_data",
                sample_size_note=(
                    f"The recent window ({window_desc}) has only "
                    f"{recent.treated_group_size} use(s) of {action_type!r} "
                    f"(or no control events) -- below "
                    f"{MIN_DISTINCT_EVENTS_PER_ACTION}, the same floor the "
                    "all-time incrementality tool enforces. No trend is reported "
                    "rather than inferring one from too few recent points."
                ),
                **base,
                **_blank,
            )
        if not prior.computable:
            return ActionLiftTrendResponse(
                computable=False,
                reason="insufficient_baseline_data",
                trend_direction="stable_or_insufficient_data",
                recent_window_lift=recent.observed_incremental_lift,
                recent_window_action_recovery_rate=(
                    recent.observed_recovery_rate_for_action
                ),
                recent_window_control_recovery_rate=(
                    recent.baseline_control_recovery_rate
                ),
                baseline_window_lift=None,
                baseline_window_action_recovery_rate=None,
                baseline_window_control_recovery_rate=None,
                trend_confidence_interval=None,
                confidence_method="not_computed",
                sample_size_note=(
                    f"The prior (pre-{cutoff.date().isoformat()}) window has only "
                    f"{prior.treated_group_size} use(s) of {action_type!r} to "
                    "compare the recent window against, so no reliable trend "
                    "baseline exists."
                ),
                **base,
            )

        ci = newcombe_difference_interval(
            recent.recovered_treated_events,
            recent.treated_group_size,
            prior.recovered_treated_events,
            prior.treated_group_size,
        )
        if ci.low > 0.0:
            direction = "improving"
        elif ci.high < 0.0:
            direction = "declining"
        else:
            direction = "stable_or_insufficient_data"

        return ActionLiftTrendResponse(
            computable=True,
            reason=None,
            recent_window_lift=recent.observed_incremental_lift,
            baseline_window_lift=prior.observed_incremental_lift,
            recent_window_action_recovery_rate=(
                recent.observed_recovery_rate_for_action
            ),
            baseline_window_action_recovery_rate=(
                prior.observed_recovery_rate_for_action
            ),
            recent_window_control_recovery_rate=(
                recent.baseline_control_recovery_rate
            ),
            baseline_window_control_recovery_rate=(
                prior.baseline_control_recovery_rate
            ),
            trend_direction=direction,
            trend_confidence_interval=ci.as_list(),
            confidence_method="newcombe_wilson_95_difference",
            sample_size_note=self._lift_trend_note(
                action_type, recent, prior, ci, direction
            ),
            **base,
        )

    @staticmethod
    def _lift_trend_note(
        action_type: str, recent, prior, ci, direction: str
    ) -> str:
        lo, hi = ci.as_list()
        parts = [
            f"Compares {action_type}'s recovery rate in the recent window "
            f"({recent.observed_recovery_rate_for_action} over "
            f"{recent.treated_group_size} uses) against the prior window "
            f"({prior.observed_recovery_rate_for_action} over "
            f"{prior.treated_group_size} uses).",
            f"95% CI for (recent - prior) recovery rate: [{lo}, {hi}] "
            "(Newcombe/Wilson, the same method as the all-time incrementality "
            "and batch recovery-impact metrics; not a p-value).",
        ]
        if direction == "stable_or_insufficient_data":
            parts.append(
                "The interval includes zero, so there is no trend distinguishable "
                "from sampling noise at this volume."
            )
        else:
            parts.append(
                f"The interval excludes zero, so the {direction} trend is "
                "unlikely to be sampling noise at this volume."
            )
        rc = recent.baseline_control_recovery_rate
        pc = prior.baseline_control_recovery_rate
        if rc is not None and pc is not None:
            parts.append(
                f"Control-arm recovery rate over the same windows: {pc} -> {rc}. "
                "A large control-arm shift would point to a system-wide change "
                f"rather than something specific to {action_type!r}."
            )
        return " ".join(parts)

    @staticmethod
    def _assignment_warnings(
        actions: list[AssignmentActionStats],
        total_interventions: int,
        total_logged: int,
    ) -> list[str]:
        warnings: list[str] = []
        if total_interventions == 0:
            warnings.append(
                "no treatment interventions recorded yet; nothing to assess"
            )
            return warnings
        if total_logged == 0:
            warnings.append(
                "no intervention carries a logged assignment propensity; "
                "run the orchestrator with epsilon>0 or regenerate data with "
                "--randomized-assignment before attempting causal estimation"
            )
        elif total_logged < total_interventions:
            warnings.append(
                f"{total_interventions - total_logged} of {total_interventions} "
                "treatment interventions have no logged propensity; "
                "inverse-propensity estimators can only use the logged subset"
            )

        actions_with_data = [a for a in actions if a.interventions > 0]
        if len(actions_with_data) < 2:
            warnings.append(
                "assignment shows no variation across actions (only "
                f"{len(actions_with_data)} action observed); no action contrast "
                "is estimable"
            )

        for a in actions:
            if a.distinct_recovery_events < MIN_DISTINCT_EVENTS_PER_ACTION:
                warnings.append(
                    f"action '{a.action_type}' has only "
                    f"{a.distinct_recovery_events} distinct events "
                    f"(< {MIN_DISTINCT_EVENTS_PER_ACTION}); insufficient "
                    "representation for a stable per-action estimate"
                )
            if (
                a.min_propensity is not None
                and a.min_propensity < MIN_PROPENSITY_FOR_OVERLAP
            ):
                warnings.append(
                    f"action '{a.action_type}' has a minimum assignment "
                    f"propensity of {a.min_propensity} "
                    f"(< {MIN_PROPENSITY_FOR_OVERLAP}); near-deterministic "
                    "assignment breaks positivity / inflates IPW variance"
                )
        return warnings
