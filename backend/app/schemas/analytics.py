"""Response schemas for the read-only analytics API.

Monetary fields are ``Decimal`` **rupees** (paise / 100, quantized to 2 dp).
Rate fields are plain ``float`` ratios in the range ``[0, 1]`` (multiply by
100 for a percentage). Lift fields are differences / ratios of those rates.
"""
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel


class SummaryResponse(BaseModel):
    total_failed_payments: int
    total_recovery_events: int
    open_events: int
    closed_events: int
    abandoned_events: int
    recovered_events: int
    total_recovered_value: Decimal
    total_failed_payment_value: Decimal
    overall_recovery_rate: float
    average_payment_amount: Decimal
    total_interventions: int
    total_outcomes: int


class GroupStats(BaseModel):
    group: str  # "control" | "treatment"
    recovery_events: int
    recovered_events: int
    recovery_rate: float
    total_payment_value: Decimal
    recovered_value: Decimal


class ControlVsTreatmentResponse(BaseModel):
    control: GroupStats
    treatment: GroupStats
    absolute_lift: float  # treatment_rate - control_rate
    relative_lift: float | None  # (treatment_rate - control_rate) / control_rate


class ActionStats(BaseModel):
    action_type: str
    interventions: int
    distinct_recovery_events: int
    recovered_events: int
    recovery_rate: float
    recovered_value: Decimal
    intervention_cost: Decimal
    cost_per_recovery: Decimal | None  # None when there were 0 recoveries


class ActionsResponse(BaseModel):
    actions: list[ActionStats]


class ExperimentVariantStats(BaseModel):
    experiment_id: int | None
    experiment_name: str | None
    variant: str | None
    recovery_events: int
    recovered_events: int
    recovery_rate: float
    payment_value: Decimal
    recovered_value: Decimal


class ExperimentResult(BaseModel):
    experiment_id: int | None
    experiment_name: str | None
    variants: list[ExperimentVariantStats]
    absolute_lift: float | None  # treatment_rate - control_rate, if both present
    relative_lift: float | None


class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentResult]


class AssignmentActionStats(BaseModel):
    action_type: str
    interventions: int
    proportion: float  # interventions for this action / all treatment interventions
    distinct_recovery_events: int
    distinct_customers: int
    recovered: int
    observed_recovery_rate: float
    # how many of these interventions carry a logged assignment propensity
    propensity_logged: int
    avg_propensity: float | None
    min_propensity: float | None
    max_propensity: float | None
    exploration_count: int


class AssignmentCoverageResponse(BaseModel):
    total_recovery_events: int
    control_events: int
    treatment_events: int
    treatment_interventions_total: int
    treatment_interventions_with_logged_propensity: int
    propensity_coverage: float  # logged / total treatment interventions
    actions: list[AssignmentActionStats]
    warnings: list[str]
    note: str


class ActionIncrementalityResponse(BaseModel):
    """Measured, observed incremental recovery lift for a single ``action_type``,
    across all historical RecoveryEvents where that action was executed, vs the
    same randomized control baseline the batch-level ``recovery_impact`` endpoint
    uses (``RecoveryEvent.is_control``).

    Rates are floats in ``[0, 1]``. ``observed_incremental_lift`` is
    ``observed_recovery_rate_for_action - baseline_control_recovery_rate`` -- the
    exact same subtraction ``recovery_impact.incremental_recovery_rate`` performs,
    only grouped by action type instead of by the whole treated arm.

    ``computable`` is ``False`` (numeric estimates ``None``, ``reason`` set) when
    the control group is empty or the action has fewer than
    ``MIN_DISTINCT_EVENTS_PER_ACTION`` historical uses -- the same small-sample
    discipline enforced on the batch endpoint, so a rate is never fabricated from
    a near-empty sample.
    """

    action_type: str
    computable: bool
    reason: str | None = None

    treated_group_size: int          # distinct RecoveryEvents with this action executed
    control_group_size: int          # is_control events (the global control baseline)
    recovered_treated_events: int
    recovered_control_events: int

    observed_recovery_rate_for_action: float | None
    baseline_control_recovery_rate: float | None
    observed_incremental_lift: float | None
    observed_incremental_lift_ci_95: list[float] | None  # [low, high], Newcombe/Wilson

    sample_size_note: str
    confidence_method: str  # "newcombe_wilson_95_difference" | "not_computed"


class RecoveryImpactResponse(BaseModel):
    """Measured money recovered across a batch: the treated group's recovery
    performance above the randomized control baseline.

    Monetary fields on THIS endpoint are integer **paise** (not Decimal rupees)
    to match the metric the project's problem statement asks for. Rates are
    floats in ``[0, 1]``.

    ``computable`` is ``False`` (with ``reason`` set and the numeric estimates
    ``None``) when there is no control group or no treated group to compare --
    the endpoint never divides by zero or fabricates a rate.
    """

    computable: bool
    reason: str | None = None
    filters: dict  # {"since": str | None, "experiment_id": int | None}

    control_group_size: int
    treated_group_size: int
    recovered_control_events: int
    recovered_treated_events: int

    control_recovery_rate: float | None
    treated_recovery_rate: float | None
    incremental_recovery_rate: float | None  # treated_recovery_rate - control_recovery_rate
    incremental_recovery_rate_ci_95: list[float] | None  # [low, high], Newcombe/Wilson

    control_at_risk_amount_paise: int
    treated_at_risk_amount_paise: int
    control_recovered_revenue_paise: int
    treated_recovered_revenue_paise: int
    total_recovered_revenue_paise: int  # == treated_recovered_revenue_paise (the interveneable pool)
    incremental_revenue_recovered_paise: int | None

    confidence_note: str
    confidence_method: str  # "newcombe_wilson_95_difference" | "not_computed"
