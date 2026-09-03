from dataclasses import asdict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.analytics import (
    ActionsResponse,
    AssignmentCoverageResponse,
    ControlVsTreatmentResponse,
    ExperimentsResponse,
    PortfolioAllocationResponse,
    RecoveryImpactResponse,
    SummaryResponse,
)
from app.services.analytics_service import AnalyticsService
from app.services.policy_factory import DEFAULT_POLICY, POLICY_NAMES
from app.services.portfolio_allocator import PortfolioAllocator

router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["analytics"],
)


@router.get(
    "/summary",
    response_model=SummaryResponse,
    status_code=status.HTTP_200_OK,
)
def analytics_summary(db: Session = Depends(get_db)) -> SummaryResponse:
    """Overall recovery performance: event counts by state, recovered vs
    failed value, overall recovery rate, intervention/outcome totals."""
    return AnalyticsService(db).summary()


@router.get(
    "/control-vs-treatment",
    response_model=ControlVsTreatmentResponse,
    status_code=status.HTTP_200_OK,
)
def analytics_control_vs_treatment(
    db: Session = Depends(get_db),
) -> ControlVsTreatmentResponse:
    """Recovery rate / value for the control and treatment groups, plus the
    absolute and relative lift of treatment over control."""
    return AnalyticsService(db).control_vs_treatment()


@router.get(
    "/actions",
    response_model=ActionsResponse,
    status_code=status.HTTP_200_OK,
)
def analytics_actions(db: Session = Depends(get_db)) -> ActionsResponse:
    """Intervention performance grouped by action_type: volume, recovery rate,
    recovered value, spend and cost-per-recovery."""
    return AnalyticsService(db).actions()


@router.get(
    "/experiments",
    response_model=ExperimentsResponse,
    status_code=status.HTTP_200_OK,
)
def analytics_experiments(db: Session = Depends(get_db)) -> ExperimentsResponse:
    """Results grouped by experiment and variant, with treatment-vs-control
    lift per experiment where both variants are present."""
    return AnalyticsService(db).experiments()


@router.get(
    "/recovery-impact",
    response_model=RecoveryImpactResponse,
    status_code=status.HTTP_200_OK,
)
def analytics_recovery_impact(
    since: datetime | None = Query(
        default=None,
        description=(
            "ISO date/datetime. Only count RecoveryEvents with "
            "created_at >= since."
        ),
    ),
    experiment_id: int | None = Query(
        default=None,
        ge=1,
        description=(
            "Filter to RecoveryEvents whose experiment_id matches (a real "
            "indexed FK column). NOTE: the synthetic generator assigns "
            "experiment_id uniformly at random and independently of the "
            "control/treatment split, so a per-experiment slice is expected to "
            "track the global lift within sampling noise -- it is a real "
            "filter, not a designed experimental contrast."
        ),
    ),
    db: Session = Depends(get_db),
) -> RecoveryImpactResponse:
    """Measured money recovered across a batch: the treated group's recovery
    rate and revenue above the randomized control baseline, with a
    Newcombe/Wilson 95% interval on the incremental rate.

    Control vs treatment = ``RecoveryEvent.is_control``. Returns
    ``computable: false`` (numbers ``null``, never divide-by-zero) when the
    control or treated group is empty. See
    ``AnalyticsService.recovery_impact`` for the exact incremental-revenue
    formula."""
    return AnalyticsService(db).recovery_impact(
        since=since, experiment_id=experiment_id
    )


@router.get(
    "/portfolio-allocation",
    response_model=PortfolioAllocationResponse,
    status_code=status.HTTP_200_OK,
)
def analytics_portfolio_allocation(
    capacity: int | None = Query(
        default=None,
        ge=1,
        description=(
            "How many of the currently-open eligible recovery events can be "
            "acted on this batch. Optional; a small sensible default is used "
            "when omitted."
        ),
    ),
    policy: str = Query(
        default=DEFAULT_POLICY,
        description=f"Scoring policy for the expected-value ranking: one of {POLICY_NAMES}.",
    ),
    db: Session = Depends(get_db),
) -> PortfolioAllocationResponse:
    """READ-ONLY. Rank the batch of currently-open eligible recovery events by
    expected value (net of action cost, from the same policy scoring the
    orchestrator uses), apply the capacity constraint, and report exactly which
    events to act on, which to skip and precisely why, plus the expected value
    the capacity limit costs you.

    Executes nothing: no Intervention is created, no state is mutated. Acting on
    an "act"-ranked event still goes through the normal agent-run endpoint and
    the existing guardrails. Returns ``computable: false`` when no open
    non-control event currently passes the guardrail actionability check."""
    try:
        allocator = PortfolioAllocator(db, policy_name=policy)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    result = allocator.allocate(capacity=capacity)
    return PortfolioAllocationResponse(**asdict(result))


@router.get(
    "/assignment-coverage",
    response_model=AssignmentCoverageResponse,
    status_code=status.HTTP_200_OK,
)
def analytics_assignment_coverage(
    db: Session = Depends(get_db),
) -> AssignmentCoverageResponse:
    """Action-assignment coverage for causal/uplift readiness: per-action
    volume, distinct events/customers, observed recovery rate, and the
    assignment-propensity distribution, plus overlap/positivity warnings."""
    return AnalyticsService(db).assignment_coverage()
