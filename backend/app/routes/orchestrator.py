from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.orchestrator import OrchestratorRunResponse
from app.services.policy_factory import (
    DEFAULT_POLICY,
    POLICY_NAMES,
    resolve_assigner,
    resolve_policy,
)
from app.services.recovery_orchestrator import RecoveryOrchestratorService

router = APIRouter(
    prefix="/api/v1/orchestrator",
    tags=["orchestrator"],
)


@router.post(
    "/run",
    response_model=OrchestratorRunResponse,
    status_code=status.HTTP_200_OK,
)
def run_orchestrator(
    limit: int | None = Query(
        default=None,
        ge=1,
        description="Max number of eligible recovery events to process.",
    ),
    policy: str = Query(
        default=DEFAULT_POLICY,
        description=f"Decision policy: one of {POLICY_NAMES}.",
    ),
    epsilon: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Exploration rate for the action-assignment layer. 0 (default) = "
            "pure exploitation of the top-ranked candidate (unchanged "
            "behaviour). >0 enables epsilon-greedy exploration and records the "
            "assignment propensity on every intervention."
        ),
    ),
    experiment_id: str | None = Query(
        default=None,
        description="Logical experiment id recorded on each assignment.",
    ),
    assignment_seed: int | None = Query(
        default=None,
        description="Deterministic seed for the exploration RNG (reproducible runs).",
    ),
    db: Session = Depends(get_db),
) -> OrchestratorRunResponse:
    """Process eligible open, non-control recovery events (highest priority
    first). Each event is orchestrated in its own transaction."""
    try:
        chosen_policy = resolve_policy(policy, db)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    assigner = resolve_assigner(
        epsilon=epsilon, experiment_id=experiment_id, seed=assignment_seed
    )
    service = RecoveryOrchestratorService(
        db, policy=chosen_policy, assigner=assigner
    )
    result = service.run_batch(limit=limit)
    return OrchestratorRunResponse(
        considered=result.considered,
        interventions_created=result.interventions_created,
        closed=result.closed,
        abandoned=result.abandoned,
        skipped=result.skipped,
        errors=result.errors,
        results=[outcome.as_response_dict() for outcome in result.outcomes],
    )
