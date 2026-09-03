from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.schemas.orchestrator import OrchestrateResponse
from app.schemas.recovery_events import (
    RecoveryEventCreate,
    RecoveryEventListResponse,
    RecoveryEventResponse,
)
from app.services.policy_factory import (
    DEFAULT_POLICY,
    POLICY_NAMES,
    resolve_assigner,
    resolve_policy,
)
from app.services.recovery_orchestrator import (
    RecoveryEventNotFound,
    RecoveryOrchestratorService,
)

router = APIRouter(
    prefix="/api/v1/recovery-events",
    tags=["recovery-events"],
)


@router.post(
    "/",
    response_model=RecoveryEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_recovery_event(
    payload: RecoveryEventCreate,
    db: Session = Depends(get_db),
) -> RecoveryEvent:
    payment = db.get(Payment, payload.payment_id)
    if payment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found.",
        )

    recovery_event = RecoveryEvent(**payload.model_dump())
    db.add(recovery_event)
    db.commit()
    db.refresh(recovery_event)
    return recovery_event


@router.get(
    "/",
    response_model=list[RecoveryEventListResponse],
    status_code=status.HTTP_200_OK,
)
def list_recovery_events(db: Session = Depends(get_db)) -> list[RecoveryEvent]:
    return list(db.scalars(select(RecoveryEvent)).all())


@router.get(
    "/{recovery_event_id}",
    response_model=RecoveryEventResponse,
    status_code=status.HTTP_200_OK,
)
def get_recovery_event(
    recovery_event_id: int,
    db: Session = Depends(get_db),
) -> RecoveryEvent:
    recovery_event = db.get(RecoveryEvent, recovery_event_id)
    if recovery_event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recovery event not found.",
        )
    return recovery_event


@router.post(
    "/{recovery_event_id}/orchestrate",
    response_model=OrchestrateResponse,
    status_code=status.HTTP_200_OK,
)
def orchestrate_recovery_event(
    recovery_event_id: int,
    policy: str = Query(
        default=DEFAULT_POLICY,
        description=f"Decision policy: one of {POLICY_NAMES}. "
        "'ml' falls back to 'rules' if the model artifact is unavailable.",
    ),
    epsilon: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Exploration rate for the action-assignment layer. 0 (default) = "
            "pure exploitation (unchanged behaviour); >0 enables epsilon-greedy "
            "exploration and records the assignment propensity."
        ),
    ),
    experiment_id: str | None = Query(
        default=None,
        description="Logical experiment id recorded on the assignment.",
    ),
    assignment_seed: int | None = Query(
        default=None,
        description="Deterministic seed for the exploration RNG.",
    ),
    db: Session = Depends(get_db),
) -> OrchestrateResponse:
    """Run the recovery orchestrator for a single recovery event: decide
    whether/what intervention to attempt, persist the resulting Intervention /
    AgentEvent / AuditLog records, and update the recovery event state."""
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
    try:
        outcome = service.orchestrate_event(recovery_event_id)
    except RecoveryEventNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recovery event not found.",
        )
    return OrchestrateResponse(**outcome.as_response_dict())
