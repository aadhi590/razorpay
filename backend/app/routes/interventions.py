from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.interventions import Intervention
from app.models.recovery_events import RecoveryEvent
from app.schemas.interventions import (
    InterventionCreate,
    InterventionListResponse,
    InterventionResponse,
)

router = APIRouter(
    prefix="/api/v1/interventions",
    tags=["interventions"],
)


@router.post(
    "/",
    response_model=InterventionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_intervention(
    payload: InterventionCreate,
    db: Session = Depends(get_db),
) -> Intervention:
    recovery_event = db.get(RecoveryEvent, payload.recovery_event_id)
    if recovery_event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recovery event not found.",
        )

    intervention = Intervention(**payload.model_dump())
    db.add(intervention)
    db.commit()
    db.refresh(intervention)
    return intervention


@router.get(
    "/",
    response_model=list[InterventionListResponse],
    status_code=status.HTTP_200_OK,
)
def list_interventions(db: Session = Depends(get_db)) -> list[Intervention]:
    return list(db.scalars(select(Intervention)).all())


@router.get(
    "/{intervention_id}",
    response_model=InterventionResponse,
    status_code=status.HTTP_200_OK,
)
def get_intervention(
    intervention_id: int,
    db: Session = Depends(get_db),
) -> Intervention:
    intervention = db.get(Intervention, intervention_id)
    if intervention is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Intervention not found.",
        )
    return intervention
