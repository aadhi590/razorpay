from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.schemas.outcome import (
    OutcomeCreate,
    OutcomeListResponse,
    OutcomeResponse,
)

router = APIRouter(
    prefix="/api/v1/outcomes",
    tags=["outcomes"],
)


@router.post(
    "/",
    response_model=OutcomeResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_outcome(
    payload: OutcomeCreate,
    db: Session = Depends(get_db),
) -> Outcome:
    intervention = db.get(Intervention, payload.intervention_id)
    if intervention is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Intervention not found.",
        )

    outcome = Outcome(**payload.model_dump())
    db.add(outcome)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Outcome already exists for this intervention.",
        )
    db.refresh(outcome)
    return outcome


@router.get(
    "/",
    response_model=list[OutcomeListResponse],
    status_code=status.HTTP_200_OK,
)
def list_outcomes(db: Session = Depends(get_db)) -> list[Outcome]:
    return list(db.scalars(select(Outcome)).all())


@router.get(
    "/{outcome_id}",
    response_model=OutcomeResponse,
    status_code=status.HTTP_200_OK,
)
def get_outcome(
    outcome_id: int,
    db: Session = Depends(get_db),
) -> Outcome:
    outcome = db.get(Outcome, outcome_id)
    if outcome is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Outcome not found.",
        )
    return outcome
