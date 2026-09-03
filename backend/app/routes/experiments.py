from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.experiment import Experiment
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentResponse,
)

router = APIRouter(
    prefix="/api/v1/experiments",
    tags=["experiments"],
)


@router.post(
    "/",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_experiment(
    payload: ExperimentCreate,
    db: Session = Depends(get_db),
) -> Experiment:
    data = payload.model_dump(exclude_none=True)
    experiment = Experiment(**data)
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


@router.get(
    "/",
    response_model=list[ExperimentListResponse],
    status_code=status.HTTP_200_OK,
)
def list_experiments(db: Session = Depends(get_db)) -> list[Experiment]:
    return list(db.scalars(select(Experiment)).all())


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    status_code=status.HTTP_200_OK,
)
def get_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
) -> Experiment:
    experiment = db.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found.",
        )
    return experiment
