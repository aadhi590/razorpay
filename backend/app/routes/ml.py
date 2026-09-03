"""Read-only ML inference endpoints.

These expose the recovery-response model as a quantitative tool (the shape a
future AI agent will call). They never write to the database and never trigger
training.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.config import ACTIONS
from app.ml.inference.predictor import RecoveryModel
from app.models.interventions import Intervention
from app.models.recovery_events import RecoveryEvent
from app.schemas.ml import (
    ActionScoreSchema,
    ActionScoresResponse,
    ModelInfoResponse,
)

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])

_SYNTHETIC_NOTE = (
    "SYNTHETIC BENCHMARK: probabilities come from a model trained on generated "
    "data with a known outcome process; treat as illustrative, not production."
)


@router.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = RecoveryModel.try_load()
    if model is None:
        return ModelInfoResponse(
            available=False,
            detail="no model artifact; run `python -m app.scripts.train_recovery_model`",
        )
    a = model.artifact
    return ModelInfoResponse(
        available=True,
        model_name=a.model_name,
        model_version=a.model_version,
        algorithm=a.algorithm,
        feature_version=a.feature_version,
        created_at=a.created_at,
        selected_reason=a.selected_reason,
        dataset=a.dataset,
        test_metrics=a.test_metrics,
        calibration={
            "method": a.calibration.get("method"),
            "test_calibrated": a.calibration.get("test_calibrated", {}).get(
                "expected_calibration_error"
            ),
        },
    )


@router.get(
    "/recovery-events/{recovery_event_id}/action-scores",
    response_model=ActionScoresResponse,
)
def action_scores(
    recovery_event_id: int,
    db: Session = Depends(get_db),
) -> ActionScoresResponse:
    event = db.get(RecoveryEvent, recovery_event_id)
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recovery event not found.",
        )

    tried = set(
        db.scalars(
            select(Intervention.action_type).where(
                Intervention.recovery_event_id == recovery_event_id
            )
        )
    )
    untried = [a for a in ACTIONS if a not in tried]
    as_of = datetime.now(timezone.utc)

    model = RecoveryModel.try_load()
    if model is None:
        return ActionScoresResponse(
            recovery_event_id=recovery_event_id,
            model_available=False,
            model_version=None,
            as_of=as_of.isoformat(),
            untried_actions=untried,
            scores=[],
            recommended_action=None,
            note="model artifact unavailable; the orchestrator would use the rules policy",
        )

    if not untried:
        return ActionScoresResponse(
            recovery_event_id=recovery_event_id,
            model_available=True,
            model_version=model.version,
            as_of=as_of.isoformat(),
            untried_actions=[],
            scores=[],
            recommended_action=None,
            note="all actions already attempted on this event",
        )

    try:
        scored = model.predict_for_event(
            db.connection(), recovery_event_id, actions=untried, as_of=as_of
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"could not score event: {exc}",
        )

    ranked = sorted(
        scored.values(), key=lambda s: (-s.expected_value_paise, s.cost_paise)
    )
    return ActionScoresResponse(
        recovery_event_id=recovery_event_id,
        model_available=True,
        model_version=model.version,
        as_of=as_of.isoformat(),
        untried_actions=untried,
        scores=[
            ActionScoreSchema(
                action=s.action,
                probability=round(s.probability, 6),
                cost_paise=s.cost_paise,
                expected_value_paise=s.expected_value_paise,
            )
            for s in ranked
        ],
        recommended_action=ranked[0].action if ranked else None,
        note=_SYNTHETIC_NOTE,
    )
