"""Read-only uplift / causal-recovery inference endpoints.

Mirrors ``app/routes/ml.py``: never writes to the database, never triggers
training. Returns the baseline (control) probability, per-action treatment
probability, uplift, and incremental economic value for a recovery event -- the
shape a future AI agent will call.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.uplift.config import TREATMENT_ACTIONS
from app.ml.uplift.inference.predictor import UpliftModel
from app.models.interventions import Intervention
from app.models.recovery_events import RecoveryEvent
from app.schemas.uplift import (
    ActionUpliftSchema,
    UpliftModelInfoResponse,
    UpliftScoresResponse,
)

router = APIRouter(prefix="/api/v1/uplift", tags=["uplift"])

_SYNTHETIC_NOTE = (
    "SYNTHETIC BENCHMARK: baseline and uplift come from meta-learners trained on "
    "randomized-assignment synthetic data with a known outcome process. The "
    "control arm is small, so treat these as illustrative, not production "
    "causal estimates. See app/ml/uplift/README.md."
)


@router.get("/model", response_model=UpliftModelInfoResponse)
def uplift_model_info() -> UpliftModelInfoResponse:
    model = UpliftModel.try_load()
    if model is None:
        return UpliftModelInfoResponse(
            available=False,
            detail="no uplift artifact; run `python -m app.scripts.train_uplift_model`",
        )
    a = model.artifact
    return UpliftModelInfoResponse(
        available=True,
        model_name=a.model_name,
        model_version=a.model_version,
        learner_type=a.learner_type,
        base_algorithm=a.base_algorithm,
        feature_version=a.feature_version,
        created_at=a.created_at,
        champion_reason=a.champion_reason,
        dataset=a.dataset,
        propensity_diagnostics=a.propensity_diagnostics,
        test_evaluation=a.evaluation.get("test"),
        limitations=a.limitations,
    )


@router.get(
    "/recovery-events/{recovery_event_id}/uplift-scores",
    response_model=UpliftScoresResponse,
)
def uplift_scores(
    recovery_event_id: int,
    db: Session = Depends(get_db),
) -> UpliftScoresResponse:
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
    untried = [a for a in TREATMENT_ACTIONS if a not in tried]
    as_of = datetime.now(timezone.utc)

    model = UpliftModel.try_load()
    if model is None:
        return UpliftScoresResponse(
            recovery_event_id=recovery_event_id,
            available=False,
            model_version=None,
            as_of=as_of.isoformat(),
            baseline_probability=None,
            amount_paise=None,
            untried_actions=untried,
            actions=[],
            recommended_action=None,
            note="uplift artifact unavailable; the orchestrator would use the ML/rules policy",
        )

    if not untried:
        return UpliftScoresResponse(
            recovery_event_id=recovery_event_id,
            available=True,
            model_version=model.version,
            as_of=as_of.isoformat(),
            baseline_probability=None,
            amount_paise=None,
            untried_actions=[],
            actions=[],
            recommended_action=None,
            note="all actions already attempted on this event",
        )

    try:
        result = model.predict_for_event(
            db.connection(), recovery_event_id, actions=untried, as_of=as_of
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"could not score event: {exc}",
        )

    d = result.as_dict()
    return UpliftScoresResponse(
        recovery_event_id=recovery_event_id,
        available=True,
        model_version=result.model_version,
        as_of=as_of.isoformat(),
        baseline_probability=d["baseline_probability"],
        amount_paise=d["amount_paise"],
        untried_actions=untried,
        actions=[ActionUpliftSchema(**a) for a in d["actions"]],
        recommended_action=result.recommended_action,
        note=_SYNTHETIC_NOTE,
    )
