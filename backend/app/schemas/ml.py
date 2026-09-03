from __future__ import annotations

from pydantic import BaseModel


class ActionScoreSchema(BaseModel):
    action: str
    probability: float          # P(recovery | context, action), calibrated
    cost_paise: int
    expected_value_paise: float  # probability * payment_amount - cost


class ActionScoresResponse(BaseModel):
    recovery_event_id: int
    model_available: bool
    model_version: str | None
    as_of: str
    untried_actions: list[str]
    scores: list[ActionScoreSchema]        # ranked best-first by expected_value_paise
    recommended_action: str | None
    note: str


class ModelInfoResponse(BaseModel):
    available: bool
    model_name: str | None = None
    model_version: str | None = None
    algorithm: str | None = None
    feature_version: str | None = None
    created_at: str | None = None
    selected_reason: str | None = None
    dataset: dict | None = None
    test_metrics: dict | None = None
    calibration: dict | None = None
    synthetic_benchmark: bool = True
    detail: str | None = None
