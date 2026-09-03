from __future__ import annotations

from pydantic import BaseModel


class CandidateActionSchema(BaseModel):
    action_type: str
    cost_paise: int
    estimated_recovery_probability: float
    expected_value_paise: float
    score: float
    confidence: float
    reason: str


class OrchestrateResponse(BaseModel):
    recovery_event_id: int
    payment_id: int
    is_control: bool
    variant: str | None
    priority: int

    previous_status: str
    recovery_event_status: str  # resulting state
    disposition: str

    action_taken: bool
    attempt_number: int | None
    selected_action: str | None
    decision_reason: str
    confidence: float | None

    intervention_id: int | None
    agent_event_id: int | None

    candidates: list[CandidateActionSchema]

    # Action-assignment / experimentation metadata. Defaults preserve the
    # pre-experimentation response shape: pure exploitation, propensity 1.0.
    propensity: float | None = None
    exploration: bool = False
    assignment: dict | None = None

    error: str | None = None


class OrchestratorRunResponse(BaseModel):
    considered: int
    interventions_created: int
    closed: int
    abandoned: int
    skipped: int
    errors: int
    results: list[OrchestrateResponse]
