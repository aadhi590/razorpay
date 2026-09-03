from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AgentEventCreate(BaseModel):
    recovery_event_id: int
    event_type: str
    input_context: dict | None = None
    decision: str | None = None
    confidence: float | None = None


class AgentEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    recovery_event_id: int
    event_type: str
    input_context: dict | None
    decision: str | None
    confidence: float | None
    created_at: datetime


class AgentEventListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    recovery_event_id: int
    event_type: str
    decision: str | None
    confidence: float | None
