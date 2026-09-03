from datetime import datetime

from pydantic import BaseModel, ConfigDict


class InterventionCreate(BaseModel):
    recovery_event_id: int
    action_type: str
    status: str = "pending"
    cost_paise: int = 0
    agent_reason: str | None = None
    executed_at: datetime | None = None


class InterventionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    recovery_event_id: int
    action_type: str
    status: str
    cost_paise: int
    agent_reason: str | None
    executed_at: datetime | None


class InterventionListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    recovery_event_id: int
    action_type: str
    status: str
    cost_paise: int
