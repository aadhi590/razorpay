from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RecoveryEventCreate(BaseModel):
    payment_id: int
    status: str = "open"
    priority: int = 0


class RecoveryEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    payment_id: int
    status: str
    priority: int
    created_at: datetime
    closed_at: datetime | None


class RecoveryEventListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    payment_id: int
    status: str
    priority: int
    created_at: datetime
