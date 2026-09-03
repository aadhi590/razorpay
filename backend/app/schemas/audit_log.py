from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AuditLogCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    recovery_event_id: int | None = None
    actor: str
    action: str
    reason: str | None = None
    metadata: dict | None = Field(default=None, alias="event_metadata")


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    recovery_event_id: int | None
    actor: str
    action: str
    reason: str | None
    metadata: dict | None = Field(default=None, alias="event_metadata")
    created_at: datetime


class AuditLogListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    actor: str
    action: str
    created_at: datetime
