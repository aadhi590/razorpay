from datetime import datetime

from pydantic import BaseModel, ConfigDict


class OutcomeCreate(BaseModel):
    intervention_id: int
    payment_recovered: bool = False
    recovered_amount_paise: int = 0
    recovery_time_seconds: int | None = None


class OutcomeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    intervention_id: int
    payment_recovered: bool
    recovered_amount_paise: int
    recovery_time_seconds: int | None
    observed_at: datetime


class OutcomeListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    intervention_id: int
    payment_recovered: bool
    recovered_amount_paise: int
