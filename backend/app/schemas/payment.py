from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PaymentCreate(BaseModel):
    subscription_id: int
    amount: int
    currency: str = "INR"
    status: str
    failure_reason: str | None = None
    failed_at: datetime | None = None


class PaymentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    subscription_id: int
    amount: int
    currency: str
    status: str
    failure_reason: str | None
    failed_at: datetime | None
    retry_count: int
    recovered_at: datetime | None


class PaymentListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    subscription_id: int
    amount: int
    currency: str
    status: str
    retry_count: int
