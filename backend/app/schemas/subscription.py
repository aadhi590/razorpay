from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SubscriptionCreate(BaseModel):
    customer_id: int
    external_subscription_id: str
    amount: int
    currency: str = "INR"
    status: str = "active"
    started_at: datetime
    next_payment_at: datetime | None = None


class SubscriptionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    customer_id: int
    external_subscription_id: str
    amount: int
    currency: str
    status: str
    started_at: datetime
    next_payment_at: datetime | None


class SubscriptionListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    customer_id: int
    external_subscription_id: str
    amount: int
    currency: str
    status: str
