from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CustomerCreate(BaseModel):
    external_customer_id: str
    email: str | None = None
    phone: str | None = None


class CustomerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_customer_id: str
    email: str | None
    phone: str | None
    total_successful_payments: int
    total_failed_payments: int
    created_at: datetime


class CustomerListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_customer_id: str
    email: str | None
    phone: str | None
    total_successful_payments: int
    total_failed_payments: int
