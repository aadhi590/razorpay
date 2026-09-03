from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ExperimentCreate(BaseModel):
    name: str
    intervention_type: str
    control_percentage: int = 50
    treatment_percentage: int = 50
    status: str = "active"
    started_at: datetime | None = None
    ended_at: datetime | None = None


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    intervention_type: str
    control_percentage: int
    treatment_percentage: int
    status: str
    started_at: datetime
    ended_at: datetime | None


class ExperimentListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    intervention_type: str
    status: str
