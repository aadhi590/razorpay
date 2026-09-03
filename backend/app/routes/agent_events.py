from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.agent_events import AgentEvent
from app.models.recovery_events import RecoveryEvent
from app.schemas.agent_event import (
    AgentEventCreate,
    AgentEventListResponse,
    AgentEventResponse,
)

router = APIRouter(
    prefix="/api/v1/agent-events",
    tags=["agent-events"],
)


@router.post(
    "/",
    response_model=AgentEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_agent_event(
    payload: AgentEventCreate,
    db: Session = Depends(get_db),
) -> AgentEvent:
    recovery_event = db.get(RecoveryEvent, payload.recovery_event_id)
    if recovery_event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recovery event not found.",
        )

    agent_event = AgentEvent(**payload.model_dump())
    db.add(agent_event)
    db.commit()
    db.refresh(agent_event)
    return agent_event


@router.get(
    "/",
    response_model=list[AgentEventListResponse],
    status_code=status.HTTP_200_OK,
)
def list_agent_events(db: Session = Depends(get_db)) -> list[AgentEvent]:
    return list(db.scalars(select(AgentEvent)).all())


@router.get(
    "/{agent_event_id}",
    response_model=AgentEventResponse,
    status_code=status.HTTP_200_OK,
)
def get_agent_event(
    agent_event_id: int,
    db: Session = Depends(get_db),
) -> AgentEvent:
    agent_event = db.get(AgentEvent, agent_event_id)
    if agent_event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent event not found.",
        )
    return agent_event
