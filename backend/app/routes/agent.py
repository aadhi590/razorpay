"""Gemini recovery-agent endpoint.

    POST /api/v1/agent/recovery-events/{recovery_event_id}/run?dry_run=true|false

Runs the autonomous, tool-driven Gemini agent for one recovery event and
returns the full turn-by-turn tool trace alongside the final decision.

This does NOT replace the existing policy endpoints -- ``/orchestrate?policy=``
and ``/api/v1/orchestrator/run`` are untouched.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.agent.runner import AgentRunError, run_recovery_agent
from app.agent.schemas import AgentRunResult
from app.database import get_db

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post(
    "/recovery-events/{recovery_event_id}/run",
    response_model=AgentRunResult,
    status_code=status.HTTP_200_OK,
)
def run_agent(
    recovery_event_id: int,
    dry_run: bool = Query(
        default=True,
        description=(
            "true (default): read-only tools run normally, action execution is "
            "simulated and nothing is persisted to the recovery workflow. "
            "false: an eligible action, if chosen, is persisted as a pending "
            "Intervention (still no external payment call or customer message)."
        ),
    ),
    db: Session = Depends(get_db),
) -> AgentRunResult:
    try:
        return run_recovery_agent(
            db, recovery_event_id, dry_run=dry_run, triggered_by="manual"
        )
    except AgentRunError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
            if exc.code == "event_not_found"
            else status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message,
        )
