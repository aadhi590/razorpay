"""Recovery Scheduler status + demo trigger.

    GET  /api/v1/scheduler/status     -- enabled state, config, recent cycles
    POST /api/v1/scheduler/run-once   -- run exactly one cycle now (demo/ops)

The scheduler itself (``app.services.recovery_scheduler.scheduler``) is a
process-wide singleton started/stopped by the FastAPI lifespan in
``app/main.py`` when ``SCHEDULER_ENABLED`` is true. These endpoints only observe
it and, for ``run-once``, drive one cycle synchronously through the *same*
``run_cycle`` the timer uses.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.scheduler import (
    SchedulerRunOnceResponse,
    SchedulerStatusResponse,
)
from app.services.recovery_scheduler import TRIGGER_RUN_ONCE, scheduler

router = APIRouter(prefix="/api/v1/scheduler", tags=["scheduler"])


@router.get(
    "/status",
    response_model=SchedulerStatusResponse,
    status_code=status.HTTP_200_OK,
)
def scheduler_status() -> SchedulerStatusResponse:
    """Whether the scheduler is enabled and running, its effective config, and
    the most recent cycles (newest first): timestamps, how many open eligible
    events were considered, how many agent runs were triggered, and every skip
    with its precise reason."""
    return SchedulerStatusResponse(**scheduler.status())


@router.post(
    "/run-once",
    response_model=SchedulerRunOnceResponse,
    status_code=status.HTTP_200_OK,
)
def scheduler_run_once(
    db: Session = Depends(get_db),
) -> SchedulerRunOnceResponse:
    """Run one scheduler cycle immediately and return its full record.

    This is the exact code path the periodic timer runs -- it asks the
    PortfolioAllocator for the ranked "act" set (capped at
    ``SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE``) and triggers real agent runs on those
    events, tagged ``triggered_by="scheduler"``. It works regardless of
    ``SCHEDULER_ENABLED`` (that flag only controls the background timer), and
    respects ``SCHEDULER_DRY_RUN`` (default: simulate, persist no Intervention).
    """
    record = scheduler.run_cycle(db=db, trigger=TRIGGER_RUN_ONCE)
    return SchedulerRunOnceResponse(
        enabled=scheduler.config.enabled,
        cycle=record.as_dict(),
    )
