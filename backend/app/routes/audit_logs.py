from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.recovery_events import RecoveryEvent
from app.schemas.audit_log import (
    AuditLogCreate,
    AuditLogListResponse,
    AuditLogResponse,
)

router = APIRouter(
    prefix="/api/v1/audit-logs",
    tags=["audit-logs"],
)


@router.post(
    "/",
    response_model=AuditLogResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_audit_log(
    payload: AuditLogCreate,
    db: Session = Depends(get_db),
) -> AuditLog:
    if payload.recovery_event_id is not None:
        recovery_event = db.get(RecoveryEvent, payload.recovery_event_id)
        if recovery_event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Recovery event not found.",
            )

    data = payload.model_dump(by_alias=True)
    audit_log = AuditLog(**data)
    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)
    return audit_log


@router.get(
    "/",
    response_model=list[AuditLogListResponse],
    status_code=status.HTTP_200_OK,
)
def list_audit_logs(db: Session = Depends(get_db)) -> list[AuditLog]:
    return list(db.scalars(select(AuditLog)).all())


@router.get(
    "/{audit_log_id}",
    response_model=AuditLogResponse,
    status_code=status.HTTP_200_OK,
)
def get_audit_log(
    audit_log_id: int,
    db: Session = Depends(get_db),
) -> AuditLog:
    audit_log = db.get(AuditLog, audit_log_id)
    if audit_log is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit log not found.",
        )
    return audit_log
