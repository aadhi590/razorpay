from __future__ import annotations

from pydantic import BaseModel, Field


class SchedulerConfigView(BaseModel):
    interval_seconds: float
    max_auto_runs_per_cycle: int
    dry_run: bool
    policy: str
    cycle_history_size: int
    ai_judgment_enabled: bool = False


class CycleJudgmentView(BaseModel):
    """The AI cycle-judgment decision for one cycle. ``effective_capacity`` is
    always ``<= configured_cap`` -- the judgment can only ever reduce it."""

    enabled: bool
    decision: str
    source: str  # disabled | gemini | fail_safe_default
    configured_cap: int
    suggested_capacity: int | None
    effective_capacity: int
    applied: bool
    reason: str
    error: str | None = None


class TriggeredRunView(BaseModel):
    recovery_event_id: int
    rank: int | None
    best_action: str | None
    expected_value_paise: float | None
    run_status: str
    decision: str | None
    stop_reason: str | None
    chosen_action: str | None
    error: str | None = None


class SkippedEventView(BaseModel):
    recovery_event_id: int
    rank: int | None
    reason: str


class CycleRecordView(BaseModel):
    cycle_id: int
    trigger: str
    started_at: str
    finished_at: str
    enabled: bool
    dry_run: bool
    policy: str
    hard_cap: int
    effective_cap: int
    ai_judgment: CycleJudgmentView | None = None
    allocation_computable: bool
    allocation_reason: str | None
    events_considered: int
    events_ranked: int
    act_set_size: int
    events_triggered: int
    eligible_beyond_cap: int
    hard_cap_enforced: bool
    expected_value_captured_paise: float
    expected_value_forgone_to_capacity_paise: float
    triggered: list[TriggeredRunView] = Field(default_factory=list)
    skipped: list[SkippedEventView] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SchedulerStatusResponse(BaseModel):
    enabled: bool
    running: bool
    config: SchedulerConfigView
    cycles_run: int
    last_cycle_at: str | None
    next_cycle_estimated_at: str | None
    cycle_history: list[CycleRecordView] = Field(default_factory=list)


class SchedulerRunOnceResponse(BaseModel):
    """The single cycle produced by POST /api/v1/scheduler/run-once."""

    ran: bool = True
    enabled: bool
    cycle: CycleRecordView
