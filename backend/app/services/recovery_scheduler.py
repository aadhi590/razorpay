"""Recovery Scheduler -- bounded, in-process auto-trigger.

The recovery agent and the orchestrator only ever act when something *outside*
them asks: a human hitting ``POST /api/v1/agent/.../run`` or
``POST /api/v1/orchestrator/run``. Nothing makes the system act on its own
backlog on a schedule.

:class:`RecoveryScheduler` is that missing piece, kept deliberately small:

* **No new infrastructure.** No Celery, no Redis, no task queue. One daemon
  ``threading.Thread`` that sleeps ``SCHEDULER_INTERVAL_SECONDS`` between cycles.
* **No new decision logic.** Each cycle asks the existing
  :class:`~app.services.portfolio_allocator.PortfolioAllocator` for the ranked
  "act" set (capacity = ``SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE``) and then runs the
  *exact* existing agent-run code path
  (:func:`app.agent.runner.run_recovery_agent`) on each of those events -- the
  same function the manual endpoint calls. No execution logic is duplicated
  here.
* **Bounded.** At most ``SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE`` agent runs per
  cycle, enforced here *and* by the allocator's capacity argument, even if more
  eligible events exist.
* **Tagged.** Every scheduler-triggered run is persisted with
  ``triggered_by="scheduler"`` in ``AgentEvent.input_context`` and every
  ``AuditLog`` row, distinct from the manual path's ``"manual"``.
* **Off by default.** When ``SCHEDULER_ENABLED`` is false the thread never
  starts and behaviour is identical to before this module existed.

Cycle history (timestamp, events considered, events triggered, and every skip
with its reason) is kept in a bounded in-memory ring and exposed verbatim by
``GET /api/v1/scheduler/status``.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy.orm import Session

from app.agent.providers.base import LLMProvider
from app.agent.runner import AgentRunError, run_recovery_agent
from app.config import settings
from app.database import SessionLocal
from app.services.portfolio_allocator import PortfolioAllocator

logger = logging.getLogger("app.recovery_scheduler")

SCHEDULER_TRIGGER = "scheduler"

TRIGGER_TIMER = "timer"
TRIGGER_RUN_ONCE = "run_once"

ProviderFactory = Callable[[], LLMProvider]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool
    interval_seconds: float
    max_auto_runs_per_cycle: int
    dry_run: bool
    policy: str
    history_size: int

    @classmethod
    def from_settings(cls) -> "SchedulerConfig":
        return cls(
            enabled=bool(settings.SCHEDULER_ENABLED),
            # a zero/negative interval would busy-loop the thread
            interval_seconds=max(1.0, float(settings.SCHEDULER_INTERVAL_SECONDS)),
            max_auto_runs_per_cycle=max(
                0, int(settings.SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE)
            ),
            dry_run=bool(settings.SCHEDULER_DRY_RUN),
            policy=str(settings.SCHEDULER_POLICY or "rules"),
            history_size=max(1, int(settings.SCHEDULER_CYCLE_HISTORY_SIZE)),
        )

    def as_public_dict(self) -> dict:
        return {
            "interval_seconds": self.interval_seconds,
            "max_auto_runs_per_cycle": self.max_auto_runs_per_cycle,
            "dry_run": self.dry_run,
            "policy": self.policy,
            "cycle_history_size": self.history_size,
        }


# ---------------------------------------------------------------------------
# Cycle records
# ---------------------------------------------------------------------------


@dataclass
class TriggeredRun:
    recovery_event_id: int
    rank: int | None
    best_action: str | None
    expected_value_paise: float | None
    # agent-run outcome, straight from run_recovery_agent's result
    run_status: str            # completed | escalated | failed_safe | error
    decision: str | None
    stop_reason: str | None
    chosen_action: str | None
    error: str | None = None


@dataclass
class SkippedEvent:
    recovery_event_id: int
    rank: int | None
    reason: str


@dataclass
class CycleRecord:
    cycle_id: int
    trigger: str                       # timer | run_once
    started_at: str
    finished_at: str
    enabled: bool
    dry_run: bool
    policy: str
    hard_cap: int                      # SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE this cycle
    allocation_computable: bool
    allocation_reason: str | None
    events_considered: int             # open, non-control, guardrail-actionable
    events_ranked: int                 # of those, had >=1 candidate action
    act_set_size: int                  # allocator's "act" set (already <= hard_cap)
    events_triggered: int              # agent runs actually started
    eligible_beyond_cap: int           # positive-EV ranked events left unacted due to the cap
    hard_cap_enforced: bool            # eligible_beyond_cap > 0
    expected_value_captured_paise: float
    expected_value_forgone_to_capacity_paise: float
    triggered: list[TriggeredRun] = field(default_factory=list)
    skipped: list[SkippedEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class RecoveryScheduler:
    def __init__(
        self,
        *,
        config: SchedulerConfig | None = None,
        session_factory: Callable[[], Session] = SessionLocal,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self._config = config or SchedulerConfig.from_settings()
        self._session_factory = session_factory
        # Injected only by tests, to supply a fake LLM provider. When None, the
        # runner builds its normal GeminiProvider (and fails safe without a key).
        self._provider_factory = provider_factory

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._cycle_seq = 0            # monotonic cycle id
        self._cycles_completed = 0     # cycles that finished (started and recorded)
        self._history: deque[CycleRecord] = deque(maxlen=self._config.history_size)
        self._last_cycle_started_at: datetime | None = None

    # -- configuration -------------------------------------------------
    @property
    def config(self) -> SchedulerConfig:
        return self._config

    def reload_config(self, config: SchedulerConfig | None = None) -> None:
        """Re-read settings. Only takes full effect on the next (re)start."""
        with self._lock:
            new = config or SchedulerConfig.from_settings()
            self._config = new
            # keep the ring buffer bound in sync
            if new.history_size != self._history.maxlen:
                self._history = deque(self._history, maxlen=new.history_size)

    # -- lifecycle ---------------------------------------------------
    def start(self) -> bool:
        """Start the periodic thread. No-op (returns False) when the scheduler
        is disabled or already running."""
        with self._lock:
            if not self._config.enabled:
                logger.info("recovery scheduler disabled (SCHEDULER_ENABLED=false)")
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="recovery-scheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "recovery scheduler started: interval=%.0fs cap=%d dry_run=%s policy=%s",
                self._config.interval_seconds,
                self._config.max_auto_runs_per_cycle,
                self._config.dry_run,
                self._config.policy,
            )
            return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        # First cycle fires after one full interval, not at startup, so it never
        # races app initialisation.
        while not self._stop_event.wait(self._config.interval_seconds):
            try:
                self.run_cycle(trigger=TRIGGER_TIMER)
            except Exception:  # noqa: BLE001 - a bad cycle must not kill the loop
                logger.exception("recovery scheduler cycle raised; loop continues")

    # -- one cycle -------------------------------------------------
    def run_cycle(
        self,
        *,
        db: Session | None = None,
        trigger: str = TRIGGER_TIMER,
    ) -> CycleRecord:
        """Run exactly one scheduler cycle and return its record.

        Safe to call directly (the ``run-once`` endpoint and the test-suite do).
        When ``db`` is supplied the caller owns the session; otherwise a session
        is opened and closed here.
        """
        owns_session = db is None
        session = db or self._session_factory()
        with self._lock:
            self._cycle_seq += 1
            cycle_id = self._cycle_seq
        started_at = datetime.now(timezone.utc)
        self._last_cycle_started_at = started_at
        cap = self._config.max_auto_runs_per_cycle

        record = CycleRecord(
            cycle_id=cycle_id,
            trigger=trigger,
            started_at=started_at.isoformat(),
            finished_at=started_at.isoformat(),
            enabled=self._config.enabled,
            dry_run=self._config.dry_run,
            policy=self._config.policy,
            hard_cap=cap,
            allocation_computable=False,
            allocation_reason=None,
            events_considered=0,
            events_ranked=0,
            act_set_size=0,
            events_triggered=0,
            eligible_beyond_cap=0,
            hard_cap_enforced=False,
            expected_value_captured_paise=0.0,
            expected_value_forgone_to_capacity_paise=0.0,
        )

        try:
            self._execute_cycle(session, record, cap)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised, per cycle
            logger.exception("recovery scheduler cycle %d failed", cycle_id)
            record.errors.append(f"cycle aborted: {type(exc).__name__}: {exc}")
            if owns_session:
                session.rollback()
        finally:
            record.finished_at = _utcnow_iso()
            with self._lock:
                self._history.append(record)
                self._cycles_completed += 1
            if owns_session:
                session.close()

        return record

    def _execute_cycle(
        self, session: Session, record: CycleRecord, cap: int
    ) -> None:
        allocator = PortfolioAllocator(session, policy_name=self._config.policy)
        allocation = allocator.allocate(capacity=max(1, cap))

        record.allocation_computable = allocation.computable
        record.allocation_reason = allocation.reason
        record.events_considered = allocation.total_open_eligible_events
        record.events_ranked = allocation.events_ranked
        record.expected_value_captured_paise = (
            allocation.expected_value_captured_paise
        )
        record.expected_value_forgone_to_capacity_paise = (
            allocation.expected_value_forgone_to_capacity_paise
        )

        # Every allocator "skip" is surfaced verbatim -- both the ones below the
        # capacity cutoff and the non-positive-EV / no-action ones.
        for s in allocation.skip:
            record.skipped.append(
                SkippedEvent(
                    recovery_event_id=s.recovery_event_id,
                    rank=s.rank,
                    reason=s.reason,
                )
            )

        # Positive-EV, ranked events that the capacity cap left unacted. This is
        # the concrete count behind "never exceeds the cap even if more eligible
        # events exist".
        record.eligible_beyond_cap = sum(
            1
            for s in allocation.skip
            if s.rank is not None and (s.expected_value_paise or 0.0) > 0.0
        )
        record.hard_cap_enforced = record.eligible_beyond_cap > 0

        if cap <= 0:
            record.act_set_size = 0
            record.errors.append(
                "SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE is 0; nothing is triggered"
            )
            for a in allocation.act:
                record.skipped.append(
                    SkippedEvent(
                        recovery_event_id=a.recovery_event_id,
                        rank=a.rank,
                        reason="hard cap: SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE=0",
                    )
                )
            return

        # Defence in depth: the allocator already capped its "act" set to
        # `capacity`, but slice again so the hard cap is enforced *here* too,
        # independent of the allocator.
        act = list(allocation.act)
        record.act_set_size = len(act)
        to_run = act[:cap]
        for extra in act[cap:]:
            record.skipped.append(
                SkippedEvent(
                    recovery_event_id=extra.recovery_event_id,
                    rank=extra.rank,
                    reason=(
                        f"hard cap: SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE={cap} "
                        "already consumed by higher-ranked events this cycle"
                    ),
                )
            )

        for allocated in to_run:
            record.triggered.append(
                self._trigger_one(session, allocated)
            )
        record.events_triggered = sum(
            1 for t in record.triggered if t.error is None
        )

    def _trigger_one(self, session: Session, allocated) -> TriggeredRun:
        eid = allocated.recovery_event_id
        provider = (
            self._provider_factory() if self._provider_factory is not None else None
        )
        try:
            result = run_recovery_agent(
                session,
                eid,
                dry_run=self._config.dry_run,
                provider=provider,
                triggered_by=SCHEDULER_TRIGGER,
            )
        except AgentRunError as exc:
            session.rollback()
            logger.warning("scheduler run for event %s could not start: %s", eid, exc)
            return TriggeredRun(
                recovery_event_id=eid,
                rank=allocated.rank,
                best_action=allocated.best_action,
                expected_value_paise=allocated.expected_value_paise,
                run_status="error",
                decision=None,
                stop_reason=None,
                chosen_action=None,
                error=f"{exc.code}: {exc.message}",
            )
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop the cycle
            session.rollback()
            logger.exception("scheduler run for event %s raised", eid)
            return TriggeredRun(
                recovery_event_id=eid,
                rank=allocated.rank,
                best_action=allocated.best_action,
                expected_value_paise=allocated.expected_value_paise,
                run_status="error",
                decision=None,
                stop_reason=None,
                chosen_action=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        return TriggeredRun(
            recovery_event_id=eid,
            rank=allocated.rank,
            best_action=allocated.best_action,
            expected_value_paise=allocated.expected_value_paise,
            run_status=result.status,
            decision=result.decision,
            stop_reason=result.stop_reason,
            chosen_action=result.chosen_action,
        )

    # -- status -----------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            history = [r.as_dict() for r in reversed(self._history)]
            cycles_run = self._cycles_completed
        last_at = (
            self._last_cycle_started_at.isoformat()
            if self._last_cycle_started_at is not None
            else None
        )
        next_at = None
        if self.running and self._last_cycle_started_at is not None:
            next_at = (
                self._last_cycle_started_at
                + timedelta(seconds=self._config.interval_seconds)
            ).isoformat()
        elif self.running:
            # started, no cycle yet
            next_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=self._config.interval_seconds)
            ).isoformat()
        return {
            "enabled": self._config.enabled,
            "running": self.running,
            "config": self._config.as_public_dict(),
            "cycles_run": cycles_run,
            "last_cycle_at": last_at,
            "next_cycle_estimated_at": next_at,
            "cycle_history": history,
        }


# Process-wide singleton wired into the FastAPI lifespan in app/main.py.
scheduler = RecoveryScheduler()
