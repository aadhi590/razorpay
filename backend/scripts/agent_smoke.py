"""ONE real multi-turn Gemini agent run against a safe, throwaway recovery event.

* Uses GEMINI_API_KEY from .env (via app.config.settings). Never prints it.
* dry_run=True  -- no recovery action is really executed.
* Creates an isolated open treatment recovery event, runs the agent, prints the
  full turn-by-turn tool trace, then deletes everything it created.

Usage (from the project root):
    .venv/Scripts/python.exe scripts/agent_smoke.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from app.agent.config import AgentConfig
from app.agent.runner import run_recovery_agent
from app.database import SessionLocal
from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription

TAG = "smoke_agent_gemini_"


def _make_event(db) -> int:
    now = datetime.now(timezone.utc)
    cust = Customer(
        external_customer_id=f"{TAG}{now.timestamp()}",
        email="rahul@example.com",
        total_successful_payments=7,
        total_failed_payments=2,
        created_at=now - timedelta(days=150),
    )
    db.add(cust); db.flush()
    sub = Subscription(
        customer_id=cust.id, external_subscription_id=f"{TAG}sub_{cust.id}",
        amount=99900, currency="INR", status="active",
        started_at=now - timedelta(days=90), next_payment_at=now + timedelta(days=15),
    )
    db.add(sub); db.flush()
    pay = Payment(
        subscription_id=sub.id, amount=99900, currency="INR", status="failed",
        failure_reason="insufficient_funds", failed_at=now - timedelta(hours=5),
        retry_count=0,
    )
    db.add(pay); db.flush()
    ev = RecoveryEvent(
        payment_id=pay.id, status="open", priority=2, is_control=False,
        variant="treatment", created_at=now - timedelta(hours=5),
    )
    db.add(ev); db.commit()
    return ev.id


def _cleanup(db) -> None:
    cust_ids = [c.id for c in db.scalars(
        select(Customer).where(Customer.external_customer_id.like(f"{TAG}%")))]
    if not cust_ids:
        return
    sub_ids = [s for s in db.scalars(select(Subscription.id).where(
        Subscription.customer_id.in_(cust_ids)))] or [-1]
    pay_ids = [p for p in db.scalars(select(Payment.id).where(
        Payment.subscription_id.in_(sub_ids)))] or [-1]
    re_ids = [r for r in db.scalars(select(RecoveryEvent.id).where(
        RecoveryEvent.payment_id.in_(pay_ids)))] or [-1]
    iv_ids = [i for i in db.scalars(select(Intervention.id).where(
        Intervention.recovery_event_id.in_(re_ids)))] or [-1]
    db.execute(delete(Outcome).where(Outcome.intervention_id.in_(iv_ids)))
    db.execute(delete(AuditLog).where(AuditLog.recovery_event_id.in_(re_ids)))
    db.execute(delete(AgentEvent).where(AgentEvent.recovery_event_id.in_(re_ids)))
    db.execute(delete(Intervention).where(Intervention.recovery_event_id.in_(re_ids)))
    db.execute(delete(RecoveryEvent).where(RecoveryEvent.id.in_(re_ids)))
    db.execute(delete(Payment).where(Payment.id.in_(pay_ids)))
    db.execute(delete(Subscription).where(Subscription.id.in_(sub_ids)))
    db.execute(delete(Customer).where(Customer.id.in_(cust_ids)))
    db.commit()


def main() -> None:
    cfg = AgentConfig.from_settings()
    if not cfg.has_key:
        sys.exit("STOP: GEMINI_API_KEY is not set in .env")
    print(f"model={cfg.model}  max_turns={cfg.max_turns}  timeout={cfg.timeout_seconds}s")
    print("API key loaded from .env (not displayed).\n")

    db = SessionLocal()
    try:
        _cleanup(db)
        event_id = _make_event(db)
        print(f"created throwaway open treatment recovery event id={event_id}\n")

        result = run_recovery_agent(db, event_id, dry_run=True, persist=True)

        print("=" * 70)
        print(f"authentication        : {'OK' if not any('auth' in e.lower() for e in result.errors) else 'FAILED'}")
        print(f"status                : {result.status}")
        print(f"stop_reason           : {result.stop_reason}")
        print(f"model                 : {result.model}")
        print(f"latency_ms            : {result.latency_ms}")
        print(f"turns_used            : {result.turns_used}  (multi-turn: {result.turns_used > 1})")
        print(f"token_usage           : {result.token_usage}")
        print(f"decision              : {result.decision}")
        print(f"chosen_action         : {result.chosen_action}")
        print(f"escalation_required   : {result.escalation_required}")
        print(f"customer_message      : {result.customer_message}")
        print(f"reasoning_summary     : {result.reasoning_summary}")
        print(f"actions_executed      : {result.actions_executed}  (dry-run => simulated)")
        print(f"errors                : {result.errors}")
        print("\n--- tool calls, in the order Gemini requested them ---")
        for t in result.tool_trace:
            print(
                f"  turn {t.turn}: {t.tool}({json.dumps(t.arguments)[:80]}) "
                f"-> ok={t.ok} terminal={t.terminal} :: {t.result_summary}"
            )
        if result.quantitative_scores:
            print("\n--- quantitative scores Gemini was given (unmodified) ---")
            for s in result.quantitative_scores:
                print(f"  {s}")
        print("=" * 70)

        # sanity: agent respected the available actions
        if result.chosen_action:
            from app.agent.guardrails import eligible_action_types, load_event
            elig = eligible_action_types(load_event(db, event_id))
            print(f"chosen action in eligible set {elig}: "
                  f"{result.chosen_action in elig}")
    finally:
        _cleanup(db)
        db.close()
        print("\nthrowaway event and trace rows deleted.")


if __name__ == "__main__":
    main()
