"""Real Razorpay TEST MODE end-to-end smoke test (Section 24).

Phases (each guarded, none run without explicit config):

  default        create a throwaway open treatment recovery event, run the agent
                 in dry_run (assert ZERO Razorpay side effects), then run it LIVE
                 (create ONE real Test Mode Payment Link), print the link + ids,
                 and assert payment_recovered is still FALSE. Leaves the event
                 open for the manual payment + webhook step.
  --observe ID   run observe_recovery_outcome against an existing event and print
                 the real recovery state (use after paying the link).
  --cleanup ID   delete the throwaway event and everything under it.

Never prints RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET / RAZORPAY_WEBHOOK_SECRET.
TEST MODE ONLY -- refuses to run if the key is not an rzp_test_ key.

    .venv/Scripts/python.exe scripts/razorpay_smoke.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from app.agent.config import AgentConfig
from app.agent.guardrails import load_event, payment_recovered
from app.agent.runner import run_recovery_agent
from app.database import SessionLocal
from app.integrations.razorpay import RazorpayConfig
from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription

TAG = "smoke_razorpay_"


class _ScriptedExecute:
    """Tiny provider: context -> scores -> execute -> stop. Not a real LLM;
    used only to drive the live Payment Link creation deterministically."""

    model = "smoke-script"

    def __init__(self) -> None:
        self._queue = [
            ("get_recovery_event_context", {}),
            ("get_action_scores", {}),
            ("execute_recovery_action", {
                "action_type": "whatsapp_nudge",
                "customer_message": (
                    "Hi, aapka subscription payment complete nahi ho paaya. "
                    "Neeche diye secure link se aap ise abhi complete kar sakte hain."
                ),
                "reason": "smoke test: create a Razorpay Test Mode payment link",
            }),
            ("stop_recovery", {
                "stop_reason": "action_executed_awaiting_outcome",
                "reasoning_summary": "payment link created; awaiting customer payment",
            }),
        ]

    def generate(self, *, system_prompt, conversation, tools):
        from app.agent.schemas import ProviderTurn, ToolCall

        name, args = self._queue.pop(0) if self._queue else (
            "stop_recovery", {"stop_reason": "other", "reasoning_summary": "done"}
        )
        return ProviderTurn(
            tool_call=ToolCall(name=name, arguments=args),
            model=self.model, latency_ms=0,
        )


def _make_event(db) -> int:
    now = datetime.now(timezone.utc)
    cust = Customer(
        external_customer_id=f"{TAG}{now.timestamp()}",
        email="rahul.smoke@example.com", phone="+919000000000",
        total_successful_payments=6, total_failed_payments=2,
        created_at=now - timedelta(days=120),
    )
    db.add(cust); db.flush()
    sub = Subscription(
        customer_id=cust.id, external_subscription_id=f"{TAG}sub_{cust.id}",
        amount=49900, currency="INR", status="active",
        started_at=now - timedelta(days=60), next_payment_at=now + timedelta(days=20),
    )
    db.add(sub); db.flush()
    pay = Payment(
        subscription_id=sub.id, amount=49900, currency="INR", status="failed",
        failure_reason="insufficient_funds", failed_at=now - timedelta(hours=4),
        retry_count=0,
    )
    db.add(pay); db.flush()
    ev = RecoveryEvent(
        payment_id=pay.id, status="open", priority=1, is_control=False,
        variant="treatment", created_at=now - timedelta(hours=4),
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


def _print_razorpay_state(db, event_id: int) -> None:
    ev = load_event(db, event_id)
    print(f"  recovery_event {event_id}: status={ev.status} "
          f"payment_recovered={payment_recovered(ev)} payment_status={ev.payment.status}")
    for i in sorted(ev.interventions, key=lambda x: x.id):
        print(f"  intervention {i.id}: action={i.action_type} "
              f"link_id={i.razorpay_payment_link_id} "
              f"short_url={i.razorpay_short_url} "
              f"razorpay_status={i.last_razorpay_status} "
              f"payment_id={i.razorpay_payment_id} "
              f"outcome={'recovered' if (i.outcome and i.outcome.payment_recovered) else None}")


def cmd_default(real_gemini: bool = True) -> None:
    cfg = RazorpayConfig.from_settings()
    print("Razorpay config:", cfg.status())
    if not cfg.is_ready:
        sys.exit(
            "STOP: Razorpay is not configured for a live Test Mode call.\n"
            "Need RAZORPAY_KEY_ID (rzp_test_...), RAZORPAY_KEY_SECRET, "
            "RAZORPAY_TEST_MODE=true in .env."
        )

    live_provider = None if real_gemini else _ScriptedExecute()
    if real_gemini:
        acfg = AgentConfig.from_settings()
        if not acfg.has_key:
            sys.exit("STOP: GEMINI_API_KEY not set; cannot run the real agent.")
        print(f"agent: real Gemini ({acfg.model}), max_turns={acfg.max_turns}\n")

    db = SessionLocal()
    try:
        _cleanup(db)
        event_id = _make_event(db)
        print(f"created throwaway open treatment recovery event id={event_id} "
              f"(amount 49900 paise)\n")

        # -- phase 1: DRY RUN safety gate (scripted, deterministic, 0 quota) --
        print("--- phase 1: DRY RUN safety gate (must make no Razorpay call) ---")
        dr = run_recovery_agent(db, event_id, dry_run=True,
                                provider=_ScriptedExecute(), persist=False)
        ev = load_event(db, event_id)
        assert not ev.interventions, "DRY RUN created an intervention!"
        print(f"  tools={[t.tool for t in dr.tool_trace]}")
        print("  OK: no intervention, no Razorpay call.\n")

        # -- phase 2: LIVE, agent-driven -- Gemini chooses, ONE real link --
        print("--- phase 2: LIVE (agent-driven; creates ONE real Test Mode link) ---")
        lr = run_recovery_agent(db, event_id, dry_run=False,
                                provider=live_provider, persist=True)
        print(f"  status={lr.status} turns={lr.turns_used}")
        print(f"  tools requested (in order): {[t.tool for t in lr.tool_trace]}")
        print(f"  decision={lr.decision} chosen_action={lr.chosen_action}")
        print(f"  reasoning_summary={lr.reasoning_summary}")
        if lr.errors:
            print(f"  errors={lr.errors}")
        _print_razorpay_state(db, event_id)

        ev = load_event(db, event_id)
        assert payment_recovered(ev) is False, "payment marked recovered on link creation!"
        assert not any(i.outcome for i in ev.interventions), "Outcome created on link creation!"

        links = [i for i in ev.interventions if i.razorpay_payment_link_id]
        assert len(links) <= 1, f"more than one payment link created: {len(links)}"
        if not links:
            print("\n  NO Payment Link was created -- Gemini did not execute an action.")
            print(f"  final status={lr.status}, decision={lr.decision}")
            return
        iv = links[0]
        print("\n  >>> OPEN THIS LINK AND PAY WITH A RAZORPAY TEST CARD <<<")
        print(f"  Payment Link URL : {iv.razorpay_short_url}")
        print(f"  payment_link_id  : {iv.razorpay_payment_link_id}")
        print(f"  reference_id     : {iv.razorpay_reference_id}")
        print(f"  recovery_event_id: {event_id}   intervention_id: {iv.id}")
        print(f"\n  after paying, re-run:  scripts/razorpay_smoke.py --observe {event_id}")
        print("\n  payment_recovered is FALSE (correct: link created != recovered).")
    finally:
        db.close()


def cmd_observe(event_id: int) -> None:
    db = SessionLocal()
    try:
        ev = load_event(db, event_id)
        if ev is None:
            sys.exit(f"recovery event {event_id} not found")
        res = run_recovery_agent(
            db, event_id, dry_run=False, provider=_ObserveOnly(), persist=False
        )
        print(f"observe result: status={res.status} stop_reason={res.stop_reason}")
        print(f"tools={[t.tool for t in res.tool_trace]}")
        for t in res.tool_trace:
            print(f"  {t.tool}: {t.result_summary}")
        _print_razorpay_state(db, event_id)
    finally:
        db.close()


class _ObserveOnly:
    model = "smoke-observe"

    def __init__(self) -> None:
        self._q = [
            ("observe_recovery_outcome", {}),
            ("stop_recovery", {"stop_reason": "payment_recovered",
                               "reasoning_summary": "observed final state"}),
        ]

    def generate(self, *, system_prompt, conversation, tools):
        from app.agent.schemas import ProviderTurn, ToolCall
        name, args = self._q.pop(0) if self._q else (
            "stop_recovery", {"stop_reason": "other", "reasoning_summary": "done"})
        return ProviderTurn(tool_call=ToolCall(name=name, arguments=args),
                            model=self.model, latency_ms=0)


def cmd_cleanup() -> None:
    db = SessionLocal()
    try:
        _cleanup(db)
        print("throwaway smoke rows deleted.")
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observe", type=int, metavar="RECOVERY_EVENT_ID")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--scripted", action="store_true",
                    help="use the deterministic scripted provider instead of real Gemini")
    args = ap.parse_args()
    if args.cleanup:
        cmd_cleanup()
    elif args.observe:
        cmd_observe(args.observe)
    else:
        cmd_default(real_gemini=not args.scripted)


if __name__ == "__main__":
    main()
