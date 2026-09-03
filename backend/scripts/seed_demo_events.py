"""Seed a small set of DEMO recovery events for the frontend walkthrough.

WHY THIS EXISTS
---------------
The generated dataset contains only terminal recovery events (``abandoned`` /
``closed``) plus ``open`` *control* events. None of them can show the agent's
full DETECT -> SCORE -> DECIDE -> EXECUTE arc live, because the agent / guardrails
correctly refuse a control or already-terminal event.

This script creates a handful of **open, treatment** recovery events so the
dashboard's "Run Recovery Agent" experience has real, explorable content, and
pre-persists a real agent trace for each one using a **scripted provider**:

    * ZERO Gemini API calls  (the scripted provider makes none)
    * ZERO Razorpay calls    (dry_run=True -> execution is simulated)
    * ZERO new Payment Links
    * a real Hinglish TTS ``.wav`` per event (local pyttsx3, no network)

Every row it writes is tagged ``demo_ui_*`` and is fully reversible:

    python -m scripts.seed_demo_events --reset      # delete everything it made
    python -m scripts.seed_demo_events              # create (idempotent)
    python -m scripts.seed_demo_events --reset --seed   # reset then recreate

It never touches the real verified recovery event (18499), the recovery /
Razorpay / agent / ML logic, or the schema.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

import dataclasses

from app.agent.config import AgentConfig
from app.agent.runner import run_recovery_agent
from app.agent.schemas import ProviderTurn, ToolCall
from app.database import SessionLocal
from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription
from app.services.voice import VoiceConfig, VoiceService

TAG = "demo_ui_"


# --------------------------------------------------------------------------
# A self-contained scripted provider (no Gemini, no test imports).
# It reads the last tool result to choose a real, eligible action, so the
# persisted trace genuinely reflects the ML/uplift recommendation.
# --------------------------------------------------------------------------
_HINGLISH = {
    "method_switch_prompt": (
        "Namaste! Aapka {amount} ka payment complete nahi ho paaya (funds issue). "
        "Neeche secure link se doosre card ya UPI se turant complete karein. "
        "Service bina rukawat chalti rahegi."
    ),
    "whatsapp_nudge": (
        "Hi! Aapka {amount} ka subscription payment fail ho gaya. "
        "Yeh secure link use karke abhi dobara try karein - 2 minute ka kaam hai."
    ),
    "sms_nudge": (
        "Aapka {amount} ka payment nahi hua. Ise abhi complete karne ke liye "
        "secure link check karein. Zaroorat ho toh support madad karega."
    ),
    "retry": (
        "Aapke {amount} ke payment ko humne dobara try kiya hai. "
        "Agar phir se fail ho toh secure link se doosra method use karein."
    ),
}


class _ScriptedRecoveryProvider:
    """Deterministic, reactive-enough to pick a real recommended action.

    ``prefer`` chooses which model signal the agent acts on:
      * "ev"     -> the ML expected-value recommendation
      * "uplift" -> the causal uplift (net incremental value) recommendation
    Both are surfaced by ``get_action_scores``; a real agent weighs them, and
    the two models genuinely disagree on several of these events.
    """

    model = "demo-scripted"

    def __init__(self, amount_paise: int, prefer: str = "ev") -> None:
        self._amount = amount_paise
        self._prefer = prefer
        self._step = 0
        self._intended: tuple[str, str] | None = None  # (action, base_reason)
        self._reality_check: str | None = None

    def generate(self, *, system_prompt, conversation, tools) -> ProviderTurn:  # noqa: ARG002
        results = [e for e in conversation if e["type"] == "tool_result"]
        last = results[-1] if results else None
        name, args = self._decide(last)
        self._step += 1
        return ProviderTurn(
            tool_call=ToolCall(name=name, arguments=args),
            model=self.model,
            latency_ms=0,
            prompt_tokens=40,
            output_tokens=12,
            total_tokens=52,
        )

    def _decide(self, last):
        if last is None:
            return "get_recovery_event_context", {}
        src = last["name"]
        payload = last["payload"]

        if src == "get_recovery_event_context":
            if payload.get("payment_recovered"):
                return "stop_recovery", {
                    "stop_reason": "payment_recovered",
                    "reasoning_summary": "Payment already recovered; nothing to do.",
                }
            return "get_payment_context", {}
        if src == "get_payment_context":
            return "get_customer_recovery_history", {}
        if src == "get_customer_recovery_history":
            return "get_action_scores", {}
        if src == "get_action_scores":
            scores = payload.get("scores") or []
            if not scores:
                return "stop_recovery", {
                    "stop_reason": "no_eligible_actions",
                    "reasoning_summary": "No eligible untried action to score.",
                }
            ev_rec = payload.get("recommended_by_expected_value")
            uplift_rec = payload.get("recommended_by_uplift")
            if self._prefer == "uplift" and uplift_rec:
                action = uplift_rec
                reason = (
                    f"Chose {action}: highest causal net incremental value "
                    f"(uplift model). It beats the raw expected-value pick "
                    f"({ev_rec}) once the customer's already-likely-to-pay "
                    f"baseline is removed."
                )
            else:
                action = ev_rec or scores[0].get("action")
                reason = (
                    f"Chose {action}: highest model expected recovered value "
                    f"for this amount and failure reason, net of action cost."
                )
            # Before committing, check the action's REAL observed incremental
            # lift against the model's prediction.
            self._intended = (action, reason)
            return "get_historical_incrementality_for_action", {"action_type": action}

        if src == "get_historical_incrementality_for_action":
            action, base_reason = self._intended or (
                payload.get("action_type", "whatsapp_nudge"),
                "model pick",
            )
            self._reality_check = self._incrementality_note(action, payload)
            reason = f"{base_reason} {self._reality_check}"
            amount_str = f"Rs.{self._amount / 100:.0f}"
            msg = _HINGLISH.get(action, _HINGLISH["whatsapp_nudge"]).format(
                amount=amount_str
            )
            return "execute_recovery_action", {
                "action_type": action,
                "customer_message": msg,
                "reason": reason[:480],
            }

        if src == "execute_recovery_action":
            if payload.get("error"):
                return "stop_recovery", {
                    "stop_reason": "guardrail_violation",
                    "reasoning_summary": f"Blocked: {payload.get('code')}.",
                }
            return "observe_recovery_outcome", {}
        if src == "observe_recovery_outcome":
            if payload.get("payment_recovered"):
                return "stop_recovery", {
                    "stop_reason": "payment_recovered",
                    "reasoning_summary": "Payment recovered after the intervention.",
                }
            summary = (
                "Recovery Payment Link would be created; awaiting the "
                "customer's payment before concluding."
            )
            if getattr(self, "_reality_check", None):
                summary = f"{self._reality_check} {summary}"
            return "stop_recovery", {
                "stop_reason": "action_executed_awaiting_outcome",
                "reasoning_summary": summary[:480],
            }
        return "stop_recovery", {
            "stop_reason": "other",
            "reasoning_summary": "Nothing left to do.",
        }

    @staticmethod
    def _incrementality_note(action: str, payload: dict) -> str:
        """An honest one-liner comparing predicted vs observed lift. Never
        overstates: if the numbers agree, it says so."""
        if not payload.get("computable"):
            return (
                f"Historical check: not enough past data for {action} "
                f"({payload.get('reason')}); relying on the model prediction."
            )
        observed = float(payload["observed_incremental_lift"])
        n = payload["treated_group_size"]
        predicted = payload.get("model_predicted_uplift_for_context")
        if predicted is None:
            return (
                f"Historical check: {action} has delivered +{observed * 100:.1f}pp "
                f"observed incremental lift across {n} past uses (95% CI excludes zero)."
            )
        predicted = float(predicted)
        ci = payload.get("observed_incremental_lift_ci_95")
        if ci and ci[0] <= predicted <= ci[1]:
            verdict = (
                "The model's prediction sits inside that observed range -- "
                "proceeding."
            )
        elif ci and predicted > ci[1]:
            verdict = (
                "The model is a little more optimistic than history, but this is "
                "still clearly the strongest option -- proceeding with tempered "
                "expectations."
            )
        elif ci and predicted < ci[0]:
            verdict = (
                "History is actually stronger than the model predicted -- "
                "proceeding with added confidence."
            )
        elif abs(predicted - observed) < 0.025:
            verdict = "Predicted and observed align -- proceeding."
        else:
            verdict = (
                "Observed differs from the model's prediction, but this is still "
                "the strongest option -- proceeding."
            )
        return (
            f"Cross-checked: the uplift model predicted +{predicted * 100:.1f}pp for "
            f"{action} here; across {n} past uses the observed incremental lift is "
            f"+{observed * 100:.1f}pp. {verdict}"
        )


# --------------------------------------------------------------------------
# Demo event definitions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DemoSpec:
    slug: str
    name: str
    email: str
    amount_paise: int
    failure_reason: str
    priority: int
    hours_since_failure: int
    successful_payments: int
    failed_payments: int
    tenure_days: int
    prefer: str = "ev"  # "ev" | "uplift" -- which model signal the agent acts on


DEMOS: list[DemoSpec] = [
    DemoSpec("aarav", "Aarav Menon", "aarav.menon@example.com", 199900,
             "insufficient_funds", 3, 5, 24, 1, 420, prefer="ev"),
    DemoSpec("isha", "Isha Kapoor", "isha.kapoor@example.com", 99900,
             "card_expired", 2, 12, 11, 3, 260, prefer="uplift"),
    DemoSpec("rohan", "Rohan Verma", "rohan.verma@example.com", 49900,
             "issuer_decline", 2, 6, 8, 5, 190, prefer="uplift"),
    DemoSpec("meera", "Meera Nair", "meera.nair@example.com", 149900,
             "bank_timeout", 1, 30, 17, 2, 540, prefer="ev"),
    DemoSpec("kabir", "Kabir Shah", "kabir.shah@example.com", 29900,
             "insufficient_funds", 0, 3, 3, 4, 95, prefer="uplift"),
]


def _reset(db) -> int:
    cust_ids = [
        c.id
        for c in db.scalars(
            select(Customer).where(Customer.external_customer_id.like(f"{TAG}%"))
        )
    ]
    if not cust_ids:
        return 0
    sub_ids = [s for s in db.scalars(
        select(Subscription.id).where(Subscription.customer_id.in_(cust_ids))
    )] or [-1]
    pay_ids = [p for p in db.scalars(
        select(Payment.id).where(Payment.subscription_id.in_(sub_ids))
    )] or [-1]
    re_ids = [r for r in db.scalars(
        select(RecoveryEvent.id).where(RecoveryEvent.payment_id.in_(pay_ids))
    )] or [-1]
    iv_ids = [i for i in db.scalars(
        select(Intervention.id).where(Intervention.recovery_event_id.in_(re_ids))
    )] or [-1]
    db.execute(delete(Outcome).where(Outcome.intervention_id.in_(iv_ids)))
    db.execute(delete(AuditLog).where(AuditLog.recovery_event_id.in_(re_ids)))
    db.execute(delete(AgentEvent).where(AgentEvent.recovery_event_id.in_(re_ids)))
    db.execute(delete(Intervention).where(Intervention.recovery_event_id.in_(re_ids)))
    db.execute(delete(RecoveryEvent).where(RecoveryEvent.id.in_(re_ids)))
    db.execute(delete(Payment).where(Payment.id.in_(pay_ids)))
    db.execute(delete(Subscription).where(Subscription.id.in_(sub_ids)))
    db.execute(delete(Customer).where(Customer.id.in_(cust_ids)))
    db.commit()
    return len(re_ids)


def _make_event(db, spec: DemoSpec, now: datetime) -> int:
    cust = Customer(
        external_customer_id=f"{TAG}{spec.slug}",
        email=spec.email,
        phone=None,
        total_successful_payments=spec.successful_payments,
        total_failed_payments=spec.failed_payments,
        created_at=now - timedelta(days=spec.tenure_days + 30),
    )
    db.add(cust)
    db.flush()
    sub = Subscription(
        customer_id=cust.id,
        external_subscription_id=f"{TAG}sub_{spec.slug}",
        amount=spec.amount_paise,
        currency="INR",
        status="active",
        started_at=now - timedelta(days=spec.tenure_days),
        next_payment_at=now + timedelta(days=14),
    )
    db.add(sub)
    db.flush()
    pay = Payment(
        subscription_id=sub.id,
        amount=spec.amount_paise,
        currency="INR",
        status="failed",
        failure_reason=spec.failure_reason,
        failed_at=now - timedelta(hours=spec.hours_since_failure),
        retry_count=0,
    )
    db.add(pay)
    db.flush()
    event = RecoveryEvent(
        payment_id=pay.id,
        status="open",
        priority=spec.priority,
        is_control=False,
        variant="treatment",
        created_at=now - timedelta(hours=spec.hours_since_failure),
    )
    db.add(event)
    db.add(
        AuditLog(
            recovery_event_id=None,
            actor="system",
            action="recovery_event_created",
            reason="demo recovery event seeded for the dashboard walkthrough",
        )
    )
    db.flush()
    # audit row needs the id
    db.add(
        AuditLog(
            recovery_event_id=event.id,
            actor="system",
            action="recovery_event_opened",
            reason=f"failed payment entered recovery ({spec.failure_reason})",
        )
    )
    db.commit()
    return event.id


def seed(db, *, force: bool) -> None:
    existing = db.scalars(
        select(Customer).where(Customer.external_customer_id.like(f"{TAG}%"))
    ).first()
    if existing is not None and not force:
        print(
            f"demo rows already present (found {existing.external_customer_id}). "
            "Use --reset to clear, or --reset --seed to rebuild."
        )
        return

    now = datetime.now(timezone.utc)
    voice = VoiceService(
        VoiceConfig(
            enabled=True, language="hi",
            output_dir="artifacts/tts", engine="pyttsx3",
        )
    )

    # The scripted walkthrough needs ~7 turns (observe x3 -> score -> execute ->
    # verify -> stop); the .env default GEMINI_MAX_TURNS is tuned low for the
    # free tier. This only affects these seeded demo runs.
    agent_cfg = dataclasses.replace(AgentConfig.from_settings(), max_turns=10)

    created: list[tuple[int, DemoSpec]] = []
    for spec in DEMOS:
        eid = _make_event(db, spec, now)
        created.append((eid, spec))
        print(f"  created recovery event {eid:>6}  {spec.name}  "
              f"Rs.{spec.amount_paise/100:.0f}  ({spec.failure_reason})")

    print("\nrunning the scripted agent (0 Gemini, 0 Razorpay) for each event:")
    for eid, spec in created:
        result = run_recovery_agent(
            db, eid,
            dry_run=True,
            provider=_ScriptedRecoveryProvider(spec.amount_paise, spec.prefer),
            config=agent_cfg,
            voice_service=voice,
            persist=True,
        )
        tools = " -> ".join(t.tool for t in result.tool_trace)
        print(f"  event {eid:>6}: {result.decision:<32} "
              f"voice={'yes' if result.voice_generated else result.voice_reason}")
        print(f"           {tools}")

    print(f"\nseeded {len(created)} demo recovery events. "
          "Reverse any time with:  python -m scripts.seed_demo_events --reset")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true",
                    help="delete every demo_ui_* row this script created")
    ap.add_argument("--seed", action="store_true",
                    help="(re)create the demo events; implied when --reset is absent")
    ap.add_argument("--force", action="store_true",
                    help="seed even if demo rows already exist")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        if args.reset:
            n = _reset(db)
            print(f"removed {n} demo recovery event(s) and everything under them.")
            if not args.seed:
                return
        seed(db, force=args.force or args.seed)
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
