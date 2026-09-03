"""
Synthetic data generator for the Razorpay AI Recovery Orchestrator.

Populates Customer -> Subscription -> Payment -> (failed) -> RecoveryEvent
-> Intervention -> Outcome, plus Experiment / AgentEvent / AuditLog,
using ONLY fields that exist on the actual SQLAlchemy models.

Control vs treatment is assigned explicitly on every RecoveryEvent via the
is_control / experiment_id / variant columns. Control events additionally
have no Intervention rows (their outcome reflects only natural recovery),
but the classification no longer depends on that absence -- it is stored
directly, which is what a later baseline-vs-intervention incrementality
comparison relies on.

Two modes:
  * full-pipeline (default) -- simulates the entire recovery lifecycle, so
    treatment RecoveryEvents end up closed/abandoned.
  * --initial-state -- generates only OPEN RecoveryEvents (no interventions,
    outcomes or simulated recovery) so the live RecoveryOrchestratorService
    has real work to process.

Opt-in flag (full-pipeline only):
  * --randomized-assignment -- treatment intervention actions are drawn
    uniformly at random from the still-eligible actions (sampling without
    replacement across an event's attempts) using a dedicated RNG stream, and
    the exact assignment propensity is recorded on each AgentEvent's
    input_context['assignment']. This produces data suitable for later
    action-level causal / uplift estimation. Observed outcomes are still the
    only outcomes written -- no counterfactuals are fabricated. Without the
    flag the generator is byte-for-byte unchanged.

Usage:
    python -m app.scripts.generate_data --customers 1000 --seed 42
    python -m app.scripts.generate_data --reset --customers 1000 --seed 42
    python -m app.scripts.generate_data --reset --customers 100 --seed 42 --initial-state
    python -m app.scripts.generate_data --reset --customers 1000 --seed 42 --randomized-assignment
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.customer import Customer
from app.models.subscription import Subscription
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.experiment import Experiment
from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog


# ---------------------------------------------------------------------------
# Reference data / distributions
# ---------------------------------------------------------------------------

FAILURE_REASONS: dict[str, float] = {
    # failure_reason -> baseline recoverability multiplier (relative, not a probability)
    "insufficient_funds": 0.55,
    "card_expired": 0.20,
    "bank_timeout": 0.75,
    "issuer_decline": 0.35,
}

ACTION_TYPES: dict[str, dict[str, float | int]] = {
    # action_type -> {cost_paise, effectiveness multiplier}
    "retry": {"cost_paise": 50, "effectiveness": 0.35},
    "sms_nudge": {"cost_paise": 20, "effectiveness": 0.45},
    "whatsapp_nudge": {"cost_paise": 80, "effectiveness": 0.55},
    "method_switch_prompt": {"cost_paise": 150, "effectiveness": 0.65},
}

SUBSCRIPTION_STATUSES = ["active", "active", "active", "cancelled"]

CONTROL_GROUP_FRACTION = 0.20  # fraction of failed-payment RecoveryEvents assigned to control (no intervention)
BASELINE_NATURAL_RECOVERY_PROB = 0.12  # control-group recovery probability floor
DIMINISHING_RETURNS_DECAY = 0.6  # multiplier applied per additional intervention attempt

# Default seed for the dedicated randomized-action-assignment RNG stream. Kept
# separate from the main seed so turning --randomized-assignment on/off never
# shifts any other generated value.
DEFAULT_ASSIGNMENT_SEED = 20260902


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_datetime_within(rng: random.Random, days_back: int) -> datetime:
    offset_seconds = rng.randint(0, days_back * 24 * 3600)
    return datetime.utcnow() - timedelta(seconds=offset_seconds)


def _customer_reliability(rng: random.Random) -> float:
    """A synthetic 0-1 reliability score driving correlated behavior.
    Not stored on the model -- used only to shape generated data."""
    return min(max(rng.gauss(0.6, 0.2), 0.05), 0.97)


def choose_randomized_action(
    rng: random.Random, eligible_actions: list[str]
) -> tuple[str, float]:
    """Uniform random assignment over the currently eligible actions.

    Returns ``(action_type, propensity)`` where ``propensity`` is the exact
    probability with which this action was assigned given the eligible set --
    i.e. ``1 / len(eligible_actions)``. This is the value a later
    inverse-propensity / doubly-robust estimator needs.
    """
    if not eligible_actions:
        raise ValueError("no eligible actions to assign")
    action = rng.choice(eligible_actions)
    return action, 1.0 / len(eligible_actions)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset_application_data(db: Session) -> None:
    """Delete existing application data in FK-safe (child-first) order.
    Never touches alembic_version and never drops/alters schema."""
    print("Resetting existing application data (schema is preserved)...")
    db.execute(delete(AuditLog))
    db.execute(delete(AgentEvent))
    db.execute(delete(Outcome))
    db.execute(delete(Intervention))
    db.execute(delete(RecoveryEvent))
    db.execute(delete(Payment))
    db.execute(delete(Subscription))
    db.execute(delete(Customer))
    db.execute(delete(Experiment))
    db.commit()
    print("Existing application data cleared.\n")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_experiments(db: Session, rng: random.Random) -> list[Experiment]:
    """Experiment has no FK to anything else in the real schema, so these
    are standalone hypothetical experiment configs, not per-row linked."""
    configs = [
        ("SMS vs WhatsApp nudge", "sms_nudge", 50, 50),
        ("Retry timing test", "retry", 40, 60),
        ("Method switch prompt trial", "method_switch_prompt", 50, 50),
    ]
    experiments = []
    for name, intervention_type, control_pct, treatment_pct in configs:
        exp = Experiment(
            name=name,
            intervention_type=intervention_type,
            control_percentage=control_pct,
            treatment_percentage=treatment_pct,
            status=rng.choice(["active", "active", "completed"]),
            started_at=_random_datetime_within(rng, 90),
        )
        db.add(exp)
        experiments.append(exp)
    db.flush()
    return experiments


def generate_customers(db: Session, rng: random.Random, count: int) -> list[Customer]:
    customers = []
    for i in range(count):
        customer = Customer(
            external_customer_id=f"cust_{i:06d}",
            email=f"customer{i}@example.com",
            phone=f"+91{rng.randint(6000000000, 9999999999)}",
            total_successful_payments=0,  # filled in after payments are generated
            total_failed_payments=0,
            created_at=_random_datetime_within(rng, 365),
        )
        db.add(customer)
        customers.append(customer)
    db.flush()
    return customers


def generate_subscriptions(
    db: Session, rng: random.Random, customers: list[Customer]
) -> list[Subscription]:
    subscriptions = []
    for customer in customers:
        num_subs = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for j in range(num_subs):
            started_at = _random_datetime_within(rng, 300)
            sub = Subscription(
                customer_id=customer.id,
                external_subscription_id=f"sub_{customer.id:06d}_{j}",
                amount=rng.choice([9900, 19900, 49900, 99900, 199900]),  # paise
                currency="INR",
                status=rng.choice(SUBSCRIPTION_STATUSES),
                started_at=started_at,
                next_payment_at=started_at + timedelta(days=30),
            )
            db.add(sub)
            subscriptions.append(sub)
    db.flush()
    return subscriptions


def generate_payments(
    db: Session,
    rng: random.Random,
    subscriptions: list[Subscription],
    customer_reliability: dict[int, float],
) -> list[Payment]:
    payments = []
    for sub in subscriptions:
        reliability = customer_reliability[sub.customer_id]
        num_payments = rng.randint(2, 6)
        cursor = sub.started_at
        for _ in range(num_payments):
            cursor = cursor + timedelta(days=30)
            if cursor > datetime.utcnow():
                break

            succeeds = rng.random() < reliability
            if succeeds:
                payment = Payment(
                    subscription_id=sub.id,
                    amount=sub.amount,
                    currency=sub.currency,
                    status="success",
                    retry_count=0,
                )
            else:
                reason = rng.choices(
                    list(FAILURE_REASONS.keys()),
                    weights=[0.4, 0.25, 0.2, 0.15],
                )[0]
                payment = Payment(
                    subscription_id=sub.id,
                    amount=sub.amount,
                    currency=sub.currency,
                    status="failed",
                    failure_reason=reason,
                    failed_at=cursor,
                    retry_count=0,
                )
            db.add(payment)
            payments.append(payment)
    db.flush()
    return payments


def backfill_customer_payment_counts(
    db: Session, customers: list[Customer], payments: list[Payment], subscriptions: list[Subscription]
) -> None:
    sub_to_customer = {s.id: s.customer_id for s in subscriptions}
    counts: dict[int, dict[str, int]] = {c.id: {"success": 0, "failed": 0} for c in customers}
    for p in payments:
        customer_id = sub_to_customer[p.subscription_id]
        if p.status == "success":
            counts[customer_id]["success"] += 1
        elif p.status == "failed":
            counts[customer_id]["failed"] += 1

    for customer in customers:
        customer.total_successful_payments = counts[customer.id]["success"]
        customer.total_failed_payments = counts[customer.id]["failed"]
    db.flush()


def generate_recovery_pipeline(
    db: Session,
    rng: random.Random,
    payments: list[Payment],
    subscriptions: list[Subscription],
    customer_reliability: dict[int, float],
    experiments: list[Experiment],
    initial_state: bool = False,
    randomized_assignment: bool = False,
    assignment_seed: int = DEFAULT_ASSIGNMENT_SEED,
) -> dict[str, int]:
    """For every failed Payment: create a RecoveryEvent with an explicit
    control/treatment assignment (is_control / experiment_id / variant).

    Default (full-pipeline) mode then simulates the whole recovery lifecycle:
    control events get a natural-recovery roll, treatment events get 1-3
    diminishing-returns Interventions with Outcomes, AgentEvents and AuditLogs.

    ``initial_state=True`` stops right after the RecoveryEvent (and its
    "recovery_event_created" AuditLog) are written: NO interventions, NO
    outcomes, NO simulated recovery, NO intervention loop. Both control and
    treatment events are left OPEN with ``payment.retry_count = 0`` so the
    live RecoveryOrchestratorService has real work to process."""

    sub_to_customer = {s.id: s.customer_id for s in subscriptions}

    # Experiment/variant assignment uses its own RNG stream so that adding it
    # does not perturb the existing main-RNG sequence -- every other generated
    # value (reliability draws, failure reasons, intervention outcomes, ...)
    # stays exactly as it was before this column was introduced.
    assignment_rng = random.Random(20260901)

    # Dedicated stream for randomized ACTION assignment. Constructed
    # unconditionally (cheap) but only consumed when randomized_assignment is
    # on, so the default path's main-RNG sequence is completely unaffected.
    action_assignment_rng = random.Random(assignment_seed)

    counts = {
        "recovery_events": 0,
        "control_events": 0,
        "treatment_events": 0,
        "open_events": 0,
        "interventions": 0,
        "outcomes": 0,
        "agent_events": 0,
        "audit_logs": 0,
        "randomized_assignments": 0,
    }

    failed_payments = [p for p in payments if p.status == "failed"]

    for payment in failed_payments:
        customer_id = sub_to_customer[payment.subscription_id]
        reliability = customer_reliability[customer_id]
        failure_reason = payment.failure_reason
        if failure_reason is None:
            raise ValueError(f"Failed payment {payment.id} has no failure reason")
        failed_at = payment.failed_at
        if failed_at is None:
            raise ValueError(f"Failed payment {payment.id} has no failure timestamp")
        reason_multiplier = FAILURE_REASONS[failure_reason]

        priority = 2 if payment.amount >= 99900 else (1 if payment.amount >= 19900 else 0)

        # Control/treatment is assigned explicitly on the RecoveryEvent. The
        # is_control draw still comes from the main RNG with the same
        # probability and at the same point in the stream as before, so the
        # control/treatment split is identical to the previous implementation;
        # only the experiment_id / variant metadata is newly recorded.
        is_control = rng.random() < CONTROL_GROUP_FRACTION
        experiment = (
            assignment_rng.choice(experiments) if experiments else None
        )
        variant = "control" if is_control else "treatment"

        recovery_event = RecoveryEvent(
            payment_id=payment.id,
            status="open",
            priority=priority,
            created_at=failed_at,
            is_control=is_control,
            experiment_id=experiment.id if experiment is not None else None,
            variant=variant,
        )
        db.add(recovery_event)
        db.flush()
        counts["recovery_events"] += 1
        counts["control_events" if is_control else "treatment_events"] += 1

        db.add(
            AuditLog(
                recovery_event_id=recovery_event.id,
                actor="system",
                action="recovery_event_created",
                reason=f"Payment failed: {payment.failure_reason}",
            )
        )
        counts["audit_logs"] += 1

        if initial_state:
            # Initial-state mode: hand the freshly-opened event to the
            # orchestrator untouched. No interventions, no outcomes, no
            # simulated recovery -- the payment stays "failed" and the
            # RecoveryEvent stays "open" for both control and treatment.
            payment.retry_count = 0
            counts["open_events"] += 1
            continue

        if is_control:
            # No Intervention rows are created for the control group -- its
            # outcome reflects only natural (un-nudged) recovery.
            natural_recovery_prob = BASELINE_NATURAL_RECOVERY_PROB * reliability * reason_multiplier
            if rng.random() < natural_recovery_prob:
                payment.status = "success"
                payment.recovered_at = failed_at + timedelta(
                    hours=rng.randint(1, 72)
                )
                recovery_event.status = "closed"
                recovery_event.closed_at = payment.recovered_at
            else:
                counts["open_events"] += 1
            continue

        # Treated group: 1-3 intervention attempts, diminishing effectiveness.
        num_attempts = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        recovered_this_event = False
        used_actions: set[str] = set()

        for attempt in range(1, num_attempts + 1):
            if recovered_this_event:
                break  # stopping rule: don't keep intervening once recovered

            if randomized_assignment:
                # Assign uniformly at random from the still-eligible actions
                # (no repeats within an event) using the dedicated stream.
                eligible_actions = [
                    a for a in ACTION_TYPES if a not in used_actions
                ]
                if not eligible_actions:
                    break
                action_type, assignment_propensity = choose_randomized_action(
                    action_assignment_rng, eligible_actions
                )
                used_actions.add(action_type)
            else:
                eligible_actions = None
                assignment_propensity = None
                action_type = rng.choices(
                    list(ACTION_TYPES.keys()),
                    weights=[0.4, 0.25, 0.2, 0.15],
                )[0]
            action_config = ACTION_TYPES[action_type]

            executed_at = recovery_event.created_at + timedelta(
                hours=rng.randint(1, 48) * attempt
            )

            intervention = Intervention(
                recovery_event_id=recovery_event.id,
                action_type=action_type,
                status="executed",
                cost_paise=int(action_config["cost_paise"]),
                agent_reason=(
                    f"attempt {attempt}: reliability={reliability:.2f}, "
                    f"failure_reason={payment.failure_reason}"
                ),
                executed_at=executed_at,
            )
            db.add(intervention)
            db.flush()
            counts["interventions"] += 1

            confidence = min(
                max(reliability * float(action_config["effectiveness"]) * reason_multiplier, 0.05),
                0.95,
            )
            input_context: dict = {
                "attempt": attempt,
                "failure_reason": payment.failure_reason,
                "action_type": action_type,
            }
            if assignment_propensity is not None:
                input_context["selected_action"] = action_type
                input_context["assignment"] = {
                    "chosen_action": action_type,
                    "propensity": assignment_propensity,
                    "exploration": True,
                    "eligible_actions": eligible_actions,
                    "policy_ranking": eligible_actions,
                    "strategy": "uniform",
                    "epsilon": 1.0,
                    "assignment_mechanism": (
                        f"uniform_random(k={len(eligible_actions)})"
                    ),
                    "experiment_id": (
                        str(experiment.id) if experiment is not None else None
                    ),
                    "variant": "treatment",
                    "policy_name": "generator:randomized",
                    "model_version": None,
                    "rng_seed": assignment_seed,
                    "notes": [],
                }
                counts["randomized_assignments"] += 1

            db.add(
                AgentEvent(
                    recovery_event_id=recovery_event.id,
                    event_type="intervention_decision",
                    input_context=input_context,
                    decision=f"execute_{action_type}",
                    confidence=round(confidence, 3),
                )
            )
            counts["agent_events"] += 1

            decay = DIMINISHING_RETURNS_DECAY ** (attempt - 1)
            recovery_prob = min(
                max(
                    reliability
                    * float(action_config["effectiveness"])
                    * reason_multiplier
                    * decay,
                    0.02,
                ),
                0.9,
            )
            recovered = rng.random() < recovery_prob

            outcome = Outcome(
                intervention_id=intervention.id,
                payment_recovered=recovered,
                recovered_amount_paise=payment.amount if recovered else 0,
                recovery_time_seconds=(
                    rng.randint(60, 259200) if recovered else None
                ),
                observed_at=executed_at + timedelta(hours=rng.randint(1, 24)),
            )
            db.add(outcome)
            counts["outcomes"] += 1

            db.add(
                AuditLog(
                    recovery_event_id=recovery_event.id,
                    actor="agent",
                    action=f"intervention_{action_type}_executed",
                    reason=intervention.agent_reason,
                )
            )
            counts["audit_logs"] += 1

            if recovered:
                payment.status = "success"
                payment.recovered_at = outcome.observed_at
                recovery_event.status = "closed"
                recovery_event.closed_at = outcome.observed_at
                recovered_this_event = True

            payment.retry_count += 1

        if not recovered_this_event:
            recovery_event.status = "abandoned"

    db.flush()
    return counts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(
    db: Session,
    customers_n: int,
    subscriptions: list[Subscription],
    payments: list[Payment],
    pipeline_counts: dict[str, int],
    initial_state: bool = False,
) -> None:
    recovered_payments = [p for p in payments if p.recovered_at is not None]
    total_payment_value = sum(p.amount for p in payments)
    total_recovered_value = sum(p.amount for p in recovered_payments)

    # Every payment that ever failed is counted once, regardless of current status.
    ever_failed = [
        p for p in payments if p.failure_reason is not None
    ]
    recovery_rate = (
        (len(recovered_payments) / len(ever_failed)) * 100 if ever_failed else 0.0
    )

    print("Synthetic data generation complete.\n")
    print(f"Mode: {'initial-state' if initial_state else 'full-pipeline'}")
    print(f"Customers: {customers_n}")
    print(f"Subscriptions: {len(subscriptions)}")
    print(f"Payments: {len(payments)}")
    print(f"Failed payments: {len(ever_failed)}")
    print(f"Recovery events: {pipeline_counts['recovery_events']}")
    print(f"  control events: {pipeline_counts['control_events']}")
    print(f"  treatment events: {pipeline_counts['treatment_events']}")
    print(f"  open events: {pipeline_counts['open_events']}")
    print(f"Interventions: {pipeline_counts['interventions']}")
    if pipeline_counts.get("randomized_assignments"):
        print(
            f"  randomized action assignments (with logged propensity): "
            f"{pipeline_counts['randomized_assignments']}"
        )
    print(f"Outcomes: {pipeline_counts['outcomes']}")
    print(f"Agent events: {pipeline_counts['agent_events']}")
    print(f"Audit logs: {pipeline_counts['audit_logs']}")
    print()
    print(f"Total payment value: \u20b9{total_payment_value / 100:,.2f}")
    if initial_state:
        print(
            "All recovery events left OPEN with no interventions/outcomes; "
            "ready for the RecoveryOrchestratorService."
        )
    else:
        print(f"Recovered payments: {len(recovered_payments)}")
        print(f"Total recovered value: \u20b9{total_recovered_value / 100:,.2f}")
        print(f"Recovery rate: {recovery_rate:.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    customers_count: int,
    seed: int,
    do_reset: bool,
    initial_state: bool = False,
    randomized_assignment: bool = False,
    assignment_seed: int = DEFAULT_ASSIGNMENT_SEED,
) -> None:
    rng = random.Random(seed)
    db: Session = SessionLocal()

    try:
        if do_reset:
            reset_application_data(db)

        mode = "initial-state" if initial_state else "full-pipeline"
        print(
            f"Generating synthetic data (mode={mode}, "
            f"customers={customers_count}, seed={seed})...\n"
        )

        experiments = generate_experiments(db, rng)
        customers = generate_customers(db, rng, customers_count)

        customer_reliability = {c.id: _customer_reliability(rng) for c in customers}

        subscriptions = generate_subscriptions(db, rng, customers)
        payments = generate_payments(db, rng, subscriptions, customer_reliability)

        backfill_customer_payment_counts(db, customers, payments, subscriptions)

        if randomized_assignment and initial_state:
            raise ValueError(
                "--randomized-assignment applies to the full-pipeline mode "
                "only; --initial-state creates no interventions to assign"
            )

        pipeline_counts = generate_recovery_pipeline(
            db, rng, payments, subscriptions, customer_reliability, experiments,
            initial_state=initial_state,
            randomized_assignment=randomized_assignment,
            assignment_seed=assignment_seed,
        )

        db.commit()
        print_summary(
            db, customers_count, subscriptions, payments, pipeline_counts,
            initial_state=initial_state,
        )

    except Exception as exc:
        db.rollback()
        print(f"\nData generation FAILED and was rolled back.\nError: {exc}")
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic recovery data.")
    parser.add_argument(
        "--customers", type=int, default=1000, help="Number of customers to generate."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic random seed."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing application data before generating (schema/migrations untouched).",
    )
    parser.add_argument(
        "--initial-state",
        action="store_true",
        help=(
            "Generate only the initial recovery state: customers, subscriptions, "
            "payments and OPEN RecoveryEvents (with control/treatment assignment), "
            "but no interventions, outcomes or simulated recovery. Use this to feed "
            "the RecoveryOrchestratorService."
        ),
    )
    parser.add_argument(
        "--randomized-assignment",
        action="store_true",
        help=(
            "Full-pipeline only: assign treatment intervention actions uniformly "
            "at random from the eligible set and record the assignment propensity "
            "on each AgentEvent (input_context['assignment']). Produces data "
            "suitable for later causal/uplift estimation. Default off (unchanged)."
        ),
    )
    parser.add_argument(
        "--assignment-seed",
        type=int,
        default=DEFAULT_ASSIGNMENT_SEED,
        help=(
            "Deterministic seed for the randomized-assignment RNG stream "
            f"(default {DEFAULT_ASSIGNMENT_SEED}). Independent of --seed."
        ),
    )
    args = parser.parse_args()

    if args.reset:
        print("WARNING: --reset will delete existing application data (not the schema).")

    run(
        customers_count=args.customers,
        seed=args.seed,
        do_reset=args.reset,
        initial_state=args.initial_state,
        randomized_assignment=args.randomized_assignment,
        assignment_seed=args.assignment_seed,
    )


if __name__ == "__main__":
    main()