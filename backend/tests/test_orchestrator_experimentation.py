"""Integration tests: the experimentation layer inside the orchestrator.

Uses the isolated ``open_treatment_event`` fixture (one open treatment event +
one control event) from ``tests/conftest.py`` -- no full-pipeline data needed.
"""
from __future__ import annotations

import random

import pytest
from sqlalchemy import select

from app.models.agent_events import AgentEvent
from app.models.interventions import Intervention
from app.models.recovery_events import RecoveryEvent
from app.services.experimentation import ActionAssigner, ExperimentConfig
from app.services.experimentation.config import STRATEGY_EPSILON_GREEDY
from app.services.recovery_orchestrator import RecoveryOrchestratorService
from app.services.recovery_policy import (
    PolicyContext,
    RulesBasedRecoveryPolicy,
)

_ASSIGNMENT_KEYS = {
    "chosen_action",
    "propensity",
    "exploration",
    "eligible_actions",
    "policy_ranking",
    "strategy",
    "epsilon",
    "assignment_mechanism",
    "experiment_id",
    "variant",
    "policy_name",
    "model_version",
    "rng_seed",
}


def _context_for(db_session, recovery_event_id: int) -> PolicyContext:
    """Rebuild the PolicyContext the orchestrator would construct for a
    first-attempt decision on this event."""
    event = db_session.get(RecoveryEvent, recovery_event_id)
    payment = event.payment
    customer = payment.subscription.customer
    return PolicyContext(
        failure_reason=payment.failure_reason,
        amount_paise=payment.amount,
        priority=event.priority,
        is_control=event.is_control,
        attempt_number=1,
        prior_action_types=[],
        customer_successful_payments=customer.total_successful_payments,
        customer_failed_payments=customer.total_failed_payments,
        recovery_event_id=event.id,
    )


def _agent_event(db_session, recovery_event_id: int) -> AgentEvent | None:
    return db_session.scalars(
        select(AgentEvent).where(
            AgentEvent.recovery_event_id == recovery_event_id,
            AgentEvent.event_type == "intervention_decision",
        )
    ).first()


# --- A. epsilon = 0 -> always exploit the top-ranked candidate --------

def test_epsilon_zero_orchestrate_exploits_top_ranked(
    db_session, open_treatment_event
):
    tid = open_treatment_event["treatment_id"]
    policy = RulesBasedRecoveryPolicy()
    expected_top = policy.decide(_context_for(db_session, tid)).selected.action_type

    service = RecoveryOrchestratorService(db_session, policy=policy)  # default assigner
    outcome = service.orchestrate_event(tid)

    assert outcome.disposition == "intervention_created"
    assert outcome.selected_action == expected_top
    assert outcome.propensity == 1.0
    assert outcome.exploration is False
    assert outcome.assignment["assignment_mechanism"].startswith("exploit")

    ae = _agent_event(db_session, tid)
    assert ae.input_context["assignment"]["propensity"] == 1.0
    assert ae.input_context["assignment"]["chosen_action"] == expected_top


# --- B. epsilon > 0 -> exploration occurs ---------------------------

def test_epsilon_one_orchestrate_explores_off_top(db_session, open_treatment_event):
    tid = open_treatment_event["treatment_id"]
    policy = RulesBasedRecoveryPolicy()
    decision = policy.decide(_context_for(db_session, tid))
    top = decision.selected.action_type
    k = len(decision.candidates)
    assert k >= 2  # this fixture's context yields all four actions

    cfg = ExperimentConfig(
        enabled=True,
        strategy=STRATEGY_EPSILON_GREEDY,
        epsilon=1.0,
        seed=4242,
        experiment_id="exp_test",
    )
    service = RecoveryOrchestratorService(
        db_session, policy=policy, assigner=ActionAssigner(cfg)
    )
    outcome = service.orchestrate_event(tid)

    assert outcome.disposition == "intervention_created"
    assert outcome.exploration is True
    assert outcome.selected_action != top
    assert outcome.selected_action in {c.action_type for c in decision.candidates}
    # C. exact propensity for epsilon=1 uniform-over-others
    assert outcome.propensity == round(1.0 / (k - 1), 8)
    assert outcome.assignment["experiment_id"] == "exp_test"


# --- D. only eligible actions are ever sampled ----------------------

def test_orchestrate_never_assigns_an_already_tried_action(
    db_session, open_treatment_event
):
    tid = open_treatment_event["treatment_id"]
    policy = RulesBasedRecoveryPolicy()
    cfg = ExperimentConfig(
        enabled=True, strategy=STRATEGY_EPSILON_GREEDY, epsilon=1.0, seed=7
    )
    service = RecoveryOrchestratorService(
        db_session, policy=policy, assigner=ActionAssigner(cfg)
    )

    tried: list[str] = []
    for _ in range(3):  # MAX_INTERVENTION_ATTEMPTS
        outcome = service.orchestrate_event(tid)
        if outcome.disposition != "intervention_created":
            break
        assert outcome.selected_action not in tried
        tried.append(outcome.selected_action)

    assert len(tried) == len(set(tried))  # never repeated an action


# --- E. control events create no intervention & no assignment -------

def test_control_event_gets_no_intervention_or_assignment(
    db_session, open_treatment_event
):
    cid = open_treatment_event["control_id"]
    cfg = ExperimentConfig(
        enabled=True, strategy=STRATEGY_EPSILON_GREEDY, epsilon=1.0, seed=1
    )
    service = RecoveryOrchestratorService(
        db_session,
        policy=RulesBasedRecoveryPolicy(),
        assigner=ActionAssigner(cfg),
    )
    outcome = service.orchestrate_event(cid)

    assert outcome.disposition == "skipped_control"
    assert outcome.action_taken is False
    assert outcome.assignment is None
    assert outcome.propensity is None

    assert (
        db_session.scalars(
            select(Intervention).where(Intervention.recovery_event_id == cid)
        ).first()
        is None
    )
    assert _agent_event(db_session, cid) is None


# --- F. assignment metadata is persisted ---------------------------

def test_assignment_metadata_is_persisted_in_agent_event(
    db_session, open_treatment_event
):
    tid = open_treatment_event["treatment_id"]
    cfg = ExperimentConfig(
        enabled=True,
        strategy=STRATEGY_EPSILON_GREEDY,
        epsilon=0.5,
        seed=11,
        experiment_id="exp_persist",
    )
    service = RecoveryOrchestratorService(
        db_session,
        policy=RulesBasedRecoveryPolicy(),
        assigner=ActionAssigner(cfg),
    )
    service.orchestrate_event(tid)

    block = _agent_event(db_session, tid).input_context["assignment"]
    assert _ASSIGNMENT_KEYS <= set(block)
    assert block["experiment_id"] == "exp_persist"
    assert block["variant"] == "treatment"
    assert block["policy_name"] == "rules_v1"
    assert block["rng_seed"] == 11
    assert 0.0 < block["propensity"] <= 1.0
    assert block["chosen_action"] in block["eligible_actions"]


# --- G. deterministic for a given (seed, event) -------------------

def test_assignment_is_deterministic_for_seed_and_event():
    """The per-decision RNG is Random(f'{seed}:{event_id}') -> reproducible."""
    from tests.test_experimentation import _decision

    d = _decision("method_switch_prompt", "retry", "sms_nudge", "whatsapp_nudge")
    cfg = ExperimentConfig(
        enabled=True, strategy=STRATEGY_EPSILON_GREEDY, epsilon=0.6, seed=2026
    )
    first = ActionAssigner(cfg).assign(
        decision=d, recovery_event_id=900, policy_name="rules_v1"
    )
    second = ActionAssigner(cfg).assign(
        decision=d, recovery_event_id=900, policy_name="rules_v1"
    )
    assert (first.chosen_action, first.propensity) == (
        second.chosen_action,
        second.propensity,
    )


# --- H / I. existing policy behaviour unchanged when disabled -----

def test_rules_policy_selection_unchanged_by_disabled_experimentation(
    db_session, open_treatment_event
):
    tid = open_treatment_event["treatment_id"]
    policy = RulesBasedRecoveryPolicy()
    expected = policy.decide(_context_for(db_session, tid)).selected.action_type

    service = RecoveryOrchestratorService(db_session, policy=policy)
    outcome = service.orchestrate_event(tid)
    assert outcome.selected_action == expected


@pytest.mark.needs_data
def test_ml_policy_selection_unchanged_when_experimentation_disabled(
    db_session, open_treatment_event, trained_model
):
    from app.services.ml_recovery_policy import MLRecoveryPolicy

    tid = open_treatment_event["treatment_id"]
    policy = MLRecoveryPolicy(db_session, model=trained_model)
    expected = policy.decide(_context_for(db_session, tid)).selected.action_type

    service = RecoveryOrchestratorService(db_session, policy=policy)  # default assigner
    outcome = service.orchestrate_event(tid)
    assert outcome.disposition == "intervention_created"
    assert outcome.selected_action == expected
    assert outcome.propensity == 1.0
    assert outcome.exploration is False
