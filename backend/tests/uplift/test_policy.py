"""Phase 8: UpliftRecoveryPolicy -- ranking, control, safety, fallback."""
from __future__ import annotations

import pytest

from app.services.recovery_policy import PolicyContext, RulesBasedRecoveryPolicy
from app.services.uplift_recovery_policy import UpliftRecoveryPolicy

pytestmark = pytest.mark.needs_data


def _ctx(**kw) -> PolicyContext:
    base = dict(
        failure_reason="insufficient_funds",
        amount_paise=99900,
        priority=2,
        is_control=False,
        attempt_number=1,
        prior_action_types=[],
        customer_successful_payments=9,
        customer_failed_payments=2,
        recovery_event_id=None,
    )
    base.update(kw)
    return PolicyContext(**base)


def test_ranks_by_incremental_value(db_session, open_treatment_event, uplift_model):
    policy = UpliftRecoveryPolicy(db_session, model=uplift_model)
    decision = policy.decide(_ctx(recovery_event_id=open_treatment_event["treatment_id"]))
    assert decision.should_intervene
    assert len(decision.candidates) == 4
    scores = [c.score for c in decision.candidates]
    assert scores == sorted(scores, reverse=True)
    assert "uplift" in decision.candidates[0].reason
    assert "baseline" in decision.candidates[0].reason
    assert policy.name.startswith("uplift:")
    assert policy.model_version is not None


def test_control_event_no_intervention(db_session, open_treatment_event, uplift_model):
    policy = UpliftRecoveryPolicy(db_session, model=uplift_model)
    decision = policy.decide(
        _ctx(is_control=True, recovery_event_id=open_treatment_event["control_id"])
    )
    assert decision.should_intervene is False
    assert decision.candidates == []


def test_already_tried_actions_excluded(db_session, open_treatment_event, uplift_model):
    policy = UpliftRecoveryPolicy(db_session, model=uplift_model)
    decision = policy.decide(
        _ctx(
            recovery_event_id=open_treatment_event["treatment_id"],
            prior_action_types=["retry", "sms_nudge"],
        )
    )
    kinds = {c.action_type for c in decision.candidates}
    assert "retry" not in kinds and "sms_nudge" not in kinds


def test_fallback_when_no_recovery_event_id(db_session, uplift_model):
    policy = UpliftRecoveryPolicy(db_session, model=uplift_model)
    decision = policy.decide(_ctx(recovery_event_id=None))
    assert "uplift fallback" in decision.rationale


def test_fallback_when_model_unavailable(db_session):
    policy = UpliftRecoveryPolicy(db_session, model=None)
    policy._model = None
    policy.name = "rules_v1+uplift_unavailable"
    policy._fallback = RulesBasedRecoveryPolicy()
    decision = policy.decide(_ctx(recovery_event_id=1))
    assert "uplift fallback" in decision.rationale
    rules = RulesBasedRecoveryPolicy().decide(_ctx(recovery_event_id=1))
    assert [c.action_type for c in decision.candidates] == [
        c.action_type for c in rules.candidates
    ]


def test_never_raises_on_bad_event(db_session, uplift_model):
    policy = UpliftRecoveryPolicy(db_session, model=uplift_model)
    decision = policy.decide(_ctx(recovery_event_id=10**9))
    assert "uplift fallback" in decision.rationale


def test_declines_when_no_positive_incremental_value(db_session, open_treatment_event, uplift_model):
    """With an absurd min threshold the policy must decline, not intervene."""
    policy = UpliftRecoveryPolicy(
        db_session, model=uplift_model, min_net_incremental_value_paise=10**12
    )
    decision = policy.decide(_ctx(recovery_event_id=open_treatment_event["treatment_id"]))
    assert decision.should_intervene is False
    assert "incremental" in decision.rationale


def test_orchestrator_respects_uplift_policy(db_session, open_treatment_event, uplift_model):
    from app.services.recovery_orchestrator import (
        DISP_INTERVENTION_CREATED,
        RecoveryOrchestratorService,
    )

    svc = RecoveryOrchestratorService(
        db_session, policy=UpliftRecoveryPolicy(db_session, model=uplift_model)
    )
    outcome = svc.orchestrate_event(open_treatment_event["treatment_id"])
    assert outcome.disposition == DISP_INTERVENTION_CREATED
    assert outcome.selected_action in {
        "retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"
    }
    # control event is skipped
    c_outcome = svc.orchestrate_event(open_treatment_event["control_id"])
    assert c_outcome.action_taken is False
