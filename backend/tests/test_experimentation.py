"""Unit tests for the action-assignment / experimentation layer.

No database: these exercise the strategy maths, propensity correctness,
determinism and eligibility guarantees in isolation.
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from app.services.experimentation import (
    DEFAULT_EXPERIMENT_CONFIG,
    ActionAssigner,
    ExperimentConfig,
    load_experiment_config,
)
from app.services.experimentation.config import (
    STRATEGY_EPSILON_GREEDY,
    STRATEGY_EXPLOIT,
    STRATEGY_UNIFORM,
)
from app.services.recovery_policy import CandidateAction, PolicyDecision


def _decision(*actions: str) -> PolicyDecision:
    """A ranked PolicyDecision over the given action names (best-first)."""
    cands = [
        CandidateAction(
            action_type=a,
            cost_paise=10,
            estimated_recovery_probability=0.1,
            expected_value_paise=100.0 - i,
            score=100.0 - i,
            confidence=0.1,
            reason=f"stub {a}",
        )
        for i, a in enumerate(actions)
    ]
    return PolicyDecision(should_intervene=True, candidates=cands, rationale="stub")


FOUR = ("method_switch_prompt", "retry", "sms_nudge", "whatsapp_nudge")


# --- config ------------------------------------------------------------

def test_default_config_is_disabled_and_exploit():
    c = DEFAULT_EXPERIMENT_CONFIG
    assert c.enabled is False
    assert c.effective_strategy == STRATEGY_EXPLOIT
    assert c.explores is False


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        ExperimentConfig(epsilon=1.5)
    with pytest.raises(ValueError):
        ExperimentConfig(strategy="bandit_9000")
    with pytest.raises(ValueError):
        ExperimentConfig(allowed_actions=("retry", "not_a_real_action"))


def test_disabled_config_ignores_strategy_and_epsilon():
    c = ExperimentConfig(enabled=False, strategy=STRATEGY_EPSILON_GREEDY, epsilon=0.9)
    assert c.effective_strategy == STRATEGY_EXPLOIT
    assert c.explores is False


def test_load_experiment_config_from_env():
    env = {
        "EXPERIMENTATION_ENABLED": "true",
        "EXPERIMENTATION_EPSILON": "0.2",
        "EXPERIMENTATION_SEED": "123",
        "EXPERIMENTATION_EXPERIMENT_ID": "exp_a",
    }
    c = load_experiment_config(env=env)
    assert c.enabled and c.epsilon == 0.2 and c.seed == 123
    assert c.experiment_id == "exp_a"
    # epsilon > 0 with no explicit strategy implies epsilon-greedy
    assert c.strategy == STRATEGY_EPSILON_GREEDY


def test_load_experiment_config_overrides_beat_env():
    env = {"EXPERIMENTATION_EPSILON": "0.2"}
    c = load_experiment_config({"enabled": True, "epsilon": 0.5}, env=env)
    assert c.epsilon == 0.5 and c.enabled


# --- epsilon = 0 : always exploit ------------------------------------

def test_epsilon_zero_always_exploits_top_ranked():
    d = _decision(*FOUR)
    assigner = ActionAssigner(DEFAULT_EXPERIMENT_CONFIG)
    for re_id in range(50):
        r = assigner.assign(
            decision=d, recovery_event_id=re_id, policy_name="rules_v1"
        )
        assert r.chosen_action == "method_switch_prompt"
        assert r.propensity == 1.0
        assert r.exploration is False
        assert r.assignment_mechanism.startswith("exploit")


def test_single_eligible_action_has_propensity_one():
    d = _decision("retry")
    cfg = ExperimentConfig(enabled=True, strategy=STRATEGY_EPSILON_GREEDY, epsilon=0.5)
    r = ActionAssigner(cfg, rng=random.Random(0)).assign(
        decision=d, recovery_event_id=1, policy_name="rules_v1"
    )
    assert r.chosen_action == "retry"
    assert r.propensity == 1.0
    assert r.exploration is False


# --- epsilon > 0 : exploration occurs & propensity is correct --------

def test_epsilon_greedy_exploration_frequency_and_propensity():
    d = _decision(*FOUR)
    epsilon = 0.4
    k = 4
    cfg = ExperimentConfig(
        enabled=True, strategy=STRATEGY_EPSILON_GREEDY, epsilon=epsilon
    )
    rng = random.Random(20260902)
    assigner = ActionAssigner(cfg, rng=rng)

    n = 40_000
    counts: Counter[str] = Counter()
    explore_hits = 0
    prop_by_action: dict[str, float] = {}
    for _ in range(n):
        r = assigner.assign(
            decision=d, recovery_event_id=1, policy_name="rules_v1"
        )
        counts[r.chosen_action] += 1
        explore_hits += int(r.exploration)
        prop_by_action[r.chosen_action] = r.propensity

    # exploration rate ~ epsilon
    assert abs(explore_hits / n - epsilon) < 0.02

    # theoretical marginal probabilities
    top = "method_switch_prompt"
    p_top = 1 - epsilon
    p_other = epsilon / (k - 1)
    assert abs(counts[top] / n - p_top) < 0.02
    for a in FOUR[1:]:
        assert abs(counts[a] / n - p_other) < 0.02

    # recorded propensity matches the mechanism exactly
    assert prop_by_action[top] == round(p_top, 8)
    for a in FOUR[1:]:
        assert prop_by_action[a] == round(p_other, 8)


def test_uniform_strategy_propensity():
    d = _decision(*FOUR)
    cfg = ExperimentConfig(enabled=True, strategy=STRATEGY_UNIFORM)
    rng = random.Random(1)
    assigner = ActionAssigner(cfg, rng=rng)
    seen = set()
    for _ in range(2000):
        r = assigner.assign(
            decision=d, recovery_event_id=1, policy_name="rules_v1"
        )
        assert r.propensity == round(1 / 4, 8)
        seen.add(r.chosen_action)
    assert seen == set(FOUR)  # every eligible action gets sampled


# --- eligibility : only policy candidates are ever chosen ------------

def test_only_eligible_actions_are_sampled():
    d = _decision("retry", "sms_nudge")  # policy offered only 2
    cfg = ExperimentConfig(
        enabled=True, strategy=STRATEGY_EPSILON_GREEDY, epsilon=0.9
    )
    assigner = ActionAssigner(cfg, rng=random.Random(3))
    for _ in range(1000):
        r = assigner.assign(
            decision=d, recovery_event_id=1, policy_name="rules_v1"
        )
        assert r.chosen_action in {"retry", "sms_nudge"}
        assert set(r.eligible_actions) == {"retry", "sms_nudge"}


def test_allowed_actions_narrows_eligible_set():
    d = _decision(*FOUR)
    cfg = ExperimentConfig(
        enabled=True,
        strategy=STRATEGY_EPSILON_GREEDY,
        epsilon=0.9,
        allowed_actions=("sms_nudge", "whatsapp_nudge"),
    )
    assigner = ActionAssigner(cfg, rng=random.Random(4))
    for _ in range(500):
        r = assigner.assign(
            decision=d, recovery_event_id=1, policy_name="rules_v1"
        )
        assert r.chosen_action in {"sms_nudge", "whatsapp_nudge"}
        assert r.eligible_actions == ["sms_nudge", "whatsapp_nudge"]


def test_allowed_actions_disjoint_from_candidates_falls_back_safely():
    d = _decision("retry", "sms_nudge")
    cfg = ExperimentConfig(
        enabled=True,
        strategy=STRATEGY_EPSILON_GREEDY,
        epsilon=0.9,
        allowed_actions=("whatsapp_nudge",),  # not offered by the policy
    )
    r = ActionAssigner(cfg, rng=random.Random(5)).assign(
        decision=d, recovery_event_id=1, policy_name="rules_v1"
    )
    assert r.chosen_action == "retry"  # top of the policy ranking
    assert r.propensity == 1.0
    assert r.exploration is False
    assert any("fell back" in n for n in r.notes)


def test_eligible_set_preserves_policy_rank_order():
    d = _decision("whatsapp_nudge", "retry", "sms_nudge")
    r = ActionAssigner(DEFAULT_EXPERIMENT_CONFIG).assign(
        decision=d, recovery_event_id=1, policy_name="rules_v1"
    )
    assert r.eligible_actions == ["whatsapp_nudge", "retry", "sms_nudge"]
    assert r.policy_ranking == ["whatsapp_nudge", "retry", "sms_nudge"]


# --- determinism -----------------------------------------------------

def test_same_seed_same_sequence():
    d = _decision(*FOUR)
    cfg = ExperimentConfig(
        enabled=True, strategy=STRATEGY_EPSILON_GREEDY, epsilon=0.5, seed=99
    )

    def run() -> list[str]:
        assigner = ActionAssigner(cfg)  # derives Random(f"{seed}:{re_id}")
        return [
            assigner.assign(
                decision=d, recovery_event_id=i % 7, policy_name="rules_v1"
            ).chosen_action
            for i in range(200)
        ]

    assert run() == run()


def test_per_event_rng_is_reproducible_for_a_given_event():
    d = _decision(*FOUR)
    cfg = ExperimentConfig(
        enabled=True, strategy=STRATEGY_EPSILON_GREEDY, epsilon=0.7, seed=1234
    )
    a1 = ActionAssigner(cfg).assign(
        decision=d, recovery_event_id=555, policy_name="rules_v1"
    )
    a2 = ActionAssigner(cfg).assign(
        decision=d, recovery_event_id=555, policy_name="rules_v1"
    )
    assert a1.chosen_action == a2.chosen_action
    assert a1.propensity == a2.propensity


def test_assign_raises_without_candidates():
    empty = PolicyDecision(should_intervene=False, candidates=[], rationale="none")
    with pytest.raises(ValueError):
        ActionAssigner(DEFAULT_EXPERIMENT_CONFIG).assign(
            decision=empty, recovery_event_id=1, policy_name="rules_v1"
        )
