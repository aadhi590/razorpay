"""AnalyticsService.action_incrementality -- the observed, per-action-type
incremental lift consumed by the agent tool.

It deliberately reuses the recovery-impact machinery, so these tests mirror
``test_recovery_impact.py``'s pattern: hand-computable constructed datasets via
``build_dataset`` + ``?experiment_id`` isolation, plus the real global batch.
"""
from __future__ import annotations

from app.services.analytics_service import AnalyticsService
from app.services.experimentation import MIN_DISTINCT_EVENTS_PER_ACTION
from app.services.proportion_stats import newcombe_difference_interval
from tests.analytics.conftest import build_dataset


def _inc(db, action_type, experiment_id):
    return AnalyticsService(db).action_incrementality(
        action_type, experiment_id=experiment_id
    )


# --- 1. exact calculation, reusing the same control + formula ----------
def test_exact_calculation_matches_recovery_impact_shape(impact_db):
    # control: 40 events, 4 recovered            -> control rate 0.10
    # treated-with-whatsapp: 40 events, 14 recovered -> action rate 0.35
    # observed incremental lift = 0.35 - 0.10 = 0.25
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 40,
        control_recovered=[True] * 4 + [False] * 36,
        treated=[10000] * 40,
        treated_recovered=[True] * 14 + [False] * 26,
        treated_action="whatsapp_nudge",
    )
    r = _inc(impact_db, "whatsapp_nudge", exp_id)

    assert r.computable is True
    assert r.action_type == "whatsapp_nudge"
    assert r.treated_group_size == 40
    assert r.control_group_size == 40
    assert r.recovered_treated_events == 14
    assert r.recovered_control_events == 4
    assert r.observed_recovery_rate_for_action == 0.35
    assert r.baseline_control_recovery_rate == 0.1
    assert r.observed_incremental_lift == 0.25

    # same subtraction recovery_impact.incremental_recovery_rate performs
    assert r.observed_incremental_lift == round(
        r.observed_recovery_rate_for_action - r.baseline_control_recovery_rate, 6
    )
    # same Newcombe helper, treated = a, control = b
    ci = newcombe_difference_interval(14, 40, 4, 40)
    assert r.observed_incremental_lift_ci_95 == ci.as_list()
    assert r.confidence_method == "newcombe_wilson_95_difference"


def test_only_the_named_action_is_counted(impact_db):
    """Two treated arms with different actions in one experiment; the tool must
    isolate the requested action_type and not blend them."""
    build_dataset(
        impact_db,
        control=[10000] * 40, control_recovered=[True] * 4 + [False] * 36,
        treated=[10000] * 40, treated_recovered=[True] * 20 + [False] * 20,
        treated_action="whatsapp_nudge",
    )
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 40, control_recovered=[True] * 4 + [False] * 36,
        treated=[10000] * 40, treated_recovered=[True] * 2 + [False] * 38,
        treated_action="retry",
    )
    # ask about retry in the second experiment: 2/40 recovered
    r = _inc(impact_db, "retry", exp_id)
    assert r.treated_group_size == 40
    assert r.recovered_treated_events == 2
    assert r.observed_recovery_rate_for_action == 0.05


# --- 2. small-sample discipline (same threshold as assignment-coverage) --
def test_insufficient_history_is_explicit_not_fabricated(impact_db):
    n = MIN_DISTINCT_EVENTS_PER_ACTION - 1  # just below the stable-estimate floor
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 100, control_recovered=[True] * 3 + [False] * 97,
        treated=[10000] * n, treated_recovered=[True] * (n // 2) + [False] * (n - n // 2),
        treated_action="method_switch_prompt",
    )
    r = _inc(impact_db, "method_switch_prompt", exp_id)
    assert r.computable is False
    assert r.reason == "insufficient_historical_data"
    assert r.treated_group_size == n
    assert r.observed_recovery_rate_for_action is None
    assert r.observed_incremental_lift is None
    assert r.observed_incremental_lift_ci_95 is None
    assert r.confidence_method == "not_computed"
    assert str(MIN_DISTINCT_EVENTS_PER_ACTION) in r.sample_size_note


def test_zero_history_for_action(impact_db):
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 50, control_recovered=[True] * 2 + [False] * 48,
        treated=[10000] * 40, treated_recovered=[True] * 10 + [False] * 30,
        treated_action="sms_nudge",
    )
    # no 'retry' interventions exist in this experiment
    r = _inc(impact_db, "retry", exp_id)
    assert r.computable is False
    assert r.reason == "no_historical_data"
    assert r.treated_group_size == 0


def test_empty_control_baseline(impact_db):
    exp_id = build_dataset(
        impact_db,
        control=[], control_recovered=[],
        treated=[10000] * 40, treated_recovered=[True] * 10 + [False] * 30,
        treated_action="whatsapp_nudge",
    )
    r = _inc(impact_db, "whatsapp_nudge", exp_id)
    assert r.computable is False
    assert r.reason == "no_control_baseline"
    assert r.control_group_size == 0
    assert r.confidence_method == "not_computed"


# --- 3. note wording is consistent with the batch endpoint's standard --
def test_note_mentions_newcombe_and_batch_consistency(impact_db):
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 200, control_recovered=[True] * 4 + [False] * 196,
        treated=[10000] * 200, treated_recovered=[True] * 60 + [False] * 140,
        treated_action="method_switch_prompt",
    )
    r = _inc(impact_db, "method_switch_prompt", exp_id)
    note = r.sample_size_note
    assert "Newcombe/Wilson" in note and "not a p-value" in note
    assert "recovery-impact" in note                     # cross-references the batch metric
    assert "excludes zero" in note                       # strong lift here
    assert "200 historical uses" in note


def test_note_flags_a_wide_interval_as_directional(impact_db):
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 40, control_recovered=[True] * 10 + [False] * 30,   # 0.25
        treated=[10000] * 40, treated_recovered=[True] * 12 + [False] * 28,   # 0.30
        treated_action="retry",
    )
    r = _inc(impact_db, "retry", exp_id)
    lo, hi = r.observed_incremental_lift_ci_95
    assert lo < 0.0 < hi
    assert "includes zero" in r.sample_size_note
    assert "directional" in r.sample_size_note


# --- 4. against the real global dataset -------------------------------
def test_global_uses_the_shared_control_arm(impact_db):
    """No experiment filter: the control group must be the whole is_control arm,
    identical to what recovery_impact reports."""
    svc = AnalyticsService(impact_db)
    impact = svc.recovery_impact()
    r = svc.action_incrementality("whatsapp_nudge")

    assert r.computable is True
    assert r.control_group_size == impact.control_group_size
    assert r.recovered_control_events == impact.recovered_control_events
    assert r.baseline_control_recovery_rate == impact.control_recovery_rate
    # every real action type is well-sampled
    assert r.treated_group_size >= MIN_DISTINCT_EVENTS_PER_ACTION
    assert r.observed_incremental_lift == round(
        r.observed_recovery_rate_for_action - r.baseline_control_recovery_rate, 6
    )
