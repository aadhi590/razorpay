"""GET /api/v1/analytics/recovery-impact -- correctness against hand-computable
datasets, empty-control handling, and the confidence note/method."""
from __future__ import annotations

import math

from app.services.analytics_service import AnalyticsService
from app.services.proportion_stats import (
    newcombe_difference_interval,
    wilson_interval,
)
from tests.analytics.conftest import build_dataset


def _impact(db, experiment_id):
    return AnalyticsService(db).recovery_impact(experiment_id=experiment_id)


# --- 1. exact calculation on a constructed dataset -------------------
def test_exact_calculation(impact_db):
    # control: 10 events @ 10000p, 2 recovered  -> rate 0.20, at_risk 100000, rev 20000
    # treated: 20 events @ 10000p, 8 recovered  -> rate 0.40, at_risk 200000, rev 80000
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 10,
        control_recovered=[True, True] + [False] * 8,
        treated=[10000] * 20,
        treated_recovered=[True] * 8 + [False] * 12,
    )
    r = _impact(impact_db, exp_id)

    assert r.computable is True
    assert r.control_group_size == 10
    assert r.treated_group_size == 20
    assert r.recovered_control_events == 2
    assert r.recovered_treated_events == 8
    assert r.control_recovery_rate == 0.2
    assert r.treated_recovery_rate == 0.4
    assert r.incremental_recovery_rate == 0.2

    assert r.control_at_risk_amount_paise == 100000
    assert r.treated_at_risk_amount_paise == 200000
    assert r.control_recovered_revenue_paise == 20000
    assert r.treated_recovered_revenue_paise == 80000
    assert r.total_recovered_revenue_paise == 80000

    # incremental_revenue = 80000 - round(0.2 * 200000) = 80000 - 40000 = 40000
    assert r.incremental_revenue_recovered_paise == 40000

    # CI matches the standalone Newcombe helper
    ci = newcombe_difference_interval(8, 20, 2, 10)
    assert r.incremental_recovery_rate_ci_95 == ci.as_list()


def test_calculation_with_varying_amounts(impact_db):
    # treated recovered events skew high -> treated_recovered_revenue is NOT
    # just rate * at_risk, so the formula must use the ACTUAL recovered revenue.
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 4,
        control_recovered=[True] + [False] * 3,          # rate 0.25
        treated=[10000, 10000, 50000, 50000],
        treated_recovered=[False, False, True, True],    # rate 0.50, rev 100000
    )
    r = _impact(impact_db, exp_id)
    assert r.control_recovery_rate == 0.25
    assert r.treated_recovery_rate == 0.5
    assert r.treated_at_risk_amount_paise == 120000
    assert r.treated_recovered_revenue_paise == 100000
    # 100000 - round(0.25 * 120000) = 100000 - 30000 = 70000
    assert r.incremental_revenue_recovered_paise == 70000


def test_negative_incremental_is_left_signed(impact_db):
    # treatment does WORSE than control -> incremental revenue negative, not clamped
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 10,
        control_recovered=[True] * 5 + [False] * 5,       # rate 0.5
        treated=[10000] * 10,
        treated_recovered=[True] * 2 + [False] * 8,       # rate 0.2, rev 20000
    )
    r = _impact(impact_db, exp_id)
    assert r.incremental_recovery_rate == -0.3
    # 20000 - round(0.5 * 100000) = 20000 - 50000 = -30000
    assert r.incremental_revenue_recovered_paise == -30000


# --- 2. empty control group ---------------------------------------
def test_empty_control_group_is_explicit_not_fabricated(impact_db):
    exp_id = build_dataset(
        impact_db,
        control=[],
        control_recovered=[],
        treated=[10000] * 5,
        treated_recovered=[True] * 2 + [False] * 3,
    )
    r = _impact(impact_db, exp_id)
    assert r.computable is False
    assert r.control_group_size == 0
    assert r.treated_group_size == 5
    assert r.control_recovery_rate is None
    assert r.treated_recovery_rate is None
    assert r.incremental_recovery_rate is None
    assert r.incremental_recovery_rate_ci_95 is None
    assert r.incremental_revenue_recovered_paise is None
    assert r.confidence_method == "not_computed"
    assert "control" in r.reason.lower()


def test_empty_treated_group_is_explicit(impact_db):
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 5,
        control_recovered=[True] + [False] * 4,
        treated=[],
        treated_recovered=[],
    )
    r = _impact(impact_db, exp_id)
    assert r.computable is False
    assert r.confidence_method == "not_computed"
    assert "treated" in r.reason.lower()


# --- 3. confidence note / method --------------------------------
def test_confidence_note_small_control_sample(impact_db):
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 12,                              # n < 30
        control_recovered=[True] + [False] * 11,
        treated=[10000] * 40,
        treated_recovered=[True] * 20 + [False] * 20,
    )
    r = _impact(impact_db, exp_id)
    assert r.confidence_method == "newcombe_wilson_95_difference"
    assert "n=12" in r.confidence_note and "< 30" in r.confidence_note
    assert "Newcombe/Wilson" in r.confidence_note
    assert "not a p-value" in r.confidence_note


def test_confidence_note_interval_excludes_zero_for_strong_lift(impact_db):
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 200,
        control_recovered=[True] * 4 + [False] * 196,      # 0.02
        treated=[10000] * 200,
        treated_recovered=[True] * 60 + [False] * 140,     # 0.30
    )
    r = _impact(impact_db, exp_id)
    lo, hi = r.incremental_recovery_rate_ci_95
    assert lo > 0.0                                        # excludes zero
    assert "excludes zero" in r.confidence_note
    assert "n=200" not in r.confidence_note                # >= 30, no small-sample note


def test_confidence_note_interval_includes_zero_for_weak_signal(impact_db):
    # tiny groups, near-identical rates -> interval straddles zero
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 8,
        control_recovered=[True] * 2 + [False] * 6,        # 0.25
        treated=[10000] * 8,
        treated_recovered=[True] * 3 + [False] * 5,        # 0.375
    )
    r = _impact(impact_db, exp_id)
    lo, hi = r.incremental_recovery_rate_ci_95
    assert lo < 0.0 < hi
    assert "includes zero" in r.confidence_note
    assert "directional" in r.confidence_note


# --- 4. proportion_stats unit tests ----------------------------
def test_wilson_known_values():
    lo, hi = wilson_interval(0, 10).low, wilson_interval(0, 10).high
    assert lo == 0.0
    assert round(hi, 3) == 0.278                           # standard Wilson 0/10 @95%
    i = wilson_interval(1, 1)
    assert round(i.low, 3) == 0.207 and i.high == 1.0


def test_newcombe_symmetry_and_zero_diff():
    # identical proportions -> difference interval brackets 0 symmetrically-ish
    i = newcombe_difference_interval(5, 20, 5, 20)
    assert i.low < 0.0 < i.high
    assert not i.excludes_zero


def test_newcombe_matches_hand_computation():
    # p_a = 8/20 = 0.4, p_b = 2/10 = 0.2, diff = 0.2
    i = newcombe_difference_interval(8, 20, 2, 10)
    ci_a = wilson_interval(8, 20)
    ci_b = wilson_interval(2, 10)
    exp_low = 0.2 - math.sqrt((0.4 - ci_a.low) ** 2 + (ci_b.high - 0.2) ** 2)
    exp_high = 0.2 + math.sqrt((ci_a.high - 0.4) ** 2 + (0.2 - ci_b.low) ** 2)
    assert abs(i.low - exp_low) < 1e-12
    assert abs(i.high - exp_high) < 1e-12


# --- 5. filters map to real fields -----------------------------
def test_experiment_id_filter_isolates_the_batch(impact_db):
    exp_a = build_dataset(
        impact_db,
        control=[10000] * 10, control_recovered=[True] * 2 + [False] * 8,
        treated=[10000] * 10, treated_recovered=[True] * 5 + [False] * 5,
    )
    # a second, different experiment in the same DB must not leak in
    r = _impact(impact_db, exp_a)
    assert r.control_group_size == 10 and r.treated_group_size == 10


def test_since_filter_is_a_real_created_at_filter(impact_db):
    from datetime import datetime, timedelta, timezone

    exp_id = build_dataset(
        impact_db,
        control=[10000] * 5, control_recovered=[True] + [False] * 4,
        treated=[10000] * 5, treated_recovered=[True] * 2 + [False] * 3,
    )
    # events were created ~10h ago; a since far in the future excludes them all
    future = datetime.now(timezone.utc) + timedelta(days=1)
    r = AnalyticsService(impact_db).recovery_impact(
        since=future, experiment_id=exp_id
    )
    assert r.control_group_size == 0 and r.treated_group_size == 0
    assert r.computable is False


# --- 6. HTTP endpoint ------------------------------------------
def _get(path: str):
    import asyncio
    import json

    from app.main import app

    p, _, q = path.partition("?")
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": p, "raw_path": p.encode(),
        "query_string": q.encode(), "root_path": "",
        "headers": [(b"host", b"t")], "client": ("t", 1), "server": ("t", 80),
    }
    inbox = [{"type": "http.request", "body": b"", "more_body": False}]
    out = {"body": b""}

    async def recv():
        return inbox.pop(0)

    async def send(m):
        if m["type"] == "http.response.start":
            out["status"] = m["status"]
        elif m["type"] == "http.response.body":
            out["body"] += m.get("body", b"")

    asyncio.run(app(scope, recv, send))
    return out["status"], json.loads(out["body"] or b"null")


def test_http_endpoint_with_experiment_filter(impact_db):
    exp_id = build_dataset(
        impact_db,
        control=[10000] * 10, control_recovered=[True] * 2 + [False] * 8,
        treated=[10000] * 20, treated_recovered=[True] * 8 + [False] * 12,
    )
    sc, body = _get(f"/api/v1/analytics/recovery-impact?experiment_id={exp_id}")
    assert sc == 200
    assert body["computable"] is True
    assert body["control_group_size"] == 10
    assert body["incremental_recovery_rate"] == 0.2
    assert body["incremental_revenue_recovered_paise"] == 40000
    assert body["confidence_method"] == "newcombe_wilson_95_difference"
    assert body["filters"] == {"since": None, "experiment_id": exp_id}


def test_http_endpoint_no_filters_returns_global_batch():
    sc, body = _get("/api/v1/analytics/recovery-impact")
    assert sc == 200
    # the real seeded dataset has both groups
    assert body["control_group_size"] > 0
    assert body["treated_group_size"] > 0
    assert body["computable"] is True
    assert body["incremental_recovery_rate"] == round(
        body["treated_recovery_rate"] - body["control_recovery_rate"], 6
    )
