"""Leakage & split-integrity checks specific to the uplift dataset.

Layered on top of the predictive layer's ``validate_training_frame`` (which
already asserts: feature SQL references no ``outcomes``/post-treatment token;
``hours_since_failure >= 0``; ``attempt_number >= 1``; non-negative counts;
ratios in ``[0, 1]``; binary label). This module adds the checks that only make
sense once control events and an explicit arm/propensity are in the frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.ml.features.schema import ALL_FEATURES, LABEL
from app.ml.training.validation import DataQualityError, validate_training_frame
from app.ml.uplift.config import CONTROL_ACTION, TREATMENT_ACTIONS
from app.ml.uplift.dataset.decision_points import uplift_feature_sql

# Columns that describe the decision's *outcome* or its *assignment* and must
# therefore never appear in the model feature matrix.
POST_DECISION_COLUMNS = [
    LABEL,
    "recovered",
    "propensity",
    "raw_action_propensity",
    "exploration",
    "treatment",
    "arm",
    "assignment_strategy",
]


@dataclass
class IntegrityReport:
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.failures.append(msg)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "checks": self.checks,
        }


def validate_uplift_frame(
    frame: pd.DataFrame,
    *,
    split_event_overlap: dict | None = None,
) -> IntegrityReport:
    report = IntegrityReport()

    # 1. reuse the predictive-layer data-quality / leakage validator on the
    #    treatment slice (its action-distribution / temporal checks assume a
    #    real action per row). Raises DataQualityError on a hard failure.
    treat = frame[frame["treatment"].astype(bool)]
    validate_training_frame(treat[ALL_FEATURES + [LABEL]])
    report.checks["predictive_validator"] = "passed on treatment slice"

    # 2. the uplift feature SQL must not touch the outcomes table either.
    # ``recovered_at`` is intentionally NOT forbidden: the shared body reads
    # ``payment.recovered_at`` only to count a customer's *prior* recoveries
    # (filtered ``< as_of``), which is a legitimate point-in-time feature. The
    # forbidden set matches the predictive layer: the ``outcomes`` table and its
    # post-decision columns.
    sql = uplift_feature_sql().lower()
    forbidden = [t for t in ("outcomes", "payment_recovered", "recovered_amount_paise") if t in sql]
    report.checks["uplift_feature_sql_forbidden_tokens"] = forbidden
    if forbidden:
        report.fail(f"uplift feature SQL references forbidden token(s): {forbidden}")

    # 3. no post-decision / assignment column is in the feature list.
    contaminated = [c for c in POST_DECISION_COLUMNS if c in ALL_FEATURES]
    report.checks["post_decision_in_features"] = contaminated
    if contaminated:
        report.fail(f"post-decision columns present in ALL_FEATURES: {contaminated}")

    # 4. arm domain.
    arms = sorted(map(str, frame["arm"].unique()))
    report.checks["arms"] = arms
    unexpected = [a for a in arms if a not in [CONTROL_ACTION, *TREATMENT_ACTIONS]]
    if unexpected:
        report.fail(f"unexpected arm values: {unexpected}")

    # 5. control rows carry the control pseudo-action and no propensity < design.
    ctrl = frame[~frame["treatment"].astype(bool)]
    if not (ctrl["arm"] == CONTROL_ACTION).all():
        report.fail("some control rows do not carry arm == 'none'")
    if (ctrl[LABEL].isna()).any():
        report.fail("control rows contain a null label")

    # 6. one row per recovery event (v1 uses only the first decision).
    dup = int(frame["recovery_event_id"].duplicated().sum())
    report.checks["duplicate_events"] = dup
    if dup:
        report.fail(f"{dup} recovery events appear on more than one row")

    # 7. split integrity: no event shared across train/val/test.
    if split_event_overlap is not None:
        report.checks["split_event_overlap"] = split_event_overlap
        if sum(split_event_overlap.values()):
            report.fail(
                f"train/val/test share recovery events: {split_event_overlap}"
            )

    if not report.passed:
        raise DataQualityError(
            "uplift integrity validation failed:\n  - " + "\n  - ".join(report.failures)
        )
    return report
