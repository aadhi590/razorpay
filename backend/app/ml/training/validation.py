"""Data-quality & leakage validation. Runs BEFORE any model is fit.

Severity model:
  * FAIL  -> raises ``DataQualityError``; training stops. Reserved for things
             that make the model invalid (temporal violations, non-binary
             label, a feature sourced from ``outcomes``, impossible values,
             train/test contamination).
  * WARN  -> recorded in the report, training continues (thin action cells,
             extreme imbalance, high-null nullable features, duplicates).

Nothing is silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.ml.config import (
    MAX_UNEXPECTED_NULL_FRACTION,
    MIN_EXAMPLES_PER_ACTION,
    NULLABLE_NUMERIC_FEATURES,
    TARGET_RATE_WARN_HIGH,
    TARGET_RATE_WARN_LOW,
)
from app.ml.features.point_in_time import PointInTimeFeatureExtractor
from app.ml.features.schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_LEAKAGE_SOURCES,
    LABEL,
    NUMERIC_FEATURES,
)


class DataQualityError(RuntimeError):
    """Severe data-quality / leakage problem -- training must not proceed."""


@dataclass
class DataQualityReport:
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "checks": self.checks,
        }


# --- individual checks --------------------------------------------------

def _check_leakage_sql(report: DataQualityReport) -> None:
    for mode in ("training", "inference"):
        sql = PointInTimeFeatureExtractor.feature_sql(mode).lower()
        hits = [tok for tok in FORBIDDEN_LEAKAGE_SOURCES if tok.lower() in sql]
        report.checks[f"feature_sql_forbidden_tokens_{mode}"] = hits
        if hits:
            report.fail(
                f"feature SQL ({mode}) references forbidden leakage source(s): "
                f"{hits}"
            )


def _check_schema(report: DataQualityReport, df: pd.DataFrame) -> None:
    missing = [c for c in ALL_FEATURES + [LABEL] if c not in df.columns]
    report.checks["missing_columns"] = missing
    if missing:
        report.fail(f"dataset is missing required columns: {missing}")


def _check_label(report: DataQualityReport, df: pd.DataFrame) -> None:
    if LABEL not in df.columns:
        return
    vals = set(pd.unique(df[LABEL].dropna()))
    report.checks["label_values"] = sorted(map(int, vals)) if vals else []
    if not vals <= {0, 1}:
        report.fail(f"label '{LABEL}' is not binary: found {sorted(vals)}")
    if df[LABEL].isna().any():
        report.fail(f"label '{LABEL}' contains nulls")
    pos_rate = float(df[LABEL].mean()) if len(df) else 0.0
    report.checks["positive_rate"] = round(pos_rate, 6)
    if pos_rate < TARGET_RATE_WARN_LOW or pos_rate > TARGET_RATE_WARN_HIGH:
        report.warn(
            f"target is extremely imbalanced (positive rate {pos_rate:.4f})"
        )


def _check_missing(report: DataQualityReport, df: pd.DataFrame) -> None:
    null_frac = df[ALL_FEATURES].isna().mean().round(6).to_dict()
    report.checks["null_fraction"] = null_frac
    for feat, frac in null_frac.items():
        if feat in NULLABLE_NUMERIC_FEATURES:
            continue
        if frac > 0:
            report.fail(
                f"non-nullable feature '{feat}' has {frac:.4f} nulls"
            )
    for feat in NULLABLE_NUMERIC_FEATURES:
        if null_frac.get(feat, 0.0) > MAX_UNEXPECTED_NULL_FRACTION:
            report.warn(
                f"nullable feature '{feat}' is {null_frac[feat]:.2f} null "
                "(imputed at train & inference)"
            )


def _check_impossible_values(report: DataQualityReport, df: pd.DataFrame) -> None:
    problems: dict[str, int] = {}

    def flag(name: str, mask: pd.Series) -> None:
        c = int(mask.sum())
        if c:
            problems[name] = c

    flag("hours_since_failure_negative", df["hours_since_failure"] < 0)
    flag("attempt_number_lt_1", df["attempt_number"] < 1)
    flag("amount_paise_non_positive", df["amount_paise"] <= 0)
    flag("subscription_amount_non_positive", df["subscription_amount"] <= 0)
    flag("subscription_tenure_negative", df["subscription_tenure_days"] < 0)
    flag("success_ratio_out_of_range",
         (df["cust_prior_success_ratio"] < 0) | (df["cust_prior_success_ratio"] > 1))
    flag("failure_ratio_out_of_range",
         (df["cust_prior_failure_ratio"] < 0) | (df["cust_prior_failure_ratio"] > 1))
    flag("negative_counts",
         (df[[c for c in NUMERIC_FEATURES if c.startswith(("cust_prior", "cust_failures"))]] < 0).any(axis=1))

    report.checks["impossible_values"] = problems
    if problems:
        report.fail(f"impossible feature values detected: {problems}")


def _check_temporal_ordering(report: DataQualityReport, df: pd.DataFrame) -> None:
    # hours_since_failure >= 0 already covers executed_at >= failed_at.
    bad = int((df["hours_since_failure"] < 0).sum())
    report.checks["temporal_violations"] = bad
    if bad:
        report.fail(f"{bad} rows have decision time before the payment failure")


def _check_duplicates(report: DataQualityReport, df: pd.DataFrame) -> None:
    exact = int(df.duplicated(subset=ALL_FEATURES + [LABEL]).sum())
    report.checks["exact_duplicate_rows"] = exact
    if exact:
        report.warn(
            f"{exact} exact-duplicate feature+label rows (kept -- they are "
            "distinct observed interventions)"
        )


def _check_action_distribution(report: DataQualityReport, df: pd.DataFrame) -> None:
    counts = df["action"].value_counts().to_dict()
    report.checks["action_counts"] = {str(k): int(v) for k, v in counts.items()}
    for action, n in counts.items():
        if n < MIN_EXAMPLES_PER_ACTION:
            report.warn(
                f"action '{action}' has only {n} examples "
                f"(< {MIN_EXAMPLES_PER_ACTION}); estimates for it are unreliable"
            )


def _check_categorical_domains(report: DataQualityReport, df: pd.DataFrame) -> None:
    report.checks["categorical_domains"] = {
        c: sorted(map(str, pd.unique(df[c].dropna()))) for c in CATEGORICAL_FEATURES
    }


# --- entry point -----------------------------------------------------

def validate_training_frame(
    df: pd.DataFrame, *, split_contamination: dict | None = None
) -> DataQualityReport:
    report = DataQualityReport()

    _check_leakage_sql(report)
    _check_schema(report, df)
    if not report.passed:
        # cannot run value checks without the required columns
        raise DataQualityError(
            "data-quality validation failed:\n  - "
            + "\n  - ".join(report.failures)
        )
    _check_label(report, df)
    _check_missing(report, df)
    _check_impossible_values(report, df)
    _check_temporal_ordering(report, df)
    _check_duplicates(report, df)
    _check_action_distribution(report, df)
    _check_categorical_domains(report, df)

    if split_contamination is not None:
        report.checks["split_contamination"] = split_contamination
        overlap = sum(v for v in split_contamination.values())
        if overlap:
            report.fail(
                f"train/val/test share recovery events: {split_contamination}"
            )

    if not report.passed:
        raise DataQualityError(
            "data-quality validation failed:\n  - "
            + "\n  - ".join(report.failures)
        )
    return report
