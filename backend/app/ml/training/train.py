"""Training driver: DB -> dataset -> validate -> split -> compare -> calibrate
-> evaluate -> select -> artifact.

Invoked only by ``app.scripts.train_recovery_model`` (a CLI). Never at import
or API startup.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ml.config import (
    ARTIFACT_DIR,
    CALIBRATION_CV_FOLDS,
    ISOTONIC_MIN_POSITIVES,
    MODEL_NAME,
    MODEL_VERSION,
    RANDOM_STATE,
)
from app.ml.datasets.builder import build_training_dataset
from app.ml.datasets.coverage import action_coverage
from app.ml.evaluation.metrics import calibration_metrics, evaluate_predictions
from app.ml.features.schema import ALL_FEATURES, CATEGORICAL_FEATURES, LABEL, NUMERIC_FEATURES
from app.ml.models.artifact import ModelArtifact, save_artifact
from app.ml.models.registry import build_candidate_estimators
from app.ml.monitoring.drift import build_feature_baseline
from app.ml.preprocessing.pipeline import build_preprocessor
from app.ml.training.splitting import grouped_chronological_split
from app.ml.training.validation import validate_training_frame

_PR_AUC_TIE = 0.02  # models within this of the best PR-AUC are "equivalent"


@dataclass
class TrainingRun:
    artifact_path: Path
    report_path: Path
    artifact: ModelArtifact
    report: dict = field(default_factory=dict)


def _calibration_method(n_positive_train: int) -> str:
    return "isotonic" if n_positive_train >= ISOTONIC_MIN_POSITIVES else "sigmoid"


def _fit_calibrated(base_pipeline: Pipeline, X, y, method: str, cv: int):
    model = CalibratedClassifierCV(
        estimator=clone(base_pipeline), method=method, cv=cv
    )
    model.fit(X, y)
    return model


def _logreg_coefficients(X: pd.DataFrame, y: pd.Series) -> list[dict]:
    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    coefs = pipe.named_steps["clf"].coef_[0]
    rows = [
        {"feature": n, "coefficient": round(float(c), 6)}
        for n, c in zip(names, coefs)
    ]
    rows.sort(key=lambda r: -abs(r["coefficient"]))
    return rows


def run_training(
    conn,
    *,
    out_dir: Path | None = None,
    fast: bool = False,
    threshold: float = 0.5,
) -> TrainingRun:
    out_dir = Path(out_dir) if out_dir is not None else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = 3 if fast else CALIBRATION_CV_FOLDS
    perm_repeats = 3 if fast else 10

    # 1. dataset -----------------------------------------------------
    dataset = build_training_dataset(conn)
    frame = dataset.frame
    coverage = action_coverage(frame)

    # 2. split (before value validation so contamination can be checked) --
    split = grouped_chronological_split(frame)
    contamination = {
        k: v for k, v in split.report["event_overlap"].items() if v
    }

    # 3. data-quality / leakage validation (FAILS LOUDLY) -----------
    dq = validate_training_frame(frame, split_contamination=split.report["event_overlap"])

    train_df, val_df, test_df = split.frames(frame)
    Xtr, ytr = train_df[ALL_FEATURES], train_df[LABEL].astype(int)
    Xva, yva = val_df[ALL_FEATURES], val_df[LABEL].astype(int)
    Xte, yte = test_df[ALL_FEATURES], test_df[LABEL].astype(int)

    method = _calibration_method(int(ytr.sum()))

    # 4. model comparison on validation ---------------------------
    candidates = build_candidate_estimators()
    if fast:
        candidates["hist_gb"]["estimator"].named_steps["clf"].set_params(max_iter=60)
        candidates["random_forest"]["estimator"].named_steps["clf"].set_params(n_estimators=80)

    comparison: dict[str, dict] = {}
    for name, spec in candidates.items():
        model = _fit_calibrated(spec["estimator"], Xtr, ytr, method, cv)
        p_val = model.predict_proba(Xva)[:, 1]
        val_eval = evaluate_predictions(val_df, p_val, threshold)
        comparison[name] = {
            "algorithm": spec["algorithm"],
            "hyperparameters": spec["hyperparameters"],
            "validation": val_eval,
        }

    # 5. selection: best validation PR-AUC, break ties by calibration --
    def pr_auc(name: str) -> float:
        v = comparison[name]["validation"]["classification"]["pr_auc"]
        return v if v is not None else -1.0

    def ece(name: str) -> float:
        v = comparison[name]["validation"]["calibration"]["expected_calibration_error"]
        return v if v is not None else 1.0

    best_pr = max(pr_auc(n) for n in comparison)
    contenders = [n for n in comparison if best_pr - pr_auc(n) <= _PR_AUC_TIE]
    selected_name = min(contenders, key=ece)
    selected_reason = (
        f"best validation PR-AUC={best_pr:.4f}; among models within "
        f"{_PR_AUC_TIE} PR-AUC ({contenders}) selected '{selected_name}' for "
        f"lowest validation ECE ({ece(selected_name):.4f})"
    )

    # 6. refit selected on train+val, evaluate on untouched test ----
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    Xtv, ytv = trainval_df[ALL_FEATURES], trainval_df[LABEL].astype(int)
    final_model = _fit_calibrated(
        candidates[selected_name]["estimator"], Xtv, ytv, method, cv
    )

    p_test = final_model.predict_proba(Xte)[:, 1]
    test_eval = evaluate_predictions(test_df, p_test, threshold)

    # calibration: uncalibrated vs calibrated on test (shows calibration helps)
    uncal = clone(candidates[selected_name]["estimator"])
    uncal.fit(Xtv, ytv)
    p_test_uncal = uncal.predict_proba(Xte)[:, 1]
    calibration_block = {
        "method": method,
        "cv_folds": cv,
        "isotonic_rejected_reason": (
            f"only {int(ytr.sum())} positives in train "
            f"(< {ISOTONIC_MIN_POSITIVES}); isotonic would overfit"
        )
        if method == "sigmoid"
        else None,
        "test_uncalibrated": calibration_metrics(yte, p_test_uncal),
        "test_calibrated": calibration_metrics(yte, p_test),
    }

    # 7. interpretability -----------------------------------------
    perm = permutation_importance(
        final_model, Xte, yte, scoring="average_precision",
        n_repeats=perm_repeats, random_state=RANDOM_STATE,
    )
    perm_rows = sorted(
        (
            {
                "feature": f,
                "importance_mean": round(float(m), 6),
                "importance_std": round(float(s), 6),
            }
            for f, m, s in zip(ALL_FEATURES, perm.importances_mean, perm.importances_std)
        ),
        key=lambda r: -r["importance_mean"],
    )
    interpretability = {
        "permutation_importance_test": perm_rows,
        "logistic_regression_coefficients": _logreg_coefficients(Xtv, ytv),
        "note": (
            "permutation importance is on the untouched test set; the logistic "
            "coefficient table is a standalone interpretable reference model "
            "(sign = direction of effect on recovery probability)."
        ),
    }

    # 8. feature baseline for monitoring -------------------------
    baseline = build_feature_baseline(frame)

    # 9. artifact -----------------------------------------------
    artifact = ModelArtifact(
        estimator=final_model,
        algorithm=candidates[selected_name]["algorithm"],
        selected_reason=selected_reason,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        all_features=ALL_FEATURES,
        label=LABEL,
        dataset={
            "dataset_version": dataset.dataset_version,
            "feature_version": dataset.feature_version,
            **dataset.stats,
        },
        split=split.report,
        data_quality=dq.as_dict(),
        action_coverage=coverage.as_dict(),
        model_comparison={
            n: {
                "algorithm": comparison[n]["algorithm"],
                "val_roc_auc": comparison[n]["validation"]["classification"]["roc_auc"],
                "val_pr_auc": comparison[n]["validation"]["classification"]["pr_auc"],
                "val_log_loss": comparison[n]["validation"]["classification"]["log_loss"],
                "val_brier": comparison[n]["validation"]["classification"]["brier"],
                "val_ece": comparison[n]["validation"]["calibration"]["expected_calibration_error"],
            }
            for n in comparison
        },
        calibration=calibration_block,
        validation_metrics=comparison[selected_name]["validation"],
        test_metrics=test_eval,
        interpretability=interpretability,
        feature_baseline=baseline,
        config={
            "random_state": RANDOM_STATE,
            "split_fractions": list(split.report["fractions"]),
            "calibration_method": method,
            "calibration_cv_folds": cv,
            "decision_threshold": threshold,
            "selected_model": selected_name,
            "fast_mode": fast,
        },
    )

    artifact_path = out_dir / f"{MODEL_NAME}_{MODEL_VERSION}.joblib"
    save_artifact(artifact, artifact_path)

    report = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "created_at": artifact.created_at,
        "synthetic_benchmark": True,
        "synthetic_warning": (
            "Trained on synthetic data from a generator with a known outcome "
            "process. Metrics are a SYNTHETIC BENCHMARK, not real-world "
            "predictive performance; the model may be recovering the "
            "simulator's rules."
        ),
        "dataset": artifact.dataset,
        "action_coverage": artifact.action_coverage,
        "data_quality": artifact.data_quality,
        "split": artifact.split,
        "model_comparison": artifact.model_comparison,
        "selected_model": selected_name,
        "selected_reason": selected_reason,
        "calibration": artifact.calibration,
        "validation_metrics": artifact.validation_metrics,
        "test_metrics": artifact.test_metrics,
        "interpretability": {
            "permutation_importance_top10": interpretability["permutation_importance_test"][:10],
            "logreg_coefficients_top15": interpretability["logistic_regression_coefficients"][:15],
        },
        "config": artifact.config,
    }
    report_path = out_dir / f"{MODEL_NAME}_{MODEL_VERSION}.report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / f"{MODEL_NAME}_{MODEL_VERSION}.report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )

    return TrainingRun(
        artifact_path=artifact_path,
        report_path=report_path,
        artifact=artifact,
        report=report,
    )


def _render_markdown(r: dict) -> str:
    c = r["test_metrics"]["classification"]
    cal = r["calibration"]
    lines = [
        f"# {r['model_name']} {r['model_version']} -- training report",
        "",
        f"_Created: {r['created_at']}_",
        "",
        "> **SYNTHETIC BENCHMARK.** " + r["synthetic_warning"],
        "",
        "## Dataset",
        f"- rows: {r['dataset']['n_rows']}  |  recovery events: {r['dataset']['n_recovery_events']}  |  customers: {r['dataset']['n_customers']}",
        f"- positives: {r['dataset']['n_positive']} ({r['dataset']['positive_rate']:.4f})",
        f"- action counts: {r['dataset']['action_counts']}",
        f"- as_of range: {r['dataset']['as_of_min']} .. {r['dataset']['as_of_max']}",
        "",
        "## Split (grouped by recovery event, chronological)",
    ]
    for k in ("train", "validation", "test"):
        s = r["split"][k]
        lines.append(
            f"- {k}: {s['rows']} rows / {s['events']} events / {s['positives']} pos "
            f"({s['positive_rate']}), {s['as_of_min']} .. {s['as_of_max']}"
        )
    lines += [
        f"- event overlap: {r['split']['event_overlap']}",
        f"- customer overlap: {r['split']['customer_overlap']}",
        "",
        "## Model comparison (validation)",
        "| model | ROC-AUC | PR-AUC | log loss | Brier | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in r["model_comparison"].items():
        lines.append(
            f"| {name} | {m['val_roc_auc']} | {m['val_pr_auc']} | "
            f"{m['val_log_loss']} | {m['val_brier']} | {m['val_ece']} |"
        )
    lines += [
        "",
        f"**Selected: {r['selected_model']}** -- {r['selected_reason']}",
        "",
        "## Test metrics (untouched)",
        f"- n={c['n']}  positives={c['n_positive']} ({c['positive_rate']})",
        f"- ROC-AUC={c['roc_auc']}  PR-AUC={c['pr_auc']}",
        f"- log loss={c['log_loss']}  Brier={c['brier']}",
        f"- precision={c['precision']}  recall={c['recall']}  F1={c['f1']}  (threshold {c['threshold']})",
        f"- confusion matrix: {c['confusion_matrix']}",
        "",
        "## Calibration (test)",
        f"- method: {cal['method']}  ({cal.get('isotonic_rejected_reason') or 'isotonic'})",
        f"- uncalibrated: Brier={cal['test_uncalibrated']['brier']}  ECE={cal['test_uncalibrated']['expected_calibration_error']}",
        f"- calibrated:   Brier={cal['test_calibrated']['brier']}  ECE={cal['test_calibrated']['expected_calibration_error']}",
        "",
        "## Per-action (test)",
    ]
    for action, m in r["test_metrics"]["per_action"].items():
        lines.append(f"- {action}: n={m['n']} pos={m['n_positive']} "
                     f"observed={m['observed_rate']} predicted={m['mean_predicted']} "
                     f"{m.get('note', '')}")
    lines += ["", "## Top permutation importances (test)"]
    for row in r["interpretability"]["permutation_importance_top10"]:
        lines.append(f"- {row['feature']}: {row['importance_mean']:.5f} ± {row['importance_std']:.5f}")
    return "\n".join(lines) + "\n"
