"""CLI: train the recovery-response model from the current database.

    python -m app.scripts.train_recovery_model
    python -m app.scripts.train_recovery_model --fast          # quick, for tests
    python -m app.scripts.train_recovery_model --out artifacts/ml

Writes:
    artifacts/ml/recovery_response_ml_v1.joblib
    artifacts/ml/recovery_response_latest.joblib
    artifacts/ml/recovery_response_ml_v1.report.json
    artifacts/ml/recovery_response_ml_v1.report.md

Training NEVER runs implicitly -- only via this command.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.database import engine
from app.ml.inference.predictor import reset_default_model
from app.ml.training.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the recovery-response model.")
    parser.add_argument("--out", type=str, default=None, help="Output directory for artifacts.")
    parser.add_argument("--fast", action="store_true", help="Fast config (fewer trees/folds) -- for CI/tests.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for reported precision/recall.")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else None
    with engine.connect() as conn:
        run = run_training(conn, out_dir=out_dir, fast=args.fast, threshold=args.threshold)

    reset_default_model()
    r = run.report
    c = r["test_metrics"]["classification"]
    print("\n=== training complete (SYNTHETIC BENCHMARK) ===")
    print(f"artifact : {run.artifact_path}")
    print(f"report   : {run.report_path}")
    print(f"dataset  : {r['dataset']['n_rows']} rows, {r['dataset']['n_positive']} positive "
          f"({r['dataset']['positive_rate']:.4f}), {r['dataset']['n_recovery_events']} events")
    print(f"selected : {r['selected_model']}  -- {r['selected_reason']}")
    print(f"test     : ROC-AUC={c['roc_auc']}  PR-AUC={c['pr_auc']}  "
          f"log_loss={c['log_loss']}  Brier={c['brier']}")
    cal = r["calibration"]
    print(f"calib    : {cal['method']}  ECE {cal['test_uncalibrated']['expected_calibration_error']} "
          f"-> {cal['test_calibrated']['expected_calibration_error']}")


if __name__ == "__main__":
    main()
